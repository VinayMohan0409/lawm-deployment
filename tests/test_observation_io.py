import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lawm_deploy import config
from lawm_deploy.cell import RealSense
from lawm_deploy.hardware_contract import HardwareError
from lawm_deploy.model_contract import validate_model_contract, timing_metadata
from lawm_deploy.observation_io import back_project, export_clouds, save_observation


@pytest.mark.parametrize('reject_setting', [False, True])
def test_camera_profile_is_applied_logged_and_verified(monkeypatch, reject_setting):
    values = dict(visual_preset=3., emitter_enabled=0., enable_auto_exposure=0., laser_power=50., exposure=8500., gain=16.)
    def set_option(key, value):
        if not (reject_setting and key == 'laser_power'):
            values[key] = value
    sensor = SimpleNamespace(supports=lambda key: key in values, set_option=set_option,
                             get_option=values.__getitem__, get_depth_scale=lambda: .001)
    intrinsics = SimpleNamespace(fx=1., fy=1., ppx=0., ppy=0., coeffs=[0]*5, width=2, height=2, model='none')
    stream = SimpleNamespace(get_intrinsics=lambda: intrinsics)
    stream.as_video_stream_profile = lambda: stream
    profile = SimpleNamespace(get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: sensor), get_stream=lambda _: stream)
    stops = []
    pipe = SimpleNamespace(start=lambda _: profile, wait_for_frames=lambda _: None, stop=lambda: stops.append(True))
    rs = SimpleNamespace(option=SimpleNamespace(**{k:k for k in values}), stream=SimpleNamespace(depth=0),
                         format=SimpleNamespace(z16=0), pipeline=lambda: pipe,
                         config=lambda: SimpleNamespace(enable_device=lambda _: None, enable_stream=lambda *a: None))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', rs)
    args = (['camera'], [np.eye(4)], 2, 2, 30, 1, 0)
    kwargs = dict(emitter='on', sensor_options={'enable_auto_exposure': 1., 'laser_power': 150.})
    if reject_setting:
        with pytest.raises(HardwareError, match='laser_power'):
            RealSense(*args, **kwargs)
        assert stops  # failed setup releases the camera
    else:
        cameras = RealSense(*args, **kwargs)
        logged = cameras.metadata[0]['sensor_settings_at_start']
        assert logged['visual_preset'] == 0 and logged['emitter_enabled'] == 1
        assert logged['enable_auto_exposure'] == 1 and logged['laser_power'] == 150
        cameras.close()


def test_projection_uses_y_focal_length_and_camera_to_base():
    K = np.array([[2.,0,0],[0,4.,0],[0,0,1]])
    T = np.eye(4); T[:3,3] = [1,2,3]
    depth = np.array([[0.,0.],[0.,2.]])
    np.testing.assert_allclose(back_project(depth,K,T,0,np.inf), [[2,2.5,5]])


def test_raw_capture_retains_native_frames_before_median():
    raw = [np.array([[1000,0]],dtype=np.uint16), np.array([[1001,0]],dtype=np.uint16)]
    class Pipe:
        index = -1
        def poll_for_frames(self): return False
        def wait_for_frames(self, timeout):
            self.index += 1
            i = self.index
            frame = SimpleNamespace(get_frame_number=lambda:i+1, get_timestamp=lambda:100+i,
                get_frame_timestamp_domain=lambda:'hardware_clock', get_data=lambda:raw[i])
            return SimpleNamespace(get_depth_frame=lambda:frame)
    cams = RealSense.__new__(RealSense)
    cams.cams = [(Pipe(),0.00025,np.eye(3),np.eye(4))]
    cams._numbers, cams.n_frames, cams.timeout_ms = {}, 2, 2000
    views = cams.views()
    np.testing.assert_array_equal(cams.last_raw_depth[0][0],np.stack(raw))
    assert cams.last_raw_depth[0][0].dtype == np.uint16
    assert views[0][0][0,0] == pytest.approx(0.250125)
    assert views[0][0][0,1] == 0
    assert [f['number'] for f in cams.last_frames[0]] == [1,2]


def test_archive_export_preserves_raw_exact_inputs_and_metadata(tmp_path):
    depth = np.array([[.250125,0]],dtype=np.float32)
    raw = np.array([[[1000,0]],[[1001,0]]],dtype=np.uint16)
    K,T = np.eye(3),np.eye(4)
    coords = np.array([[.123456, .654321, .8]], dtype=np.float32)
    obs = SimpleNamespace(coords=coords,segment=np.array([0],dtype=np.int8),q_arm=np.zeros(6),
        q_hand=np.zeros(22),flange=T,info={'camera_metadata':[{'serial':'test'}]},
        views=[(depth,K,T)],raw_depth=[(raw,.00025)])
    source=tmp_path/'start.npz'
    save_observation(source,obs,{'model':{'spin_deg':315}},{'checkpoint_sha256':'test'})
    with np.load(source,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['depth_median_m_0'],depth)
        np.testing.assert_array_equal(a['sensor_depth_z16_0'],raw)
        assert a['depth_0'][0,0] == 250
        assert json.loads(str(a['timing_json']))['training_step_s'] == .2
    output=tmp_path/'export'
    manifest=export_clouds(source,output)
    assert manifest['model_input_rotation_applied'] is False
    assert manifest['cameras'][0]['sensor_z16_available'] is True
    assert manifest['cameras'][0]['source'] == 'full_precision_median_depth'
    assert (output/'observation.npz').read_bytes() == source.read_bytes()
    np.testing.assert_allclose(np.loadtxt(output/'model_input_base.ply',skiprows=7).reshape(-1,3),coords)
    with pytest.raises(FileExistsError): export_clouds(source,output)


def test_legacy_export_does_not_invent_raw_frames(tmp_path):
    source=tmp_path/'old.npz'
    np.savez(source,depth_0=np.array([[1000]],dtype=np.uint16),cam_0=np.r_[np.eye(3).ravel(),np.eye(4).ravel()])
    result=export_clouds(source,tmp_path/'legacy')
    assert not result['cameras'][0]['sensor_z16_available']
    assert result['cameras'][0]['source'] == 'legacy_rounded_millimetre_median'
    np.testing.assert_allclose(np.loadtxt(tmp_path/'legacy/camera_0_base.ply',skiprows=7),[0,0,1])


def test_preflight_export_needs_no_model_input(tmp_path):
    source=tmp_path/'depth_001.npz'
    np.savez(source,depth_0=np.array([[1.]],dtype=np.float32),K_0=np.eye(3),T_base_depth_0=np.eye(4))
    result=export_clouds(source,tmp_path/'preflight')
    assert 'model_points' not in result
    assert result['cameras'][0]['source'] == 'saved_median_depth_metres'


def test_owner_contract_freezes_model_settings_but_not_execution():
    cfg=config.load(config.REPO/'configs/real.yaml')
    cfg['cell']['motion']['speed'] = .001
    cfg['perception']['box_lo'] = [-1,-1,-1]
    validate_model_contract(cfg)
    for section,key,value in [('model','spin_deg',0),('model','training_up',[0,0,1]),
                               ('perception','n_hand',512),('perception','n_scene',1024)]:
        bad=copy.deepcopy(cfg); bad[section][key]=value
        with pytest.raises(ValueError): validate_model_contract(bad)
    assert timing_metadata()['execution_duration_is_training_step'] is False
