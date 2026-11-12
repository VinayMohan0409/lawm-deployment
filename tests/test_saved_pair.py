import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lawm_deploy import config
from lawm_deploy.hardware_contract import HardwareError, hand_contract
from lawm_deploy.robot_model import ARM_JOINTS
from lawm_deploy.saved_pair import load_pair, read_sample


@pytest.fixture
def sample(tmp_path):
    cfg = config.load(config.REPO / 'configs/real.yaml')
    cfg['perception'].update(use_box=False, voxel_m=0., min_scene_points=1)
    names = hand_contract()[0]
    robot = SimpleNamespace(hand_joint_names=names, set=lambda *a: None,
                            flange_pose=lambda: np.eye(4), root_pose=lambda: np.eye(4),
                            surface=lambda *a: np.zeros((0, 3)))
    hand = SimpleNamespace(points=lambda *a: np.zeros((1024, 3), dtype=np.float32))
    serials = [d['serial'] for d in cfg['cell']['cameras']['devices']]
    calibration = tmp_path / 'calibration.yaml'
    calibration.write_text(yaml.safe_dump(dict(frame='depth_optical', T_base_camera={s: np.eye(4).tolist() for s in serials})))
    cfg['cell']['cameras']['calibration'] = str(calibration)
    for label in ('start', 'goal'):
        p = tmp_path / label
        p.mkdir()
        def write(name, data):
            (p / name).write_text(json.dumps(data))
        write('frames.json', dict(point_cloud_frame='UR_controller_base'))
        write('capture_config.json', cfg)
        write('joint_state.json', dict(angle_units='radians', stationarity={'passed': True},
            robot_ip=cfg['cell']['robot_ip'], hand_serial=cfg['cell']['sharpa']['serial'],
            arm_joint_names=list(ARM_JOINTS), q_arm_rad=[0]*6,
            hand_joint_names=names[::-1], q_hand_rad=list(np.arange(22, dtype=float)[::-1])))
        arrays = {}
        for i, role in enumerate(('ego', 'third')):
            (p / role).mkdir()
            write(f'{role}/camera.json', dict(serial=serials[i], depth_settings_at_start={'emitter_enabled': 1.0}))
            arrays.update({f'raw_depth_z16_{i}': np.full((2, 2), 1000, dtype=np.uint16),
                           f'depth_scale_m_{i}': .001, f'K_depth_{i}': np.eye(3), f'T_base_depth_{i}': np.eye(4)})
        np.savez(p / 'scene_raw.npz', **arrays)
    return tmp_path, cfg, robot, hand


def test_sample_reorders_joints_preserves_raw_and_never_becomes_live(sample):
    path, cfg, robot, hand = sample
    obs = read_sample(path / 'goal', cfg, robot, hand)
    np.testing.assert_array_equal(obs.q_hand, np.arange(22))
    assert obs.coords.shape == (6144, 3)
    assert obs.raw_depth[0][0].dtype == np.uint16
    assert not obs.info['live_observation']
    assert obs.info['source_capture_config'] == cfg
    saved = []
    session = SimpleNamespace(finished=True, cfg=cfg, robot=robot, hand=hand, current='stale',
                              save=lambda name, obs: saved.append(name))
    # No cell attribute: loading cannot query or move a device.
    load_pair(session, path)
    assert session.current is None and saved == ['goal', 'start']
    session.finished = False
    with pytest.raises(HardwareError, match='Finish'): load_pair(session, path)


@pytest.mark.parametrize('change', ['frame', 'mount', 'calibration', 'serial', 'stationarity', 'nan'])
def test_incompatible_saved_observation_rejected(sample, change):
    path, cfg, robot, hand = sample
    cfg = copy.deepcopy(cfg)
    if change == 'mount':
        cfg['mount']['xyz_m'][2] += .01
    elif change == 'calibration':
        p = path / 'calibration.yaml'; c = yaml.safe_load(p.read_text())
        c['T_base_camera'][cfg['cell']['cameras']['devices'][0]['serial']][0][3] = .02
        p.write_text(yaml.safe_dump(c))
    elif change == 'serial':
        cfg['cell']['cameras']['devices'][0]['serial'] = 'wrong'
    else:
        p = path / 'goal' / ('frames.json' if change == 'frame' else 'joint_state.json')
        d = json.loads(p.read_text())
        if change == 'frame': d['point_cloud_frame'] = 'base_link'
        if change == 'stationarity': d['stationarity']['passed'] = False
        if change == 'nan': d['q_arm_rad'][0] = float('nan')
        p.write_text(json.dumps(d))
    with pytest.raises(HardwareError): read_sample(path / 'goal', cfg, robot, hand)
