import sys
import types

import numpy as np
import pytest

from lawm_deploy.cell import SharpaHand, MotionSettings, move_duration, camera_poses
from lawm_deploy.hardware_contract import HardwareError, hand_contract, transform
from lawm_deploy.safety import SafetySettings, check


def test_sharpa_units_mapping_and_read_only_connection(monkeypatch):
    names, _ = hand_contract()
    ok = types.SimpleNamespace(code=0, message='OK')
    calls = []
    wave = types.SimpleNamespace(
        get_device_info=lambda: types.SimpleNamespace(hand_side='right'),
        get_joint_position_degree=lambda: (ok, np.arange(22, dtype=float)),
        set_joint_position=lambda q, blocking: calls.append(('write', q, blocking)) or ok,
    )
    for name in ('set_control_mode','set_speed_coeff','set_current_coeff','set_control_source'):
        setattr(wave, name, lambda value, n=name: calls.append((n, value)) or ok)
    manager = types.SimpleNamespace(
        get_all_devices=lambda: [types.SimpleNamespace(sn='test', device_type='hand', hand_side='right')],
        connect=lambda serial: wave, disconnect=lambda serial: calls.append(('disconnect',serial)))
    sdk = types.SimpleNamespace(SharpaWaveManager=types.SimpleNamespace(get_instance=lambda:manager),
                               DeviceType=types.SimpleNamespace(HAND='hand'), HandSide=types.SimpleNamespace(RIGHT='right'),
                               ControlMode=types.SimpleNamespace(POSITION='position'), ControlSource=types.SimpleNamespace(SDK='sdk'))
    monkeypatch.setitem(sys.modules, 'sharpa', sdk)
    hand = SharpaHand({'serial':'test'}, names[::-1])
    np.testing.assert_allclose(hand.read(), np.deg2rad(np.arange(22)[::-1]))
    assert not calls
    with pytest.raises(HardwareError, match='disabled'): hand.write(np.zeros(22))
    hand.enable()
    q = np.linspace(0,0.1,22)
    hand.write(q)
    np.testing.assert_allclose(calls[-1][1], q[::-1])
    with pytest.raises(HardwareError): hand.write(np.full(22,np.nan))
    with pytest.raises(HardwareError): hand.write(np.full(22,10))
    hand.close()
    assert calls[-1] == ('disconnect','test')


def test_minimum_jerk_finger_limits():
    m = MotionSettings(.02,.1,.15,.3,.5,100,0,1.0)
    delta = .2
    duration = move_duration(m,np.eye(4),np.eye(4),np.zeros(22),np.full(22,delta))
    assert 1.875 * delta/duration <= m.finger_speed + 1e-12
    assert 5.774 * delta/duration**2 <= m.finger_accel + 1e-12


def test_nan_and_malformed_commands_are_not_safe():
    s = SafetySettings(60,.5,30,60,30,15,[-1,-1,-1],[1,1,1],-.1)
    args = dict(target=np.eye(4), q_hand=np.zeros(22), q_hand_now=np.zeros(22), flange_now=np.eye(4),
                q_arm_now=np.zeros(6),q_arm_ik=np.zeros(6),fit_info={'mean_residual_mm':0},hand_points=np.zeros((5,3)))
    assert check(s, **args) is None
    for field in ('target','q_hand','q_hand_now','q_arm_now','q_arm_ik','hand_points'):
        assert check(s, **{**args,field:np.full_like(args[field],np.nan)}) is not None
    assert check(s, **{**args,'fit_info':{'mean_residual_mm':np.nan}}) is not None
    with pytest.raises(HardwareError): transform(np.zeros((4,4)))


def test_color_calibration_rejected(tmp_path):
    f = tmp_path/'calibration.yaml'
    f.write_text('frame: color_optical\nT_base_camera: {}\n')
    with pytest.raises(HardwareError, match='depth_optical'):
        camera_poses({'calibration':str(f),'devices':[{'serial':'a'}]},True)
