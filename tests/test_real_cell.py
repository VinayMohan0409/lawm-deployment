"""Fake hardware tests. No robot network, SDK discovery or USB access."""
import copy
import sys
import types

import numpy as np
import pytest

from lawm_deploy import config
from lawm_deploy.hardware_contract import HardwareError, hand_contract
from lawm_deploy import cell as backend


class Clock:
    def __init__(self):
        self.now = 100.0
    def monotonic(self):
        return self.now
    def time_ns(self):
        return int(self.now * 1e9)
    def sleep(self, seconds):
        self.now += seconds


class FakeReceive:
    instances = []
    def __init__(self, *args):
        self.q = np.zeros(6)
        self.tcp = [0.49, 0.13, 0.49, np.pi, 0, 0]
        self.protective, self.connected, self.stamp = False, True, 0.0
        self.__class__.instances.append(self)
    def getActualQ(self): return self.q
    def getActualQd(self): return np.zeros(6)
    def getActualTCPPose(self): return self.tcp
    def getActualTCPSpeed(self): return np.zeros(6)
    def isProtectiveStopped(self): return self.protective
    def isEmergencyStopped(self): return False
    def isConnected(self): return self.connected
    def getTimestamp(self):
        self.stamp += 0.008
        return self.stamp
    def getRuntimeState(self): return 2 if any(c.connected for c in FakeControl.instances) else 1
    def getSpeedScaling(self): return 1
    def getTargetSpeedFraction(self): return 1
    def disconnect(self): self.connected = False


class FakeControl:
    instances = []
    reject_move = False
    protective_on_move = False
    def __init__(self, *args):
        self.calls, self.connected = [], True
        self.rx = FakeReceive.instances[-1]
        self.__class__.instances.append(self)
    def getTCPOffset(self): return [0] * 6
    def isPoseWithinSafetyLimits(self, x): return True
    def isJointsWithinSafetyLimits(self, q): return True
    def getInverseKinematicsHasSolution(self, x, q): return True
    def getInverseKinematics(self, x, q): return list(q)
    def setWatchdog(self, hz): self.calls.append('watchdog'); return True
    def kickWatchdog(self): return True
    def moveL(self, pose, speed, accel, asynchronous):
        self.calls.append('moveL')
        if self.reject_move: return False
        self.rx.tcp = pose
        self.rx.protective = self.protective_on_move
        return True
    def getAsyncOperationProgress(self): return -1
    def stopL(self, a): self.calls.append('stopL'); return True
    def stopScript(self): self.calls.append('stopScript'); return True
    def disconnect(self): self.connected = False
    def teachMode(self): return True
    def endTeachMode(self): return True


class FakeHand:
    instances = []
    def __init__(self, *args):
        self.q, self.writes, self.enabled, self.closed = np.zeros(22), [], False, False
        self.__class__.instances.append(self)
    def read(self): return self.q.copy()
    def enable(self): self.enabled = True
    def write(self, q):
        assert self.enabled
        self.q = np.array(q)
        self.writes.append(self.q.copy())
    def hold(self): pass
    def close(self): self.closed = True


@pytest.fixture
def rig(monkeypatch, tmp_path):
    FakeControl.instances.clear()
    FakeReceive.instances.clear()
    FakeHand.instances.clear()
    FakeControl.reject_move = FakeControl.protective_on_move = False
    monkeypatch.setitem(sys.modules, 'rtde_control', types.SimpleNamespace(RTDEControlInterface=FakeControl))
    monkeypatch.setitem(sys.modules, 'rtde_receive', types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    monkeypatch.setattr(backend, 'SharpaHand', FakeHand)
    clock = Clock()
    monkeypatch.setattr(backend, 'time', clock)
    cfg = config.load(config.REPO / 'configs/real.yaml')
    cfg['commissioning'] = {k: True for k in cfg['commissioning']}
    cfg['cell']['motion']['settle_s'] = 0
    cfg['log_dir'] = str(tmp_path / 'logs')
    cells = []
    def make(execute=False, trial_step=False, trial_run=False):
        cell = backend.RealCell(copy.deepcopy(cfg['cell']), hand_contract()[0], execute=execute,
                                full_config=copy.deepcopy(cfg), cameras=False, trial_step=trial_step, trial_run=trial_run)
        cell.validate_hand_path = lambda *args: None
        cells.append(cell)
        return cell
    yield make, cfg
    for cell in cells: cell.close()


def test_read_only_never_creates_control_or_sends_hand(rig):
    make, _ = rig
    cell = make()
    q, h, flange = cell.state()
    assert q.shape == (6,) and h.shape == (22,)
    assert cell.ik(flange, q) is None
    for action in (cell.take_control, lambda: cell.move(flange, h), lambda: cell.freedrive(True)):
        with pytest.raises(HardwareError): action()
    cell.close()
    assert not FakeControl.instances
    assert not FakeHand.instances[-1].enabled and not FakeHand.instances[-1].writes


def test_execute_guard_before_any_connection(rig):
    _, cfg = rig
    cfg['commissioning']['depth_verified'] = False
    with pytest.raises(HardwareError, match='depth_verified'):
        backend.RealCell(cfg['cell'], hand_contract()[0], execute=True, full_config=cfg)
    assert not FakeReceive.instances and not FakeControl.instances


def test_trial_is_explicit_does_not_certify_setup_and_allows_only_one_move(rig):
    make, cfg = rig
    cfg['commissioning'] = {k: False for k in cfg['commissioning']}
    with pytest.raises(HardwareError, match='site checks pending'): make(True)
    with pytest.raises(HardwareError, match='explicitly enable'): make(trial_step=True)
    cell = make(True, trial_step=True)
    assert not FakeControl.instances and not cell.hand.enabled
    assert not cell.full_config['commissioning']['mount_verified']
    with pytest.raises(HardwareError, match='Freedrive'): cell.freedrive(True)
    _, h, target = cell.state()
    target[0, 3] += .005
    cell.move(target, h + .05)
    assert cell.trial_attempted and sum(c.calls.count('moveL') for c in FakeControl.instances) == 1
    with pytest.raises(HardwareError, match='already attempted'): cell.move(target, h)
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 1


def test_trial_rejected_command_consumes_attempt_and_cannot_relax_limits(rig):
    make, cfg = rig
    cfg['safety']['max_step_mm'] = 11.
    with pytest.raises(HardwareError, match='max_step_mm'): make(True, trial_step=True)
    assert not FakeReceive.instances
    cfg['safety']['max_step_mm'] = 10.
    cell = make(True, trial_step=True)
    _, h, target = cell.state(); target[0, 3] += .005
    FakeControl.reject_move = True
    with pytest.raises(HardwareError, match='moveL rejected'): cell.move(target, h + .05)
    with pytest.raises(HardwareError, match='already attempted'): cell.move(target, h + .05)
    assert cell.trial_attempted and not cell.hand.writes


def test_trial_continuous_session_run_blocked_before_any_motion():
    from lawm_deploy.session import Session
    session = Session.__new__(Session)
    session.cell = types.SimpleNamespace(trial_step=True)
    with pytest.raises(HardwareError, match='Continuous Run'): session.run()


def test_trial_run_allows_multiple_checked_moves_and_keeps_ordinary_execute_guard(rig):
    make, cfg = rig
    cfg['commissioning'] = {k: False for k in cfg['commissioning']}
    with pytest.raises(HardwareError, match='site checks pending'): make(True)
    with pytest.raises(HardwareError, match='explicitly enable'): make(trial_run=True)
    with pytest.raises(HardwareError, match='either'): make(True, trial_step=True, trial_run=True)
    cell = make(True, trial_run=True)
    for _ in range(2):
        _, h, target = cell.state()
        target[0, 3] += .005
        cell.move(target, h + .02)
        assert cell.ctrl is None
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 2
    assert not cell.full_config['commissioning']['mount_verified']
    _, h, target = cell.state(); target[0, 3] += .02
    with pytest.raises(HardwareError, match='step limit'): cell.move(target, h)
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 2


def test_move_checks_arrival_and_releases_arm(rig):
    make, _ = rig
    cell = make(True)
    q, hand, target = cell.state()
    target[0, 3] += 0.005
    cell.move(target, hand + 0.05)
    ctrl = FakeControl.instances[-1]
    assert 'moveL' in ctrl.calls and 'watchdog' in ctrl.calls and 'stopScript' in ctrl.calls
    assert cell.ctrl is None and not ctrl.connected
    assert np.allclose(cell.hand.q, hand + 0.05)


def test_stop_before_move_never_sends_target(rig):
    make, _ = rig
    cell = make(True)
    q, hand, target = cell.state()
    cell.stop_requested.set()
    with pytest.raises(HardwareError, match='Stop requested'):
        cell.move(target, hand)
    assert not FakeControl.instances and not cell.hand.writes


def test_rejected_movel_does_not_advance_fingers(rig):
    make, _ = rig
    cell = make(True)
    _, hand, target = cell.state()
    target[0, 3] += 0.005
    FakeControl.reject_move = True
    with pytest.raises(HardwareError, match='moveL rejected'):
        cell.move(target, hand + 0.05)
    assert not cell.hand.writes and cell.ctrl is None


def test_protective_stop_does_not_continue_finger_loop(rig):
    make, _ = rig
    cell = make(True)
    _, hand, target = cell.state()
    target[0, 3] += 0.005
    FakeControl.protective_on_move = True
    with pytest.raises(HardwareError, match='protective'):
        cell.move(target, hand + 0.05)
    assert not cell.hand.writes and cell.ctrl is None


def test_release_disconnects_hand_and_reconnects_without_output(rig):
    make, _ = rig
    cell = make(True)
    old = cell.hand
    cell.release()
    assert cell.released and old.closed
    with pytest.raises(HardwareError, match='released'):
        cell.state()
    cell.take_control()
    assert not cell.released and cell.hand is not old
    assert not cell.hand.enabled and cell.ctrl is None


def test_camera_failure_closes_prior_devices(rig, monkeypatch):
    _, cfg = rig
    monkeypatch.setattr(backend, 'camera_poses', lambda *a: [np.eye(4)] * 2)
    def fail(*a): raise HardwareError('camera failed')
    monkeypatch.setattr(backend, 'RealSense', fail)
    with pytest.raises(HardwareError, match='camera failed'):
        backend.RealCell(cfg['cell'], hand_contract()[0])
    assert FakeHand.instances[-1].closed
    assert not FakeReceive.instances[-1].connected
    assert not FakeControl.instances


def test_nonfinite_and_large_targets_refused(rig):
    make, _ = rig
    cell = make(True)
    _, hand, target = cell.state()
    with pytest.raises(HardwareError, match='finite'):
        cell.move(target, hand + np.nan)
    target[0,3] += 0.05
    with pytest.raises(HardwareError, match='step limit'):
        cell.move(target, hand)
    assert all('moveL' not in c.calls for c in FakeControl.instances)


def test_running_external_program_is_not_replaced(rig):
    make, _ = rig
    cell = make(True)
    cell.recv.getRuntimeState = lambda: 2
    with pytest.raises(HardwareError, match='Another UR program'):
        cell.control()
    assert not FakeControl.instances
