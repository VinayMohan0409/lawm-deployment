"""Fake hardware tests. No robot network, SDK discovery or USB access."""
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from lawm_deploy import config
from lawm_deploy.hardware_contract import HardwareError, StepTranslationRejected, hand_contract
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


def test_trial_is_explicit_and_allows_only_one_move(rig):
    make, cfg = rig
    with pytest.raises(HardwareError, match='explicitly enable'): make(trial_step=True)
    cell = make(True, trial_step=True)
    assert not FakeControl.instances and not cell.hand.enabled
    with pytest.raises(HardwareError, match='Freedrive'): cell.freedrive(True)
    _, h, target = cell.state()
    target[0, 3] += .005
    cell.move(target, h + .05)
    assert cell.trial_attempted and sum(c.calls.count('moveL') for c in FakeControl.instances) == 1
    with pytest.raises(HardwareError, match='already attempted'): cell.move(target, h)
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 1


def test_trial_rejected_command_consumes_attempt(rig):
    make, cfg = rig
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


def test_trial_run_allows_multiple_checked_moves(rig):
    make, cfg = rig
    with pytest.raises(HardwareError, match='explicitly enable'): make(trial_run=True)
    with pytest.raises(HardwareError, match='either'): make(True, trial_step=True, trial_run=True)
    cell = make(True, trial_run=True)
    for _ in range(2):
        _, h, target = cell.state()
        target[0, 3] += .025  # normal predicted step must not hit the old 10 mm cap
        cell.move(target, h + .02)
        assert cell.ctrl is None
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 2
    _, h, target = cell.state(); target[0, 3] += .063
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


@pytest.mark.parametrize('max_step_mm', [10., 60.])
def test_translation_margin_accepts_fk_discrepancy_but_rejects_large_move(rig, max_step_mm):
    make, cfg = rig
    cfg['safety']['max_step_mm'] = max_step_mm
    cell = make(True, trial_run=True)
    _, hand, target = cell.state()
    target[0, 3] += (max_step_mm + .077) / 1000  # nominal FK/controller discrepancy
    cell.move(target, hand)
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 1
    before_writes = len(cell.hand.writes)
    _, hand, target = cell.state()
    target[0, 3] += (max_step_mm + 2.1) / 1000
    with pytest.raises(StepTranslationRejected, match='translation exceeds step limit'):
        cell.move(target, hand)
    assert cell.ctrl is None
    assert len(cell.hand.writes) == before_writes
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 1
    journal = next(Path(cfg['log_dir']).glob('hardware_*.jsonl'))
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    rejected = [e for e in events if e['event'] == 'translation_rejected'][-1]
    assert rejected['distance_mm'] == pytest.approx(max_step_mm + 2.1)
    assert rejected['limit_mm'] == max_step_mm + 2.0
    assert np.asarray(rejected['measured_flange']).shape == (4, 4)


@pytest.mark.parametrize('margin', [-1., float('nan'), 2.1])
def test_invalid_translation_margin_rejected_before_connection(rig, margin):
    make, cfg = rig
    cfg['safety']['execution_step_margin_mm'] = margin
    with pytest.raises(HardwareError, match='execution_step_margin_mm'):
        make(True, trial_run=True)
    assert not FakeReceive.instances


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('excess_deg', [.221, .506, .95])
def test_small_feedback_overrun_preserves_feedback_and_clamps_commands(rig, side, excess_deg):
    make, cfg = rig
    cell = make(True, trial_run=True)
    i = cell.joint_names.index('right_index_MCP_FE')
    bound = cell.hand_bounds[i, 0 if side == 'lower' else 1]
    sign = -1 if side == 'lower' else 1
    cell.hand.q[i] = bound + sign * np.radians(excess_deg)
    _, measured, target = cell.state()
    assert measured[i] == cell.hand.q[i]  # reads retain the actual overrun
    fingers = measured.copy(); fingers[i] = bound - sign * np.radians(1.)
    target[0, 3] += .005
    paths = []
    cell.validate_hand_path = lambda f0, f1, h0, h1: paths.append(h0.copy())
    cell.move(target, fingers)
    writes = np.array(cell.hand.writes)
    assert len(writes) and (writes >= cell.hand_bounds[:, 0]).all() and (writes <= cell.hand_bounds[:, 1]).all()
    assert writes[0, i] == bound
    assert paths[0][i] == measured[i] and paths[1][i] == bound
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    tolerated = next(r for r in rows if r['event'] == 'hand_feedback_tolerated')
    assert tolerated['measured_hand'][i] == measured[i]
    assert tolerated['excess_deg'][i] == pytest.approx(excess_deg)
    assert rows[-1]['event'] == 'move_arrived'


def test_large_feedback_overrun_and_outside_target_are_still_rejected(rig):
    make, _ = rig
    cell = make(True)
    i = cell.joint_names.index('right_index_MCP_FE')
    cell.hand.q[i] = cell.hand_bounds[i, 0] - np.radians(1.1)
    _, measured, target = cell.state()
    fingers = np.clip(measured, cell.hand_bounds[:, 0], cell.hand_bounds[:, 1])
    with pytest.raises(HardwareError, match='right_index_MCP_FE.*excess 1.100 deg > 1.000'):
        cell.move(target, fingers)
    assert not cell.hand.enabled and not cell.hand.writes
    cell.hand.q[:] = fingers
    with pytest.raises(HardwareError, match='Hand target outside joint limits'):
        cell.move(target, measured)
    assert sum(c.calls.count('moveL') for c in FakeControl.instances) == 0


def test_feedback_overrun_during_motion_stops_further_commands(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    original = FakeControl.moveL
    def move_then_overrun(ctrl, *args):
        result = original(ctrl, *args)
        cell.hand.q[5] = cell.hand_bounds[5, 0] - np.radians(1.1)
        return result
    monkeypatch.setattr(FakeControl, 'moveL', move_then_overrun)
    with pytest.raises(HardwareError, match='feedback tolerance'):
        cell.move(target, fingers)
    assert not cell.hand.writes and cell.ctrl is None
    assert 'stopL' in FakeControl.instances[-1].calls


def test_recorded_ring_feedback_overrun_during_motion_remains_raw(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    i = cell.joint_names.index('right_ring_MCP_FE')
    bound = cell.hand_bounds[i, 0]
    cell.hand.q[i] = bound
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[i] += np.radians(1.)
    measured = bound - np.radians(.506)
    move = FakeControl.moveL
    def move_with_overrun(ctrl, *args):
        accepted = move(ctrl, *args)
        cell.hand.q[i] = measured
        return accepted
    monkeypatch.setattr(FakeControl, 'moveL', move_with_overrun)
    assert cell.move(target, fingers)['hand_target_reached']
    writes = np.asarray(cell.hand.writes)
    assert (writes >= cell.hand_bounds[:, 0]).all() and (writes <= cell.hand_bounds[:, 1]).all()
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    feedback = next(r for r in rows if r['event'] == 'feedback')
    assert feedback['hand_q'][i] == measured


@pytest.mark.parametrize('joint,actual_deg,target_deg', [
    ('right_thumb_CMC_AA', 2.01, 11.89),
    ('right_index_MCP_AA', 6.75, 14.80),
    ('right_pinky_MCP_AA', -2.15, -12.44),
])
def test_hand_shortfall_holds_measured_pose_and_returns_partial(rig, monkeypatch, joint, actual_deg, target_deg):
    make, cfg = rig
    cell = make(True)
    index = cell.joint_names.index(joint)
    cell.hand.q[index] = np.radians(actual_deg)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[index] = np.radians(target_deg)
    monkeypatch.setattr(cell.hand, 'write', lambda q: cell.hand.writes.append(np.array(q)))
    result = cell.move(target, fingers)
    assert result['outcome'] == 'hand_target_not_reached' and not result['hand_target_reached']
    assert result['arrival']['worst_hand_joint'] == joint
    assert result['arrival']['hand_error_deg'] == pytest.approx(abs(actual_deg-target_deg))
    np.testing.assert_allclose(cell.hand.writes[-1], cell.hand.q)
    np.testing.assert_allclose(result['hand_actual'], cell.hand.q)
    np.testing.assert_allclose(result['hand_target'], fingers)
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    start = next(r for r in rows if r['event'] == 'move_start')
    partial = next(r for r in rows if r['event'] == 'move_partial')
    assert partial['elapsed_s'] == pytest.approx(start['duration_s'] + cfg['cell']['guards']['hand_settle_timeout_s'], abs=.03)
    assert not any(r['event'] in ('motion_timeout', 'move_arrived', 'move_failed') for r in rows)
    assert cell.ctrl is None and 'stopScript' in FakeControl.instances[-1].calls


def test_delayed_hand_can_still_arrive_during_settle(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[5] = .2
    def delayed_write(q):
        cell.hand.writes.append(np.array(q))
        if len(cell.hand.writes) >= 150:
            cell.hand.q = np.array(q)
    monkeypatch.setattr(cell.hand, 'write', delayed_write)
    assert cell.move(target, fingers)['hand_target_reached']
    assert len(cell.hand.writes) > 150


@pytest.mark.parametrize('deadline_offset', [-.2, .005])
@pytest.mark.parametrize('hand_follows', [False, True])
def test_arm_arriving_at_deadline_gets_independent_hand_settle(rig, monkeypatch, deadline_offset, hand_follows):
    """A slow moveL must not consume the hand's settling time or use stale arrival."""
    make, _ = rig
    cell = make(True)
    _, fingers, flange = cell.state()
    target = flange.copy(); target[0, 3] += .005
    fingers[1] = .18
    duration = backend.move_duration(cell.motion, flange, target, cell.hand.q, fingers)
    arm_duration = max(5., 3*duration+2) + deadline_offset
    motion = {}
    def slow_move(ctrl, pose, *args):
        ctrl.calls.append('moveL')
        motion.update(start=backend.time.monotonic(), initial=list(ctrl.rx.tcp), target=list(pose))
        return True
    def pose_feedback():
        if motion and backend.time.monotonic()-motion['start'] >= arm_duration:
            return motion['target']
        return cell.recv.tcp
    def progress(ctrl):
        return -1 if backend.time.monotonic()-motion['start'] >= arm_duration else 0
    monkeypatch.setattr(FakeControl, 'moveL', slow_move)
    monkeypatch.setattr(FakeControl, 'getAsyncOperationProgress', progress)
    monkeypatch.setattr(cell.recv, 'getActualTCPPose', pose_feedback)
    if not hand_follows:
        monkeypatch.setattr(cell.hand, 'write', lambda q: cell.hand.writes.append(np.array(q)))
    result = cell.move(target, fingers)
    assert result['hand_target_reached'] is hand_follows
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    completion = rows[-1]
    assert completion['event'] == ('move_arrived' if hand_follows else 'move_partial')
    extra = 0 if hand_follows else 1.0
    assert completion['elapsed_s'] == pytest.approx(arm_duration + extra, abs=.03)
    assert result['hand_settle_elapsed_s'] == pytest.approx(extra, abs=.02)
    assert not any(r['event'] == 'motion_timeout' for r in rows)
    assert cell.ctrl is None


def test_failed_partial_hold_command_is_fatal(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[5] = .2
    def fail_hold(q):
        if cell.hand.writes and cell.hand.writes[-1][5] > .1 and q[5] == 0:
            raise HardwareError('hold command failed')
        cell.hand.writes.append(np.array(q))
    monkeypatch.setattr(cell.hand, 'write', fail_hold)
    with pytest.raises(HardwareError, match='hold command failed'):
        cell.move(target, fingers)
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    assert not any(r['event'] == 'move_partial' for r in rows)
    assert cell.ctrl is None and 'stopL' in FakeControl.instances[-1].calls


def test_stop_during_extra_hand_settle_is_respected(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[5] = .2
    def request_stop(q):
        cell.hand.writes.append(np.array(q))
        if len(cell.hand.writes) == 150:
            cell.stop_requested.set()
    monkeypatch.setattr(cell.hand, 'write', request_stop)
    with pytest.raises(HardwareError, match='stopped by user'):
        cell.move(target, fingers)
    assert len(cell.hand.writes) == 150
    assert cell.ctrl is None and 'stopL' in FakeControl.instances[-1].calls


@pytest.mark.parametrize('failure', ['arm_position', 'arm_rotation', 'arm_still_moving'])
def test_arm_failure_is_not_recovered_as_partial_hand_motion(rig, monkeypatch, failure):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[5] = .2
    monkeypatch.setattr(cell.hand, 'write', lambda q: cell.hand.writes.append(np.array(q)))
    if failure == 'arm_still_moving':
        monkeypatch.setattr(FakeControl, 'getAsyncOperationProgress', lambda ctrl: 0)
    else:
        move = FakeControl.moveL
        def inaccurate_move(ctrl, *args):
            accepted = move(ctrl, *args)
            ctrl.rx.tcp[0 if failure == 'arm_position' else 3] += .01 if failure == 'arm_position' else .05
            return accepted
        monkeypatch.setattr(FakeControl, 'moveL', inaccurate_move)
    with pytest.raises(HardwareError, match='Motion arrival timeout'):
        cell.move(target, fingers)
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    assert not any(r['event'] == 'move_partial' for r in rows)
    timeout = next(r for r in rows if r['event'] == 'motion_timeout')
    assert timeout['timeout_phase'] == 'arm'
    assert cell.ctrl is None and 'stopL' in FakeControl.instances[-1].calls


def test_large_hand_following_error_still_stops(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    fingers[6] = .2
    def wrong_direction(q):
        cell.hand.writes.append(np.array(q))
        cell.hand.q[6] = -.2
    monkeypatch.setattr(cell.hand, 'write', wrong_direction)
    with pytest.raises(HardwareError, match='Sharpa following error'):
        cell.move(target, fingers)
    assert cell.ctrl is None and 'stopL' in FakeControl.instances[-1].calls


def test_control_loop_delay_reports_gap_separately(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    sleep = backend.time.sleep
    monkeypatch.setattr(backend.time, 'sleep', lambda seconds: sleep(seconds + .16))
    with pytest.raises(HardwareError, match='Control loop stalled: gap'):
        cell.move(target, fingers)
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    event = next(r for r in rows if r['event'] == 'control_loop_stalled')
    assert event['loop_gap_s'] > event['limit_s']
    assert not any(r['event'] == 'motion_timeout' for r in rows)
    assert cell.ctrl is None


def test_107ms_host_delay_is_tolerated_below_watchdog(rig, monkeypatch):
    make, cfg = rig
    assert cfg['cell']['guards']['control_gap_s'] < cfg['cell']['guards']['watchdog_timeout_s']
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    sleep = backend.time.sleep
    monkeypatch.setattr(backend.time, 'sleep', lambda seconds: sleep(seconds + .097))
    cell.move(target, fingers)
    rows = [json.loads(s) for s in cell.log_path.read_text().splitlines()]
    assert rows[-1]['event'] == 'move_arrived'
    assert rows[-1]['max_control_gap_s'] == pytest.approx(.107)


def test_loop_sleep_accounts_for_command_processing_time(rig, monkeypatch):
    make, _ = rig
    cell = make(True)
    _, fingers, target = cell.state(); target[0, 3] += .005
    timestamps = []
    write = cell.hand.write
    def slow_write(q):
        timestamps.append(backend.time.monotonic())
        write(q)
        backend.time.sleep(.004)
    monkeypatch.setattr(cell.hand, 'write', slow_write)
    cell.move(target, fingers)
    np.testing.assert_allclose(np.diff(timestamps), .01, atol=1e-9)


def test_loop_guard_cannot_exceed_watchdog(rig):
    make, cfg = rig
    cfg['cell']['guards']['control_gap_s'] = cfg['cell']['guards']['watchdog_timeout_s']
    with pytest.raises(HardwareError, match='below watchdog_timeout_s'):
        make(True)
    assert not FakeReceive.instances


@pytest.mark.parametrize('tolerance', [-.1, float('nan'), 1.1])
def test_invalid_hand_feedback_tolerance_rejected_before_connect(rig, tolerance):
    make, cfg = rig
    cfg['cell']['guards']['hand_feedback_limit_tolerance_deg'] = tolerance
    with pytest.raises(HardwareError, match='hand_feedback_limit_tolerance_deg'):
        make(True)
    assert not FakeReceive.instances


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
    target[0,3] += 0.07
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
