"""RealCell against fakes of ur_rtde, pyrealsense2 and the Sharpa SDK: connect, observe, IK, move, release to teleoperation and take control back."""
import sys
import types

import numpy as np
import pytest

from lawm_deploy import config
from lawm_deploy.robot_model import RobotModel


class FakeControl:
    """rtde_control.RTDEControlInterface: records calls; moves finish at once."""

    def __init__(self, ip):
        """Connected, with no calls yet."""
        self.calls, self.connected = [], True

    def __getattr__(self, name):
        """Any call is recorded; IK answers a fixed solution, async progress says done."""
        def call(*args):
            self.calls.append(name)
            return {"getInverseKinematicsHasSolution": True, "getInverseKinematics": [0.1] * 6, "getAsyncOperationProgress": -1}.get(name)
        return call


class FakeReceive:
    """rtde_receive.RTDEReceiveInterface at a fixed pose."""

    def __init__(self, ip):
        """Nothing to connect."""

    def getActualQ(self):
        """Arm angles."""
        return [np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]

    def getActualTCPPose(self):
        """Flange pose vector."""
        return [0.49, 0.13, 0.49, np.pi, 0.0, 0.0]

    def isProtectiveStopped(self):
        """Never."""
        return False


class FakeHand:
    """SharpaHand: remembers the last target."""

    def __init__(self, settings, names):
        """Open hand."""
        self.q, self.writes = np.zeros(22), 0

    def read(self):
        """Measured angles."""
        return self.q

    def write(self, q):
        """Take the target at once."""
        self.q, self.writes = np.asarray(q, float), self.writes + 1


@pytest.fixture()
def cell(monkeypatch):
    """A RealCell on the fakes, with one fake camera returning a flat table."""
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=FakeControl))
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    from lawm_deploy import cell as cell_module
    monkeypatch.setattr(cell_module, "SharpaHand", FakeHand)

    class FakeCameras:
        """One camera 1 m above the table looking down."""

        def __init__(self, *args):
            """Nothing to start."""

        def views(self):
            """A flat depth image at 1 m."""
            K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
            return [(np.full((240, 320), 1.0), K, np.array([[1.0, 0, 0, 0.5], [0, -1, 0, 0.15], [0, 0, -1, 1.0], [0, 0, 0, 1]]))]

        def close(self):
            """Nothing to stop."""
    monkeypatch.setattr(cell_module, "RealSense", FakeCameras)
    monkeypatch.setattr(cell_module, "camera_poses", lambda cams, calibrated: [np.eye(4)])
    cfg = config.load(config.REPO / "configs/real.yaml")
    return cell_module.RealCell(cfg["cell"], RobotModel(np.eye(4), n_mask_points=100).hand_joint_names)


def test_moves_and_reads(cell):
    """A move sends moveL and finger targets; state and IK come back."""
    q_arm, q_hand, flange = cell.state()
    assert q_arm.shape == (6,) and q_hand.shape == (22,) and np.allclose(flange[:3, 3], [0.49, 0.13, 0.49])
    assert cell.ik(flange, q_arm) is not None
    cell.move(flange, np.full(22, 0.3))
    assert "moveL" in cell.ctrl.calls and np.allclose(cell.hand.q, 0.3) and cell.hand.writes > 1


def test_release_and_take_control(cell):
    """Released: state still reads, motion and IK refuse; taking control back allows them again."""
    ctrl = cell.ctrl
    cell.release()
    assert cell.released and "stopScript" in ctrl.calls
    cell.state()
    with pytest.raises(RuntimeError, match="released"):
        cell.move(np.eye(4), np.zeros(22))
    with pytest.raises(RuntimeError, match="released"):
        cell.ik(np.eye(4), np.zeros(6))
    cell.stop()  # harmless while released
    cell.take_control()
    assert not cell.released and cell.ik(np.eye(4), np.zeros(6)) is not None
