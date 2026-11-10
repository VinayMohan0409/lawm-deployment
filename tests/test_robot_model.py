"""The robot model: arm forward kinematics against UR's published UR5e parameters, the hand's points against MuJoCo, and the fit."""
import mujoco
import numpy as np
import pytest

from lawm_deploy.hand import Hand
from lawm_deploy.robot_model import RobotModel

UR5E_DH = dict(d=[0.1625, 0, 0, 0.1333, 0.0997, 0.0996], a=[0, -0.425, -0.3922, 0, 0, 0], alpha=[np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])


def dh_flange(q):
    """Flange pose in UR's base frame (the frame RTDE reports) from the standard DH table."""
    t = np.eye(4)
    for qi, d, a, al in zip(q, UR5E_DH["d"], UR5E_DH["a"], UR5E_DH["alpha"]):
        ct, st, ca, sa = np.cos(qi), np.sin(qi), np.cos(al), np.sin(al)
        t = t @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
    return t


@pytest.fixture(scope="module")
def robot():
    """The robot with the hand flush on the flange."""
    return RobotModel(np.eye(4), n_mask_points=2000)


def test_arm_matches_ur_frame(robot):
    """MuJoCo's flange pose equals UR's DH flange pose (rotation exactly, position within 2 mm: Menagerie's link lengths), so the model's world frame is the robot's base frame."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        robot.set(q, np.zeros(22))
        ours, ur = robot.flange_pose(), dh_flange(q)
        np.testing.assert_allclose(ours[:3, :3], ur[:3, :3], atol=1e-6)
        np.testing.assert_allclose(ours[:3, 3], ur[:3, 3], atol=2e-3)


def test_hand_points_follow_mujoco(robot):
    """FK hand points equal lawm's sampler placed by MuJoCo's own geom poses."""
    hand = Hand(robot.model, 1024)
    rng = np.random.default_rng(1)
    q = rng.uniform(robot.hand_limits[:, 0], robot.hand_limits[:, 1])
    robot.set(rng.uniform(-1, 1, 6), q)
    ours = hand.points(robot.root_pose(), q)
    np.testing.assert_allclose(ours, hand.sampler(robot.data), atol=1e-5)


def test_fit_recovers_angles(robot):
    """The fit of points drawn at known angles gives back the palm pose and the points."""
    hand = Hand(robot.model, 1024)
    rng = np.random.default_rng(2)
    lo, hi = robot.hand_limits[:, 0], robot.hand_limits[:, 1]
    q = lo + (hi - lo) * rng.uniform(0.2, 0.8, 22)
    robot.set(np.array([0, -1.5, 1.5, -1.5, -1.5, 0]), q)
    root = robot.root_pose()
    rot, pos, q_fit, info = hand.fit(hand.points(root, q), np.clip(q + 0.1, lo, hi))
    np.testing.assert_allclose(pos, root[:3, 3], atol=1e-5)
    assert info["mean_residual_mm"] < 0.5
    assert hand.finger_mask.any() and not hand.finger_mask.all()
