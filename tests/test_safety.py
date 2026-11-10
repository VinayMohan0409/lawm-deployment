"""The executor rule and the checks that hold the robot."""
import numpy as np

from lawm_deploy.robot_model import pose
from lawm_deploy.safety import Executor, SafetySettings, check

S = SafetySettings(max_step_mm=60, ori_alpha=0.5, max_rot_deg=30, max_finger_deg=60, max_joint_deg=30, max_fit_residual_mm=15,
                   box_lo=(-1, -1, 0), box_hi=(1, 1, 1), floor_z_m=0.0)
FLANGE = pose(np.diag([1.0, -1.0, -1.0]), [0.5, 0.0, 0.5])
GOOD = dict(q_hand=np.zeros(22), q_hand_now=np.zeros(22), flange_now=FLANGE, q_arm_now=np.zeros(6), q_arm_ik=np.zeros(6),
            fit_info=dict(mean_residual_mm=1.0), hand_points=np.array([[0.5, 0.0, 0.3]]))


def test_translation_is_limited():
    """A 20 cm palm motion becomes a 6 cm flange motion in the same direction."""
    root = FLANGE.copy()
    target, _, mm = Executor(S, FLANGE).command(root[:3, :3], root[:3, 3] + [0.2, 0, 0], np.zeros(22), root, FLANGE)
    assert round(mm) == 200 and np.allclose(target[:3, 3], FLANGE[:3, 3] + [0.06, 0, 0])


def test_checks():
    """A good command passes; each bad one gives its reason."""
    assert check(S, FLANGE, **GOOD) is None
    assert "residual" in check(S, FLANGE, **{**GOOD, "fit_info": dict(mean_residual_mm=20.0)})
    assert "IK" in check(S, FLANGE, **{**GOOD, "q_arm_ik": None})
    assert "joint jump" in check(S, FLANGE, **{**GOOD, "q_arm_ik": np.r_[1.0, np.zeros(5)]})
    assert "finger" in check(S, FLANGE, **{**GOOD, "q_hand": np.full(22, 1.5)})
    assert "floor" in check(S, FLANGE, **{**GOOD, "hand_points": np.array([[0.5, 0.0, -0.01]])})
    assert "box" in check(S, pose(FLANGE[:3, :3], [2.0, 0, 0.5]), **GOOD)
