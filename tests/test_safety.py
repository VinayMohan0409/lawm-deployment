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


def test_model_frame_round_trip():
    """The training-frame rotation is a proper rotation taking z to the training up, and the wrapped model's displacements come back in the base frame."""
    from lawm_deploy.model import FramedModel, StandInModel, model_frame
    up = np.array([-0.022, -0.546, -0.837])
    R = model_frame(up, 315)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 0, 1.0], up / np.linalg.norm(up))
    coords, seg = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32), np.r_[np.zeros(20), np.ones(30)].astype(np.int8)
    z = np.random.default_rng(1).normal(size=(4, 5, 2, 4)).astype(np.float32)
    inner = StandInModel()
    out = FramedModel(inner, R).decoder(coords, seg)(z)
    assert np.allclose(out, inner.decoder(coords @ R.T, seg)(z) @ R, atol=1e-6)
