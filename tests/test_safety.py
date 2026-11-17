"""The executor rule and the checks that hold the robot."""
import numpy as np
import pytest

from lawm_deploy.robot_model import pose
from lawm_deploy.safety import Executor, SafetySettings, check, limit_finger_step
from lawm_deploy.hardware_contract import hand_contract, HardwareError

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


def test_missing_ik_does_not_hide_other_rejections():
    from scipy.spatial.transform import Rotation
    unknown_ik = {**GOOD, 'q_arm_ik': None}
    turned = pose(Rotation.from_euler('z', 40, degrees=True).as_matrix() @ FLANGE[:3, :3], FLANGE[:3, 3])
    assert 'wrist rotation' in check(S, turned, **unknown_ik)
    assert 'finger' in check(S, FLANGE, **{**unknown_ik, 'q_hand': np.full(22, 1.5)})
    assert 'floor' in check(S, FLANGE, **{**unknown_ik, 'hand_points': np.array([[0., 0., -.01]])})
    assert 'IK' in check(S, FLANGE, **unknown_ik)


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


@pytest.mark.parametrize('margin_deg', [.5, 1.])
def test_finger_increment_clamp_preserves_limits_and_feedback(margin_deg):
    names, bounds = hand_contract()
    measured = np.zeros(22)
    i = names.index('right_index_MCP_FE')
    measured[i] = bounds[i, 0] - np.radians(.221)
    target = measured.copy(); target[i] = np.radians(36)
    original = measured.copy()
    command, info = limit_finger_step(target, measured, bounds, 20., margin_deg)
    assert info['finger_step_limited'] and info['finger_clipped_indices'] == [i]
    assert info['commanded_finger_step_deg'] == pytest.approx(20. - margin_deg)
    assert (command >= bounds[:, 0]).all() and (command <= bounds[:, 1]).all()
    np.testing.assert_array_equal(measured, original)
    normal, info = limit_finger_step(np.zeros(22), np.zeros(22), bounds, 20., margin_deg)
    assert not info['finger_step_limited']
    np.testing.assert_array_equal(normal, 0)


def test_finger_clamp_absolute_bounds_nan_and_impossible_step():
    _, bounds = hand_contract()
    near_upper = bounds[:, 1] - np.radians(1)
    command, info = limit_finger_step(bounds[:, 1] + .1, near_upper, bounds, 20., .5)
    np.testing.assert_array_equal(command, bounds[:, 1])
    assert info['finger_step_limited']
    with pytest.raises(HardwareError):
        limit_finger_step(np.full(22, np.nan), near_upper, bounds, 20.)
    with pytest.raises(HardwareError, match='both absolute and step limits'):
        limit_finger_step(np.zeros(22), bounds[:, 0] - 1, bounds, 20.)
