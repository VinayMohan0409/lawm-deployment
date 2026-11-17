"""Hardware contracts shared by preflight, adapters and tests; no device imports."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class HardwareError(RuntimeError):
    """A device, command or measured state failed validation."""


class StepTranslationRejected(HardwareError):
    """Translation rejected before sending arm or finger motion; safe to replan."""


def vector(value, n, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise HardwareError(f"{name}: expected {n} finite values")
    return value


def transform(value, name="transform"):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise HardwareError(f"{name}: expected finite 4x4 matrix")
    R = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        raise HardwareError(f"{name}: invalid rigid transform")
    return value


def hand_contract():
    """SDK order, as verified against supplements/teleop_hand.py in the local SDK repo."""
    suffixes = ["thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP"]
    for finger in ("index", "middle", "ring", "pinky"):
        if finger == "pinky":
            suffixes.append("pinky_CMC")
        suffixes += [f"{finger}_{joint}" for joint in ("MCP_FE", "MCP_AA", "PIP", "DIP")]
    names = ["right_" + name for name in suffixes]
    path = Path(__file__).resolve().parents[1] / "assets/sharpa_wave/right_hand.xml"
    joints = {j.get("name"): np.fromstring(j.get("range"), sep=" ")
              for j in ET.parse(path).findall(".//worldbody//joint")}
    if set(joints) != set(names):
        raise HardwareError("Sharpa MJCF and SDK joint names differ")
    return names, np.asarray([joints[n] for n in names])


def validate_cell_settings(cfg):
    if not np.array_equal(vector(cfg["expected_tcp_offset"], 6, "expected TCP offset"), np.zeros(6)):
        raise HardwareError("This backend currently requires the reference setup's zero TCP offset")
    for name, value in cfg["guards"].items():
        if name == 'hand_feedback_limit_tolerance_deg':
            if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1.0:
                raise HardwareError(f"Invalid runtime guard: {name} (expected 0..1 degrees)")
            continue
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise HardwareError(f"Invalid runtime guard: {name}")
    hand = cfg["sharpa"]
    if cfg['guards']['control_gap_s'] >= cfg['guards']['watchdog_timeout_s']:
        raise HardwareError('control_gap_s must remain below watchdog_timeout_s')
    for name in ("speed_coeff", "current_coeff"):
        if not np.isfinite(hand[name]) or not 0 < hand[name] <= 1:
            raise HardwareError(f"Invalid Sharpa {name}")


def require_execution_setup(cfg):
    """Validate configured motion limits before opening a command interface."""
    s = cfg["safety"]
    margin = s.get("execution_step_margin_mm", 0.0)
    if not np.isfinite(margin) or not 0 <= margin <= 2.0:
        raise HardwareError("execution_step_margin_mm must be between 0 and 2 mm")
    lo, hi = vector(s["box_lo"], 3, "box_lo"), vector(s["box_hi"], 3, "box_hi")
    if not (lo < hi).all() or not np.isfinite(s["floor_z_m"]):
        raise HardwareError("Invalid measured workspace or table height")
    for k in ("max_step_mm", "max_rot_deg", "max_finger_deg", "max_joint_deg", "max_fit_residual_mm"):
        if not np.isfinite(s[k]) or s[k] <= 0:
            raise HardwareError(f"Invalid safety setting: {k}")
