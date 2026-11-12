"""Hardware contracts shared by preflight, adapters and tests; no device imports."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class HardwareError(RuntimeError):
    """A device, command or measured state failed validation."""


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


REQUIRED_SITE_CHECKS = ("mount_verified", "depth_verified", "workspace_verified", "tcp_payload_verified")


def validate_cell_settings(cfg):
    if not np.array_equal(vector(cfg["expected_tcp_offset"], 6, "expected TCP offset"), np.zeros(6)):
        raise HardwareError("This backend currently requires the reference setup's zero TCP offset")
    for name, value in cfg["guards"].items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise HardwareError(f"Invalid runtime guard: {name}")
    hand = cfg["sharpa"]
    for name in ("speed_coeff", "current_coeff"):
        if not np.isfinite(hand[name]) or not 0 < hand[name] <= 1:
            raise HardwareError(f"Invalid Sharpa {name}")


def require_execution_setup(cfg, *, trial_step=False):
    """Never treat provisional geometry or camera calibration as commissioned hardware."""
    missing = [key for key in REQUIRED_SITE_CHECKS if cfg.get("commissioning", {}).get(key) is not True]
    if missing and not trial_step:
        raise HardwareError("Execution blocked: site checks pending: " + ", ".join(missing))
    s = cfg["safety"]
    lo, hi = vector(s["box_lo"], 3, "box_lo"), vector(s["box_hi"], 3, "box_hi")
    if not (lo < hi).all() or not np.isfinite(s["floor_z_m"]):
        raise HardwareError("Invalid measured workspace or table height")
    for k in ("max_step_mm", "max_rot_deg", "max_finger_deg", "max_joint_deg", "max_fit_residual_mm"):
        if not np.isfinite(s[k]) or s[k] <= 0:
            raise HardwareError(f"Invalid safety setting: {k}")
    if trial_step:
        # Explicit operator-supervised trial of the existing setup, not a claim
        # that its outstanding physical commissioning checks were completed.
        for name, cap in dict(max_step_mm=10., max_rot_deg=8., max_finger_deg=20.,
                              max_joint_deg=10., max_fit_residual_mm=15.).items():
            if s[name] > cap:
                raise HardwareError(f"Trial step requires {name} <= {cap}")
        for name, cap in dict(speed=.02, accel=.1, rot_speed=.15, finger_speed=.3, finger_accel=1.).items():
            value = cfg['cell']['motion'][name]
            if not np.isfinite(value) or not 0 < value <= cap:
                raise HardwareError(f"Trial step requires motion.{name} <= {cap}")
