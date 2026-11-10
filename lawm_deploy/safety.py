"""From a fitted hand state to a safe command: lawm's executor rule (rotation smoothing, translation limit) for 22 fingers, and the checks that hold the robot still instead of sending a bad target."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lawm.sim.dexwm.control.hand_fit import hand_state_to_action

from lawm_deploy.robot_model import pose


@dataclass(frozen=True)
class SafetySettings:
    """Executor rule (as in lawm's DexWM config) and the outlier guards; distances in mm, angles in degrees, flange box and hand floor height in metres (base frame, set by the operator)."""

    max_step_mm: float
    ori_alpha: float
    max_rot_deg: float
    max_finger_deg: float
    max_joint_deg: float
    max_fit_residual_mm: float
    box_lo: tuple
    box_hi: tuple
    floor_z_m: float


class Executor:
    """lawm's DexWM executor rule for the real robot: the palm's fitted motion applied to the flange, the rotation moved ori_alpha of the way from the last target, the translation limited to max_step_mm."""

    def __init__(self, settings, flange_now):
        """Start the rotation smoothing from the flange's rotation at the start of the episode."""
        self.s = settings
        self.last_rotvec = Rotation.from_matrix(flange_now[:3, :3]).as_rotvec()

    def command(self, root_rot, root_pos, q, root_now, flange_now):
        """(4x4 flange target, 22 finger targets, unclipped translation in mm); the smoothing state only changes once the target is sent (sent())."""
        pos, rotvec, fingers = hand_state_to_action(root_rot, root_pos, q, root_now[:3, :3], root_now[:3, 3], flange_now[:3, 3], flange_now[:3, :3])
        keys = Rotation.concatenate([Rotation.from_rotvec(self.last_rotvec), Rotation.from_rotvec(rotvec)])
        rotvec = Slerp([0, 1], keys)([self.s.ori_alpha])[0].as_rotvec()
        move = pos - flange_now[:3, 3]
        move_mm = float(np.linalg.norm(move) * 1000.0)
        if move_mm > self.s.max_step_mm:
            pos = flange_now[:3, 3] + move * (self.s.max_step_mm / move_mm)
        return pose(Rotation.from_rotvec(rotvec).as_matrix(), pos), np.asarray(fingers, dtype=float), move_mm

    def sent(self, target):
        """Record that a flange target was sent: the next rotation is smoothed from it."""
        self.last_rotvec = Rotation.from_matrix(target[:3, :3]).as_rotvec()


def check(settings, target, q_hand, q_hand_now, flange_now, q_arm_now, q_arm_ik, fit_info, hand_points):
    """None if the command is safe to send, else the reason to hold: bad fit, no IK or a joint jump, a large rotation or finger jump, the flange outside the box, a hand point below the floor height."""
    s = settings
    if fit_info["mean_residual_mm"] > s.max_fit_residual_mm:
        return f"hand fit residual {fit_info['mean_residual_mm']:.1f} mm > {s.max_fit_residual_mm}"
    if q_arm_ik is None:
        return "no arm IK solution for the target"
    jump = float(np.degrees(np.abs(np.asarray(q_arm_ik) - q_arm_now).max()))
    if jump > s.max_joint_deg:
        return f"arm joint jump {jump:.1f} deg > {s.max_joint_deg}"
    rot = float(np.degrees(Rotation.from_matrix(target[:3, :3] @ flange_now[:3, :3].T).magnitude()))
    if rot > s.max_rot_deg:
        return f"wrist rotation {rot:.1f} deg > {s.max_rot_deg}"
    finger = float(np.degrees(np.abs(q_hand - q_hand_now).max()))
    if finger > s.max_finger_deg:
        return f"finger jump {finger:.1f} deg > {s.max_finger_deg}"
    if not ((target[:3, 3] >= s.box_lo).all() and (target[:3, 3] <= s.box_hi).all()):
        return f"flange target {np.round(target[:3, 3], 3).tolist()} outside the box"
    low = float(hand_points[:, 2].min())
    if low < s.floor_z_m:
        return f"hand reaches z = {low * 1000:.0f} mm, below the floor height {s.floor_z_m * 1000:.0f} mm"
    return None
