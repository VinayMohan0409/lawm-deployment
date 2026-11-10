"""The UR7e + Sharpa Wave model: UR5e kinematics from MuJoCo Menagerie with the right Sharpa hand mounted on the flange."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
HAND_PREFIX = "hand_right_"  # body prefix of the mounted hand
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
FLANGE_SITE = "attachment_site"  # the UR tool flange (tool0)


def hand_spec():
    """The right Sharpa hand spec with its 22 joints renamed joint_0..joint_21 in MJCF order, and the original names in that order."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / "sharpa_wave/right_hand.xml"))
    names = [j.name for j in spec.joints]
    for k, joint in enumerate(spec.joints):
        joint.name = f"joint_{k}"
    for actuator in spec.actuators:
        actuator.target = f"joint_{names.index(actuator.target)}"
    return spec, names


def quat_of(rot):
    """MuJoCo (w, x, y, z) quaternion of a rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=np.float64).ravel())
    return q


def pose(rot, pos):
    """4x4 pose of a rotation and a position."""
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, pos
    return out


def cell_spec(mount):
    """The arm with the hand attached at flange @ mount (mount = 4x4 pose of the hand's MJCF origin in the flange frame), gravity compensated as the real controllers are."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / "universal_robots_ur5e/ur5e.xml"))
    for key in list(spec.keys):  # Menagerie's arm-only keyframe does not fit the hand
        key.delete()
    flange = next(s for s in spec.sites if s.name == FLANGE_SITE)
    f_quat = np.asarray(flange.quat, dtype=np.float64) / np.linalg.norm(flange.quat)
    f_rot = np.zeros(9)
    mujoco.mju_quat2Mat(f_rot, f_quat)
    at = pose(f_rot.reshape(3, 3), flange.pos) @ np.asarray(mount, dtype=np.float64)
    site = spec.find_body("wrist_3_link").add_site()
    site.name, site.pos, site.quat = "hand_mount", at[:3, 3], quat_of(at[:3, :3])
    hand, _ = hand_spec()
    site.attach(hand.worldbody.first_body(), "hand_", "")
    for body in spec.bodies:
        body.gravcomp = 1.0
    return spec


class RobotModel:
    """Kinematics of the whole robot for given joint angles: the flange pose and dense surface points of the arm and hand (for masking them out of the camera clouds)."""

    def __init__(self, mount, n_mask_points=20000, seed=0):
        """Compile the robot and draw the masking points once on all visual geoms."""
        from lawm.sim import clouds
        self.model = cell_spec(mount).compile()
        self.data = mujoco.MjData(self.model)
        _, self.hand_joint_names = hand_spec()
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qpos = np.array([self.model.jnt_qposadr[j] for j in jid])
        hand = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"hand_joint_{k}") for k in range(len(self.hand_joint_names))]
        self.hand_qpos = np.array([self.model.jnt_qposadr[j] for j in hand])
        self.hand_limits = np.array([self.model.jnt_range[j] for j in hand])
        self.flange = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, FLANGE_SITE)
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_PREFIX + "hand_C_MC")
        visual = [g for b in range(1, self.model.nbody) for g in clouds.body_geom_ids(self.model, b, "visual")]
        self.mask_sampler = clouds.SurfaceSampler(self.model, visual, n_mask_points, np.random.default_rng(seed))
        self.set(np.zeros(6), np.zeros(len(hand)))
        self.root_in_flange = np.linalg.inv(self.flange_pose()) @ self.root_pose()

    def set(self, q_arm, q_hand):
        """Place the robot at joint angles (6 arm, 22 hand) by forward kinematics."""
        self.data.qpos[self.arm_qpos], self.data.qpos[self.hand_qpos] = q_arm, q_hand
        mujoco.mj_kinematics(self.model, self.data)

    def flange_pose(self):
        """4x4 flange pose in the base frame at the current angles."""
        return pose(self.data.site_xmat[self.flange].reshape(3, 3), self.data.site_xpos[self.flange])

    def root_pose(self):
        """4x4 pose of the hand's root body (the palm) at the current angles."""
        return pose(self.data.xmat[self.root].reshape(3, 3), self.data.xpos[self.root])

    def surface(self, q_arm, q_hand):
        """Dense points (n, 3) on the visual surface of the arm and the hand at these angles."""
        self.set(q_arm, q_hand)
        return self.mask_sampler(self.data)
