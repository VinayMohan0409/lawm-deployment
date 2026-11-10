"""The Sharpa hand as the planner sees it: n points drawn once on its visual surface (row i is the same point in every cloud), placed by forward kinematics, and the fit of a palm pose and 22 angles to points."""
from __future__ import annotations

import numpy as np

from lawm.sim import clouds
from lawm.sim.dexwm.control.hand_fit import HandKinematics
from lawm.sim.dexwm.scene.hand_surface import hand_finger_mask, hand_geom_ids

from lawm_deploy.robot_model import HAND_PREFIX


class Hand:
    """Points of the mounted hand from its palm pose and joint angles, with lawm's own kinematics and fit."""

    def __init__(self, model, n, seed=0):
        """Draw n points over the hand's visual geoms of a compiled robot model."""
        sampler = clouds.SurfaceSampler(model, hand_geom_ids(model, HAND_PREFIX), n, np.random.default_rng(seed))
        self.sampler = sampler
        self.kinematics = HandKinematics(model, sampler, HAND_PREFIX)
        self.finger_mask = hand_finger_mask(model, sampler.owner, HAND_PREFIX)
        self.n = n

    def points(self, root_pose, q):
        """(n, 3) float32 hand points for a 4x4 palm pose and 22 angles."""
        return self.kinematics.points(root_pose[:3, :3], root_pose[:3, 3], np.asarray(q, dtype=float)).astype(np.float32)

    def fit(self, points, q_init):
        """(palm rotation, palm position, 22 angles, info with mean_residual_mm) that best explain the points."""
        return self.kinematics.fit(np.asarray(points, dtype=np.float64), q_init)
