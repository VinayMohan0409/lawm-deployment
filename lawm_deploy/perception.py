"""Depth images to the planner's cloud: the camera points with the robot removed (by its measured joint angles), plus the Sharpa mesh points placed by forward kinematics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from lawm.sim.costs import SEG_HAND, SEG_OBJECT

SEG_SCENE = SEG_OBJECT  # every camera point gets one label (no object/table split); lawm's costs read it as the object


class PerceptionError(RuntimeError):
    """The cameras give too few points."""


@dataclass(frozen=True)
class PerceptionSettings:
    """Depth range (m), workspace box in the base frame (m) and whether to crop to it, robot mask margin (m), voxel size (m), hand and scene point counts, fewest scene points accepted."""

    near_m: float
    far_m: float
    use_box: bool
    box_lo: tuple
    box_hi: tuple
    robot_margin_m: float
    voxel_m: float
    n_hand: int
    n_scene: int
    min_scene_points: int


def back_project(depth, K, cam_pose, near, far):
    """Base-frame points (n, 3) of the valid pixels of a depth image (m) with intrinsics K and the 4x4 pose of the optical frame (x right, y down, z forward)."""
    v, u = np.nonzero((depth > near) & (depth < far))
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return cam @ cam_pose[:3, :3].T + cam_pose[:3, 3]


def voxel_down(points, size):
    """One measured point (the first) per occupied voxel, as PointWorld's grid_sample_transform in test mode; unchanged for size 0."""
    if size <= 0 or not len(points):
        return points
    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


class Perception:
    """The cloud the planner reads: n_hand mesh points (FK), then exactly n_scene camera points (lawm's planner compares clouds row by row, so the size is fixed)."""

    def __init__(self, settings, seed=0):
        """Keep the settings and a random generator for the subsampling."""
        self.s, self.rng = settings, np.random.default_rng(seed)
        self.segment = np.r_[np.full(settings.n_hand, SEG_HAND), np.full(settings.n_scene, SEG_SCENE)].astype(np.int8)

    def scene_points(self, views, robot_points):
        """Camera points (inside the workspace box when use_box), one per voxel, without the robot; views = [(depth m, K, 4x4 optical pose)]."""
        s = self.s
        pts = np.concatenate([back_project(d, K, T, s.near_m, s.far_m) for d, K, T in views])
        if s.use_box:
            pts = pts[(pts >= s.box_lo).all(1) & (pts <= s.box_hi).all(1)]
        pts = voxel_down(pts, s.voxel_m)
        if len(robot_points) and len(pts):
            pts = pts[cKDTree(robot_points).query(pts, distance_upper_bound=s.robot_margin_m)[0] > s.robot_margin_m]
        return pts

    def __call__(self, views, robot_points, hand_points):
        """(coords (n_hand + n_scene, 3) float32, segment, info) for one observation."""
        s = self.s
        pts = self.scene_points(views, robot_points)
        if len(pts) < s.min_scene_points:
            raise PerceptionError(f"only {len(pts)} camera points, need {s.min_scene_points}")
        assert len(hand_points) == s.n_hand, f"{len(hand_points)} hand points, config says {s.n_hand}"
        if len(pts) >= s.n_scene:  # more voxels than the planner takes: random subsample, as PointWorld's enforce_max_num_points
            keep = np.sort(self.rng.choice(len(pts), s.n_scene, replace=False))
        else:  # fewer: every point once, then repeats (the model's 5 mm voxel step merges repeats)
            keep = np.r_[np.arange(len(pts)), self.rng.choice(len(pts), s.n_scene - len(pts))]
        scene = pts[keep]
        return np.concatenate([hand_points, scene]).astype(np.float32), self.segment, dict(n_scene_seen=len(pts), n_repeated=max(0, s.n_scene - len(pts)))
