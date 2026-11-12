"""Camera calibration against the robot itself: over several arm poses, the pixels whose depth differs from the per-pixel median (the static scene) are the robot; their points are matched (ICP) to the robot's surface at its measured joint angles, from a rough tape-measured camera pose."""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import yaml
from scipy.spatial import cKDTree

from lawm.sim.dexwm.control.hand_fit import kabsch

from lawm_deploy.perception import back_project, voxel_down
from lawm_deploy.robot_model import pose

RADII_M = (0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.012, 0.008, 0.005)  # pairs farther apart than this are ignored, shrinking as the fit improves
MOVED_M = 0.02  # a pixel this much off its median depth over the poses shows the (moving) robot
SKIP_BODIES = ("base", "shoulder_link")  # they barely move between poses, so their pixels look static


def look_at_optical(eye, target):
    """4x4 pose of a camera's optical frame (x right, y down, z forward) at eye looking at target, z up in the world."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return pose(np.stack([right, np.cross(f, right), f], axis=1), eye)


def fit_camera(clouds, surfaces, rough, iters=8):
    """Camera pose that best puts its points (camera frame, one cloud per arm pose) onto the robot surfaces, starting from rough; returns (4x4 pose, median residual mm, share of the points matched)."""
    trees = [cKDTree(s) for s in surfaces]
    T = np.array(rough, dtype=float)
    for radius in RADII_M:
        for _ in range(iters):
            src, dst, dist = [], [], []
            for pts, tree, surf in zip(clouds, trees, surfaces):
                world = pts @ T[:3, :3].T + T[:3, 3]
                d, i = tree.query(world, distance_upper_bound=radius)
                ok = d < radius
                src.append(world[ok]), dst.append(surf[i[ok]]), dist.append(d[ok])
            src, dst = np.concatenate(src), np.concatenate(dst)
            if len(src) < 100:
                raise RuntimeError(f"only {len(src)} camera points within {radius * 100:.0f} cm of the robot: is the rough pose right and the robot in view?")
            r, t = kabsch(src, dst)
            T = pose(r, t) @ T
    return T, float(np.median(np.concatenate(dist)) * 1000), len(src) / sum(len(c) for c in clouds)


def moving_points(depths, K, near, far, voxel):
    """Per pose, the camera-frame points of the pixels whose depth is more than MOVED_M off the per-pixel median over all poses."""
    stack = np.stack(depths)
    valid = (stack > near) & (stack < far)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels with no depth in any pose
        median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    out = []
    for d, ok in zip(stack, valid):
        moved = ok & ~(np.abs(d - median) <= MOVED_M)
        out.append(voxel_down(back_project(np.where(moved, d, 0.0), K, np.eye(4), near, far), voxel))
    return out


def calibrate(robot, frames, rough_poses, near=0.2, far=2.0, voxel=0.005):
    """Fit every camera; frames = [(q_arm, q_hand, [(depth, K, _) per camera])] at different arm poses (at least 3, the arm well apart), rough_poses = one rough 4x4 per camera."""
    import mujoco
    body = robot.model.geom_bodyid[robot.mask_sampler.owner]
    keep = ~np.isin(body, [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in SKIP_BODIES])
    surfaces = [robot.surface(q_arm, q_hand)[keep] for q_arm, q_hand, _ in frames]
    out = []
    for c, rough in enumerate(rough_poses):
        clouds = moving_points([views[c][0] for _, _, views in frames], frames[0][2][c][1], near, far, voxel)
        out.append(fit_camera(clouds, surfaces, rough))
    return out


def main():
    """python -m lawm_deploy.calibrate --config configs/real.yaml [--poses 5]: move the arm by hand to each pose, then write the camera poses."""
    from lawm_deploy import config
    from lawm_deploy.cell import RealCell, camera_poses
    from lawm_deploy.robot_model import RobotModel, hand_spec
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--poses", type=int, default=5)
    args = parser.parse_args()
    cfg = config.load(args.config)
    robot = RobotModel(config.mount_pose(cfg), n_mask_points=100000)
    frames = []
    cell = RealCell(cfg["cell"], hand_spec()[1], calibrated=False)
    try:
        for k in range(args.poses):
            input(f"Pose {k + 1}/{args.poses}: use the pendant to arrange the arm, END pendant freedrive and let it hold still, then press Enter. This tool is read-only. ")
            time.sleep(1.0)
            q_arm, q_hand, _ = cell.state()
            views = cell.views()
            qa, qh, _ = cell.state()
            if np.max(np.abs(qa-q_arm)) > 0.01 or np.max(np.abs(qh-q_hand)) > 0.03:
                raise RuntimeError("Robot moved during calibration capture")
            frames.append((q_arm, q_hand, views))
    finally:
        cell.close()
    results = calibrate(robot, frames, camera_poses(cfg["cell"]["cameras"], calibrated=False))
    out = {}
    for d, (T, residual, share) in zip(cfg["cell"]["cameras"]["devices"], results):
        verdict = "OK" if share > 0.7 else "SUSPECT: re-measure the rough pose (eye within 3 cm, aim within 5 deg) and rerun"
        print(f"camera {d['serial']}: {share:.0%} of the moving (robot) points matched within 5 mm, median {residual:.1f} mm -> {verdict}")
        out[str(d["serial"])] = np.round(T, 6).tolist()
    path = config.repo_path(cfg["cell"]["cameras"]["calibration"])
    # Do not silently replace a previous calibration, or accept a failed ICP fit.
    if any(share <= 0.7 for _, _, share in results):
        raise RuntimeError("Calibration SUSPECT; not writing calibration.yaml")
    if path.exists():
        raise FileExistsError(f"Preserve/review the existing calibration before replacing {path}")
    path.write_text(yaml.safe_dump(dict(frame="depth_optical", T_base_camera=out,
                                      provenance=f"robot-surface ICP {time.strftime('%Y-%m-%d %H:%M')}",
                                      physical_validation="PENDING")))
    print(f"written to {path}")


if __name__ == "__main__":
    main()
