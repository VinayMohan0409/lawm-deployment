"""Run one task in a MuJoCo copy of a real run's cell, through the deployment Session (the same loop, perception, fit and safety as the robot).

The cell comes from a real run directory (run.json, goal.npz, start.npz):
- the hand mount;
- the two calibrated cameras;
- the table height;
- a box object fitted to the start cloud (the scene points above the table nearest the path of the hand from start to goal).

Goals are made in the sim:
- push: the hand is driven in one straight line from the run's start pose to the run's goal pose, and the physics pushes the box.
- grasp: the robot is placed at the run's goal angles around the object, as an operator arranges it, without simulating. The object is a cylinder fitted to the
  start depth images of both cameras: RANSAC removes the table plane, the cluster above it nearest the goal hand is the object, a circle fit gives its centre
  and radius, and it is moved/shrunk (at most 1 cm / to the fit) until no goal hand point is inside it.
- grasp_lift: the grasp goal, then the hand and the box raised by --lift-m (the lift goal).

PYTHONPATH=<lawm>/src:<lawm-deployment> MUJOCO_GL=egl python scripts/sim_task.py --run RUN_DIR --task push --out OUT [--standin]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

try:  # torch before mujoco's renderer libraries (lawm-model env segfaults otherwise)
    import torch  # noqa: F401
except ImportError:
    pass
import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from lawm_deploy import config  # noqa: E402
from lawm_deploy.replica import ReplicaCell, table_height  # noqa: E402

GOAL_KIND = {"push": "push", "grasp": "grasp", "grasp_lift": "grasp"}


def fit_box(coords, segment, table_z, hand_from, hand_to, eps=0.02, min_points=30):
    """The object box (centre xy, size xyz, yaw deg) of the scene cluster above the table nearest the hand's path (xy) from hand_from to hand_to."""
    p = coords[(segment != 0) & (coords[:, 2] > table_z + 0.012) & (coords[:, 2] < table_z + 0.3)]
    p = np.unique(p, axis=0)
    pairs = cKDTree(p).query_pairs(eps, output_type="ndarray")
    n, label = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p))), directed=False)
    a, b = hand_from[:2], hand_to[:2]

    def to_path(c):
        t = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        return np.linalg.norm(c - (a + t * (b - a)))
    clusters = [p[label == k] for k in range(n) if (label == k).sum() >= min_points]
    obj = min(clusters, key=lambda c: to_path(c[:, :2].mean(0)))
    xy = obj[:, :2] - obj[:, :2].mean(0)
    axes = np.linalg.eigh(xy.T @ xy)[1][:, ::-1]
    proj = xy @ axes
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    centre = obj[:, :2].mean(0) + axes @ ((lo + hi) / 2)
    size = np.maximum(np.r_[hi - lo, np.percentile(obj[:, 2], 98) - table_z], 0.04)
    return dict(shape="box", pos=centre.tolist(), size=size.tolist(), yaw_deg=float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0]))), mass=0.2), obj


def depth_points(npz):
    """All points (base frame) of both cameras' median depth images."""
    out = []
    for i in (0, 1):
        cam, d = npz[f"cam_{i}"], npz[f"depth_median_m_{i}"].astype(float)
        K, T = cam[:9].reshape(3, 3), cam[9:].reshape(4, 4)
        v, u = np.nonzero((d > 0.2) & (d < 1.5))
        out.append(np.stack([(u - K[0, 2]) * d[v, u] / K[0, 0], (v - K[1, 2]) * d[v, u] / K[1, 1], d[v, u]], 1) @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(out)


def ransac_cylinder(npz, near_xy, hand, iters=500, plane_mm=6.0, seed=0):
    """(centre xy, radius, top z, n points) of the upright object nearest near_xy: RANSAC finds the table plane in both depth images, the points
    8 mm to 25 cm above it (minus the hand's) are clustered, and a circle is fitted to the cluster nearest near_xy."""
    rng = np.random.default_rng(seed)
    p = depth_points(npz)
    p = p[(np.abs(p[:, 0]) < 0.45) & (p[:, 1] > -0.95) & (p[:, 1] < -0.3) & (p[:, 2] < 0.3)]
    low = p[p[:, 2] < 0.0][::20]
    best = None
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9 or abs(n[2]) < 0.9 * np.linalg.norm(n):
            continue
        n = n / np.linalg.norm(n)
        inl = np.abs((low - a) @ n) < plane_mm / 1000
        if best is None or inl.sum() > best.sum():
            best = inl
    q = low[best]
    n = np.linalg.svd(q - q.mean(0))[2][-1]
    n = n * np.sign(n[2])
    h = (p - q.mean(0)) @ n
    obj = p[(h > 0.008) & (h < 0.25)]
    obj = obj[cKDTree(hand).query(obj)[0] > 0.015][::3]
    pairs = cKDTree(obj).query_pairs(0.01, output_type="ndarray")
    _, label = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(obj), len(obj))), directed=False)
    clusters = [obj[label == k] for k in np.unique(label) if (label == k).sum() >= 200]
    o = min(clusters, key=lambda c: np.linalg.norm(c[:, :2].mean(0) - near_xy))
    top = float(np.percentile(o[:, 2], 98))
    xy = o[o[:, 2] < top - 0.015, :2]  # the sides (the top face would pull the circle in)
    sol = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
    return sol[:2], float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)), top, len(o)


def box_pose(cell):
    """(position, quaternion w x y z) of the box."""
    a = cell.object_qpos[0]
    q = cell.data.qpos[a:a + 7].copy()
    return q[:3], q[3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a real run directory (run.json, goal.npz, start.npz)")
    ap.add_argument("--task", required=True, choices=sorted(GOAL_KIND))
    ap.add_argument("--out", required=True)
    ap.add_argument("--standin", action="store_true", help="stand-in model, no GPU (smoke test)")
    ap.add_argument("--max-replans", type=int, default=40)
    ap.add_argument("--execute-steps", type=int, default=None, help="default: configs/real.yaml loop.<task>.execute_steps (push 5, grasp 3)")
    ap.add_argument("--arm", choices=["async", "sync"], default="async", help="async: as the real UR (moveL, fingers on their own timer)")
    ap.add_argument("--lift-m", type=float, default=0.10)
    ap.add_argument("--object-mass", type=float, default=0.3, help="grasp tasks: mass of the cylinder (kg)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lawm_deploy.model import StandInModel, load_model
    from lawm_deploy.robot_model import RobotModel
    from lawm_deploy.session import Session, task_distance_mm

    run = Path(args.run)
    real = json.loads((run / "run.json").read_text())
    goal_npz, start_npz = np.load(run / "goal.npz"), np.load(run / "start.npz")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # her configuration, with this repo's tasks/weights and a simulated cell
    cfg = copy.deepcopy(real["config"])
    mine = config.load(config.REPO / "configs/real.yaml")
    cfg["planner"], cfg["loop"] = mine["planner"], mine["loop"]
    cfg["loop"][args.task]["max_replans"] = args.max_replans
    cfg["seed"], cfg["log_dir"] = args.seed, str(out)
    cfg["experiment"].update(task=args.task, goal=None, sample_pair=None)
    if args.execute_steps is not None:
        cfg["experiment"]["execute_steps"] = args.execute_steps
        cfg["loop"][args.task].pop("execute_steps", None)
    table_z = table_height(goal_npz)
    cams = [(goal_npz[f"cam_{i}"][:9].reshape(3, 3), goal_npz[f"cam_{i}"][9:].reshape(4, 4)) for i in (0, 1)]
    mount = config.mount_pose(cfg)
    robot = RobotModel(mount, seed=cfg["seed"])
    robot.set(start_npz["q_arm"], start_npz["q_hand"])
    start_flange = robot.flange_pose()
    robot.set(goal_npz["q_arm"], goal_npz["q_hand"])
    goal_flange = robot.flange_pose()
    h = start_npz["segment"] == 0
    if GOAL_KIND[args.task] == "push":  # the box the camera sees between the start and the goal hand
        box, box_points = fit_box(start_npz["coords"], start_npz["segment"], table_z, start_npz["coords"][h].mean(0), goal_npz["coords"][h].mean(0))
    else:  # the object the goal hand wraps, from the start depth images (RANSAC table removal); the camera and FK hand disagree by about a cm,
        # so the cylinder is then moved (at most 1 cm) and shrunk until no goal hand point is inside it, as the operator's hand wraps the real one
        from lawm_deploy.replica import fit_cup
        gh = goal_npz["coords"][goal_npz["segment"] == 0]
        sh = start_npz["coords"][start_npz["segment"] == 0]
        centre, radius, top, n_obj = ransac_cylinder(start_npz, gh.mean(0)[:2], sh)
        height = top - table_z
        print(f"object from RANSAC: {n_obj} points, centre {np.round(centre, 3).tolist()}, radius {radius:.3f} m, height {height:.3f} m", flush=True)
        centre, radius = fit_cup(gh, centre, table_z, height, search_m=0.01, max_radius=radius)
        assert radius > 0.02, f"no graspable cylinder inside the goal hand (radius {radius:.3f} m)"
        box = dict(pos=list(centre), radius=radius, height=height, mass=args.object_mass)
    target = [*box["pos"], table_z + 0.05]
    cfg["cell"] = dict(type="sim", width=848, height=480, noise_at_1m=0.002, dropout=0.02, motion=real["config"]["cell"]["motion"],
                       guards=dict(hand_feedback_limit_tolerance_deg=real["config"]["cell"].get("guards", {}).get("hand_feedback_limit_tolerance_deg", 0.0)),
                       video=dict(eye=[target[0] + 0.45, target[1] - 0.6, target[2] + 0.35], target=target, fovy_deg=50, width=640, height=480, fps=20))
    cell = ReplicaCell(cfg["cell"], mount, cams, table_z, box, args.seed)
    cell.arm_mode = args.arm
    model = StandInModel() if args.standin else load_model(cfg)
    session = Session(cell, cfg, model, out / "log")

    # the scene at the start: robot at the run's start angles, the box where the start cloud shows it, settled
    cell.place(start_npz["q_arm"], start_npz["q_hand"])
    cell.wait(0.5)
    box0 = box_pose(cell)
    kind = GOAL_KIND[args.task]
    if kind == "push":  # one straight-line move from the start to the goal pose; the box is pushed by the physics
        cell.arm_mode = "sync"
        cell.move(goal_flange, goal_npz["q_hand"])
        cell.arm_mode = args.arm
        session.capture_goal()
    else:  # the robot placed around the box at the goal angles, nothing simulated (as an operator arranges the real cell)
        cell.place(goal_npz["q_arm"], goal_npz["q_hand"], box0[0], box0[1])
        session.capture_goal()
        if args.task == "grasp_lift":
            lifted = goal_flange.copy()
            lifted[2, 3] += args.lift_m
            q_lift = cell.ik(lifted, goal_npz["q_arm"])
            assert q_lift is not None, "no IK for the lifted goal"
            cell.place(q_lift, goal_npz["q_hand"], box0[0] + [0, 0, args.lift_m], box0[1])
            session.capture_goal_lift()
    box_goal = box_pose(cell)
    cell.place(start_npz["q_arm"], start_npz["q_hand"], box0[0], box0[1])
    cell.wait(0.5)
    session.capture_start()

    trace = []

    def on_step(record):
        q_arm, q_hand, flange = cell.state()
        b = box_pose(cell)
        trace.append(dict(replan=record["replan"], stage=record.get("stage", 0), distance_mm=record["distance_mm"], moved=record["moved"],
                          hold=record["hold"], flange=flange[:3, 3].tolist(), q_hand=q_hand.tolist(), box=b[0].tolist(), box_quat=b[1].tolist()))
        print(f"replan {record['replan']:2d} stage {record.get('stage', 0)} dist {record['distance_mm']:6.1f} mm  box {np.round(b[0] * 1000).astype(int).tolist()} mm"
              f"{'  HOLD ' + record['hold'] if record['hold'] else ''}", flush=True)

    session.begin(args.task)
    reason = session.run(execute=True, on_step=on_step)

    # evaluation (none of it is in the cost)
    final_q = cell.state()[1]
    names = robot.hand_joint_names
    pip = [i for i, n in enumerate(names) if n.endswith("_PIP")]
    flange_path = np.array([t["flange"] for t in trace]) if trace else np.zeros((0, 3))
    line = goal_flange[:3, 3] - start_flange[:3, 3]
    lateral = [float(np.linalg.norm(np.cross(p - start_flange[:3, 3], line)) / np.linalg.norm(line)) for p in flange_path]
    push_dir = box_goal[0][:2] - box0[0][:2]
    final_box = box_pose(cell)[0]
    summary = dict(
        task=args.task, model="standin" if args.standin else "checkpoint", reason=reason, replans=len(trace), arm=args.arm,
        execute_steps=session.execute_steps, box_fit=box, table_z=table_z,
        final_distance_mm=trace[-1]["distance_mm"] if trace else None,
        min_distance_mm=min((t["distance_mm"] for t in trace), default=None),
        initial_distance_mm=task_distance_mm(session.start, session.goal),
        hand_path_max_lateral_mm=1000 * max(lateral, default=0.0),
        box_start=box0[0].tolist(), box_goal=box_goal[0].tolist(), box_final=final_box.tolist(),
        box_goal_displacement_mm=1000 * float(np.linalg.norm(push_dir)),
        push_progress=float(np.dot(final_box[:2] - box0[0][:2], push_dir) / max(np.dot(push_dir, push_dir), 1e-12)) if np.linalg.norm(push_dir) > 0.005 else None,
        box_lift_mm=1000 * float(final_box[2] - box0[0][2]),
        finger_error_deg_mean=float(np.degrees(np.abs(final_q - goal_npz["q_hand"]).mean())),
        pip_final_deg=np.degrees(final_q[pip]).round(1).tolist(), pip_goal_deg=np.degrees(goal_npz["q_hand"][pip]).round(1).tolist(),
        run_dir=str(session.run_dir))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "trace.json").write_text(json.dumps(trace))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("box_fit",)}, indent=1))
    cell.close()


if __name__ == "__main__":
    main()
