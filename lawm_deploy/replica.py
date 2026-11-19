"""A simulated copy of one real run, to test deployment-side changes before they go to the robot.

The cell is rebuilt from the run's bundle: the hand mount, the two calibrated cameras, the table height and a cylinder in place of the cup;
the robot starts at the run's start angles and the goal is captured at the goal capture's angles. The loop is the one the real run used:
a plan is reused for execute_steps moves (step k takes delta[k] of the plan), the translation is capped at max_step_mm and every finger
joint at the finger cap. Three switches change only the execution:

  --arm async        as on the real UR: moveL turns the tool at about UR_ROT_PER_SPEED rad/s per m/s of commanded speed, fingers run on
                     their own minimum-jerk timer of the planned duration (so they finish first when the wrist turns)
  --arm sync         arm and fingers on one minimum-jerk clock (what streaming servoJ would give)
  --fingers full     the whole fitted finger step every move (real run)
  --fingers scaled   the finger step scaled by the same fraction as the translation cap (arm and fingers keep the model's proportions)
  --hover off        no hover stage: the grasp goes straight for the goal hand (real run: on, 200 mm above first)
  --penetration W    add W x the penetration cost: predicted hand points may come no closer to the scene observed at planning time than
                     PENETRATION_MARGIN_M (or than the same point is in the goal, where the goal hand touches the object)
  --box-lo-x X       the perception crop's lower x (the real run's -0.05 m cut most of the cup out of the cloud)

--replay executes the real run's logged commands instead of planning (no model, no GPU), to compare execution alone.

PYTHONPATH=<lawm>/src:<lawm-deployment> MUJOCO_GL=osmesa python -m lawm_deploy.replica --bundle DIR --out DIR [options]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

try:  # torch must load before mujoco's renderer libraries: imported after them it segfaults (lawm-model env)
    import torch  # noqa: F401, E402
except ImportError:
    pass
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from lawm_deploy import config  # noqa: E402
from lawm_deploy.cell import MotionSettings, min_jerk  # noqa: E402
from lawm_deploy.robot_model import ARM_JOINTS, FLANGE_SITE, RobotModel, cell_spec, pose, quat_of  # noqa: E402
from lawm_deploy.sim_cell import SimCell, look_at  # noqa: E402

UR_ROT_PER_SPEED = 4.5  # rad/s of tool rotation per m/s of moveL speed: fit to the 41 moves of the 2026-11-15 grasp_cup run (UR7e)
HAND_ARRIVAL_RAD = 0.12  # the real cell's guards.hand_arrival_rad
MINJERK_PEAK = 1.875  # peak speed of a minimum-jerk move over its mean speed; the real cell sizes its moves with it
FINGER_GROUPS = {"thumb": range(0, 5), "index": range(5, 9), "middle": range(9, 13), "ring": range(13, 17), "pinky": range(17, 22)}


def trapezoid(t, total, ramp):
    """Progress 0..1 of a trapezoidal speed profile (constant speed between two ramps) at time t of total."""
    t, ramp = float(np.clip(t, 0.0, total)), min(ramp, total / 2)
    top = 1.0 / (total - ramp)
    if t < ramp:
        return 0.5 * top * t * t / ramp
    if t > total - ramp:
        return 1.0 - 0.5 * top * (total - t) ** 2 / ramp
    return top * (t - ramp / 2)


def real_duration(m, flange, target, q0, q1):
    """The planned move time of the real cell: minimum-jerk at the configured peak speeds of translation, rotation and fingers."""
    dist = float(np.linalg.norm(target[:3, 3] - flange[:3, 3]))
    rot = float(Rotation.from_matrix(target[:3, :3] @ flange[:3, :3].T).magnitude())
    fingers = float(np.abs(np.asarray(q1) - q0).max())
    return max(m.min_move_s, MINJERK_PEAK * max(dist / m.speed, rot / m.rot_speed, fingers / m.finger_speed))


def ur_movel_time(m, dist, rot, duration):
    """How long the UR takes for the real cell's moveL: it commands speed = distance / planned duration, and the tool turns at only UR_ROT_PER_SPEED x that speed."""
    speed = min(m.speed, max(dist / duration, 1e-5)) if dist > 1e-6 else m.rot_speed / UR_ROT_PER_SPEED
    return max(dist / speed, rot / (UR_ROT_PER_SPEED * speed)) + speed / m.accel, speed


class ReplicaCell(SimCell):
    """The run's cell in MuJoCo: SimCell's arm, hand, rendering and IK, with the run's cameras, table and a cylinder for the object."""

    def __init__(self, cfg, mount, cameras, table_z, cup, seed=0):
        """cameras = [(K, optical pose)], cup = dict(pos=(x, y), radius, height, mass), or a box: dict(shape="box", pos=(x, y), size=(sx, sy, sz), yaw_deg, mass)."""
        self.cfg, self.motion = cfg, MotionSettings(**{f.name: cfg["motion"][f.name] for f in dataclasses.fields(MotionSettings)})
        self.cam_K = [K for K, _ in cameras]
        spec = cell_spec(mount)
        vis = getattr(spec.visual, "global")
        vis.offwidth, vis.offheight = max(cfg["width"], cfg["video"]["width"]), max(cfg["height"], cfg["video"]["height"])
        world = spec.worldbody
        world.add_light(pos=[0.3, -0.4, 2.0], dir=[0, 0, -1], directional=True)
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.7, 0.7, 0.025], pos=[0.0, -0.55, table_z - 0.025], rgba=[0.6, 0.5, 0.4, 1])
        if cup.get("shape") == "box":
            body = world.add_body(name="object_0", pos=[*cup["pos"], table_z + cup["size"][2] / 2],
                                  quat=Rotation.from_euler("z", cup.get("yaw_deg", 0.0), degrees=True).as_quat()[[3, 0, 1, 2]])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[v / 2 for v in cup["size"]], mass=cup["mass"], friction=[1.0, 0.01, 0.001], rgba=[0.85, 0.6, 0.3, 1])
        else:
            body = world.add_body(name="object_0", pos=[*cup["pos"], table_z + cup["height"] / 2])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[cup["radius"], cup["height"] / 2, 0], mass=cup["mass"], friction=[1.0, 0.01, 0.001], rgba=[0.85, 0.85, 0.9, 1])
        for i, (K, T) in enumerate(cameras):
            fovy = float(np.degrees(2 * np.arctan(0.5 * cfg["height"] / K[1, 1])))
            world.add_camera(name=f"cam_{i}", pos=T[:3, 3], quat=quat_of(T[:3, :3] @ np.diag([1.0, -1.0, -1.0])), fovy=fovy)
        v = cfg["video"]
        world.add_camera(name="video", pos=v["eye"], quat=quat_of(look_at(np.asarray(v["eye"], float), v["target"])), fovy=v["fovy_deg"])
        self.model = spec.compile()
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.arm_dof = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.hand = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"hand_joint_{k}")] for k in range(22)])
        self.object_qpos = [int(self.model.jnt_qposadr[self.model.body_jntadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_0")]])]
        self.flange = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, FLANGE_SITE)
        self.cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{i}") for i in range(len(cameras))]
        self.renderer = mujoco.Renderer(self.model, cfg["height"], cfg["width"])
        self.renderer.enable_depth_rendering()
        self.video = mujoco.Renderer(self.model, v["height"], v["width"])
        self.writer, self.frame, self.markers, self.next_frame = None, None, np.zeros((0, 3)), 0.0
        self.stop_requested = threading.Event()
        self.rng = np.random.default_rng(seed)
        self.dt = self.model.opt.timestep
        self.arm_mode = "async"

    def views(self):
        """[(depth m, K, optical pose)] of both cameras, with the run's intrinsics and SimCell's depth noise and dropout."""
        h, w = self.cfg["height"], self.cfg["width"]
        out = []
        for cam in self.cams:
            self.renderer.update_scene(self.data, cam)
            depth = self.renderer.render().astype(np.float32).copy()
            depth += self.rng.normal(0, 1, depth.shape) * self.cfg["noise_at_1m"] * depth ** 2
            depth[self.rng.uniform(size=depth.shape) < self.cfg["dropout"]] = 0.0
            f = 0.5 * h / np.tan(np.radians(self.model.cam_fovy[cam]) / 2)
            K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
            out.append((depth, K, pose(self.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), self.data.cam_xpos[cam])))
        return out

    def place(self, q_arm, q_hand, cup_xyz=None, cup_quat=None):
        """Put the robot (and the cup) somewhere without simulating (a goal hand may overlap the cup it grasps); cup_quat (w, x, y, z), default upright."""
        self.data.qpos[self.arm], self.data.qpos[self.hand] = q_arm, q_hand
        if cup_xyz is not None:
            a = self.object_qpos[0]
            self.data.qpos[a:a + 7] = [*cup_xyz, *(cup_quat if cup_quat is not None else (1, 0, 0, 0))]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:6], self.data.ctrl[6:28] = q_arm, q_hand
        mujoco.mj_forward(self.model, self.data)

    def cup_pose(self):
        """(position (3,), tilt from upright in degrees) of the cup."""
        a = self.object_qpos[0]
        q = self.data.qpos[a:a + 7]
        up = Rotation.from_quat([q[4], q[5], q[6], q[3]]).apply([0, 0, 1])
        return q[:3].copy(), float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))

    def move(self, target, q_hand):
        """One move as the real cell makes it (arm_mode async) or with arm and fingers on one clock (sync); returns the timing."""
        q_arm, q0, flange = self.state()
        q_hand = np.asarray(q_hand, float)
        duration = real_duration(self.motion, flange, target, q0, q_hand)
        dist = float(np.linalg.norm(target[:3, 3] - flange[:3, 3]))
        r0, r1 = Rotation.from_matrix(flange[:3, :3]), Rotation.from_matrix(target[:3, :3])
        rel = (r1 * r0.inv()).as_rotvec()
        if self.arm_mode == "async":
            arm_time, speed = ur_movel_time(self.motion, dist, float(np.linalg.norm(rel)), duration)
            arm_s = lambda t: trapezoid(t, arm_time, speed / self.motion.accel)  # noqa: E731
            finger_time, finger_s = duration, lambda t: min_jerk(t / duration)  # noqa: E731
        else:
            arm_time = finger_time = duration
            arm_s = finger_s = lambda t: min_jerk(t / duration)  # noqa: E731
        total = max(arm_time, finger_time)
        control_dt = 1.0 / self.motion.finger_rate_hz
        q_cmd = q_arm.copy()
        for k in range(1, int(np.ceil(total / control_dt)) + 1):
            t = k * control_dt
            s = arm_s(t)
            way = pose((Rotation.from_rotvec(s * rel) * r0).as_matrix(), flange[:3, 3] + s * (target[:3, 3] - flange[:3, 3]))
            q_ik = self.ik(way, q_cmd)
            if q_ik is None:
                raise RuntimeError("IK failed along the path")
            q_cmd = q_ik
            self.data.ctrl[:6], self.data.ctrl[6:28] = q_cmd, q0 + (q_hand - q0) * finger_s(t)
            self.wait(control_dt)
        self.wait(self.motion.settle_s)
        reached = bool(np.abs(self.state()[1] - q_hand).max() < HAND_ARRIVAL_RAD)  # Session reads it: a blocked finger is a partial move
        return dict(planned_s=duration, arm_s=arm_time, fingers_s=finger_time, hand_target_reached=reached)


PENETRATION_MARGIN_M = 0.005  # a hand point should stay this far in front of the observed surface
PENETRATION_REACH_M = 0.03  # only the surface within 3 cm counts: further behind a surface is out of view, not inside an object
PENETRATION_NEIGHBOURS = 5  # behind the surface of all its 5 nearest scene points (one plane is wrong at edges, e.g. beside a rim)
PENETRATION_TOP_K = 32  # the penetration of a candidate is the mean of its 32 deepest hand points (about one fingertip's worth)


def surface_normals(points, cameras, k=10):
    """Unit normals (n, 3) of a scene cloud by local PCA, each turned towards the camera that faces it most (the camera that saw that side)."""
    from scipy.spatial import cKDTree
    nb = points[cKDTree(points).query(points, k=k)[1]]
    centred = nb - nb.mean(axis=1, keepdims=True)
    normal = np.linalg.eigh(np.einsum("nki,nkj->nij", centred, centred))[1][:, :, 0]
    view = np.stack([c - points for c in cameras], 1)
    view /= np.linalg.norm(view, axis=2, keepdims=True)
    facing = np.einsum("nj,ncj->nc", normal, view)
    return normal * np.where(facing.max(1) >= -facing.min(1), 1.0, -1.0)[:, None]


class Penetration:
    """How far the predicted hand goes into the scene, holding the scene still as observed when planning. Each scene point carries the normal of
    its surface towards the cameras; a hand point within PENETRATION_REACH_M of its nearest scene point has the signed distance (p - s) . n,
    negative behind the surface (inside the object, under the table); the largest over its PENETRATION_NEIGHBOURS nearest scene points counts. It may be no lower than PENETRATION_MARGIN_M, or than the same point's
    signed distance in the goal where the goal hand touches or wraps the object."""

    def __init__(self, goal_coords, goal_segment, cameras, floor_z):
        """The allowed signed distance of every hand point, from the goal cloud; cameras = (3,) positions of the cameras; floor_z = the safety
        floor height, also a surface (the table under the hand is missing from the cloud: hidden by the hand and cut by the robot margin)."""
        from lawm.sim.costs import SEG_HAND
        self.cameras, self.floor_z = [np.asarray(c, float) for c in cameras], floor_z
        hand = goal_segment == SEG_HAND
        self.allowed = np.minimum(PENETRATION_MARGIN_M, self.signed(goal_coords[~hand], goal_coords[hand][None])[0])

    def signed(self, scene, hand, normals=None, tree=None):
        """(pop, n_hand) signed distances of hand points (pop, n_hand, 3) to the scene surface; +inf where no surface is within reach."""
        from scipy.spatial import cKDTree
        tree = tree or cKDTree(scene)
        normals = surface_normals(scene, self.cameras) if normals is None else normals
        flat = hand.reshape(-1, 3)
        d, i = tree.query(flat, k=PENETRATION_NEIGHBOURS, workers=-1)
        sd = np.einsum("nkj,nkj->nk", flat[:, None] - scene[i], normals[i])
        sd = np.where(d < PENETRATION_REACH_M, sd, np.inf).max(axis=1)
        return np.minimum(sd, flat[:, 2] - self.floor_z).reshape(hand.shape[:2])

    def bind(self, coords0, segment):
        """Fix the scene and the start hand of this replan; returns cost(pred_deltas, horizon) -> (pop,) in mm."""
        from scipy.spatial import cKDTree
        from lawm.sim.costs import SEG_HAND
        hand = segment == SEG_HAND
        scene, start = coords0[~hand], coords0[hand]
        tree, normals = cKDTree(scene), surface_normals(scene, self.cameras)

        def cost(pred_deltas, horizon):
            total = np.zeros(pred_deltas.shape[0])
            for k, wk in enumerate(horizon):
                if wk == 0:
                    continue
                sd = self.signed(scene, start[None] + pred_deltas[:, k][:, hand], normals, tree)
                depth = np.maximum(0.0, self.allowed[None] - sd) * 1000.0
                total += wk * np.sort(depth, axis=1)[:, -PENETRATION_TOP_K:].mean(axis=1)
            return total
        return cost


class PalmFrameHandCost:
    """The hand-to-goal cost split into palm pose and finger shape, so the fingers cannot make up for a misplaced palm.
    pose:  the goal hand's shape put on the predicted palm, against the goal hand (world frame): the palm's position and rotation error,
           felt over the whole hand's reach;
    shape: the predicted hand in its own palm frame against the goal hand in the goal palm frame: the finger shape error.
    Both are the per-point weighted mean distance (fingertip weight on the finger points), in mm, like lawm's hand cost; the sum is used."""

    def __init__(self, kinematics, goal_hand, finger_mask, fingertip, shape_weight=1.0):
        from lawm.sim.dexwm.control.hand_fit import kabsch
        self.palm = kinematics.point_body == kinematics.root
        self.L = kinematics.local[self.palm]
        R, t = kabsch(self.L, goal_hand[self.palm])
        self.G, self.G_local = goal_hand, (goal_hand - t) @ R
        w = np.where(finger_mask, fingertip, 1.0)
        self.w = w / w.sum()
        self.shape_weight = shape_weight

    def palm_poses(self, P):
        """Batched Kabsch: rotations (pop, 3, 3) and translations (pop, 3) with P[:, palm] ~ L @ R.T + t."""
        Pp, lc = P[:, self.palm], self.L.mean(0)
        pc = Pp.mean(1)
        U, _, Vt = np.linalg.svd(np.einsum("ni,pnj->pij", self.L - lc, Pp - pc[:, None]))
        V = np.transpose(Vt, (0, 2, 1))
        d = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
        R = (V * np.stack([np.ones_like(d), np.ones_like(d), d], 1)[:, None, :]) @ np.transpose(U, (0, 2, 1))
        return R, pc - np.einsum("pij,j->pi", R, lc)

    def __call__(self, P):
        """(pop,) cost in mm of predicted hands P (pop, n_hand, 3): pose + shape_weight x shape."""
        pose, shape = self.terms(P)
        return pose + self.shape_weight * shape

    def terms(self, P):
        """(pose, shape) costs (pop,) in mm."""
        R, t = self.palm_poses(P)
        local = np.einsum("pnj,pjk->pnk", P - t[:, None], R)
        shape = np.linalg.norm(local - self.G_local[None], axis=2)
        pose = np.linalg.norm(np.einsum("nk,pjk->pnj", self.G_local, R) + t[:, None] - self.G[None], axis=2)
        return (pose * self.w[None]).sum(1) * 1000.0, (shape * self.w[None]).sum(1) * 1000.0


def make_planner(task, goal, finger_mask, model, seed, cem, weights, gates, hover, penetration, cameras, floor_z, kinematics=None, shape_weight=1.0):
    """lawm's TaskPlanner, without its hover gate if hover is False, with W x the Penetration cost added if penetration > 0, and with the hand
    cost split into palm pose and palm-frame finger shape (PalmFrameHandCost) if kinematics is given."""
    from lawm.sim.costs import SEG_HAND
    from lawm.sim.dexwm.planning.cem_replan import plan_replan, replan_rng
    from lawm.sim.dexwm.planning.plan_costs import PlanCost
    from lawm.sim.dexwm.planning.task_planner import EXECUTED_STEP, TaskPlanner

    from lawm.sim.dexwm.planning.plan_costs import hand_goal_cost_mm

    class PenalisedCost(PlanCost):
        def __call__(self, pred_deltas):
            """PlanCost, with the hand term replaced by the palm-frame one (if set), plus penetration x the penetration cost."""
            total = super().__call__(pred_deltas)
            if self.palm_frame is not None:
                w = self.weights
                pw = np.where(self.finger_mask, w.fingertip, 1.0).astype(np.float32)
                total = total - w.hand_scale * hand_goal_cost_mm(pred_deltas, self.coords0, self.goal, self.hand_mask, self.horizon, pw)
                start = self.coords0[self.hand_mask]
                for k, wk in enumerate(self.horizon):
                    if wk:
                        total = total + w.hand_scale * wk * self.palm_frame(start[None] + pred_deltas[:, k][:, self.hand_mask])
            if penetration:
                total = total + penetration * self.extra(pred_deltas, self.horizon)
            return total

    class Planner(TaskPlanner):
        def plan(self, coords, replan, segment=None):
            """TaskPlanner.plan with the penetration term and the palm-frame hand cost."""
            if not penetration and kinematics is None:
                return super().plan(coords, replan, segment)
            segment = self.segment if segment is None else segment
            goal_now = self.gate.goal if self.gate is not None else self.goal
            cost = PenalisedCost(coords, goal_now, segment, self.finger_mask, self.weights)
            cost.extra = self.penetration.bind(coords, segment) if penetration else None
            cost.palm_frame = None if kinematics is None else PalmFrameHandCost(kinematics, goal_now[segment == SEG_HAND], self.finger_mask, self.weights.fingertip, shape_weight)
            z_mean, z_std = self.model.prior
            plan = plan_replan(self.model.decoder(coords, self.model_segment(segment)), cost, z_mean, z_std, self.settings, replan_rng(self.seed, replan))
            hand = segment == SEG_HAND
            return (coords[hand] + plan.delta[EXECUTED_STEP][hand]).astype(np.float32), plan

    planner = Planner(task, goal.coords, goal.segment, finger_mask, model, seed, cem, weights, gates)
    if not hover:
        planner.gate = None
    planner.penetration = Penetration(goal.coords, goal.segment, cameras, floor_z)
    return planner


@dataclasses.dataclass
class Obs:
    coords: np.ndarray
    segment: np.ndarray
    q_arm: np.ndarray
    q_hand: np.ndarray
    flange: np.ndarray
    root: np.ndarray


class Replica:
    """The run's loop on a ReplicaCell."""

    def __init__(self, bundle, out, video_eye=None, cup_at="goal", seed=0, box_lo_x=None, floor_z=None):
        """Read the run's config, cameras, start and goal from the bundle and build the cell, robot model, hand and perception."""
        from lawm_deploy.hand import Hand
        from lawm_deploy.perception import Perception, PerceptionSettings
        from lawm_deploy.safety import SafetySettings
        bundle = Path(bundle)
        run = json.loads((bundle / "run/run.json").read_text())
        self.cfg, self.task = run["config"], run["task"]
        if box_lo_x is not None:
            self.cfg["perception"]["box_lo"][0] = box_lo_x
        if floor_z is not None:
            self.cfg["safety"]["floor_z_m"] = floor_z
        self.bundle, self.out = bundle, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.goal_npz, self.start_npz = np.load(bundle / "run/goal.npz"), np.load(bundle / "run/start.npz")
        cams = [(self.goal_npz[f"cam_{i}"][:9].reshape(3, 3), self.goal_npz[f"cam_{i}"][9:].reshape(4, 4)) for i in (0, 1)]
        self.table_z = table_height(self.goal_npz)
        self.cup = dict(pos=CUP_XY[cup_at], radius=0.04, height=0.12, mass=0.15)
        if cup_at == "goal":  # the largest cylinder near the camera estimate that the goal hand wraps without overlapping it
            self.cup["pos"], self.cup["radius"] = fit_cup(self.goal_npz["coords"][self.goal_npz["segment"] == 0], CUP_XY["goal"], self.table_z, self.cup["height"])
        cell_cfg = dict(width=848, height=480, noise_at_1m=0.002, dropout=0.02, motion=self.cfg["cell"]["motion"],
                        video=dict(eye=video_eye or [0.38, -1.02, 0.26], target=[-0.04, -0.56, 0.08], fovy_deg=45, width=640, height=480, fps=20))
        self.mount = config.mount_pose(self.cfg)
        self.cell = ReplicaCell(cell_cfg, self.mount, cams, self.table_z, self.cup, seed)
        self.robot = RobotModel(self.mount, seed=self.cfg["seed"])
        self.hand = Hand(self.robot.model, self.cfg["perception"]["n_hand"], self.cfg["seed"])
        self.perception = Perception(PerceptionSettings(**self.cfg["perception"]), self.cfg["seed"])
        self.safety = SafetySettings(**{f.name: self.cfg["safety"][f.name] for f in dataclasses.fields(SafetySettings)})
        self.camera_positions = [T[:3, 3] for _, T in cams]
        self.cup_xyz = np.array([*self.cup["pos"], self.table_z + self.cup["height"] / 2])

    def observe(self):
        """The planner's cloud of the cell as it is, built as the real Session builds it."""
        q_arm, q_hand, _ = self.cell.state()
        self.robot.set(q_arm, q_hand)
        flange = self.robot.flange_pose()
        root = flange @ self.robot.root_in_flange
        coords, segment, _ = self.perception(self.cell.views(), self.robot.surface(q_arm, q_hand), self.hand.points(root, q_hand))
        return Obs(coords, segment, q_arm, q_hand, flange, root)

    def setup(self):
        """Capture the goal at the goal angles with the cup in place (no simulation, so the grasping hand may touch it), then put the robot at the start and let the cup settle."""
        self.cell.place(self.goal_npz["q_arm"], self.goal_npz["q_hand"], self.cup_xyz)
        self.goal = self.observe()
        self.cell.place(self.start_npz["q_arm"], self.start_npz["q_hand"], self.cup_xyz)
        self.cell.wait(0.5)
        self.cup0 = self.cell.cup_pose()[0]
        self.cell.set_markers(self.goal.coords[self.goal.segment == 0])

    def record(self, rec, extra):
        """Append a record and print a line."""
        cup, tilt = self.cell.cup_pose()
        rec.update(cup_moved_mm=float(np.linalg.norm(cup[:2] - self.cup0[:2]) * 1000), cup_tilt_deg=tilt, **extra)
        with (self.out / "records.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(" ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if not isinstance(v, (list, dict))), flush=True)

    def replay(self, arm, max_replans, finger_alpha=None, observe=True):
        """Execute the real run's logged commands (the moves it made) in order. finger_alpha: instead of the logged finger command, move each
        finger finger_alpha of the way from where it is to the logged fitted angles (then the run's finger cap); observe: measure the distance."""
        from lawm_deploy.session import task_distance_mm
        self.cell.arm_mode = arm
        run = self.bundle / "run"
        moves = [json.loads(l)["replan"] for l in open(run / "records.jsonl") if json.loads(l)["moved"]][:max_replans]
        cap = np.radians(self.cfg["safety"]["max_finger_deg"] - self.cfg["cell"]["guards"].get("hand_feedback_limit_tolerance_deg", 0.0))
        limits = self.robot.hand_limits
        self.cell.start_recording(self.out / "video.mp4")
        for k in moves:
            c = np.load(run / f"command_replan_{k:03d}.npz")
            fingers = c["fingers"]
            if finger_alpha is not None:
                q_now = self.cell.state()[1]
                fingers = np.clip(q_now + np.clip(finger_alpha * (c["fitted_fingers"] - q_now), -cap, cap), limits[:, 0], limits[:, 1])
            timing = self.cell.move(c["target"], fingers)
            rec = dict(replan=k, fingers=np.asarray(fingers).tolist(), q_hand=self.cell.state()[1].tolist())
            if observe:
                rec["distance_mm"] = task_distance_mm(self.observe(), self.goal)
            self.record(rec, timing)
        self.cell.stop_recording()

    def run(self, model, arm, fingers, hover, penetration, max_replans, execute_steps, finger_alpha=None, palm_frame=False, shape_weight=1.0, pop=None, iters=None):
        """The real run's loop with the planner and the chosen execution."""
        from lawm.sim.costs import SEG_HAND
        from lawm_deploy.safety import Executor, check
        from lawm_deploy.session import planner_settings, task_distance_mm
        self.cell.arm_mode = arm
        cem, weights, gates = planner_settings(self.cfg, self.task)
        cem = dataclasses.replace(cem, **{k: v for k, v in (("pop", pop), ("iters", iters)) if v})
        planner = make_planner(self.task, self.goal, self.hand.finger_mask, model, self.cfg["seed"], cem, weights, gates, hover, penetration, self.camera_positions, self.safety.floor_z_m,
                               self.hand.kinematics if palm_frame else None, shape_weight)
        hand_rows = self.goal.segment == SEG_HAND
        executor = Executor(self.safety, self.cell.state()[2])
        cap = np.radians(self.cfg["safety"]["max_finger_deg"] - self.cfg["cell"]["guards"].get("hand_feedback_limit_tolerance_deg", 0.0))
        limits = self.robot.hand_limits
        loop = self.cfg["loop"][self.task]
        self.cell.start_recording(self.out / "video.mp4")
        need_plan, within, step = True, 0, 0
        for replan in range(max_replans):
            obs = self.observe()
            if need_plan:
                t0 = time.monotonic()
                hand_target, plan = planner.plan(obs.coords, replan)
                ref, step, planning_s = obs, 0, time.monotonic() - t0
                np.savez_compressed(self.out / f"plan_{replan:03d}.npz", delta=plan.delta, coords=obs.coords, segment=obs.segment, q_hand=obs.q_hand, flange=obs.flange,
                                    hover=bool(planner.gate is not None and planner.gate.active), goal=planner.gate.goal if planner.gate is not None else planner.goal)
            else:
                step += 1
                hand_target, planning_s = (ref.coords[hand_rows] + plan.delta[step][hand_rows]).astype(np.float32), 0.0
            rot, pos, q, fit = self.hand.fit(hand_target, obs.q_hand)
            target, q_fit, move_mm = executor.command(rot, pos, q, obs.root, obs.flange)
            share = min(1.0, self.safety.max_step_mm / move_mm) if move_mm > 0 else 1.0
            q_cmd = obs.q_hand + (finger_alpha if finger_alpha is not None else share if fingers == "scaled" else 1.0) * (q_fit - obs.q_hand)
            q_cmd = np.clip(obs.q_hand + np.clip(q_cmd - obs.q_hand, -cap, cap), limits[:, 0], limits[:, 1])
            q_ik = self.cell.ik(target, obs.q_arm)
            hold = check(dataclasses.replace(self.safety, max_finger_deg=999.0), target, q_cmd, obs.q_hand, obs.flange, obs.q_arm, q_ik, fit,
                         self.hand.points(target @ self.robot.root_in_flange, q_cmd))
            timing = {}
            if hold is None:
                timing = self.cell.move(target, q_cmd)
                executor.sent(target)
            new = self.observe()
            phase = bool(planner.update(new.coords, replan))
            dist = task_distance_mm(new, self.goal)
            np.savez_compressed(self.out / f"replan_{replan:03d}.npz", coords=new.coords, segment=new.segment, q_arm=new.q_arm, q_hand=new.q_hand, flange=new.flange,
                                target=target, fingers=q_cmd, fitted_fingers=q_fit, hand_target=hand_target, plan_step=step)
            flex = {f: float(np.degrees((q_cmd - self.start_npz["q_hand"])[list(ix)].sum())) for f, ix in FINGER_GROUPS.items()}
            executed = np.zeros_like(obs.coords)
            executed[hand_rows] = hand_target - obs.coords[hand_rows]
            pen = float(planner.penetration.bind(obs.coords, obs.segment)(executed[None, None], [1.0])[0])  # of the step sent (before the caps)
            self.record(dict(replan=replan, plan_step=step, penetration_mm=pen, distance_mm=dist, unclipped_mm=move_mm, finger_step_deg=float(np.degrees(np.abs(q_cmd - obs.q_hand).max())),
                             fit_mm=fit["mean_residual_mm"], hold=hold, phase_changed=phase, planning_s=planning_s, hover=bool(planner.gate is not None and planner.gate.active),
                             flex_deg=flex), timing)
            need_plan = hold is not None or phase or step + 1 >= min(execute_steps, len(plan.delta))
            within = within + 1 if dist < loop["success_mm"] else 0
            if within >= loop["success_replans"]:
                break
        self.cell.stop_recording()


CUP_XY = {"goal": (-0.03, -0.585), "run": (0.045, -0.62)}  # cup centre (m, base frame) in the goal capture and in the run (it had moved 6-8 cm), from the camera clouds


def fit_cup(hand, near, table_z, height, search_m=0.03, max_radius=0.045, clearance_m=0.003):
    """(centre (x, y), radius) of the largest upright cylinder within search_m of near that no hand point at its heights is inside."""
    hand = hand[(hand[:, 2] > table_z) & (hand[:, 2] < table_z + height)][:, :2]
    best = (tuple(near), 0.0)
    for dx in np.arange(-search_m, search_m + 1e-9, 0.002):
        for dy in np.arange(-search_m, search_m + 1e-9, 0.002):
            c = np.asarray(near) + [dx, dy]
            r = min(max_radius, float(np.linalg.norm(hand - c, axis=1).min()) - clearance_m)
            if r > best[1]:
                best = (tuple(float(v) for v in c), r)
    return best


def table_height(npz):
    """The table top (m, base frame): the most common height of the camera points under the workspace."""
    z = []
    for i in (0, 1):
        cam, d = npz[f"cam_{i}"], npz[f"depth_median_m_{i}"].astype(float)
        K, T = cam[:9].reshape(3, 3), cam[9:].reshape(4, 4)
        v, u = np.nonzero((d > 0.2) & (d < 2.0))
        p = np.stack([(u - K[0, 2]) * d[v, u] / K[0, 0], (v - K[1, 2]) * d[v, u] / K[1, 1], d[v, u]], 1) @ T[:3, :3].T + T[:3, 3]
        z.append(p[(np.abs(p[:, 0]) < 0.4) & (p[:, 1] > -0.9) & (p[:, 1] < -0.3) & (p[:, 2] < 0.0), 2])
    h, e = np.histogram(np.concatenate(z), bins=np.arange(-0.15, 0.0, 0.0025))
    return float(e[np.argmax(h)] + 0.00125)


def main():
    """Build the replica from a bundle and replay or plan."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--replay", action="store_true", help="execute the run's logged commands (no model)")
    p.add_argument("--arm", choices=("async", "sync"), default="async")
    p.add_argument("--fingers", choices=("full", "scaled"), default="full")
    p.add_argument("--hover", choices=("on", "off"), default="on")
    p.add_argument("--penetration", type=float, default=0.0, help="weight of the penetration cost (0: off, as in the real run)")
    p.add_argument("--box-lo-x", type=float, default=None)
    p.add_argument("--floor", type=float, default=None, help="safety floor height (m); the run's -0.065 sits 1 mm under its own goal fingertips")
    p.add_argument("--finger-alpha", type=float, default=None, help="blend the fingers this far towards the fitted angles each move (the smoothing fix; replay and run)")
    p.add_argument("--palm-frame", action="store_true", help="hand cost split into palm pose + finger shape in the palm frame (the finger fix)")
    p.add_argument("--shape-weight", type=float, default=1.0, help="with --palm-frame: weight of the finger-shape term against the palm-pose term")
    p.add_argument("--pop", type=int, default=None, help="CEM candidates per iteration (run config: 64)")
    p.add_argument("--iters", type=int, default=None, help="CEM iterations (run config: 4)")
    p.add_argument("--no-observe", action="store_true", help="replay: skip the cameras after each move (faster, video only)")
    p.add_argument("--chunk", type=int, default=None, help="candidates per decoder call (lawm: 16; results do not depend on it; 4 fits next to other jobs)")
    p.add_argument("--standin", action="store_true", help="lawm-deployment's stand-in model (CPU smoke test of the loop)")
    p.add_argument("--cup", choices=sorted(CUP_XY), default="goal")
    p.add_argument("--max-replans", type=int, default=60)
    p.add_argument("--execute-steps", type=int, default=None, help="default: the run's experiment.execute_steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot", action="store_true", help="only write snapshot.png of the goal and start (checks the scene)")
    a = p.parse_args()
    r = Replica(a.bundle, a.out, cup_at=a.cup, seed=a.seed, box_lo_x=a.box_lo_x, floor_z=a.floor)
    (r.out / "args.json").write_text(json.dumps(vars(a), indent=1))
    r.setup()
    if a.snapshot:
        import imageio
        r.cell.place(r.goal_npz["q_arm"], r.goal_npz["q_hand"], r.cup_xyz)
        goal = r.cell.render_frame()
        r.cell.place(r.start_npz["q_arm"], r.start_npz["q_hand"], r.cup_xyz)
        imageio.imwrite(r.out / "snapshot.png", np.concatenate([goal, r.cell.render_frame()], 1))
        np.savez_compressed(r.out / "snapshot_clouds.npz", goal=r.goal.coords, goal_segment=r.goal.segment, real_goal=r.goal_npz["coords"], real_segment=r.goal_npz["segment"])
        print(f"table z {r.table_z:.4f}; goal cloud {len(r.goal.coords)} points; wrote {r.out / 'snapshot.png'}")
    elif a.replay:
        r.replay(a.arm, a.max_replans, a.finger_alpha, not a.no_observe)
    else:
        from lawm_deploy.model import StandInModel, load_model
        model_cfg = dict(r.cfg)
        model_cfg["model"] = {**r.cfg["model"], **{k: str(config.repo_path(r.cfg["model"][k])) for k in ("checkpoint", "manifest", "prior")}}
        model = StandInModel() if a.standin else load_model(model_cfg)
        if a.chunk and not a.standin:
            model.inner.copies = a.chunk
        r.run(model, a.arm, a.fingers, a.hover == "on", a.penetration, a.max_replans, a.execute_steps or r.cfg.get("experiment", {}).get("execute_steps", 1),
              a.finger_alpha, a.palm_frame, a.shape_weight, a.pop, a.iters)
    r.cell.close()
    (r.out / "done.json").write_text(json.dumps(dict(finished_unix=time.time())))


if __name__ == "__main__":
    main()
