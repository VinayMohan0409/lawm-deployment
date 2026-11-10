"""A MuJoCo stand-in of the real cell: the same robot on a table with one object and two depth cameras at D435i resolution, behind the RealCell interface."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from lawm_deploy.cell import MotionSettings, min_jerk, move_duration
from lawm_deploy.robot_model import ARM_JOINTS, FLANGE_SITE, cell_spec, pose, quat_of


def look_at(eye, target):
    """MuJoCo camera rotation (x right, y up, looking along -z) at eye looking at target."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, f), -f], axis=1)


def build(cfg, mount):
    """The compiled cell: robot, table top at z = 0 under the robot base, free box objects, two cameras."""
    spec = cell_spec(mount)
    vis = getattr(spec.visual, "global")
    vis.offwidth, vis.offheight = cfg["width"], cfg["height"]
    world = spec.worldbody
    world.add_light(pos=[0.5, 0, 2.0], dir=[0, 0, -1], directional=True)
    t = cfg["table"]
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t["half_x"], t["half_y"], 0.025], pos=[t["x"], t["y"], -0.025], rgba=[0.6, 0.5, 0.4, 1])
    for i, o in enumerate(cfg["objects"]):
        body = world.add_body(name=f"object_{i}", pos=o["pos"])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in o["size"]], mass=o["mass"], friction=[1.5, 0.01, 0.001], rgba=o["rgba"])
    for name, cam in cfg["cameras"].items():
        world.add_camera(name=name, pos=cam["eye"], quat=quat_of(look_at(np.asarray(cam["eye"], float), cam["target"])), fovy=cfg["fovy_deg"])
    return spec.compile()


class SimCell:
    """The simulated cell: arm on IK-tracked straight-line moves, fingers on minimum-jerk paths, rendered depth with RealSense-like noise."""

    def __init__(self, cfg, mount, seed=0):
        """Build the cell, put the robot at home and the object at its start, and set up the renderer."""
        self.cfg, self.motion = cfg, MotionSettings(**cfg["motion"])
        self.model = build(cfg, mount)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.arm_dof = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.hand = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"hand_joint_{k}")] for k in range(22)])
        self.object_qpos = [int(self.model.jnt_qposadr[self.model.body_jntadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{i}")]]) for i in range(len(cfg["objects"]))]
        self.flange = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, FLANGE_SITE)
        self.cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cfg["cameras"]]
        self.renderer = mujoco.Renderer(self.model, cfg["height"], cfg["width"])
        self.renderer.enable_depth_rendering()
        self.rng = np.random.default_rng(seed)
        self.dt = self.model.opt.timestep
        self.set_state(np.radians(cfg["home_deg"]), np.zeros(22), [o["pos"] for o in cfg["objects"]])

    def set_state(self, q_arm, q_hand, object_positions):
        """Put the robot and the objects somewhere directly (scene setup only, as a person arranges the real cell), holding there."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm], self.data.qpos[self.hand] = q_arm, q_hand
        for a, pos in zip(self.object_qpos, object_positions):
            self.data.qpos[a:a + 7] = [*pos, 1, 0, 0, 0]
        self.data.ctrl[:6], self.data.ctrl[6:28] = q_arm, q_hand
        mujoco.mj_forward(self.model, self.data)
        self.wait(0.5)

    def wait(self, seconds):
        """Simulate for a while with the controls unchanged."""
        for _ in range(int(round(seconds / self.dt))):
            mujoco.mj_step(self.model, self.data)

    def state(self):
        """(6 arm angles, 22 hand angles, 4x4 flange pose)."""
        return self.data.qpos[self.arm].copy(), self.data.qpos[self.hand].copy(), pose(self.data.site_xmat[self.flange].reshape(3, 3).copy(), self.data.site_xpos[self.flange].copy())

    def object_positions(self):
        """(n_objects, 3) object positions (tests only; the planner never reads them)."""
        return np.array([self.data.qpos[a:a + 3] for a in self.object_qpos])

    def views(self):
        """[(depth m, K, optical-frame pose)] of both cameras, with depth noise growing with distance squared and dropped pixels."""
        h, w = self.cfg["height"], self.cfg["width"]
        f = 0.5 * h / np.tan(np.radians(self.cfg["fovy_deg"]) / 2)
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
        out = []
        for cam in self.cams:
            self.renderer.update_scene(self.data, cam)
            depth = self.renderer.render().astype(np.float32).copy()
            depth += self.rng.normal(0, 1, depth.shape) * self.cfg["noise_at_1m"] * depth ** 2
            depth[self.rng.uniform(size=depth.shape) < self.cfg["dropout"]] = 0.0
            T = pose(self.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), self.data.cam_xpos[cam])
            out.append((depth, K, T))
        return out

    def ik(self, target, q_near, iters=200, tol_m=1e-4, tol_rad=1e-3):
        """Damped least-squares IK of the flange from q_near, or None if it does not converge."""
        d = self.scratch
        q = np.array(q_near, dtype=float)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            d.qpos[self.arm] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            err_p = target[:3, 3] - d.site_xpos[self.flange]
            err_r = Rotation.from_matrix(target[:3, :3] @ d.site_xmat[self.flange].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(err_p) < tol_m and np.linalg.norm(err_r) < tol_rad:
                return q
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.flange)
            J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[err_p, err_r])
        return None

    def move(self, target, q_hand):
        """Straight-line flange move and minimum-jerk finger path over the same time (as RealCell), then settle."""
        q_arm, q0, flange = self.state()
        duration = move_duration(self.motion, flange, target, q0, q_hand)
        r0, r1 = Rotation.from_matrix(flange[:3, :3]), Rotation.from_matrix(target[:3, :3])
        rel = (r1 * r0.inv()).as_rotvec()
        control_dt = 1.0 / self.motion.finger_rate_hz
        q_cmd = q_arm.copy()
        for k in range(1, int(np.ceil(duration / control_dt)) + 1):
            s = min_jerk(k * control_dt / duration)
            way = pose((Rotation.from_rotvec(s * rel) * r0).as_matrix(), flange[:3, 3] + s * (target[:3, 3] - flange[:3, 3]))
            q_ik = self.ik(way, q_cmd)
            if q_ik is None:
                raise RuntimeError("IK failed along the path")
            q_cmd = q_ik
            self.data.ctrl[:6], self.data.ctrl[6:28] = q_cmd, q0 + (np.asarray(q_hand) - q0) * s
            self.wait(control_dt)
        self.wait(self.motion.settle_s)

    def stop(self):
        """Hold where the robot is."""
        q_arm, q_hand, _ = self.state()
        self.data.ctrl[:6], self.data.ctrl[6:28] = q_arm, q_hand

    def close(self):
        """Free the renderer now (a renderer freed later by the garbage collector can break another one mid-render)."""
        self.renderer.close()
