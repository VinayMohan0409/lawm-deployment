"""The robot cell behind one interface (state, views, ik, move, stop) and its real backend: UR7e over RTDE, Sharpa Wave over its SDK, two D435i over USB."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from lawm_deploy.robot_model import pose


def min_jerk(t):
    """Minimum-jerk progress 0..1 for normalised time t in 0..1."""
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def pose_from_vec(x):
    """4x4 pose of a UR pose vector [x, y, z, rx, ry, rz]."""
    return pose(Rotation.from_rotvec(x[3:6]).as_matrix(), x[:3])


def vec_from_pose(T):
    """UR pose vector [x, y, z, rx, ry, rz] of a 4x4 pose."""
    return [*map(float, T[:3, 3]), *map(float, Rotation.from_matrix(T[:3, :3]).as_rotvec())]


@dataclass(frozen=True)
class MotionSettings:
    """How a target is reached: tool speed (m/s) and acceleration (m/s^2), rotation speed (rad/s), finger speed (rad/s), shortest move (s), finger command rate (Hz), wait after a move (s)."""

    speed: float
    accel: float
    rot_speed: float
    finger_speed: float
    min_move_s: float
    finger_rate_hz: float
    settle_s: float


def move_duration(m, flange_now, target, q_now, q_target):
    """Seconds a move takes at the configured speeds (the slowest of translation, rotation and fingers)."""
    dist = float(np.linalg.norm(target[:3, 3] - flange_now[:3, 3]))
    rot = float(Rotation.from_matrix(target[:3, :3] @ flange_now[:3, :3].T).magnitude())
    fingers = float(np.abs(np.asarray(q_target) - q_now).max())
    return max(m.min_move_s, dist / m.speed + m.speed / m.accel, rot / m.rot_speed, fingers / m.finger_speed)


def camera_poses(cams, calibrated):
    """Optical-frame poses of the cameras: from the calibration file, or the rough tape-measured ones."""
    from lawm_deploy.calibrate import look_at_optical
    from lawm_deploy.config import repo_path
    if not calibrated:
        return [look_at_optical(np.asarray(d["rough_eye"], float), d["rough_target"]) for d in cams["devices"]]
    path = repo_path(cams["calibration"])
    assert path.exists(), f"{path} not found: run python -m lawm_deploy.calibrate first"
    stored = yaml.safe_load(path.read_text())
    return [np.asarray(stored[str(d["serial"])], dtype=float) for d in cams["devices"]]


class SharpaHand:
    """The Sharpa Wave SDK, reduced to read and write of the 22 joint angles in the MJCF order of assets/sharpa_wave/right_hand.xml (names in RobotModel.hand_joint_names)."""

    def __init__(self, settings, joint_names):
        """Connect to the hand. FILL IN from the SDK samples in /opt/sharpa-wave-sdk/python/ (installed by Sharpa Pilot)."""
        self.joint_names = joint_names
        raise NotImplementedError("Sharpa SDK not wired yet: implement __init__, read and write in lawm_deploy/cell.py (see README, 'Sharpa hand').")

    def read(self):
        """The 22 measured joint angles (rad), MJCF order."""
        raise NotImplementedError

    def write(self, q):
        """Send 22 absolute joint-angle targets (rad), MJCF order, to the hand's own position controller."""
        raise NotImplementedError


class RealSense:
    """D435i depth cameras by serial: each gives a depth image (m, median of a few frames), its intrinsics and its pose."""

    def __init__(self, serials, poses, width, height, fps, n_frames, preset):
        """Start one pipeline per camera; poses = 4x4 optical-frame poses in the base frame, one per serial."""
        import pyrealsense2 as rs
        self.rs, self.n_frames, self.cams = rs, n_frames, []
        for serial, cam_pose in zip(serials, poses):
            cfg = rs.config()
            cfg.enable_device(str(serial))
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            sensor = profile.get_device().first_depth_sensor()
            if preset is not None:
                sensor.set_option(rs.option.visual_preset, float(preset))
            i = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
            self.cams.append((pipe, sensor.get_depth_scale(), K, np.asarray(cam_pose, dtype=float)))
        for _ in range(30):  # let auto-exposure settle
            for pipe, *_ in self.cams:
                pipe.wait_for_frames()

    def views(self):
        """[(depth m, K, pose)] for every camera, each the per-pixel median of n_frames (0 = no depth)."""
        out = []
        for pipe, scale, K, T in self.cams:
            frames = np.stack([np.asanyarray(pipe.wait_for_frames().get_depth_frame().get_data()) for _ in range(self.n_frames)]).astype(np.float32) * scale
            frames[frames == 0] = np.nan
            out.append((np.nan_to_num(np.nanmedian(frames, axis=0)), K, T))
        return out

    def close(self):
        """Stop the pipelines."""
        for pipe, *_ in self.cams:
            pipe.stop()


class RealCell:
    """UR7e (ur_rtde, TCP = flange), Sharpa Wave (SDK) and the D435i cameras."""

    def __init__(self, cfg, hand_joint_names, calibrated=True):
        """Connect to the arm, the hand and the cameras from the config's cell section; calibrated=False uses the rough camera poses (for calibrating)."""
        import rtde_control
        import rtde_receive
        self.motion = MotionSettings(**cfg["motion"])
        self.ctrl = rtde_control.RTDEControlInterface(cfg["robot_ip"])
        self.recv = rtde_receive.RTDEReceiveInterface(cfg["robot_ip"])
        self.ctrl.setTcp([0.0] * 6)  # poses are flange poses; the hand mount lives in the config, not on the pendant
        self.hand = SharpaHand(cfg.get("sharpa", {}), hand_joint_names)
        c = cfg["cameras"]
        self.cameras = RealSense([d["serial"] for d in c["devices"]], camera_poses(c, calibrated), c["width"], c["height"], c["fps"], c["n_frames"], c.get("preset"))

    def state(self):
        """(6 arm angles, 22 hand angles, 4x4 flange pose) as measured."""
        return np.array(self.recv.getActualQ()), np.asarray(self.hand.read(), dtype=float), pose_from_vec(np.array(self.recv.getActualTCPPose()))

    def views(self):
        """The depth views of both cameras."""
        return self.cameras.views()

    def ik(self, target, q_near):
        """The UR controller's IK solution for a flange target near q_near, or None."""
        x = vec_from_pose(target)
        if not self.ctrl.getInverseKinematicsHasSolution(x, list(map(float, q_near))):
            return None
        return np.array(self.ctrl.getInverseKinematics(x, list(map(float, q_near))))

    def move(self, target, q_hand):
        """Straight-line flange move (moveL, asynchronous) with the fingers on a minimum-jerk path over the same time, then wait to settle."""
        q_arm, q0, flange = self.state()
        duration = move_duration(self.motion, flange, target, q0, q_hand)
        self.ctrl.moveL(vec_from_pose(target), self.motion.speed, self.motion.accel, True)
        start, dt = time.time(), 1.0 / self.motion.finger_rate_hz
        while (t := time.time() - start) < duration:
            self.hand.write(q0 + (np.asarray(q_hand) - q0) * min_jerk(t / duration))
            time.sleep(dt)
        self.hand.write(np.asarray(q_hand))
        deadline = time.time() + 2 * duration + 5.0
        while self.ctrl.getAsyncOperationProgress() >= 0:
            if time.time() > deadline or self.recv.isProtectiveStopped():
                self.stop()
                raise RuntimeError("arm move did not finish (timeout or protective stop)")
            time.sleep(0.01)
        time.sleep(self.motion.settle_s)

    def freedrive(self, on):
        """Let the arm be moved by hand (UR teach mode) or hold it again."""
        self.ctrl.teachMode() if on else self.ctrl.endTeachMode()

    def stop(self):
        """Stop the arm and hold the fingers where they are."""
        self.ctrl.stopL(2.0)
        self.hand.write(self.hand.read())

    def close(self):
        """Stop and disconnect."""
        self.stop()
        self.cameras.close()
        self.ctrl.stopScript()
