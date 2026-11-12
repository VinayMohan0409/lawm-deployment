"""UR7e/Sharpa/RealSense backend. Read-only by default; motion requires --execute.

SDK conventions and device lifecycle follow sharpa-vision-retargeting-sdk/
joint_policy_deployment/joint_dp/deployment/hardware.py. No reset/open on connect.
"""
from __future__ import annotations

import sys
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from lawm_deploy.hardware_contract import HardwareError, hand_contract, transform, vector, validate_cell_settings
from lawm_deploy.robot_model import pose


def min_jerk(t):
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def pose_from_vec(x):
    x = vector(x, 6, "UR pose")
    return pose(Rotation.from_rotvec(x[3:6]).as_matrix(), x[:3])


def vec_from_pose(T):
    T = transform(T)
    return [*map(float, T[:3, 3]), *map(float, Rotation.from_matrix(T[:3, :3]).as_rotvec())]


@dataclass(frozen=True)
class MotionSettings:
    speed: float
    accel: float
    rot_speed: float
    finger_speed: float
    min_move_s: float
    finger_rate_hz: float
    settle_s: float
    finger_accel: float = 2.0

    def __post_init__(self):
        for key, value in vars(self).items():
            if not np.isfinite(value) or value < 0 or (key != "settle_s" and value == 0):
                raise HardwareError(f"Invalid motion setting: {key}")


def move_duration(m, flange_now, target, q_now, q_target):
    """Conservative duration, including quintic peak speed and acceleration."""
    dist = float(np.linalg.norm(target[:3, 3] - flange_now[:3, 3]))
    rot = float(Rotation.from_matrix(target[:3, :3] @ flange_now[:3, :3].T).magnitude())
    fingers = float(np.abs(np.asarray(q_target) - q_now).max())
    return max(m.min_move_s, 1.875 * dist / m.speed, np.sqrt(5.774 * dist / m.accel),
               1.875 * rot / m.rot_speed, 1.875 * fingers / m.finger_speed,
               np.sqrt(5.774 * fingers / m.finger_accel))


def camera_poses(cams, calibrated):
    from lawm_deploy.config import repo_path
    if not calibrated:
        from lawm_deploy.calibrate import look_at_optical
        return [look_at_optical(np.asarray(d["rough_eye"], float), d["rough_target"]) for d in cams["devices"]]
    path = repo_path(cams["calibration"])
    if not path.is_file():
        raise HardwareError(f"Depth calibration missing: {path}; finish the depth check and import depth-optical extrinsics first")
    stored = yaml.safe_load(path.read_text())
    if not isinstance(stored, dict) or stored.get("frame") != "depth_optical":
        raise HardwareError("Calibration must explicitly declare frame: depth_optical; color extrinsics cannot be copied directly")
    matrices = stored["T_base_camera"]
    return [transform(matrices[str(d["serial"])], f"camera {d['serial']}") for d in cams["devices"]]


class SharpaHand:
    """Read degrees, write radians, map SDK joints to the requested MJCF order."""

    def __init__(self, settings, joint_names):
        self.settings, self.joint_names = settings, list(joint_names)
        names, bounds = hand_contract()
        if len(self.joint_names) != 22 or set(self.joint_names) != set(names):
            raise HardwareError("Expected the 22 right Sharpa joint names")
        self.to_model = np.array([names.index(n) for n in self.joint_names])
        self.to_sdk = np.argsort(self.to_model)
        self.bounds = bounds[self.to_model]
        self.serial = str(settings["serial"])
        self.manager = self.wave = None
        self.enabled = False
        self.last_receipt = None
        sdk = settings.get("sdk_path", "/opt/sharpa-wave-sdk/python")
        if sdk not in sys.path:
            sys.path.insert(0, sdk)
        from sharpa import SharpaWaveManager, DeviceType, HandSide
        try:
            self.manager = SharpaWaveManager.get_instance()
            deadline = time.monotonic() + float(settings.get("discovery_timeout_s", 5.0))
            while True:
                found = [d for d in self.manager.get_all_devices()
                         if d.sn == self.serial and d.device_type == DeviceType.HAND and d.hand_side == HandSide.RIGHT]
                if len(found) == 1:
                    break
                if time.monotonic() >= deadline:
                    raise HardwareError(f"Right Sharpa {self.serial} not uniquely discovered")
                time.sleep(0.1)
            self.wave = self.manager.connect(self.serial)
            if self.wave is None or self.wave.get_device_info().hand_side != HandSide.RIGHT:
                raise HardwareError("Sharpa connection failed or wrong side")
            self.read()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def checked(error, name):
        if error.code != 0:
            raise HardwareError(f"Sharpa {name}: {error.message}")

    def read(self):
        started = time.monotonic()
        error, degrees = self.wave.get_joint_position_degree()
        self.checked(error, "feedback")
        self.last_receipt = time.monotonic()
        if self.last_receipt - started > self.settings.get("feedback_timeout_s", 0.15):
            raise HardwareError("Sharpa feedback query timed out")
        # The SDK exposes query timing, not a verified motor-sample timestamp.
        return vector(np.deg2rad(degrees), 22, "Sharpa feedback")[self.to_model]

    def enable(self):
        if self.enabled:
            return
        from sharpa import ControlMode, ControlSource
        # Do not send a hold target if mode/source setup failed partway through.
        for name, value in (("set_control_mode", ControlMode.POSITION),
                            ("set_speed_coeff", self.settings.get("speed_coeff", 0.3)),
                            ("set_current_coeff", self.settings.get("current_coeff", 0.6)),
                            ("set_control_source", ControlSource.SDK)):
            self.checked(getattr(self.wave, name)(value), name)
        self.enabled = True

    def write(self, q):
        if not self.enabled:
            raise HardwareError("Sharpa output is disabled")
        q = vector(q, 22, "Sharpa command")
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            raise HardwareError("Sharpa command outside MJCF/URDF joint limits")
        self.checked(self.wave.set_joint_position(q[self.to_sdk].tolist(), False), "command")

    def hold(self):
        if self.enabled and self.wave is not None:
            # Feedback may exceed software bounds under contact. Do not issue an open/reset.
            self.write(np.clip(self.read(), self.bounds[:, 0], self.bounds[:, 1]))

    def close(self):
        try:
            self.hold()
        finally:
            if self.manager is not None and self.wave is not None:
                self.manager.disconnect(self.serial)
            self.wave = None
            self.enabled = False


class RealSense:
    """Native depth with explicit, logged sensor settings."""

    def __init__(self, serials, poses, width, height, fps, n_frames, preset,
                 emitter="off", timeout_ms=2000, sensor_options=None):
        import pyrealsense2 as rs
        if (emitter not in ("off", "on") or n_frames < 1 or not serials
                or len(poses) != len(serials) or len(set(serials)) != len(serials)):
            raise HardwareError("Invalid depth camera settings")
        self.rs, self.n_frames, self.cams = rs, n_frames, []
        self.timeout_ms, self.metadata, self.last_frames = timeout_ms, [], []
        self.last_raw_depth = []
        self._numbers = {}
        sensor_options = sensor_options or {}
        if set(sensor_options) - {"enable_auto_exposure", "laser_power"}:
            raise HardwareError("Unsupported configured depth sensor option")
        try:
            for serial, cam_pose in zip(serials, poses):
                cfg = rs.config()
                cfg.enable_device(str(serial))
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                pipe = rs.pipeline()
                profile = pipe.start(cfg)
                # Register immediately so any later setup failure stops this pipeline.
                self.cams.append((pipe, None, None, transform(cam_pose)))
                sensor = profile.get_device().first_depth_sensor()
                if preset is not None:
                    if not sensor.supports(rs.option.visual_preset):
                        raise HardwareError("Depth camera does not support requested preset")
                    sensor.set_option(rs.option.visual_preset, float(preset))
                if not sensor.supports(rs.option.emitter_enabled):
                    raise HardwareError("Cannot verify the depth IR emitter state")
                sensor.set_option(rs.option.emitter_enabled, float(emitter == "on"))
                if sensor.get_option(rs.option.emitter_enabled) != float(emitter == "on"):
                    raise HardwareError("Depth IR emitter setting failed")
                for name, value in sensor_options.items():
                    option = getattr(rs.option, name)
                    if not np.isfinite(value) or not sensor.supports(option):
                        raise HardwareError(f"Unsupported/invalid depth sensor option: {name}")
                    sensor.set_option(option, float(value))
                    if not np.isclose(sensor.get_option(option), value, atol=1e-6, rtol=0):
                        raise HardwareError(f"Depth sensor setting failed: {name}")
                settings = {name: float(sensor.get_option(getattr(rs.option, name)))
                            for name in ("visual_preset", "emitter_enabled", "enable_auto_exposure", "laser_power", "exposure", "gain")
                            if sensor.supports(getattr(rs.option, name))}
                if preset is not None and settings.get("visual_preset") != float(preset):
                    raise HardwareError("Depth visual preset readback differs from configured preset")
                stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                i = stream.get_intrinsics()
                if not np.allclose(i.coeffs, 0):
                    raise HardwareError("Nonzero depth distortion requires SDK deprojection; pinhole backprojection refused")
                K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
                scale = float(sensor.get_depth_scale())
                if not np.isfinite(scale) or scale <= 0:
                    raise HardwareError("Invalid depth scale")
                self.cams[-1] = (pipe, scale, K, transform(cam_pose))
                self.metadata.append(dict(serial=str(serial), frame="depth_optical", emitter=emitter,
                                          depth_scale_m=scale, width=i.width, height=i.height,
                                          distortion=str(i.model), coefficients=list(i.coeffs), fps=fps,
                                          sensor_settings_at_start=settings))
            for _ in range(30):
                for pipe, *_ in self.cams:
                    pipe.wait_for_frames(self.timeout_ms)
        except BaseException:
            self.close()
            raise

    def views(self):
        out, self.last_frames = [], []
        self.last_raw_depth = []
        for index, (pipe, scale, K, T) in enumerate(self.cams):
            # Bound flushing; a busy producer must not make this an infinite loop.
            for _ in range(60):
                if not pipe.poll_for_frames():
                    break
            else:
                raise HardwareError("Depth queue did not drain")
            frames, stamps = [], []
            for _ in range(self.n_frames):
                frame = pipe.wait_for_frames(self.timeout_ms).get_depth_frame()
                if not frame:
                    raise HardwareError("Missing depth frame")
                number = frame.get_frame_number()
                if number <= self._numbers.get(index, -1):
                    raise HardwareError("Depth frame number stopped or reversed")
                self._numbers[index] = number
                frames.append(np.array(frame.get_data(), copy=True))
                stamps.append(dict(number=number, timestamp_ms=frame.get_timestamp(),
                                   domain=str(frame.get_frame_timestamp_domain()), receipt_monotonic=time.monotonic()))
            raw = np.stack(frames)
            self.last_raw_depth.append((raw, scale))
            stack = raw.astype(np.float32) * scale
            stack[stack == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                depth = np.nan_to_num(np.nanmedian(stack, axis=0))
            out.append((depth, K.copy(), T.copy()))
            self.last_frames.append(stamps)
        return out

    def close(self):
        errors = []
        for pipe, *_ in self.cams:
            try:
                pipe.stop()
            except Exception as error:
                errors.append(str(error))
        self.cams = []
        if errors:
            warnings.warn("Camera cleanup: " + "; ".join(errors))


class RealCell:
    """Guarded moveL backend. Arm/finger timing is approximate, not a shared servo clock."""

    def __init__(self, cfg, hand_joint_names, calibrated=True, *, execute=False,
                 full_config=None, cameras=True, trial_step=False, trial_run=False):
        if trial_step and trial_run:
            raise HardwareError('Choose either trial step or trial run')
        if (trial_step or trial_run) and not execute:
            raise HardwareError('Trial mode must explicitly enable execution')
        validate_cell_settings(cfg)
        self.trial_step, self.trial_attempted = trial_step, False
        self.trial_run = trial_run
        self.cfg, self.full_config = cfg, full_config
        self.motion = MotionSettings(**cfg["motion"])
        self.ip, self.joint_names = cfg["robot_ip"], hand_joint_names
        self.read_only, self.teach, self._released = not execute, False, False
        self.ctrl = self.recv = self.hand = self.cameras = None
        self.stop_requested = threading.Event()
        self.last_stamp, self.fresh = None, time.monotonic()
        self.validate_hand_path = None
        self.guards = cfg["guards"]
        self.log_path = None
        if execute:
            from lawm_deploy.hardware_contract import require_execution_setup
            if full_config is None:
                raise HardwareError("Execution requires the complete site configuration")
            require_execution_setup(full_config, trial_step=trial_step or trial_run)
        if full_config is not None:
            from lawm_deploy.config import repo_path
            log_dir = repo_path(full_config["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = log_dir / f"hardware_{time.time_ns()}_{os.getpid()}.jsonl"
        # Validate calibration before touching any hardware.
        c = cfg["cameras"]
        poses = camera_poses(c, calibrated) if cameras else None
        import rtde_receive
        try:
            self.recv = rtde_receive.RTDEReceiveInterface(self.ip, cfg.get("arm_hz", 125))
            self.hand = SharpaHand(cfg["sharpa"], hand_joint_names)
            if cameras:
                self.cameras = RealSense([d["serial"] for d in c["devices"]], poses,
                                        c["width"], c["height"], c["fps"], c["n_frames"], c.get("preset"),
                                        c.get("emitter", "off"), c.get("timeout_ms", 2000), c.get("sensor_options"))
            self.state()
            self.event("connected", read_only=self.read_only, robot_ip=self.ip,
                       camera_metadata=self.cameras.metadata if self.cameras else [],
                       execution_mode='supervised_trial_step' if trial_step else ('supervised_trial_run' if trial_run else ('execute' if execute else 'read_only')),
                       commissioning=full_config.get('commissioning', {}) if full_config else {},
                       existing_setup_accepted_for_trial=trial_step or trial_run, max_move_attempts=1 if trial_step else None)
        except BaseException:
            self.close()
            raise

    def event(self, kind, **fields):
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(dict(event=kind, monotonic=time.monotonic(), **fields), allow_nan=False) + "\n")

    def _arm_state(self):
        r, g = self.recv, self.guards
        if not r.isConnected() or r.isProtectiveStopped() or r.isEmergencyStopped():
            raise HardwareError("UR disconnected or protective/emergency stop")
        stamp, now = float(r.getTimestamp()), time.monotonic()
        if not np.isfinite(stamp) or (self.last_stamp is not None and stamp < self.last_stamp):
            raise HardwareError("UR feedback clock invalid/reversed")
        if self.last_stamp is None or stamp > self.last_stamp:
            self.fresh = now
        self.last_stamp = stamp
        if now - self.fresh > g["feedback_timeout_s"]:
            raise HardwareError("UR feedback stopped updating")
        q, qd = vector(r.getActualQ(), 6, "arm q"), vector(r.getActualQd(), 6, "arm qd")
        tcp, speed = vector(r.getActualTCPPose(), 6, "TCP"), vector(r.getActualTCPSpeed(), 6, "TCP speed")
        if np.max(np.abs(qd)) > g["max_arm_speed_rad_s"]:
            raise HardwareError("Measured arm speed exceeds limit")
        if np.linalg.norm(speed[:3]) > g["max_tcp_speed_m_s"] or np.linalg.norm(speed[3:]) > g["max_tcp_rot_speed_rad_s"]:
            raise HardwareError("Measured TCP speed exceeds limit")
        if self.ctrl is not None:
            scaling = float(r.getSpeedScaling()) * float(r.getTargetSpeedFraction())
            if r.getRuntimeState() != 2 or not np.isfinite(scaling) or scaling <= 0:
                raise HardwareError("UR program not playing or speed scaling is zero")
        # Receive API has no getTCPOffset. Read-only logs the reported TCP and FK
        # discrepancy; the control API verifies the active offset before any command.
        return q, pose_from_vec(tcp)

    def state(self):
        if self.hand is None:
            raise HardwareError("Sharpa released for teleoperation; stop teleop and take control back before capture")
        q, flange = self._arm_state()
        return q, self.hand.read(), flange

    def take_control(self):
        if self.read_only:
            raise HardwareError("Read-only mode: restart with --execute after commissioning")
        if self.hand is None:
            self.hand = SharpaHand(self.cfg["sharpa"], self.joint_names)
        self._released = False

    def control(self):
        if self.read_only or self._released:
            raise HardwareError("Motion disabled: read-only or released to teleoperation")
        if self.ctrl is None:
            import rtde_control
            self._arm_state()
            if self.recv.getRuntimeState() == 2:
                raise HardwareError("Another UR program is running; stop teleoperation/the pendant program before taking control")
            try:
                self.ctrl = rtde_control.RTDEControlInterface(self.ip)
                offset = vector(self.ctrl.getTCPOffset(), 6, "control TCP offset")
                if not np.allclose(offset, self.cfg["expected_tcp_offset"], atol=0.001, rtol=0):
                    raise HardwareError("Control TCP offset mismatch")
                deadline = time.monotonic() + 2.0
                while self.recv.getRuntimeState() != 2:
                    if self.stop_requested.is_set() or time.monotonic() >= deadline:
                        raise HardwareError("UR control program did not become ready")
                    time.sleep(0.01)
            except BaseException:
                self._release_arm()
                raise
        return self.ctrl

    def _release_arm(self):
        ctrl, self.ctrl = self.ctrl, None
        if ctrl is not None:
            try:
                ctrl.stopScript()
            finally:
                ctrl.disconnect()

    def release(self):
        try:
            self.stop()
        finally:
            try:
                self._release_arm()
            finally:
                try:
                    if self.hand is not None:
                        self.hand.close()
                finally:
                    self.hand = None
                    self._released = True
                    self.teach = False

    @property
    def released(self):
        return self._released

    def views(self):
        if self.cameras is None:
            raise HardwareError("Cameras disabled for device-only preflight")
        return self.cameras.views()

    def ik(self, target, q_near):
        # Shadow does not upload a control script merely to ask for controller IK.
        if self.read_only:
            return None
        ctrl = self.control()
        x, q = vec_from_pose(target), vector(q_near, 6, "IK seed").tolist()
        if not ctrl.isPoseWithinSafetyLimits(x) or not ctrl.getInverseKinematicsHasSolution(x, q):
            return None
        result = vector(ctrl.getInverseKinematics(x, q), 6, "IK result")
        return result if ctrl.isJointsWithinSafetyLimits(result.tolist()) else None

    def _check_target(self, target, fingers, q0, h0, flange):
        s = self.full_config["safety"]
        _, bounds = hand_contract()
        # Hand command bounds are also checked again by the SDK adapter.
        if np.any(fingers < bounds[:, 0]) or np.any(fingers > bounds[:, 1]):
            raise HardwareError("Hand target outside joint limits")
        if np.any(h0 < bounds[:, 0]) or np.any(h0 > bounds[:, 1]):
            raise HardwareError("Measured hand starts outside joint limits; no arm motion sent")
        if not ((target[:3, 3] >= s["box_lo"]).all() and (target[:3, 3] <= s["box_hi"]).all()
                and (flange[:3, 3] >= s["box_lo"]).all() and (flange[:3, 3] <= s["box_hi"]).all()):
            raise HardwareError("Measured/target flange outside commissioned workspace")
        if np.linalg.norm(target[:3, 3] - flange[:3, 3]) * 1000 > s["max_step_mm"] + 1e-6:
            raise HardwareError("Flange translation exceeds step limit")
        angle = Rotation.from_matrix(target[:3, :3] @ flange[:3, :3].T).magnitude()
        if np.degrees(angle) > s["max_rot_deg"] or np.degrees(np.abs(fingers - h0).max()) > s["max_finger_deg"]:
            raise HardwareError("Wrist/finger step exceeds limit")
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([flange[:3, :3], target[:3, :3]])))
        last = q0
        for t in np.linspace(0, 1, 11):
            waypoint = pose(rotations(t).as_matrix(), flange[:3, 3] * (1-t) + target[:3, 3] * t)
            q = self.ik(waypoint, last)
            if q is None or np.degrees(np.abs(q - last).max()) > s["max_joint_deg"]:
                raise HardwareError("Cartesian path has no continuous safe controller IK")
            if np.degrees(np.abs(q - q0).max()) > s["max_joint_deg"]:
                raise HardwareError("Arm path exceeds joint excursion limit")
            last = q
        if self.validate_hand_path is None:
            raise HardwareError("Hand path geometry validator is not installed")
        self.validate_hand_path(flange, target, h0, fingers)
        return angle

    def move(self, target, q_hand):
        if self.trial_step and self.trial_attempted:
            raise HardwareError('Trial step already attempted; restart only after reviewing its result')
        if self.read_only or self._released or self.teach:
            raise HardwareError("Motion requires execution mode, control ownership and freedrive off")
        if self.stop_requested.is_set():
            raise HardwareError("Stop requested before motion")
        target, fingers = transform(target, "flange target"), vector(q_hand, 22, "hand target")
        ctrl = self.control()
        try:
            q0, h0, flange = self.state()
            angle = self._check_target(target, fingers, q0, h0, flange)
            # Re-read after controller path queries; do not execute a stale start.
            q1, h1, _ = self.state()
            if np.max(np.abs(q1-q0)) > 0.01 or np.max(np.abs(h1-h0)) > 0.03:
                raise HardwareError("Robot moved during path validation")
            if self.stop_requested.is_set():
                raise HardwareError("Stop requested during path validation")
            duration = move_duration(self.motion, flange, target, h0, fingers)
            dist = np.linalg.norm(target[:3, 3] - flange[:3, 3])
            # moveL interprets speed as tool translation, or rotation for a pure rotation.
            # This slows the arm to approximately match the finger duration; arrival is measured.
            length = float(dist if dist > 1e-6 else angle)
            speed = min(self.motion.speed if dist > 1e-6 else self.motion.rot_speed,
                        max(length / duration, 1e-5))
            self.event("move_start", target=target.tolist(), hand_target=fingers.tolist(),
                       hand_start=h0.tolist(), arm_start=q0.tolist(), duration_s=duration, moveL_speed=speed,
                       timing_contract="asynchronous moveL plus timed fingers; approximate synchronization")
            if self.trial_step:
                self.trial_attempted = True  # consume even if sending fails; never retry a partly sent move
            self.hand.enable()
            if not ctrl.setWatchdog(1.0 / self.guards["watchdog_timeout_s"]):
                raise HardwareError("UR watchdog setup failed")
            if self.stop_requested.is_set():
                raise HardwareError("Stop requested before moveL")
            if length > 1e-7 and not ctrl.moveL(vec_from_pose(target), speed, self.motion.accel, True):
                raise HardwareError("UR moveL rejected command")
            start = previous = time.monotonic()
            deadline = start + max(5.0, 3 * duration + 2)
            last_hand = h0
            next_log = start
            while True:
                now = time.monotonic()
                if self.stop_requested.is_set():
                    raise HardwareError("Motion stopped by user")
                if now > deadline or now - previous > self.guards["control_gap_s"]:
                    raise HardwareError("Motion timeout or control loop stalled")
                if not ctrl.kickWatchdog():
                    raise HardwareError("UR watchdog kick failed")
                qa, measured_hand, actual = self.state()
                if np.max(np.abs(qa-q0)) > np.radians(self.full_config["safety"]["max_joint_deg"]):
                    raise HardwareError("Measured arm excursion exceeds limit")
                s = self.full_config["safety"]
                if np.any(actual[:3,3] < s["box_lo"]) or np.any(actual[:3,3] > s["box_hi"]):
                    raise HardwareError("Measured flange left workspace")
                self.validate_hand_path(actual, actual, measured_hand, measured_hand)
                if np.max(np.abs(last_hand - measured_hand)) > self.guards["hand_following_error_rad"]:
                    raise HardwareError("Sharpa following error")
                fraction = min_jerk((now-start) / duration)
                command = h0 + (fingers-h0) * fraction
                self.hand.write(command)
                last_hand = command
                if now >= next_log:
                    self.event("feedback", arm_q=qa.tolist(), hand_q=measured_hand.tolist(),
                               hand_command=command.tolist(), tcp=actual.tolist())
                    next_log = now + 0.1
                moving = length > 1e-7 and ctrl.getAsyncOperationProgress() >= 0
                pos_error = np.linalg.norm(actual[:3, 3]-target[:3, 3])
                rot_error = Rotation.from_matrix(actual[:3,:3] @ target[:3,:3].T).magnitude()
                arrived = (pos_error <= self.guards["arrival_position_m"]
                           and rot_error <= self.guards["arrival_rotation_rad"]
                           and np.max(np.abs(measured_hand-fingers)) <= self.guards["hand_arrival_rad"])
                if now-start >= duration and not moving and arrived:
                    break
                previous = now
                time.sleep(1.0 / self.motion.finger_rate_hz)
            # Stop the control script at each action boundary: no idle watchdog thread,
            # no lingering move command while model inference or camera reads block.
            self._release_arm()
            self.event("move_arrived", elapsed_s=time.monotonic()-start)
            if self.stop_requested.wait(self.motion.settle_s):
                raise HardwareError("Stopped during settle")
        except BaseException as primary:
            try:
                self.stop()
            except Exception as error:
                warnings.warn(f"Secondary stop failure: {error}")
            finally:
                self._release_arm()
            try:
                self.event("move_failed", error=str(primary))
            except Exception:
                pass  # preserve the device error if the log disk failed too
            raise

    def freedrive(self, on):
        if self.trial_step or self.trial_run:
            raise HardwareError('Freedrive is unavailable in trial mode')
        ctrl = self.control()
        if not (ctrl.teachMode() if on else ctrl.endTeachMode()):
            raise HardwareError("UR freedrive command failed")
        self.teach = on

    def stop(self):
        try:
            if self.ctrl is not None:
                if self.teach:
                    self.ctrl.endTeachMode()
                    self.teach = False
                self.ctrl.stopL(2.0)
        finally:
            if self.hand is not None:
                self.hand.hold()

    def close(self):
        # Attempt every cleanup even if one device has disconnected.
        errors = []
        for action in (self.stop, self._release_arm,
                       lambda: self.hand.close() if self.hand else None,
                       lambda: self.cameras.close() if self.cameras else None,
                       lambda: self.recv.disconnect() if self.recv else None):
            try:
                action()
            except Exception as error:
                errors.append(str(error))
        self.hand = self.cameras = self.recv = None
        if errors:
            warnings.warn("Device cleanup: " + "; ".join(errors))
