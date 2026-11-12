"""One experiment on a cell: capture the goal, capture the start, then replan in a receding horizon (plan, fit, check, move, observe), logging every replan."""
from __future__ import annotations

import json
import hashlib
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lawm_deploy.hardware_contract import HardwareError
from lawm_deploy.model_contract import timing_metadata, validate_model_contract
from lawm_deploy.observation_io import save_observation

from lawm.sim.control import StallDetector
from lawm.sim.costs import SEG_HAND
from lawm.sim.dexwm.planning.cem_replan import CemSettings
from lawm.sim.dexwm.planning.phase_gates import GateSettings, HoverSettings
from lawm.sim.dexwm.planning.plan_costs import CostWeights
from lawm.sim.dexwm.planning.task_planner import TaskPlanner

from lawm_deploy.config import mount_pose
from lawm_deploy.hand import Hand
from lawm_deploy.perception import Perception, PerceptionSettings
from lawm_deploy.robot_model import RobotModel, ARM_JOINTS
from lawm_deploy.safety import Executor, SafetySettings, check

TASKS = ("reach", "grasp")  # place needs object labels, which real camera clouds do not have


@dataclass(frozen=True)
class LoopSettings:
    """When a run stops: replan cap, success (distance under success_mm for success_replans replans in a row), stall patience (None = off) and tolerance (mm), and how many held replans in a row end it."""

    max_replans: int
    success_mm: float
    success_replans: int
    stall_patience: int | None
    stall_tol_mm: float
    max_holds: int


@dataclass
class Observation:
    """One look at the cell: the planner's cloud and segment, the joint angles, flange and palm poses, the depth views and perception info."""

    coords: np.ndarray
    segment: np.ndarray
    q_arm: np.ndarray
    q_hand: np.ndarray
    flange: np.ndarray
    root: np.ndarray
    views: list
    info: dict
    raw_depth: list | None = None
    config_snapshot: dict | None = None


def task_distance_mm(obs, goal):
    """Measured distance to the goal: mean distance of the hand points (FK) to the goal hand points."""
    h = obs.segment == SEG_HAND
    return float(np.linalg.norm(obs.coords[h] - goal.coords[h], axis=1).mean() * 1000)


def planner_settings(cfg, task):
    """lawm's search, cost-weight and gate settings of a task from the config."""
    p = cfg["planner"]
    weights = CostWeights(**p["weights"][task])
    return CemSettings(**p["cem"][task]), weights, GateSettings(HoverSettings(**p["gates"]["hover"]), None)  # the secure gate is place only


class Session:
    """Goal, start and the replan loop of one experiment on one cell (real or simulated) with one model."""

    def __init__(self, cell, cfg, model, log_root):
        """Build the robot model, the hand cloud and perception; captures go to log_root/<time>, each run to its own folder inside it."""
        self.cell, self.cfg, self.model = cell, cfg, model
        validate_model_contract(cfg)
        from lawm_deploy.config import repo_path
        self.provenance = dict(checkpoint_sha256=hashlib.sha256(repo_path(cfg["model"]["checkpoint"]).read_bytes()).hexdigest(),
                               simulation_validation="shared checkpoint not the exact simulation-tested checkpoint, per owner 2026-11-11",
                               camera_observation_validation="pending", **timing_metadata())
        self.robot = RobotModel(mount_pose(cfg), seed=cfg["seed"])
        self.hand = Hand(self.robot.model, cfg["perception"]["n_hand"], cfg["seed"])
        self.perception = Perception(PerceptionSettings(**cfg["perception"]), cfg["seed"])
        self.safety = SafetySettings(**cfg["safety"])
        if hasattr(cell, "validate_hand_path"):
            cell.validate_hand_path = self.check_hand_path
        self.log = Path(log_root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        self.log.mkdir(parents=True, exist_ok=True)
        self.goal = self.start = self.current = self.planner = self.last_plan = self.run_dir = None
        self.task, self.replan, self.holds, self.records, self.finished = None, 0, 0, [], True

    def observe(self):
        """Read the joints, place the flange and hand by forward kinematics (not the controller's TCP, which a teleop system may change), read the cameras and build the cloud."""
        observation_begin_ns = time.time_ns()
        observation_begin_monotonic = time.monotonic()
        snapshot = copy.deepcopy(self.cfg)
        q_arm, q_hand, reported = self.cell.state()
        self.robot.set(q_arm, q_hand)
        flange = self.robot.flange_pose()
        root = flange @ self.robot.root_in_flange
        views = self.cell.views()
        # Sequential depth medians require a stationary robot. Never combine images
        # acquired while the operator/another process is moving the arm or fingers.
        if hasattr(self.cell, "read_only"):
            after_arm, after_hand, _ = self.cell.state()
            if np.max(np.abs(after_arm-q_arm)) > 0.01 or np.max(np.abs(after_hand-q_hand)) > 0.03:
                raise HardwareError("Robot moved while acquiring depth; capture a stationary scene")
        coords, segment, info = self.perception(views, self.robot.surface(q_arm, q_hand), self.hand.points(root, q_hand))
        if hasattr(self.cell, "read_only"):
            ep = float(np.linalg.norm(flange[:3,3]-reported[:3,3]))
            er = float(Rotation.from_matrix(flange[:3,:3] @ reported[:3,:3].T).magnitude())
            info.update(fk_tcp_position_error_m=ep, fk_tcp_rotation_error_rad=er,
                        tcp_offset_verified=self.cell.ctrl is not None,
                        camera_metadata=self.cell.cameras.metadata,
                        camera_frames=self.cell.cameras.last_frames)
            g = self.cell.guards
            if not self.cell.read_only and (ep > g["fk_position_tolerance_m"] or er > g["fk_rotation_tolerance_rad"]):
                raise HardwareError(f"Nominal FK / reported TCP disagreement: {ep:.4f} m, {er:.4f} rad")
        raw = self.cell.cameras.last_raw_depth if hasattr(self.cell, "read_only") else None
        if snapshot != self.cfg:
            raise HardwareError("Configuration changed during observation; capture again")
        info.update(observation_begin_unix_ns=observation_begin_ns, observation_end_unix_ns=time.time_ns(),
                    observation_begin_monotonic=observation_begin_monotonic,
                    observation_end_monotonic=time.monotonic())
        return Observation(coords, segment, q_arm, q_hand, flange, root, views, info, raw, snapshot)

    def check_hand_path(self, flange0, flange1, hand0, hand1):
        """Sample hand/table clearance across independent arm/finger progress.

        moveL and SDK fingers do not share a servo clock. This checks combinations,
        not just simultaneous interpolation. It is not full arm/scene collision checking.
        """
        stationary = np.allclose(flange0, flange1) and np.allclose(hand0, hand1)
        fractions = [0.0] if stationary else np.linspace(0, 1, 9)
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([flange0[:3,:3], flange1[:3,:3]])))
        for a in fractions:
            T = np.eye(4)
            T[:3,:3] = rotations(a).as_matrix()
            T[:3,3] = (1-a)*flange0[:3,3] + a*flange1[:3,3]
            for b in fractions:
                points = self.hand.points(T @ self.robot.root_in_flange, (1-b)*hand0 + b*hand1)
                if not np.isfinite(points).all() or points[:,2].min() < self.safety.floor_z_m:
                    raise HardwareError("Hand path reaches below the configured floor")

    def save(self, name, obs, folder=None, **extra):
        """Keep exact model inputs, full-precision median depth, and native raw frames when available."""
        save_observation(Path(folder or self.log) / f"{name}.npz", obs,
                         obs.config_snapshot if obs.config_snapshot is not None else self.cfg, self.provenance, **extra)

    def capture_goal(self):
        """Observe and keep it as the goal (the robot and the object arranged as they should end up)."""
        self.goal = self.observe()
        self.save("goal", self.goal)
        if hasattr(self.cell, "set_markers"):  # simulated cell: draw the goal hand in the video
            self.cell.set_markers(self.goal.coords[self.goal.segment == SEG_HAND])
        return self.goal

    def capture_start(self):
        """Observe and keep it as the start."""
        self.start = self.current = self.observe()
        self.save("start", self.current)
        return self.current

    def up_to_date(self):
        """The current observation, observed again if the robot has moved since (teleop, freedrive, finger moves, a failed step)."""
        if hasattr(self.cell, "read_only"):
            self.current = self.observe()  # objects can move even if robot joints do not
            return self.current
        q_arm, q_hand, _ = self.cell.state()
        c = self.current
        if c is None or np.degrees(np.abs(q_arm - c.q_arm).max()) > 0.5 or np.degrees(np.abs(q_hand - c.q_hand).max()) > 2.0:
            self.current = self.observe()
        return self.current

    def begin(self, task):
        """Start a run of a task towards the captured goal from the robot as it is now, in a new run folder (goal, captured start, run start)."""
        assert task in TASKS, task
        assert self.goal is not None and self.start is not None, "capture the goal, then the start"
        if self.run_dir is not None and not self.finished:
            self.finish("superseded by new run")
        self.current = self.observe()
        cem, weights, gates = planner_settings(self.cfg, task)
        seed = int(np.random.SeedSequence([self.cfg["seed"], TASKS.index(task), int(time.time())]).generate_state(1)[0])
        self.planner = TaskPlanner(task, self.goal.coords, self.goal.segment, self.hand.finger_mask, self.model, seed, cem, weights, gates)
        self.executor = Executor(self.safety, self.current.flange)
        self.loop = LoopSettings(**self.cfg["loop"][task])
        self.stall = None if self.loop.stall_patience is None else StallDetector(self.loop.stall_patience, self.loop.stall_tol_mm)
        self.task, self.replan, self.holds, self.within, self.records, self.finished, self.plan_only_count = task, 0, 0, 0, [], False, 0
        self.run_dir = self.log / f"{time.strftime('%H%M%S')}_{task}_{time.time_ns()}"
        self.run_dir.mkdir()
        (self.run_dir / "run.json").write_text(json.dumps(dict(task=task, seed=seed, config=self.cfg,
            hardware_log=str(getattr(self.cell, "log_path", "")),
            arm_joint_names=list(ARM_JOINTS), hand_joint_names=self.robot.hand_joint_names,
            provenance=self.provenance, initial_hand_distance_mm=task_distance_mm(self.current, self.goal),
            success_definition="hand_mean_point_distance_only; grasp success requires separate assessment"), indent=1))
        self.save("goal", self.goal, self.run_dir)
        self.save("start", self.start, self.run_dir)
        self.save("run_start", self.current, self.run_dir)
        if hasattr(self.cell, "start_recording"):  # simulated cell: record the run
            self.cell.start_recording(self.run_dir / "video.mp4")
        self.write_status("ready", loop_ended=False)

    def event(self, event, **fields):
        if self.run_dir is not None:
            with (self.run_dir / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=event, unix_ns=time.time_ns(), monotonic=time.monotonic(), **fields)) + '\n')

    def write_status(self, reason, *, error=None, loop_ended=False):
        if self.run_dir is not None:
            temporary = self.run_dir / 'status.json.tmp'
            temporary.write_text(json.dumps(dict(loop_ended=loop_ended, reason=reason, error=error,
                replans=self.replan, hand_goal_reached=reason == 'success', physical_grasp_success=None,
                updated_unix_ns=time.time_ns()), indent=2) + '\n')
            temporary.replace(self.run_dir / 'status.json')

    def step(self, execute=True):
        """One logged attempt. Plan only keeps execution counters/gates unchanged."""
        try:
            record = self._step(execute)
            self.write_status(record['stop'] or ('step complete' if execute else 'plan only complete'),
                              loop_ended=self.finished)
            if self.finished:
                self.finish(record['stop'])
            return record
        except BaseException as exc:
            # Preserve the original fault even if the disk or video writer also fails.
            self.finished = True
            try:
                self.event('step_failed', replan=self.replan, execute=execute,
                           stage=getattr(self, '_stage', 'before step'), error=str(exc))
                self.finish('error', str(exc))
            except Exception:
                pass
            raise

    def _step(self, execute=True):
        if execute and getattr(self.cell, "read_only", False):
            raise HardwareError("Read-only session cannot execute")
        stem = f'replan_{self.replan:03d}' if execute else f'plan_only_{self.plan_only_count:03d}'
        self._stage = 'observe'
        self.event('step_started', replan=self.replan, execute=execute, archive_stem=stem)
        obs = self.up_to_date()
        self.save(f'input_{stem}', obs, self.run_dir)
        self._stage = 'plan'
        started = time.monotonic()
        hand_target, plan = self.planner.plan(obs.coords, self.replan)
        planning_elapsed_s = time.monotonic() - started
        # Save before fitting/IK/motion: a failed step must retain its prediction.
        np.savez_compressed(self.run_dir / f'prediction_{stem}.npz', delta=plan.delta,
                            hand_target=hand_target, planning_elapsed_s=planning_elapsed_s)
        if not np.isfinite(hand_target).all():
            raise HardwareError("Model predicted nonfinite hand points")
        self._stage = 'fit'
        rot, pos, q, fit = self.hand.fit(hand_target, obs.q_hand)
        target, fingers, move_mm = self.executor.command(rot, pos, q, obs.root, obs.flange)
        commanded_mm = float(np.linalg.norm(target[:3, 3] - obs.flange[:3, 3]) * 1000)
        np.savez_compressed(self.run_dir / f'command_{stem}.npz', target=target, fingers=fingers,
                            fitted_root_rotation=rot, fitted_root_position=pos, fitted_fingers=q,
                            fit_json=json.dumps(fit))
        self._stage = 'ik_and_checks'
        q_ik = self.cell.ik(target, obs.q_arm)
        commanded_hand = self.hand.points(target @ self.robot.root_in_flange, fingers)
        hold = check(self.safety, target, fingers, obs.q_hand, obs.flange, obs.q_arm, q_ik, fit, commanded_hand)
        if getattr(self.cell, "read_only", False) and q_ik is None:
            hold = "controller IK not checked in read-only mode" if hold == "no arm IK solution for the target" else hold
        self.last_plan = dict(hand_target=hand_target, commanded_hand=commanded_hand)
        self.save(stem, obs, self.run_dir, delta=plan.delta, target=target, fingers=fingers,
                  hand_target=hand_target, q_arm_ik=np.asarray(q_ik) if q_ik is not None else np.empty(0),
                  ik_query_performed=not getattr(self.cell, 'read_only', False),
                  fit_json=json.dumps(fit), hold_json=json.dumps(hold))
        self.event('plan_ready', replan=self.replan, execute=execute, hold=hold,
                   planning_elapsed_s=planning_elapsed_s, commanded_mm=commanded_mm,
                   unclipped_command_mm=move_mm)
        if not execute:
            self.plan_only_count += 1
            record = dict(replan=self.replan, distance_mm=task_distance_mm(obs, self.goal), moved=False, hold=hold, executed=plan.executed, plan_cost=plan.cost,
                        predicted_error_mm=plan.final_hand_error_mm, commanded_mm=commanded_mm,
                        unclipped_command_mm=move_mm, planning_elapsed_s=planning_elapsed_s,
                        fit_residual_mm=fit["mean_residual_mm"], phase_changed=False,
                        within_success=self.within, stop=None, plan_only=True)
            with (self.run_dir / 'plans.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, archive_stem=stem)) + '\n')
            return record
        moved = hold is None
        if moved:
            self._stage = 'move'
            if self.cell.stop_requested.is_set():
                raise HardwareError("Stop requested during planning; no motion sent")
            current_arm, current_hand, _ = self.cell.state()
            if np.max(np.abs(current_arm-obs.q_arm)) > 0.01 or np.max(np.abs(current_hand-obs.q_hand)) > 0.03:
                raise HardwareError("Robot moved during planning; discard target")
            self.cell.move(target, fingers)
            self.executor.sent(target)
        self.holds = 0 if hold is None else self.holds + 1
        self._stage = 'observe_after'
        new = self.observe() if moved else obs
        self.save(f'after_{stem}', new, self.run_dir)
        self._stage = 'record'
        phase = bool(self.planner.update(new.coords, self.replan))
        dist = task_distance_mm(new, self.goal)
        record = dict(replan=self.replan, distance_mm=dist, moved=moved, hold=hold, executed=plan.executed, plan_cost=plan.cost,
                      predicted_error_mm=plan.final_hand_error_mm, commanded_mm=commanded_mm,
                      unclipped_command_mm=move_mm, planning_elapsed_s=planning_elapsed_s,
                      fit_residual_mm=fit["mean_residual_mm"], phase_changed=phase,
                      elapsed_s=time.monotonic()-started, success_definition="hand_mean_point_distance_only; not grasp success")
        self.current, self.replan = new, self.replan + 1
        self.within = self.within + 1 if dist < self.loop.success_mm else 0
        record["within_success"] = self.within
        record["stop"] = ("success" if self.within >= self.loop.success_replans else "too many holds" if self.holds >= self.loop.max_holds
                          else "stalled" if moved and self.stall and self.stall.update(dist) else "max replans" if self.replan >= self.loop.max_replans else None)
        self.finished = record["stop"] is not None
        self.records.append(record)
        with open(self.run_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self.event('step_completed', replan=self.replan - 1, moved=moved, hold=hold, stop=record['stop'])
        return record

    def run(self, execute=True, should_stop=lambda: False, on_step=lambda record: None):
        """Replan until a stop condition or should_stop(); on_step sees each record; the video (sim) is closed at the end."""
        if execute and getattr(self.cell, 'trial_step', False):
            raise HardwareError('Continuous Run is disabled in trial-step mode; use one Step')
        reason = "stopped by user"
        error = None
        try:
            while not should_stop():
                record = self.step(execute)
                on_step(record)
                if record["stop"]:
                    reason = record["stop"]
                    break
        except BaseException as exc:
            reason, error = "error", str(exc)
            raise
        finally:
            self.finished = True
            self.finish(reason, error)
        return reason

    def finish(self, reason=None, error=None):
        """Finalize Step/Run/Stop consistently, including the simulated video."""
        self.finished = True
        if reason is None and self.records:
            reason = self.records[-1].get('stop')
        self.write_status(reason or 'closed', error=error, loop_ended=True)
        if hasattr(self.cell, "stop_recording"):
            self.cell.stop_recording()
