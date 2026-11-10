"""One experiment on a cell: capture the goal, capture the start, then replan in a receding horizon (plan, fit, check, move, observe), logging every replan."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lawm.sim.control import StallDetector
from lawm.sim.costs import SEG_HAND
from lawm.sim.dexwm.planning.cem_replan import CemSettings
from lawm.sim.dexwm.planning.phase_gates import GateSettings, HoverSettings
from lawm.sim.dexwm.planning.plan_costs import CostWeights
from lawm.sim.dexwm.planning.task_planner import TaskPlanner

from lawm_deploy.config import mount_pose
from lawm_deploy.hand import Hand
from lawm_deploy.perception import Perception, PerceptionSettings
from lawm_deploy.robot_model import RobotModel
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
        self.robot = RobotModel(mount_pose(cfg), seed=cfg["seed"])
        self.hand = Hand(self.robot.model, cfg["perception"]["n_hand"], cfg["seed"])
        self.perception = Perception(PerceptionSettings(**cfg["perception"]), cfg["seed"])
        self.safety = SafetySettings(**cfg["safety"])
        self.log = Path(log_root) / time.strftime("%Y%m%d_%H%M%S")
        self.log.mkdir(parents=True, exist_ok=True)
        self.goal = self.start = self.current = self.planner = self.last_plan = self.run_dir = None
        self.task, self.replan, self.holds, self.records, self.finished = None, 0, 0, [], True

    def observe(self):
        """Read the joints, place the flange and hand by forward kinematics (not the controller's TCP, which a teleop system may change), read the cameras and build the cloud."""
        q_arm, q_hand, _ = self.cell.state()
        self.robot.set(q_arm, q_hand)
        flange = self.robot.flange_pose()
        root = flange @ self.robot.root_in_flange
        views = self.cell.views()
        coords, segment, info = self.perception(views, self.robot.surface(q_arm, q_hand), self.hand.points(root, q_hand))
        return Observation(coords, segment, q_arm, q_hand, flange, root, views, info)

    def save(self, name, obs, folder=None, **extra):
        """Write an observation (depth in mm as uint16) and extra arrays to folder/name.npz (default: the session folder)."""
        depths = {f"depth_{i}": np.round(d * 1000).clip(0, 65535).astype(np.uint16) for i, (d, _, _) in enumerate(obs.views)}
        cams = {f"cam_{i}": np.concatenate([K.ravel(), T.ravel()]) for i, (_, K, T) in enumerate(obs.views)}
        np.savez_compressed(Path(folder or self.log) / f"{name}.npz", coords=obs.coords, segment=obs.segment, q_arm=obs.q_arm, q_hand=obs.q_hand, flange=obs.flange, **depths, **cams, **extra)

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
        q_arm, q_hand, _ = self.cell.state()
        c = self.current
        if c is None or np.degrees(np.abs(q_arm - c.q_arm).max()) > 0.5 or np.degrees(np.abs(q_hand - c.q_hand).max()) > 2.0:
            self.current = self.observe()
        return self.current

    def begin(self, task):
        """Start a run of a task towards the captured goal from the robot as it is now, in a new run folder (goal, captured start, run start)."""
        assert task in TASKS, task
        assert self.goal is not None and self.start is not None, "capture the goal, then the start"
        self.current = self.observe()
        cem, weights, gates = planner_settings(self.cfg, task)
        seed = int(np.random.SeedSequence([self.cfg["seed"], TASKS.index(task), int(time.time())]).generate_state(1)[0])
        self.planner = TaskPlanner(task, self.goal.coords, self.goal.segment, self.hand.finger_mask, self.model, seed, cem, weights, gates)
        self.executor = Executor(self.safety, self.current.flange)
        self.loop = LoopSettings(**self.cfg["loop"][task])
        self.stall = None if self.loop.stall_patience is None else StallDetector(self.loop.stall_patience, self.loop.stall_tol_mm)
        self.task, self.replan, self.holds, self.within, self.records, self.finished, self.plan_only_count = task, 0, 0, 0, [], False, 0
        self.run_dir = self.log / f"{time.strftime('%H%M%S')}_{task}"
        self.run_dir.mkdir()
        (self.run_dir / "run.json").write_text(json.dumps(dict(task=task, seed=seed, config=self.cfg), indent=1))
        self.save("goal", self.goal, self.run_dir)
        self.save("start", self.start, self.run_dir)
        self.save("run_start", self.current, self.run_dir)
        if hasattr(self.cell, "start_recording"):  # simulated cell: record the run
            self.cell.start_recording(self.run_dir / "video.mp4")

    def step(self, execute=True):
        """One replan: plan on an up-to-date cloud, fit the hand to predicted step 1, check, move (if execute and safe), observe; returns the record, with 'stop' set when the run should end. execute=False only shows the plan: nothing moves and the run's counters, gates and logs are untouched."""
        obs = self.up_to_date()
        hand_target, plan = self.planner.plan(obs.coords, self.replan)
        rot, pos, q, fit = self.hand.fit(hand_target, obs.q_hand)
        target, fingers, move_mm = self.executor.command(rot, pos, q, obs.root, obs.flange)
        q_ik = self.cell.ik(target, obs.q_arm)
        commanded_hand = self.hand.points(target @ self.robot.root_in_flange, fingers)
        hold = check(self.safety, target, fingers, obs.q_hand, obs.flange, obs.q_arm, q_ik, fit, commanded_hand)
        self.last_plan = dict(hand_target=hand_target, commanded_hand=commanded_hand)
        if not execute:
            self.save(f"plan_only_{self.plan_only_count:03d}", obs, self.run_dir, delta=plan.delta.astype(np.float16), target=target, fingers=fingers, hand_target=hand_target)
            self.plan_only_count += 1
            return dict(replan=self.replan, distance_mm=task_distance_mm(obs, self.goal), moved=False, hold=hold, executed=plan.executed, plan_cost=plan.cost,
                        predicted_error_mm=plan.final_hand_error_mm, commanded_mm=move_mm, fit_residual_mm=fit["mean_residual_mm"], phase_changed=False,
                        within_success=self.within, stop=None, plan_only=True)
        moved = hold is None
        if moved:
            self.cell.move(target, fingers)
            self.executor.sent(target)
        self.holds = 0 if hold is None else self.holds + 1
        new = self.observe() if moved else obs
        phase = bool(self.planner.update(new.coords, self.replan))
        dist = task_distance_mm(new, self.goal)
        record = dict(replan=self.replan, distance_mm=dist, moved=moved, hold=hold, executed=plan.executed, plan_cost=plan.cost,
                      predicted_error_mm=plan.final_hand_error_mm, commanded_mm=move_mm, fit_residual_mm=fit["mean_residual_mm"], phase_changed=phase)
        self.save(f"replan_{self.replan:03d}", obs, self.run_dir, delta=plan.delta.astype(np.float16), target=target, fingers=fingers, hand_target=hand_target)
        self.current, self.replan = new, self.replan + 1
        self.within = self.within + 1 if dist < self.loop.success_mm else 0
        record["within_success"] = self.within
        record["stop"] = ("success" if self.within >= self.loop.success_replans else "too many holds" if self.holds >= self.loop.max_holds
                          else "stalled" if moved and self.stall and self.stall.update(dist) else "max replans" if self.replan >= self.loop.max_replans else None)
        self.finished = record["stop"] is not None
        self.records.append(record)
        with open(self.run_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def run(self, execute=True, should_stop=lambda: False, on_step=lambda record: None):
        """Replan until a stop condition or should_stop(); on_step sees each record; the video (sim) is closed at the end."""
        reason = "stopped by user"
        try:
            while not should_stop():
                record = self.step(execute)
                on_step(record)
                if record["stop"]:
                    reason = record["stop"]
                    break
        finally:
            self.finished = True
            self.finish()
        return reason

    def finish(self):
        """Close the run's video (simulated cell only)."""
        if hasattr(self.cell, "stop_recording"):
            self.cell.stop_recording()
