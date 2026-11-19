"""The hand-only tasks: push and grasp plan towards one goal hand, grasp_lift towards the grasp hand and then the lifted hand."""
import numpy as np
import pytest

from lawm_deploy import config
from lawm_deploy.model import StandInModel
from lawm_deploy.session import Session, TASK_SPECS, planner_settings
from lawm_deploy.sim_cell import SimCell

CFG = config.load(config.REPO / "configs/sim.yaml")


@pytest.fixture()
def cell():
    c = SimCell(CFG["cell"], config.mount_pose(CFG))
    yield c
    c.close()


@pytest.mark.parametrize("task", ["push", "grasp", "grasp_lift"])
def test_new_tasks_score_the_hand_only(task):
    """No object term and no hover gate in the new tasks."""
    _, weights, _ = planner_settings(CFG, task)
    assert weights.object_still == weights.object_goal == weights.grasp_quality == weights.rigid_attachment == weights.grasp_tight == 0.0
    assert TASK_SPECS[task][0] == "reach"  # lawm's planner without a phase gate


def test_only_the_hand_tasks_exist():
    assert set(TASK_SPECS) == {"push", "grasp", "grasp_lift"}


def test_execute_steps_per_task(cell, tmp_path):
    """push runs 5 predicted targets per model call, grasp 3; experiment.execute_steps is the fallback."""
    assert config.execution_steps(CFG, "push") == 5 and config.execution_steps(CFG, "grasp") == 3
    assert config.execution_steps({"experiment": {"execute_steps": 2}, "loop": {"push": {}}}, "push") == 2
    s = Session(cell, CFG, StandInModel(), tmp_path)
    s.capture_goal()
    s.capture_start()
    s.begin("push")
    assert s.execute_steps == 5


def test_grasp_lift_needs_the_lift_goal(cell, tmp_path):
    s = Session(cell, CFG, StandInModel(), tmp_path)
    s.capture_goal()
    s.capture_start()
    with pytest.raises(AssertionError, match="lift goal"):
        s.begin("grasp_lift")


def test_grasp_lift_switches_to_the_lifted_goal(cell, tmp_path):
    """The planner aims at the grasp hand first; once the hand is there the lifted hand takes over."""
    s = Session(cell, CFG, StandInModel(), tmp_path)
    _, _, home = cell.state()
    grasp = home.copy()
    grasp[:3, 3] += [0.03, -0.02, -0.04]
    lifted = grasp.copy()
    lifted[:3, 3] += [0.0, 0.0, 0.05]
    cell.move(grasp, np.zeros(22))
    s.capture_goal()
    cell.move(lifted, np.zeros(22))
    s.capture_goal_lift()
    cell.move(home, np.zeros(22))
    s.capture_start()
    s.begin("grasp_lift")
    stages = [s.step()["stage"] for _ in range(14)]
    assert stages[0] == 0 and 1 in stages, stages
    assert stages == sorted(stages)
    assert (s.run_dir / "goal_lift.npz").exists()


def test_grasp_lift_keeps_the_fingers_closed_while_lifting(cell, tmp_path):
    """After the switch to the lifted goal the fingers are commanded at the grasp angles (+ squeeze), not reopened."""
    from lawm_deploy.sim_demo import closure_angles
    s = Session(cell, CFG, StandInModel(), tmp_path)
    _, _, home = cell.state()
    closed = closure_angles(cell, 0.4)
    grasp = home.copy()
    grasp[:3, 3] += [0.03, -0.02, -0.04]
    lifted = grasp.copy()
    lifted[:3, 3] += [0.0, 0.0, 0.05]
    cell.move(grasp, closed)
    s.capture_goal()
    cell.move(lifted, closed)
    s.capture_goal_lift()
    cell.move(home, np.zeros(22))
    s.capture_start()
    s.begin("grasp_lift")
    records = [s.step() for _ in range(14)]
    lift = [r["replan"] for r in records if r["stage"] == 1]
    assert lift, "never reached the lift stage"
    squeeze = np.radians(s.loop.finger_squeeze_deg + 1.0)  # the squeeze past the goal angles, + 1 deg of measurement
    for i in lift:
        requested = np.load(s.run_dir / f"command_replan_{i:03d}.npz")["requested_fingers"]
        assert np.abs(requested - s.goal_lift.q_hand).max() <= squeeze
        assert np.abs(requested - s.goal.q_hand).max() <= squeeze
