"""The whole pipeline in the simulated cell: perception, smooth moves, and a reach run with the stand-in model."""
import numpy as np
import pytest
from lawm_deploy.model import StandInModel
from scipy.spatial import cKDTree

from lawm.sim.costs import SEG_HAND
from lawm_deploy import config
from lawm_deploy.perception import SEG_SCENE
from lawm_deploy.session import Session
from lawm_deploy.sim_cell import SimCell

CFG = config.load(config.REPO / "configs/sim.yaml")


@pytest.fixture()
def cell():
    """A fresh simulated cell at home, closed after the test."""
    c = SimCell(CFG["cell"], config.mount_pose(CFG))
    yield c
    c.close()


def test_perception_keeps_scene_and_drops_robot(cell, tmp_path):
    """No camera point is left on the hand, points remain on every object, and the cloud is 1024 hand + n_scene camera points."""
    s = Session(cell, CFG, StandInModel(), tmp_path)
    obs = s.observe()
    q_arm, q_hand, _ = cell.state()
    scene = s.perception.scene_points(obs.views, s.robot.surface(q_arm, q_hand))
    assert (cKDTree(obs.coords[obs.segment == SEG_HAND]).query(scene)[0] < 0.01).sum() == 0
    for pos in cell.object_positions():
        assert (np.linalg.norm(scene - pos, axis=1) < 0.05).sum() > 50
    assert [(obs.segment == k).sum() for k in (SEG_HAND, SEG_SCENE)] == [CFG["perception"]["n_hand"], CFG["perception"]["n_scene"]]


def test_move_is_smooth_and_accurate(cell):
    """A 10 cm move ends within 5 mm and the arm joints never exceed 1 rad/s."""
    peak = []
    wait = cell.wait

    def tracked(seconds):
        """Simulate and record the peak arm joint speed."""
        for _ in range(int(round(seconds / cell.dt))):
            wait(cell.dt)
            peak.append(np.abs(cell.data.qvel[cell.arm_dof]).max())
    cell.wait = tracked
    _, _, flange = cell.state()
    target = flange.copy()
    target[:3, 3] += [0.05, 0.05, -0.07]
    cell.move(target, np.full(22, 0.3))
    _, _, after = cell.state()
    assert np.linalg.norm(after[:3, 3] - target[:3, 3]) < 0.005
    assert max(peak) < 1.0


def test_reach_runs_and_logs(cell, tmp_path):
    """Goal then start capture, then replans that bring the hand closer, with a log file per replan."""
    s = Session(cell, CFG, StandInModel(), tmp_path)
    _, _, home = cell.state()
    goal = home.copy()
    goal[:3, 3] += [0.05, -0.04, -0.06]
    cell.move(goal, np.zeros(22))
    s.capture_goal()
    cell.move(home, np.zeros(22))
    s.capture_start()
    s.begin("reach")
    first = s.step()["distance_mm"]
    for _ in range(5):
        record = s.step()
    assert record["distance_mm"] < 0.5 * first
    s.finish()
    assert len(list(s.run_dir.glob("replan_*.npz"))) == 6 and (s.run_dir / "goal.npz").exists()
    assert (s.run_dir / "video.mp4").stat().st_size > 10000


def test_dry_run_does_not_move(cell, tmp_path):
    """With execute=False the robot stays where it is."""
    s = Session(cell, CFG, StandInModel(), tmp_path)
    s.capture_goal()
    s.capture_start()
    s.begin("grasp")
    before = cell.state()[0]
    s.step(execute=False)
    np.testing.assert_allclose(cell.state()[0], before)


def started(cell, tmp_path):
    """A session with a reach goal 6 cm away and the start at home, captured."""
    s = Session(cell, CFG, StandInModel(), tmp_path)
    _, _, home = cell.state()
    goal = home.copy()
    goal[:3, 3] += [0.05, -0.04, -0.06]
    cell.move(goal, np.zeros(22))
    s.capture_goal()
    cell.move(home, np.zeros(22))
    s.capture_start()
    return s, home


def test_step_after_robot_moved_uses_fresh_state(cell, tmp_path):
    """If the robot moved after the start was captured, the step plans from where it is: the flange moves at most max_step_mm."""
    s, home = started(cell, tmp_path)
    s.begin("reach")
    away = home.copy()
    away[:3, 3] += [0.0, 0.2, 0.05]
    cell.move(away, np.zeros(22))
    before = cell.state()[2][:3, 3]
    record = s.step()
    travelled = np.linalg.norm(cell.state()[2][:3, 3] - before) * 1000
    assert record["moved"] and travelled <= CFG["safety"]["max_step_mm"] + 5


def test_each_run_is_new_and_plan_only_changes_nothing(cell, tmp_path):
    """Plan only leaves the run's counters and logs alone; a second begin starts a fresh run in its own folder."""
    s, _ = started(cell, tmp_path)
    s.begin("reach")
    first_dir = s.run_dir
    s.step(execute=False)
    assert s.replan == 0 and s.records == [] and not list(first_dir.glob("replan_*.npz"))
    s.step()
    import time
    time.sleep(1.1)  # run folders are named by the second
    s.begin("reach")
    assert s.run_dir != first_dir and s.replan == 0 and s.records == [] and not s.finished


def test_stop_halts_a_move(cell):
    """With Stop requested, a long move ends early and the robot holds."""
    _, _, flange = cell.state()
    target = flange.copy()
    target[:3, 3] += [0.0, 0.0, -0.2]
    cell.stop_requested.set()
    cell.move(target, np.full(22, 0.5))
    assert np.linalg.norm(cell.state()[2][:3, 3] - flange[:3, 3]) < 0.01


def test_unsent_commands_do_not_change_the_executor():
    """The rotation smoothing starts from the last target actually sent."""
    from scipy.spatial.transform import Rotation

    from lawm_deploy.robot_model import pose
    from lawm_deploy.safety import Executor, SafetySettings
    ex = Executor(SafetySettings(**CFG["safety"]), np.eye(4))
    turned = pose(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [0, 0, 0])
    ex.command(turned[:3, :3], turned[:3, 3], np.zeros(22), np.eye(4), np.eye(4))
    assert np.allclose(ex.last_rotvec, 0)
    ex.sent(turned)
    assert np.allclose(ex.last_rotvec, Rotation.from_euler("z", 20, degrees=True).as_rotvec())
