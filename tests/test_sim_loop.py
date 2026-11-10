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
