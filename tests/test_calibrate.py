"""Calibration against the robot surface recovers the cameras of the simulated cell from rough poses several centimetres and degrees off."""
import numpy as np
from scipy.spatial.transform import Rotation

from lawm_deploy import config
from lawm_deploy.calibrate import calibrate, look_at_optical
from lawm_deploy.robot_model import RobotModel, pose
from lawm_deploy.sim_cell import SimCell

CFG = config.load(config.REPO / "configs/sim.yaml")


def test_look_at_matches_sim_camera():
    """The rough-pose helper gives the same optical frame as the simulated camera built from the same eye and target."""
    cell = SimCell(CFG["cell"], config.mount_pose(CFG))
    for (_, _, T), cam in zip(cell.views(), CFG["cell"]["cameras"].values()):
        np.testing.assert_allclose(look_at_optical(np.asarray(cam["eye"], float), cam["target"]), T, atol=1e-5)
    cell.close()


def test_recovers_camera_poses():
    """From rough poses 3 cm and 5 degrees off, over five arm poses, both cameras land within 3 mm and 0.5 degree with most points matched."""
    cell = SimCell(CFG["cell"], config.mount_pose(CFG))
    robot = RobotModel(config.mount_pose(CFG), n_mask_points=100000)
    _, _, home = cell.state()
    frames = []
    for offset, close in (([0, 0, 0], 0.0), ([0.12, -0.15, -0.1], 0.4), ([-0.12, 0.15, 0.05], 0.8), ([0.1, 0.15, 0.1], 0.2), ([-0.1, -0.15, -0.05], 0.6)):
        target = home.copy()
        target[:3, 3] += offset
        cell.move(target, np.full(22, close))
        q_arm, q_hand, _ = cell.state()
        frames.append((q_arm, q_hand, cell.views()))
    truth = [T for _, _, T in frames[0][2]]
    nudge = pose(Rotation.from_rotvec(np.radians(5) * np.array([0.6, 0.0, 0.8])).as_matrix(), [0.03, -0.02, 0.02])
    results = calibrate(robot, frames, [T @ nudge for T in truth])
    for (T, residual, share), true in zip(results, truth):
        assert np.linalg.norm(T[:3, 3] - true[:3, 3]) < 0.003
        assert np.degrees(Rotation.from_matrix(T[:3, :3] @ true[:3, :3].T).magnitude()) < 0.5
        assert residual < 3.0 and share > 0.7
    cell.close()
