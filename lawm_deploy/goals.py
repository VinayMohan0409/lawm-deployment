"""Capture and load reusable goal states: calibrated depth plus measured joints."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from lawm_deploy import config
from lawm_deploy.hardware_contract import HardwareError, hand_contract, transform, vector
from lawm_deploy.observation_io import write_ply
from lawm_deploy.perception import Perception, PerceptionSettings, SEG_SCENE
from lawm_deploy.saved_pair import read_sample


def resolve_goal(path):
    """Accept a goal NPZ, a new capture directory, or an original static sample."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if (path / 'goal.npz').is_file():
            path = path / 'goal.npz'
        elif (path / 'goal').is_dir():
            path = path / 'goal'
        if path.is_dir() and (path / 'scene_raw.npz').is_file():
            return path
    if path.is_file() and path.suffix == '.npz':
        return path
    raise HardwareError('Goal must be a captured goal.npz or goal directory with depth and joints; a PLY alone is insufficient')


def read_goal(path, cfg, robot, hand):
    """Reprocess saved depth and FK under current settings, without device access."""
    from lawm_deploy.cell import camera_poses
    from lawm_deploy.session import Observation

    path = resolve_goal(path)
    if path.is_dir():
        return read_sample(path, cfg, robot, hand)
    with np.load(path, allow_pickle=False) as saved:
        required = {'config_json', 'observation_info_json', 'q_arm', 'q_hand', 'coords', 'segment'}
        if not required.issubset(saved.files):
            raise HardwareError('Goal archive is missing saved observation metadata or joints')
        captured = json.loads(str(saved['config_json']))
        info = json.loads(str(saved['observation_info_json']))
        if info.get('point_cloud_frame', 'UR_controller_base') != 'UR_controller_base' or info.get('angle_units', 'radians') != 'radians':
            raise HardwareError('Goal requires UR_controller_base coordinates and radian joints')
        if not np.allclose(config.mount_pose(captured), config.mount_pose(cfg), atol=1e-8, rtol=0):
            raise HardwareError('Saved goal mount differs from current configuration')
        if (captured['cell']['robot_ip'] != cfg['cell']['robot_ip'] or
                captured['cell']['sharpa']['serial'] != cfg['cell']['sharpa']['serial']):
            raise HardwareError('Saved goal robot/hand identity differs from current configuration')
        serials = [str(d['serial']) for d in cfg['cell']['cameras']['devices']]
        if [str(d['serial']) for d in captured['cell']['cameras']['devices']] != serials:
            raise HardwareError('Saved goal camera identity/order differs from current configuration')
        qa = vector(saved['q_arm'], 6, 'goal arm joints').copy()
        qh = vector(saved['q_hand'], 22, 'goal hand joints').copy()
        if 'hand_joint_names' in info and info['hand_joint_names'] != robot.hand_joint_names:
            raise HardwareError('Goal hand joint order differs from current model')
        from lawm_deploy.robot_model import ARM_JOINTS
        if 'arm_joint_names' in info and info['arm_joint_names'] != list(ARM_JOINTS):
            raise HardwareError('Goal arm joint order differs from current model')
        poses = camera_poses(cfg['cell']['cameras'], True)
        views, raw = [], []
        for i, expected in enumerate(poses):
            camera = np.asarray(saved[f'cam_{i}'], dtype=float)
            if camera.shape != (25,):
                raise HardwareError('Invalid saved goal camera parameters')
            K, T = camera[:9].reshape(3, 3), transform(camera[9:].reshape(4, 4), 'goal camera pose')
            if not np.allclose(T, expected, atol=1e-7, rtol=0):
                raise HardwareError('Saved goal depth calibration differs from current calibration')
            depth = np.asarray(saved[f'depth_median_m_{i}'], dtype=np.float32)
            if (depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any() or
                    not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0):
                raise HardwareError('Invalid goal depth/intrinsics')
            views.append((depth.copy(), K.copy(), T.copy()))
            key = f'sensor_depth_z16_{i}'
            if key in saved:
                frames, scale = saved[key], float(saved[f'sensor_depth_scale_m_{i}'])
                if (frames.dtype != np.uint16 or frames.ndim != 3 or frames.shape[1:] != depth.shape or
                        not len(frames) or not np.isfinite(scale) or scale <= 0):
                    raise HardwareError('Invalid goal raw depth/scale')
                raw.append((frames.copy(), scale))
        robot.set(qa, qh)
        flange, root = robot.flange_pose(), robot.root_pose()
        coords, segment, processed = Perception(PerceptionSettings(**cfg['perception']), cfg['seed'])(
            views, robot.surface(qa, qh), hand.points(root, qh))
    info.update(processed)
    info.update(source='saved_goal_archive', source_path=str(path), live_observation=False,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_capture_config=captured)
    return Observation(coords, segment, qa, qh, flange, root, views, info,
                       raw if len(raw) == len(views) else None, copy.deepcopy(cfg))


def prepare_goal(path, cfg):
    """Validate the complete goal before opening devices or restoring the arm."""
    from lawm_deploy.robot_model import RobotModel
    from lawm_deploy.hand import Hand
    robot = RobotModel(config.mount_pose(cfg), seed=cfg['seed'])
    hand = Hand(robot.model, cfg['perception']['n_hand'], cfg['seed'])
    return read_goal(path, cfg, robot, hand)


def install_goal(session, goal):
    session.goal, session.start, session.current = goal, None, None
    session.save('goal', goal)


def capture_goal(cfg, destination):
    """One read-only capture; no model weights, UR control or hand commands."""
    from lawm_deploy.cell import RealCell
    from lawm_deploy.session import Session
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Goal destination already exists: {destination}')
    # Build the observation components and reserve the directory before connecting.
    session = Session(None, cfg, None, destination, capture_only=True)
    cell = None
    try:
        cell = RealCell(cfg['cell'], hand_contract()[0], full_config=cfg)
        session.cell = cell
        from lawm_deploy.robot_model import ARM_JOINTS
        obs = session.observe()
        obs.info.update(point_cloud_frame='UR_controller_base', angle_units='radians',
                        arm_joint_names=list(ARM_JOINTS), hand_joint_names=session.robot.hand_joint_names)
        session.save('goal', obs)
        write_ply(destination / 'model_input_base.ply', obs.coords)
        write_ply(destination / 'scene_base.ply', obs.coords[obs.segment == SEG_SCENE])
        write_ply(destination / 'hand_base.ply', obs.coords[obs.segment != SEG_SCENE])
        (destination / 'capture.json').write_text(json.dumps(dict(
            completed=True, mode='read_only_goal_capture', point_cloud_frame='UR_controller_base',
            units='metres', goal=str(destination / 'goal.npz'), hardware_log=str(cell.log_path)), indent=2) + '\n')
        print(f'Goal saved: {destination / "goal.npz"}', flush=True)
        return destination / 'goal.npz'
    finally:
        if cell is not None:
            cell.close()
