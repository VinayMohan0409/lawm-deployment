"""Import stationary RGB-D samples as goals/references, never as live state."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from lawm_deploy.config import mount_pose, repo_path
from lawm_deploy.hardware_contract import HardwareError, transform, vector
from lawm_deploy.perception import Perception, PerceptionSettings
from lawm_deploy.robot_model import ARM_JOINTS


def read_sample(path, cfg, robot, hand):
    """Rebuild FK/masked model input using current preprocessing and saved raw depth.

    No device access. Reject mismatched frames, devices, mount or calibration.
    Recorded joint feedback is preserved even if slightly outside model limits.
    """
    from lawm_deploy.session import Observation

    path = Path(path).resolve()
    read = lambda name: json.loads((path / name).read_text())
    frames, state, captured = read('frames.json'), read('joint_state.json'), read('capture_config.json')
    if frames.get('point_cloud_frame') != 'UR_controller_base' or state.get('angle_units') != 'radians':
        raise HardwareError('Saved sample requires UR_controller_base coordinates and radian joints')
    if not state.get('stationarity', {}).get('passed'):
        raise HardwareError('Saved sample did not pass stationary capture checks')
    if not np.allclose(mount_pose(captured), mount_pose(cfg), atol=1e-8, rtol=0):
        raise HardwareError('Saved sample mount differs from current configuration')
    if state.get('robot_ip') != cfg['cell']['robot_ip'] or state.get('hand_serial') != cfg['cell']['sharpa']['serial']:
        raise HardwareError('Saved sample robot/hand identity differs from current configuration')

    def reorder(names, values, expected):
        if len(names) != len(expected) or set(names) != set(expected):
            raise HardwareError('Saved joint names do not match the configured robot')
        q = vector(values, len(expected), 'saved joints')
        return q[[names.index(n) for n in expected]]

    qa = reorder(state['arm_joint_names'], state['q_arm_rad'], list(ARM_JOINTS))
    qh = reorder(state['hand_joint_names'], state['q_hand_rad'], robot.hand_joint_names)
    calibration = yaml.safe_load(repo_path(cfg['cell']['cameras']['calibration']).read_text())
    if calibration.get('frame') != 'depth_optical':
        raise HardwareError('Current calibration must be native depth optical')
    expected_serials = [str(d['serial']) for d in cfg['cell']['cameras']['devices']]
    views, raw, metadata = [], [], []
    with np.load(path / 'scene_raw.npz', allow_pickle=False) as saved:
        for i, role in enumerate(('ego', 'third')):
            camera = read(f'{role}/camera.json')
            serial = str(camera['serial'])
            if i >= len(expected_serials) or serial != expected_serials[i]:
                raise HardwareError('Saved camera identity/order differs from current configuration')
            T = transform(saved[f'T_base_depth_{i}'], 'saved depth pose')
            if not np.allclose(T, calibration['T_base_camera'][serial], atol=1e-7, rtol=0):
                raise HardwareError('Saved depth calibration differs from current calibration')
            K = np.asarray(saved[f'K_depth_{i}'], dtype=float)
            depth = saved[f'raw_depth_z16_{i}']
            scale = float(saved[f'depth_scale_m_{i}'])
            if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                    or depth.dtype != np.uint16 or depth.ndim != 2 or not np.isfinite(scale) or scale <= 0):
                raise HardwareError('Invalid native depth/intrinsics/scale in sample')
            views.append((depth.astype(np.float32) * scale, K, T))
            raw.append((depth[None].copy(), scale))
            metadata.append(dict(serial=serial, role=role, frame='depth_optical',
                                 sensor_settings_at_capture=camera['depth_settings_at_start']))
    if len(expected_serials) != len(views):
        raise HardwareError('Saved pair requires the same two cameras')
    robot.set(qa, qh)
    flange, root = robot.flange_pose(), robot.root_pose()
    coords, segment, info = Perception(PerceptionSettings(**cfg['perception']), cfg['seed'])(
        views, robot.surface(qa, qh), hand.points(root, qh))
    info.update(source='saved_static_sample', source_directory=str(path), live_observation=False,
                camera_metadata=metadata, source_capture_config=captured,
                source_sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                               for name in ('scene_raw.npz', 'joint_state.json', 'frames.json', 'capture_config.json')},
                note='Reprocessed saved single-frame depth and FK with current config; not a new capture or live state.')
    return Observation(coords, segment, qa, qh, flange, root, views, info, raw, copy.deepcopy(cfg))


def load_pair(session, path):
    """Install a saved goal/start reference. Session.begin must observe live state."""
    if not session.finished:
        raise HardwareError('Finish the current run before loading a saved pair')
    path = Path(path)
    goal = read_sample(path / 'goal', session.cfg, session.robot, session.hand)
    start = read_sample(path / 'start', session.cfg, session.robot, session.hand)
    session.goal, session.start, session.current = goal, start, None
    session.save('goal', goal)
    session.save('start', start)
    return goal, start
