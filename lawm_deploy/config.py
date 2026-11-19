"""The YAML config: one file per cell, paths relative to the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml
from scipy.spatial.transform import Rotation

from lawm_deploy.robot_model import pose

REPO = Path(__file__).resolve().parents[1]


def load(path):
    """The config dictionary of a YAML file."""
    return yaml.safe_load(Path(path).read_text())


def repo_path(p):
    """A config path made absolute against the repo root."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO / path


def execution_steps(cfg, task=None):
    """Number of successive predicted targets consumed by Run per model call: loop.<task>.execute_steps, else experiment.execute_steps, else 1."""
    value = cfg.get('experiment', {}).get('execute_steps', 1)
    if task is not None and cfg.get('loop', {}).get(task, {}).get('execute_steps') is not None:
        value = cfg['loop'][task]['execute_steps']
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError('experiment.execute_steps must be an integer from 1 to 5')
    return value


def mount_pose(cfg):
    """4x4 pose of the hand's MJCF origin in the flange frame."""
    m = cfg["mount"]
    return pose(Rotation.from_euler("xyz", m["rpy_deg"], degrees=True).as_matrix(), m["xyz_m"])
