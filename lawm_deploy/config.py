"""The YAML config: one file per cell, paths relative to the repo root."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from lawm_deploy.robot_model import pose

REPO = Path(__file__).resolve().parents[1]


def load(path):
    """The config dictionary of a YAML file."""
    return yaml.safe_load(Path(path).read_text())


def repo_path(p):
    """A config path made absolute against the repo root."""
    return Path(p) if Path(p).is_absolute() else REPO / p


def mount_pose(cfg):
    """4x4 pose of the hand's MJCF origin in the flange frame."""
    m = cfg["mount"]
    return pose(Rotation.from_euler("xyz", m["rpy_deg"], degrees=True).as_matrix(), m["xyz_m"])
