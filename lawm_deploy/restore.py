"""Reuse the collection project's UR preset restore, before LAWM owns devices."""
from __future__ import annotations

import json
import subprocess
import sys

from lawm_deploy import config
from lawm_deploy.hardware_contract import HardwareError, vector


def restore_command(cfg, preset=None, *, open_hand=False):
    """Validate files and robot identity without connecting to any device."""
    settings = cfg['restore']
    script = config.repo_path(settings['script']).resolve()
    source_config = config.repo_path(settings['config']).resolve()
    path = config.repo_path(preset or settings['preset']).resolve()
    for file in (script, source_config, path):
        if not file.is_file():
            raise FileNotFoundError(file)
    stored = json.loads(path.read_text())
    source = json.loads(source_config.read_text())
    for value in (stored, source):
        if value.get('robot_ip') != cfg['cell']['robot_ip'] or value.get('side') != 'right':
            raise HardwareError('Restore preset/config must match the configured right UR')
    if stored.get('schema') != 'ur-joint-preset-v1' or stored.get('joint_names') != [
            'base', 'shoulder', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']:
        raise HardwareError('Unsupported restore preset or joint order')
    vector(stored.get('q_rad'), 6, 'restore joints')
    return [sys.executable, str(script), '--config', str(source_config),
            '--restore', str(path), '--execute'] + ([] if open_hand else ['--arm-only'])


def restore(cfg, preset=None, *, open_hand=False):
    command = restore_command(cfg, preset, open_hand=open_hand)
    action = 'then slowly open Sharpa and verify feedback' if open_hand else 'hand unchanged'
    print(f'Restoring UR using collection preset: {command[command.index("--restore") + 1]} ({action})', flush=True)
    subprocess.run(command, check=True)
