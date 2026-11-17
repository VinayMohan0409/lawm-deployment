"""Operator commands: no hardware connections or physical motion in these tests."""
import json
import subprocess
import shlex
from types import SimpleNamespace

import pytest

from lawm_deploy import config
from lawm_deploy.__main__ import main
from lawm_deploy.hardware_contract import HardwareError
from lawm_deploy import restore as restoring


@pytest.fixture
def restore_cfg(tmp_path):
    cfg = config.load(config.REPO / 'configs/real.yaml')
    script = tmp_path / 'ur_joint_pose.py'
    script.write_text('# fake collection entrypoint\n')
    source = tmp_path / 'collection.json'
    source.write_text(json.dumps(dict(robot_ip=cfg['cell']['robot_ip'], side='right')))
    preset = tmp_path / 'pose.json'
    preset.write_text(json.dumps(dict(schema='ur-joint-preset-v1', robot_ip=cfg['cell']['robot_ip'], side='right',
        joint_names=['base', 'shoulder', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'], q_rad=[0]*6)))
    cfg['restore'] = dict(script=str(script), config=str(source), preset=str(preset))
    return cfg, preset


def test_restore_reuses_collection_arm_only_and_propagates_failure(restore_cfg, monkeypatch):
    cfg, preset = restore_cfg
    calls = []
    monkeypatch.setattr(restoring.subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw)))
    restoring.restore(cfg)
    cmd, kw = calls[0]
    assert cmd[-4:] == ['--restore', str(preset), '--execute', '--arm-only'] and kw == {'check': True}
    def fail(*a, **kw): raise subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(restoring.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError): restoring.restore(cfg)


def test_restore_wrong_robot_never_launches(restore_cfg, monkeypatch):
    cfg, preset = restore_cfg
    data = json.loads(preset.read_text()); data['robot_ip'] = 'wrong'
    preset.write_text(json.dumps(data))
    monkeypatch.setattr(restoring.subprocess, 'run', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(HardwareError, match='match'): restoring.restore(cfg)


def test_restore_open_hand_uses_collection_full_reset(restore_cfg, monkeypatch):
    cfg, preset = restore_cfg
    calls = []
    monkeypatch.setattr(restoring.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    restoring.restore(cfg, open_hand=True)
    assert calls[0][-3:] == ['--restore', str(preset), '--execute']
    assert '--arm-only' not in calls[0]


def test_open_hand_cli_requires_restore_and_forwards_option(monkeypatch):
    base = ['--config', str(config.REPO / 'configs/real.yaml')]
    with pytest.raises(SystemExit): main(base + ['--open-hand'])
    calls = []
    monkeypatch.setattr(restoring, 'restore_command', lambda *a, **kw: [])
    monkeypatch.setattr(restoring, 'restore', lambda *a, **kw: calls.append(kw))
    main(base + ['--restore', '--open-hand'])
    assert calls == [{'open_hand': True}]


def test_capture_and_restore_cli_exit_without_loading_model(monkeypatch, tmp_path):
    from lawm_deploy import goals, model
    monkeypatch.setattr(model, 'load_model', lambda *a: pytest.fail('must not load model'))
    calls = []
    monkeypatch.setattr(goals, 'capture_goal', lambda cfg, path: calls.append(('capture', path)))
    monkeypatch.setattr(restoring, 'restore_command', lambda *a: [])
    monkeypatch.setattr(restoring, 'restore', lambda cfg, path: calls.append(('restore', path)))
    base = ['--config', str(config.REPO / 'configs/real.yaml')]
    main(base + ['--capture-goal', str(tmp_path / 'goal')])
    main(base + ['--restore'])
    assert calls == [('capture', tmp_path / 'goal'), ('restore', '')]


@pytest.mark.parametrize('options', [
    ['--capture-goal', '/unused', '--trial-run'],
    ['--capture-goal', '/unused', '--restore'],
    ['--observe-only', '--restore'],
    ['--goal', '/unused', '--sample-pair', '/unused'],
])
def test_conflicting_modes_fail_before_devices(options):
    with pytest.raises(SystemExit) as error:
        main(['--config', str(config.REPO / 'configs/real.yaml')] + options)
    assert error.value.code == 2


def test_goal_validated_then_restore_then_device_ownership(monkeypatch, tmp_path):
    from lawm_deploy import goals, model, cell, session, ui
    calls = []
    goal = object()
    monkeypatch.setattr(goals, 'prepare_goal', lambda *a: calls.append('goal_validated') or goal)
    monkeypatch.setattr(model, 'load_model', lambda *a: calls.append('model') or object())
    monkeypatch.setattr(restoring, 'restore_command', lambda *a: [])
    monkeypatch.setattr(restoring, 'restore', lambda *a: calls.append('restore'))
    monkeypatch.setattr(cell, 'RealCell', lambda *a, **kw: calls.append('connect') or SimpleNamespace(close=lambda: calls.append('close')))
    monkeypatch.setattr(session, 'Session', lambda c, cfg, *a: SimpleNamespace(cfg=cfg))
    def install(s, g):
        assert g is goal and s.cfg['experiment']['sample_pair'] is None
        calls.append('install_goal')
    monkeypatch.setattr(goals, 'install_goal', install)
    monkeypatch.setattr(ui, 'App', lambda *a: SimpleNamespace(serve=lambda: calls.append('ui'), shutdown=lambda: True))
    main(['--config', str(config.REPO / 'configs/real.yaml'), '--trial-run', '--restore', '--goal', str(tmp_path / 'goal')])
    assert calls == ['goal_validated', 'model', 'restore', 'connect', 'install_goal', 'ui', 'close']


def test_invalid_goal_prevents_restore(monkeypatch):
    from lawm_deploy import goals
    monkeypatch.setattr(restoring, 'restore_command', lambda *a: [])
    monkeypatch.setattr(restoring, 'restore', lambda *a: pytest.fail('must not restore'))
    def fail(*a): raise HardwareError('Invalid goal')
    monkeypatch.setattr(goals, 'prepare_goal', fail)
    with pytest.raises(HardwareError, match='Invalid goal'):
        main(['--config', str(config.REPO / 'configs/real.yaml'), '--trial-run', '--restore', '--goal', '/unused'])


@pytest.mark.parametrize('value', [0, 6, 1.5, True, '3'])
def test_invalid_execution_steps(value):
    with pytest.raises(ValueError, match='execute_steps'):
        config.execution_steps({'experiment': {'execute_steps': value}})


@pytest.mark.parametrize('external_settings', [False, True])
def test_launcher_uses_active_python_or_external_settings(tmp_path, monkeypatch, external_settings):
    """Exercise launcher selection without importing SDKs or connecting devices."""
    import os
    prefix = tmp_path / 'environment with spaces'
    executable = prefix / 'bin/python'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    for key in ('LAWM_CONFIG', 'LAWM_CONDA_PREFIX'):
        monkeypatch.delenv(key, raising=False)
    settings = tmp_path / 'environment.sh'
    monkeypatch.setenv('LAWM_LOCAL_ENV', str(settings))
    monkeypatch.setenv('PATH', str(executable.parent) + os.pathsep + os.environ['PATH'])
    if external_settings:
        local_config = tmp_path / 'real local.yaml'
        settings.write_text(f'export LAWM_CONDA_PREFIX={shlex.quote(str(prefix))}\n'
                            f'export LAWM_CONFIG={shlex.quote(str(local_config))}\n')
    else:
        local_config = config.REPO / 'configs/real.local.yaml'
        local_config = 'configs/real.local.yaml' if local_config.exists() else 'configs/real.yaml'
    result = subprocess.run(['bash', str(config.REPO / 'run_lawm.sh'), '--check', '--help'],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ['-m', 'lawm_deploy.preflight', '--config', str(local_config), '--help']
