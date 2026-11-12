"""Conventions confirmed by the model owner on 2026-11-11.

Training time is semantic metadata, not an execution deadline. The rotation
and point counts stay fixed until the model owner revises this contract.
"""
import numpy as np

TRAINING_STEP_S = 0.2
TRAINING_UP = (-0.022, -0.546, -0.837)
SPIN_DEG = 315.0
N_HAND, N_SCENE = 1024, 5120


def validate_model_contract(cfg):
    m, p = cfg['model'], cfg['perception']
    up = np.asarray(m['training_up'], dtype=float)
    if up.shape != (3,) or not np.allclose(up, TRAINING_UP, rtol=0, atol=1e-9):
        raise ValueError('training_up is fixed for this checkpoint; confirm changes with the model owner')
    if m['spin_deg'] != SPIN_DEG or (p['n_hand'], p['n_scene']) != (N_HAND, N_SCENE):
        raise ValueError('Keep spin_deg=315, n_hand=1024 and n_scene=5120 for this checkpoint')


def timing_metadata():
    return dict(training_step_s=TRAINING_STEP_S, execution='observe_execute_settle_observe',
                execution_duration_is_training_step=False,
                arm_finger_phase_tolerance='not specified by model owner')
