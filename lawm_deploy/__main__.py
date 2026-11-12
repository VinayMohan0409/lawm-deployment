"""python -m lawm_deploy --config configs/real.yaml [--standin] [--port 8080]: connect the cell, load the model and serve the operator page."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # only the simulated cell renders; read before mujoco is imported
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse  # noqa: E402

from lawm_deploy import config  # noqa: E402


def main():
    """Parse the arguments, build everything and serve until Ctrl+C."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--standin", action="store_true", help="test the pipeline with a stand-in model (no GPU)")
    parser.add_argument("--port", type=int, default=8080)
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--sample-pair", help="saved static start/goal directory; defaults to experiment.sample_pair")
    samples.add_argument("--no-sample-pair", action="store_true", help="capture a new goal/start instead")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="enable real commands after site checks; never resets the robot")
    modes.add_argument("--trial-step", action="store_true", help="operator-supervised single move using the accepted existing setup; preserves unverified commissioning flags and all runtime checks; no Run/reset")
    modes.add_argument("--trial-run", action="store_true", help="operator-supervised multi-step closed-loop Run using the accepted existing setup; retains runtime checks and current loop stop rules")
    modes.add_argument("--observe-only", action="store_true", help="read-only real cell, no model or UR control interface")
    args = parser.parse_args()
    cfg = config.load(args.config)
    sample_pair = None if args.no_sample_pair else args.sample_pair or cfg.get('experiment', {}).get('sample_pair')
    if args.observe_only:
        from lawm_deploy.preflight import observe
        observe(cfg, cameras=True)
        return
    trial = args.trial_step or args.trial_run
    if trial and cfg['cell']['type'] != 'real':
        parser.error('--trial-step/--trial-run are for the real cell')
    if args.execute or trial:
        from lawm_deploy.hardware_contract import require_execution_setup
        if cfg['cell']['type'] == 'real':
            require_execution_setup(cfg, trial_step=trial)
    if trial:
        cfg['execution'] = dict(mode='supervised_trial_step' if args.trial_step else 'supervised_trial_run',
            max_move_attempts=1 if args.trial_step else None,
            existing_setup_accepted=True, physical_commissioning_completed=False,
            payload_policy='retain existing controller setting per operator instruction')
        print(('Trial step: one move attempt via Step; Run disabled. ' if args.trial_step else
               'Trial run: Step and closed-loop Run enabled; existing loop stop rules apply. ') +
              'Existing setup accepted for this trial, physical commissioning flags remain unverified.', flush=True)
    # Fail on missing model dependencies before opening any device.
    from lawm_deploy.model import StandInModel, load_model
    from lawm_deploy.session import Session
    from lawm_deploy.ui import App
    from lawm_deploy.hardware_contract import hand_contract
    model = StandInModel() if args.standin else load_model(cfg)
    cell = app = None
    try:
        if cfg["cell"]["type"] == "sim":
            from lawm_deploy.sim_cell import SimCell
            cell = SimCell(cfg["cell"], config.mount_pose(cfg), cfg["seed"])
        else:
            from lawm_deploy.cell import RealCell
            cell = RealCell(cfg["cell"], hand_contract()[0], execute=args.execute or trial,
                            trial_step=args.trial_step, trial_run=args.trial_run, full_config=cfg)
        session = Session(cell, cfg, model, config.repo_path(cfg["log_dir"]))
        if sample_pair:
            from lawm_deploy.saved_pair import load_pair
            load_pair(session, config.repo_path(sample_pair))
        app = App(session, args.config, args.port, cell if cfg["cell"]["type"] == "sim" else None)
        app.serve()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        if cell is not None and (app is None or app.shutdown()):
            cell.close()


if __name__ == "__main__":
    main()
