"""python -m lawm_deploy --config configs/real.yaml [--standin] [--port 8080]: connect the cell, load the model and serve the operator page."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # only the simulated cell renders; read before mujoco is imported
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse  # noqa: E402

from lawm_deploy import config  # noqa: E402


def main(argv=None):
    """Parse the arguments, build everything and serve until Ctrl+C."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--standin", action="store_true", help="test the pipeline with a stand-in model (no GPU)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--execute-steps", type=int, choices=range(1, 6),
                        help="Run executes this many successive predicted targets per inference (default: experiment.execute_steps, or 1); Step stays one move")
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--sample-pair", help="saved static start/goal directory; defaults to experiment.sample_pair")
    samples.add_argument("--no-sample-pair", action="store_true", help="capture a new goal/start instead")
    samples.add_argument("--goal", help="goal.npz, captured goal directory, or original static goal directory")
    parser.add_argument("--capture-goal", metavar="DIRECTORY", help="read-only goal capture into a new directory, then exit; no model inference")
    parser.add_argument("--restore", nargs="?", const="", metavar="PRESET_JSON",
                        help="restore UR using collection's saved preset (or specified JSON); alone exits, with --trial-run restores before opening the UI")
    parser.add_argument('--open-hand', action='store_true', help='with --restore, also slowly open Sharpa and verify arrival, using the collection reset routine')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="enable real motion and manual arrangement controls")
    modes.add_argument("--trial-step", action="store_true", help="enable one move attempt; Run and manual arrangement controls disabled")
    modes.add_argument("--trial-run", action="store_true", help="enable closed-loop Run; manual arrangement controls disabled")
    modes.add_argument("--observe-only", action="store_true", help="read-only real cell, no model or UR control interface")
    args = parser.parse_args(argv)
    cfg = config.load(args.config)
    if args.open_hand and args.restore is None:
        parser.error('--open-hand requires --restore')
    restore_options = {'open_hand': True} if args.open_hand else {}
    if args.execute_steps is not None:
        cfg.setdefault('experiment', {})['execute_steps'] = args.execute_steps
    config.execution_steps(cfg)
    execute = args.execute or args.trial_step or args.trial_run
    if args.capture_goal and (execute or args.restore is not None or args.observe_only or args.goal or args.sample_pair):
        parser.error('--capture-goal cannot be combined with motion, observation or saved-goal loading modes')
    if args.observe_only and (args.restore is not None or args.goal or args.sample_pair):
        parser.error('--observe-only cannot be combined with restore or saved-goal loading')
    if (args.restore is not None or args.capture_goal) and cfg['cell']['type'] != 'real':
        parser.error('--restore and --capture-goal require a real cell config')
    if args.capture_goal:
        from lawm_deploy.goals import capture_goal
        capture_goal(cfg, config.repo_path(args.capture_goal))
        return
    if args.restore is not None:
        from lawm_deploy.restore import restore, restore_command
        restore_command(cfg, args.restore, **restore_options)  # reject a bad preset before opening devices
        if not execute:
            if args.goal or args.sample_pair:
                parser.error('Standalone --restore exits; add --trial-run to also load a goal and run')
            restore(cfg, args.restore, **restore_options)
            return
    sample_pair = None if args.no_sample_pair or args.goal else args.sample_pair or cfg.get('experiment', {}).get('sample_pair')
    goal = None
    if args.goal:
        from lawm_deploy.goals import prepare_goal
        goal_path = config.repo_path(args.goal).resolve()
        cfg.setdefault('experiment', {}).update(goal=str(goal_path), sample_pair=None)
        goal = prepare_goal(goal_path, cfg)
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
            require_execution_setup(cfg)
    if trial:
        cfg['execution'] = dict(mode='supervised_trial_step' if args.trial_step else 'supervised_trial_run',
            max_move_attempts=1 if args.trial_step else None)
    # Fail on missing model dependencies before opening any device.
    from lawm_deploy.model import StandInModel, load_model
    from lawm_deploy.session import Session
    from lawm_deploy.ui import App
    from lawm_deploy.hardware_contract import hand_contract
    model = StandInModel() if args.standin else load_model(cfg)
    if args.restore is not None:
        restore(cfg, args.restore, **restore_options)  # collection releases devices before LAWM connects
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
        if goal is not None:
            from lawm_deploy.goals import install_goal
            install_goal(session, goal)
        app = App(session, args.config, args.port, cell if cfg["cell"]["type"] == "sim" else None)
        app.serve()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        if cell is not None and (app is None or app.shutdown()):
            cell.close()


if __name__ == "__main__":
    main()
