"""python -m lawm_deploy --config configs/real.yaml [--standin] [--port 8080]: connect the cell, load the model and serve the operator page."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # only the simulated cell renders; read before mujoco is imported
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse  # noqa: E402

from lawm_deploy import config  # noqa: E402
from lawm_deploy.model import StandInModel, load_model  # noqa: E402
from lawm_deploy.robot_model import hand_spec  # noqa: E402
from lawm_deploy.session import Session  # noqa: E402
from lawm_deploy.ui import App  # noqa: E402


def main():
    """Parse the arguments, build everything and serve until Ctrl+C."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--standin", action="store_true", help="test the pipeline with a stand-in model (no GPU)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    cfg = config.load(args.config)
    if cfg["cell"]["type"] == "sim":
        from lawm_deploy.sim_cell import SimCell
        cell = SimCell(cfg["cell"], config.mount_pose(cfg), cfg["seed"])
    else:
        from lawm_deploy.cell import RealCell
        cell = RealCell(cfg["cell"], hand_spec()[1])
    model = StandInModel() if args.standin else load_model(cfg)
    session = Session(cell, cfg, model, config.repo_path(cfg["log_dir"]))
    try:
        App(session, args.config, args.port, cell if cfg["cell"]["type"] == "sim" else None).serve()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        cell.close()  # stops the arm, ends the UR control script and the cameras


if __name__ == "__main__":
    main()
