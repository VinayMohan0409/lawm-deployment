"""python -m lawm_deploy.sim_demo --config configs/sim.yaml --task reach [--standin]: an example goal and start in the simulated cell, one Run with the model, the video and logs in runs/."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from lawm_deploy import config  # noqa: E402
from lawm_deploy.model import StandInModel, load_model  # noqa: E402
from lawm_deploy.robot_model import hand_spec  # noqa: E402
from lawm_deploy.session import Session  # noqa: E402
from lawm_deploy.sim_cell import SimCell  # noqa: E402
from lawm_deploy.ui import FLEX, SIM_PRESETS  # noqa: E402


def closure_angles(cell, closure):
    """Sharpa angles with the curling joints at closure of their upper limit, the others at 0."""
    names = hand_spec()[1]
    hi = [cell.model.jnt_range[mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_JOINT, f"hand_joint_{k}")][1] for k in range(len(names))]
    return np.array([closure * h if any(n.endswith(f) or f"_{f}" in n for f in FLEX) else 0.0 for n, h in zip(names, hi)])


def arrange(cell, offset, closure, direct):
    """Put the robot at home + offset with the fingers at closure: driven there, or set directly (objects back at their config positions)."""
    target = cell.home.copy()
    target[:3, 3] += offset
    if direct:
        cell.set_state(cell.ik(target, np.radians(cell.cfg["home_deg"])), closure_angles(cell, closure), [o["pos"] for o in cell.cfg["objects"]])
    else:
        cell.move(target, closure_angles(cell, closure))


def main():
    """Capture the example goal, return to home, capture the start, Run, print each replan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config.REPO / "configs/sim.yaml"))
    parser.add_argument("--task", choices=sorted(SIM_PRESETS), default="reach")
    parser.add_argument("--standin", action="store_true")
    args = parser.parse_args()
    cfg = config.load(args.config)
    cell = SimCell(cfg["cell"], config.mount_pose(cfg), cfg["seed"])
    cell.home = cell.state()[2]
    session = Session(cell, cfg, StandInModel() if args.standin else load_model(cfg), config.repo_path(cfg["log_dir"]))
    offset, closure, direct = SIM_PRESETS[args.task]
    arrange(cell, offset, closure, direct)
    session.capture_goal()
    arrange(cell, (0.0, 0.0, 0.0), 0.0, True)
    session.capture_start()
    session.begin(args.task)
    reason = session.run(on_step=lambda r: print(f"replan {r['replan']:2d}: distance {r['distance_mm']:6.1f} mm, {'moved' if r['moved'] else 'held: ' + str(r['hold'])}, "
                                                  f"commanded {r['commanded_mm']:5.1f} mm, fit {r['fit_residual_mm']:.1f} mm, {r['executed']}", flush=True))
    print(f"stopped: {reason}; video and logs: {session.run_dir}")
    cell.close()


if __name__ == "__main__":
    main()
