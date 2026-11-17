# LAWM deployment

Deploy the latent action world model on a UR arm, a right Sharpa Wave hand,
and two calibrated RealSense cameras. Includes a Viser interface and a MuJoCo
simulation backend.

The planner searches latent actions and predicts five steps of point-cloud
motion. Hand fitting converts the prediction into a palm pose and 22 joint
angles; the real backend checks controller IK and executes a flange `moveL`
alongside timed finger commands.

## Installation

Python 3.10 and an editable installation of `latent-action-world-model` are
required. The reference model revision is `5a2ccb8`. Model inference needs CUDA;
the stand-in model and tests run without a GPU or checkpoint.

For an existing environment:

```bash
python -m pip install -e ../latent-action-world-model
python -m pip install -e '.[real,dev]'
```

Install the model repository's CUDA dependencies and the vendor Sharpa SDK.
If the SDK is not importable, set `SHARPA_SDK_PYTHON` to its Python module directory.
For a separate Conda environment with the reference dependency versions:

```bash
./scripts/setup_environment.sh ../latent-action-world-model
conda activate ./.conda/lawm
```

`LAWM_CONDA_EXE` selects the Conda executable; `LAWM_CONDA_PREFIX` overrides the
environment location. The setup script does not connect to hardware.

Place the matched model files in `checkpoint/`; see [checkpoint/README.md](checkpoint/README.md).
The weights, raw data, calibration and generated outputs are not tracked by Git.

## Configuration

```bash
cp configs/real.yaml configs/real.local.yaml
```

Edit the local copy: robot IP, hand and camera serial numbers, mounting transform,
camera calibration, workspace/table bounds, and optional restore paths. Values in
`real.yaml` are examples, not a calibration of your installation.

The launcher uses the active Python interpreter, or `LAWM_CONDA_PREFIX/bin/python`
when configured. Configuration selection is `--config`, then `LAWM_CONFIG`, then
`configs/real.local.yaml` if present, otherwise `configs/real.yaml`. Paths inside
YAML are relative to this repository; absolute paths and `~` are also accepted.
Use local storage for `log_dir` to avoid filesystem delays in the control loop.

Optional machine-specific environment settings can live outside the repository
in `$XDG_CONFIG_HOME/lawm-deploy/environment.sh` (default
`~/.config/lawm-deploy/environment.sh`). The launcher sources this file;
`LAWM_LOCAL_ENV` selects another file. Keep personal configuration out of commits.

Camera calibration is a YAML file with `frame: depth_optical` and
`T_base_camera`, mapping each camera serial to its 4×4 camera-to-base matrix.
The base frame is the UR controller base, z-up, in metres, with zero TCP offset.
`python -m lawm_deploy.calibrate --config configs/real.local.yaml` provides a
robot-surface ICP calibration helper. Its matching score is not an independent
measurement of calibration accuracy.

## Capture and run

Arrange the hand and scene at the desired goal, then capture into a new directory:

```bash
./run_lawm.sh --capture-goal goals/cup_grasp_01
```

This reads the cameras and joints without commanding motion. It saves `goal.npz`
and PLY previews. A goal needs depth, calibration and joint state; a PLY alone
cannot replace `goal.npz`. Existing captures are not overwritten.

Arrange the start scene. Preview a plan without motion:

```bash
./run_lawm.sh --goal goals/cup_grasp_01
```

Open `http://localhost:8080`, select the task, click **Capture start**, then
**Plan only**. Enable closed-loop execution with:

```bash
./run_lawm.sh --trial-run --goal goals/cup_grasp_01 --execute-steps 3
```

Click **Capture start**, then **Run**. **Stop** ends execution.
`--trial-step` permits one move attempt. `--execute` also exposes manual arrangement
and teleoperation handover controls. All execution modes use the same configured
motion limits. No robot movement occurs automatically on startup unless `--restore`
is explicitly requested.

The model predicts five targets. `--execute-steps` selects 1–5 successive targets
per inference; the default is 1. Every target gets fresh measured state, IK and
execution checks. A held or partially tracked move, finger clipping, task-phase
change or Stop discards remaining targets and forces replanning. Manual **Step**
always makes a fresh plan.

Optional return to a recorded arm pose:

```bash
./run_lawm.sh --restore path/to/initial_pose.json --open-hand
```

Set `restore.script`, `restore.config` and `restore.preset` for the external
collection tool. The adapter expects its `ur_joint_pose.py` interface:
`--config`, `--restore`, `--execute`, and `--arm-only`. `--open-hand` requests its
full reset including finger opening; otherwise restore is arm-only. Restore can
be combined with `--trial-run --goal ...`. The collection tool and its SDK
requirements are installed separately.

## Tasks and execution

- **reach:** move the whole-hand points toward the goal, penalising predicted scene motion.
- **grasp:** approach a raised hand goal, then the captured grasp pose.
- **place/push:** not exposed here. Place in the upstream simulation requires
  object labels and contact-stage costs; real perception currently combines
  object and background points into one scene segment. Reach is not a push objective.

Keep the reference model's training-frame rotation and point counts fixed:
1024 FK hand points and 5120 scene points. Training steps represent 0.2 seconds;
real execution is observe–plan–move–settle–observe, without a 5 Hz requirement.

The example real configuration caps planned translation at 60 mm, with a 2 mm
FK/controller margin in the execution check. Wrist, arm and finger increments
have separate limits. Feedback may exceed a hand model bound by up to 1 degree;
commands remain within the joint bounds. Large finger increments are clamped and
followed by replanning. Arm and finger commands do not share a servo clock.

After arm arrival and timed finger motion, fingers get up to one extra second to
settle. A remaining hand-target error within the following-error guard yields a
partial result: hold measured fingers, observe, and replan. Arm non-arrival,
communication faults, excessive feedback errors and user Stop still terminate
motion. These checks do not provide full arm/scene collision detection.

Success means the mean distance between corresponding FK hand points and goal
points is **below 10 mm for five consecutive checks** in the real configuration.
It does not establish physical grasp, push or placement success.

## Logs

`log_dir/<session>/<run>/` contains:

- `run.json`, `status.json`, `records.jsonl`, `events.jsonl`: configuration,
  checkpoint provenance, progress and termination reason.
- `goal.npz`, `start.npz`, `run_start.npz`: the goal and initial observations.
- `input_replan_*.npz`, `prediction_replan_*.npz`, `command_replan_*.npz`,
  `replan_*.npz`, `after_replan_*.npz`: measured inputs, predictions, fitted and
  bounded targets, and post-move observations, including raw depth when available.

`run.json` links the hardware journal in `log_dir`. Include it, the run directory
and the original goal capture when sharing a trial. Real video must be recorded
separately. The simulation backend records video.

Export saved observations to PLY with:

```bash
python -m lawm_deploy.observation_io goals/cup_grasp_01/goal.npz exports/cup_goal
```

## Simulation and tests

```bash
./run_lawm.sh --config configs/sim.yaml --standin
python -m lawm_deploy.sim_demo --config configs/sim.yaml --standin --task reach
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider
```

Tests use simulated or fake devices, not real robot connections. The arm model
uses UR5e kinematics as the nominal model for this UR7e setup. Verify the FK and
camera overlay for your hardware.

`./run_lawm.sh --check` checks packages/checkpoint compatibility without connecting
to devices. `--observe-only` explicitly captures read-only hardware feedback.

Bundled robot assets retain their licenses in `assets/sharpa_wave/` and
`assets/universal_robots_ur5e/`. The world model, weights and vendor SDK are separate
dependencies.
