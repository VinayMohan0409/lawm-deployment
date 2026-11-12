# Local deployment

## Environment and launch

`run_lawm.sh` uses `/data2/xuchuan/conda_envs/lawm` (Python 3.10).
Set `LAWM_CONDA_PREFIX` to relocate the environment. The vendor Sharpa SDK is
installed separately in `/opt/sharpa-wave-sdk/python`.

```bash
bash scripts/setup_environment.sh /absolute/path/to/latent-action-world-model
./run_lawm.sh --check
./run_lawm.sh
```

The model checkout must be at the README-pinned commit `5a2ccb8`.
The setup script installs the dependencies and writes an ignored environment
report under `reports/`. Generated reports, caches and run data are not source files.

The default launch connects in read-only mode and opens the UI at
`http://localhost:8080`. It loads the configured sample pair's goal; the saved start
is a reference only. Every real run/replan captures current state. No reset is sent.
Use `--sample-pair PATH` to choose another pair or `--no-sample-pair` for fresh captures.
Stop other hardware owners before connecting.

Execution modes and the Capture start / Step / Run workflow are in [README.md](README.md).
`--trial-step` permits one supervised move attempt; `--trial-run` permits the
supervised loop using the operator-accepted setup. These modes retain runtime
checks and record the commissioning exception. Ordinary `--execute` requires
completed commissioning flags. A trial does not itself certify those checks.

Read-only device checks:

```bash
./run_lawm.sh --check --devices --duration 3
./run_lawm.sh --check --cameras --duration 3
```

The first reads UR/Sharpa feedback; the second also captures calibrated depth.
Plan only does not connect RTDEControl to query IK. Execution validates controller IK.

## Hardware and calibration conventions

- Configuration: `configs/real.yaml`; local calibration: `configs/calibration.yaml`
  (ignored by Git). Each installation needs its own native-depth extrinsics.
- Feedback from the Sharpa SDK is degrees; commands are radians. The adapter maps
  all 22 joints by name and enforces the MJCF joint limits.
- Robot coordinates are the UR controller base, not ROS `base_link`. The expected
  TCP offset is zero and is checked before motion; the adapter does not set it.
- Nominal zero-offset tool-to-palm mount: translation `[0, 0, 0.0295]` m,
  rotation `[0, 0, -90]` degrees. A wrist-root model instead needs the 0.5 mm
  mount plus its built-in 29 mm wrist-to-palm offset; do not apply 29 mm twice.
- Native depth is 848x480 at 30 Hz, one frame per live observation, Custom preset 0,
  emitter on, auto exposure on and laser power 150. Settings are read back and logged.
  Auto exposure/gain can vary; the cameras are not hardware synchronized.
- `perception.box_*` crops observations; `safety.box_*` bounds the flange, not the
  whole arm. `floor_z_m=-0.065` is based on an estimated table at about -0.075 m
  plus 10 mm clearance. Physical mount, floor, depth and TCP/payload checks remain
  represented by the commissioning flags; fitting residuals do not certify them.

Calibration YAML must declare `frame: depth_optical` and store each camera's
4x4 matrix under `T_base_camera`, keyed by its quoted serial number. Convert
color-camera calibration to native-depth extrinsics with:

```text
T_base_depth = T_base_color @ T_color_depth
```

`T_color_depth` maps native depth-optical coordinates into color-optical coordinates. Obtain it for the correct serial/profile. RealSense SDK `extrinsics.rotation` is column-major (`reshape(3,3,order='F')`). Do not use a raw color matrix or an aligned-depth convention with the native depth stream.

The optional robot-surface ICP calibration command is now read-only: arrange poses using the pendant and end pendant freedrive before capturing. It refuses SUSPECT fits and refuses to overwrite an existing calibration. Its fit score alone does not establish physical calibration accuracy.

## Model and execution limits

The training frame and model contract stay fixed: `spin_deg=315`,
`training_up=[-0.022,-0.546,-0.837]`, 1024 FK hand points and 5120 scene points.
Camera poses and observation archives use base-frame metres; the adapter applies
training rotation once. Keep checkpoint and latent prior matched.

The model's training step is 0.2 seconds. Deployment uses observe–execute–settle–observe;
this does not impose 5 Hz control. Current real limits are 10 mm translation,
8 degrees wrist rotation, 20 degrees finger change and 10 degrees arm excursion.
Translation clipping does not proportionally scale fitted finger motion.

- Finite-value checks reject invalid observations, transforms, fitted states and commands.
- Minimum-jerk finger timing accounts for the 1.875 peak-speed factor and peak acceleration. `moveL` is slowed to approximately match the finger duration. **It is not exact synchronization, and simulated and physical controller dynamics are not identical.** The model owner confirmed slow stop-and-go execution is allowed (2026-11-11); acceptable arm–finger phase error during contact remains unspecified.
- Controller pose/joint safety checks and sampled intermediate IK check the proposed Cartesian path. Hand/table clearance is sampled across independent arm/finger progress; live hand/table and flange bounds are checked as well. These are not a complete arm/environment/self-collision checker.
- Feedback clock/connection, protective/emergency state, measured arm/TCP speed, hand following error, control-loop gaps, move return values and final arrival are checked. Rotational speed is also monitored; waiting longer alone is not a controller rotation-speed guarantee.
- Motion uses asynchronous moveL so the command owner can stop it. The UR watchdog is armed during the move, and the UR control script is stopped/disconnected at each completed action boundary. No control script needs to stay alive during model inference or camera acquisition. A process failure does not provide a hardware watchdog for the Sharpa hand; its internal controller may retain its last bounded target.
- Stop is checked before motion, after path validation and during execution. The UI requests cancellation and lets the command owner stop; shutdown avoids concurrent RTDE calls. The pendant e-stop remains the emergency mechanism.
- A stationary robot is required during sequential depth capture. The current real observation is refreshed every replan, including held steps. After planning, changed robot joints invalidate the target. External object movement during slow inference remains an experimental limitation.
- `success` still means five hand-distance checks below 50 mm, preserving the friend's metric. Logs explicitly call this hand-goal attainment; `physical_grasp_success` remains unset. This is not evidence that an object was grasped or lifted.

The supplied checkpoint differs from the owner's simulation-tested artifact.
Partial real depth observations, arm/finger coordination during contact and physical
grasp outcomes require evaluation independently of software tests and hand distance.

## Run files

Logs are under `/data2/xuchuan/lawm_runs/<session>/<run>/`:

- `run.json`: config, checkpoint hash, seed, joint names and linked hardware journal;
- `goal.npz`, `start.npz`, `run_start.npz`: fixed goal, chosen reference start and actual run start;
- `input_replan_*.npz`: observed point cloud, joints, poses, native depth/scale,
  intrinsics/extrinsics, timestamps and provenance, saved before inference;
- `prediction_replan_*.npz`: full-precision predicted point displacement and hand target,
  saved before fitting and IK;
- `command_replan_*.npz`: fitted hand state and flange/finger command, saved before IK;
- `replan_*.npz`: observation, prediction, command, IK result/availability and hold reason,
  saved before movement;
- `after_replan_*.npz`: the observation after each completed move (or unchanged observation on hold),
  including the final step;
- `records.jsonl`: executed/held step metrics; `plans.jsonl`: Plan only metrics;
- `events.jsonl`: attempt start, plan ready, completion or error stage;
- `status.json`: current/final status, including individual Step, Run, Stop and clean shutdown.

Plan only uses analogous `*_plan_only_*.npz` archives and does not advance execution counters.
Exceptions retain the input and all outputs produced before the failing stage; no missing
post-action data is fabricated. Disk failure/power loss can still prevent records.
Movement feedback is in the sibling `hardware_*.jsonl` linked by `run.json`; include it
when sharing a trial. RealCell captures depth, not RGB video: record a phone video separately.
`physical_grasp_success` remains unset for manual outcome annotation.


## Observation export

New observation archives preserve:

- `coords`, `segment`: exact base-frame model input and labels, if produced by Session;
- `q_arm`, `q_hand`, intrinsics/extrinsics and configuration snapshot;
- `depth_median_m_i`: full-precision median depth for Session archives (`depth_i` is already metre-valued in preflight archives);
- `sensor_depth_z16_i`, `sensor_depth_scale_m_i`: actual native sensor frames before the median, when captured from real cameras;
- camera timestamps/serials, checkpoint SHA-256 for Session captures, and training/execution timing conventions.

The legacy Session `depth_i` keys remain rounded uint16 millimetres for compatibility. They were processed medians, not original sensor depth. Older recordings cannot recover original frames retroactively.

Export each saved sample offline (no device connections, model import or upload):

```bash
/data2/xuchuan/conda_envs/lawm/bin/python -m lawm_deploy.observation_io /path/to/start.npz /path/to/new_start_export
/data2/xuchuan/conda_envs/lawm/bin/python -m lawm_deploy.observation_io /path/to/goal.npz /path/to/new_goal_export
```

A preflight `depth_001.npz` is also accepted as input. Each output contains the original archive, per-camera optical/base-frame PLY files, a manifest, and `model_input_base.ply` when actual model input is present. Camera PLY files are backprojected from median depth, **before cropping, robot masking or voxel sampling**; individual raw Z16 frames are retained in `observation.npz`. The training rotation has not been applied to these exports: use the existing adapter exactly once. The exporter refuses to overwrite an existing output directory.

## Capture a new goal and start

Launch `./run_lawm.sh --no-sample-pair`, arrange each scene, and use **Capture goal**
then **Capture start**. Session archives are written under the configured log directory;
use the observation exporter above to inspect or share their point clouds.
The default saved static pair is still supported by `saved_pair.py`.
