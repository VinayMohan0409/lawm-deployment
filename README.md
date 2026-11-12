# lawm-deployment

**Local UR7e/Sharpa integration:** see [DEPLOYMENT_LOCAL.md](DEPLOYMENT_LOCAL.md) for the independent conda environment, SDK adapter and checks. Real hardware is read-only unless `--execute`, `--trial-step`, or `--trial-run` is supplied.

**Continuous supervised trial (operator-authorized existing setup):**
```bash
CUDA_VISIBLE_DEVICES=1 ./run_lawm.sh --trial-run
```
Open `http://localhost:8080`, keep **reach**, click **Capture start**, then **Run**.
The saved goal is loaded automatically; do not Capture goal again. Each cycle
observes the current scene and joints, replans, fits the hand, checks IK and executes
one move. Run repeats automatically until a stop condition (at most 80 replans
with the current config); **Stop** ends it. Every replan is logged. This mode uses
the same accepted existing setup, motion limits and runtime checks as `--trial-step`,
without the one-attempt limit. Commissioning flags remain unchanged.

**First supervised real move (operator-authorized existing setup):**
```bash
CUDA_VISIBLE_DEVICES=1 ./run_lawm.sh --trial-step
```
Open `http://localhost:8080`, keep **reach**, click **Capture start**, optionally
**Plan only**, then **Step (one move)**. The saved goal is loaded automatically;
do not Capture goal again. Step captures fresh state and replans, so its command
can differ from the preceding preview. It performs IK/path checks before moving.
Only one move attempt is allowed per launch; **Run** and manual jog are disabled.
If it holds, nothing moved; inspect the stated reason. Stop the process after the
trial and review its logs before another launch. This mode records acceptance of
the existing setup (including existing payload) without claiming outstanding
physical commissioning checks passed. It preserves runtime safety checks and
caps of 10 mm translation, 8 degrees wrist, 20 degrees fingers, 10 degrees arm
joint excursion and the existing slow motion settings. No automatic reset occurs.

**Current local trial (2026-11-12):** `configs/real.yaml` loads
`/data2/xuchuan/lawm_samples/grasp_20261111_171635` and selects **reach**.
The saved goal is fixed; the saved start is a reference only. Every real run/replan
captures current joints and depth. There is no automatic move back to the saved start.
Run `./run_lawm.sh` for live observation and Plan only; `--no-sample-pair` selects
fresh goal/start capture instead. Stop other hardware owners before connecting.
Ordinary `--execute` requires completed commissioning; `--trial-run` records the
operator-authorized experimental exception for the existing setup.
Environment setup, calibration conventions, log contents and execution limits
are described in [DEPLOYMENT_LOCAL.md](DEPLOYMENT_LOCAL.md).

Runs a world model (LAWM) on a **UR7e + right Sharpa Wave hand + two RealSense D435i cameras**.

You set a goal pose and a start pose, pick **reach** or **grasp**, and press **Run**. On every replan the robot:

1. reads its joint angles and both depth cameras;
2. lets the model choose the next hand motion;
3. makes that one move;
4. looks again.

Each run is logged so it can be analysed afterwards.

## What you need
- A Linux PC with an NVIDIA GPU (11 GB or more).
- **UR7e:** connected by Ethernet to the PC, on the same subnet. On the teach pendant:
  - enable **Remote Control** (Settings → System → Remote Control), then switch the Local/Remote toggle at the top right to **Remote**; this lets the PC command the robot;
  - make sure the **RTDE** service is not disabled (Settings → Security → Services); `ur_rtde` uses it to read and move the arm.

  Menu names vary slightly between PolyScope versions.
- **Sharpa Wave:** connected as its own documentation says, with **Sharpa Pilot** installed. Pilot installs the hand's Python SDK in `/opt/sharpa-wave-sdk/`.
- **2× D435i:** a USB 3 cable each. After installing, list their serial numbers with
  `python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"`.
- Read access to both GitHub repos (`latent-action-world-model`, `lawm-deployment`).

## Install (once)
```bash
conda create -n lawm python=3.10 && conda activate lawm
# torch 2.5 + CUDA, spconv, torch_scatter, timm, addict, einops: see latent-action-world-model/requirements.md, section 2 (on x86: pip install spconv-cu12x)
git clone https://github.com/VinayMohan0409/latent-action-world-model
cd latent-action-world-model && git checkout 5a2ccb8 && pip install -e . && cd ..
git clone https://github.com/VinayMohan0409/lawm-deployment
cd lawm-deployment && pip install -e ".[real,dev]"
```
The model files are already in `checkpoint/`.

## Set up (once)

**1. Sharpa SDK adapter (implemented locally).**
`SharpaHand` discovers the configured right hand, maps joints by name, converts degree feedback to radians, and sends radians. Connection is read-only; output is enabled only during an explicitly enabled move. No reset/open command is issued on startup.

**2. Fill in `configs/real.yaml`.** Every line marked `CHANGE`:
- the robot's IP address;
- how the hand is mounted on the flange;
- the camera serials (keep them in quotes), each with a rough tape-measured position and a point it looks at (metres, in the robot base frame);
- the box the flange may move in;
- the table height.

**3. Camera extrinsics** (each camera's pose in the robot base frame).
- If you already have them, write `configs/calibration.yaml`: declare `frame: depth_optical` and put matrices under `T_base_camera`, keyed by serial. Existing color-camera calibration must be converted with the camera-internal depth-to-color transform.
- If not, run:
  ```bash
  python -m lawm_deploy.calibrate --config configs/real.yaml
  ```
  This command is read-only. Use the pendant to arrange 5 different poses, end pendant freedrive and wait until stationary before each capture, with the hand visible to both cameras and clear of the table. It prints, per camera, the share of points matched and **OK** (over 70%) or **SUSPECT**. SUSPECT results are not written; existing calibration files are not overwritten. On SUSPECT, re-measure the rough pose (it needs to be within about 3 cm and 5°) and run it again.

**4. Check the hardware and geometry.**
For device-only feedback (no depth calibration or LAWM required), use `./run_lawm.sh --check --devices`. The UI below still needs the LAWM library, even with a stand-in model. It does not enable motion:
```bash
python -m lawm_deploy --config configs/real.yaml --standin
```
Open `http://localhost:8080`, then:
- **Overlay check:** tick **Show raw camera points**. The red camera points must sit on the blue hand (drawn from the joint angles) in a few different arm poses. If they're offset, fix the extrinsics or the mount before going on.
- **Workspace box:** in **1. Workspace**, set the box to the table area the hand works in and press **Save box to config**. Keep it the same for every trial.

## Each trial
After independently verifying mount geometry, depth calibration, workspace/table and TCP/payload, record those checks under `commissioning` in `configs/real.yaml`. Start with `python -m lawm_deploy --config configs/real.yaml --execute` and open `http://localhost:8080`. Without `--execute`, only observation and Plan only are available; controller IK is not queried.

1. **Goal:** put the robot and objects where they should end up, then press **Capture goal**. In **Arrange the robot**:
   - **Teleoperation mode:** releases the UR controller and disconnects the Sharpa SDK. Stop teleoperation and take control back before capturing; this integration does not assume concurrent Sharpa SDK readers.
   - **Freedrive and the finger slider:** to place it by hand instead.
2. **Start:** arrange the start the same way, then press **Capture start**. Untick **Teleoperation mode** (and stop your teleop system) before the next step.
3. **Run:** pick the task and press **Run**. The status line shows the replan number, the distance to the goal hand, and why the robot held if it did.

A run ends by itself:
- with **success** when the hand stays within 5 cm of the goal hand for 5 replans in a row (hand-goal attainment only, not proof of a successful grasp);
- when it stalls, after 5 holds in a row, or after 80 replans;
- or when you press **Stop**. The execution loop checks this during motion and before sending a planned move; this is not a physical e-stop.

**For emergencies use the pendant's e-stop.**

**Model frame:** keep `training_up=[-0.022,-0.546,-0.837]`, `spin_deg=315`, 1024 hand points and 5120 scene points fixed for this checkpoint.

**Please send back** the session folder, its linked hardware journal and a phone video of each run.

For this local installation, logs are under `/data2/xuchuan/lawm_runs`.
Inputs, predictions and commands are saved before execution; post-action observations,
Plan only metrics, failure stages and Step/Run status are saved as well. Include the linked
`hardware_*.jsonl` at the log root. Real RGB video is not recorded automatically; use a
separate video recording. File details are in [DEPLOYMENT_LOCAL.md](DEPLOYMENT_LOCAL.md).

## If something is off

| Problem | Setting in `configs/real.yaml` |
|---|---|
| Moves too fast or slow | `cell.motion` (start slow) |
| Arm or hand points left in the cloud | raise `perception.robot_margin_m` |
| Robot keeps holding | the status line says why; adjust the matching `safety` setting |
| Grasp approach too high for your space | lower `planner.gates.hover.margin_mm` (200) |

Leave the other planner settings as they are.

The robot holds instead of moving when:
- the hand fit is poor;
- the UR has no IK solution, or the arm path exceeds the configured joint excursion;
- wrist or finger motion exceeds the configured limits;
- the flange would leave the box;
- the hand would go below `floor_z_m`.

The local real configuration caps each move at 10 mm, wrist rotation at 8 degrees, and finger changes at 20 degrees (rotation/finger limits increased with user authorization on 2026-11-12 after reviewing live plans); simulation retains the original limits. `moveL` and minimum-jerk fingers use approximately matched timing, not a shared servo clock. Arrival, feedback and runtime limits are checked. See DEPLOYMENT_LOCAL.md for the remaining execution limitations.

## Not verified yet
- A supervised real trial has been performed; task success and physical commissioning remain separate checks.
- Teleoperation handover now disconnects the hand rather than assuming concurrent SDK readers. Hardware validation is pending.
- The UR7e is modelled with the UR5e's kinematics. The overlay check in setup step 4 will show if that's wrong.

## Simulation (no hardware)
`configs/sim.yaml` is a MuJoCo copy of the cell.
```bash
MUJOCO_GL=osmesa python -m lawm_deploy --config configs/sim.yaml   # same page, with a Simulation panel
MUJOCO_GL=osmesa python -m lawm_deploy.sim_demo --task grasp        # example goal and start, one run
MUJOCO_GL=osmesa pytest tests
```
For a trial in simulation:
1. In **Simulation**, press **Example reach/grasp goal**, then **Move**, then **Capture goal**.
2. Press **Reset scene**, then **Capture start**.
3. Press **Run**.

Each run saves `video.mp4` in its run folder.

## How the code fits together

One replan, start to end:
```
cell.state() ──> joint angles, flange pose ──> hand.py: 1024 hand points (FK on the Sharpa mesh)
cell.views() ──> 2 depth images ──> perception.py: base-frame points, box crop, robot removed, 1.5 cm voxels, 5120 points
                                         │
                hand + scene cloud ──────┴──> model.py: rotate into the training frame ──> lawm planner (CEM over latent actions,
                                              cost against the goal hand) ──> predicted step 1 of the hand points
                                                                                   │
safety.py <── hand.py: fit palm pose + 22 angles to those points <─────────────────┘
   │  executor: flange target (configured step cap, rotation smoothed); checks: fit, IK, jumps, box, floor ──> hold if any fails
   └──> cell.move(): moveL for the arm + smooth finger targets ──> observe again
```

```
lawm-deployment/
├── README.md
├── pyproject.toml               dependencies (pip install -e ".[real,dev]")
├── configs/
│   ├── real.yaml                every setting for the real cell (CHANGE lines must be set)
│   ├── sim.yaml                 the same for the MuJoCo copy of the cell
│   └── calibration.yaml         camera extrinsics (yours, or written by calibrate.py; not in git)
├── checkpoint/                  best.pt (model), run_manifest.json (its settings), prior.npz (latent prior)
├── assets/                      Sharpa Wave right hand and UR5e models (MuJoCo Menagerie)
├── lawm_deploy/
│   ├── __main__.py              python -m lawm_deploy: connects the cell, loads the model, serves the page
│   ├── ui.py                    the browser page (viser): workspace box, captures, Plan/Step/Run/Stop, teleop, sim panel
│   ├── session.py               one experiment: capture goal/start, the replan loop, stop rules, logs in runs/
│   ├── perception.py            depth images -> hand + scene cloud
│   ├── hand.py                  Sharpa hand points from joint angles, and the fit of angles to points
│   ├── robot_model.py           UR + Sharpa kinematics (flange, palm, surface points to mask the robot)
│   ├── model.py                 loads the checkpoint + prior, training-frame rotation, stand-in model for tests
│   ├── safety.py                predicted hand -> flange/finger command, and the checks that hold the robot
│   ├── cell.py                  real hardware: UR (ur_rtde), Sharpa (SharpaHand: implemented locally), D435i (pyrealsense2)
│   ├── sim_cell.py              MuJoCo copy of the cell with simulated depth cameras and video recording
│   ├── calibrate.py             camera extrinsics by matching depth points to the robot's own surface
│   ├── config.py                reads the YAML configs
│   └── sim_demo.py              one example run in simulation without the page
├── tests/                       pytest (CPU): kinematics, perception, safety, calibration, loop, real cell on fakes
└── runs/                        logs and videos of every run (created when you run; send these back)
```
The planner itself (search, costs, grasp hover gate, hand fit) lives in `latent-action-world-model` under `src/lawm/sim/dexwm/`; this repo only calls it.

## Credits
- Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Sharpa Wave: Apache-2.0; UR5e: BSD-3-Clause; licences in `assets/`).
- Voxel sampling: as in [PointWorld](https://github.com/NVlabs/PointWorld).
