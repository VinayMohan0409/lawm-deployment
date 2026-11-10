# lawm-deployment

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

**1. Connect the hand's SDK (the only code to write).**
In `lawm_deploy/cell.py`, class `SharpaHand`, fill in three short methods from the SDK's Python samples (`/opt/sharpa-wave-sdk/python/`):
- `__init__`: connect to the hand;
- `read()`: return the 22 joint angles in radians;
- `write(q)`: send 22 joint-angle targets.

Use the joint order of `assets/sharpa_wave/right_hand.xml` (thumb first, then index, middle, ring, pinky), matched to the SDK by joint name.

**2. Fill in `configs/real.yaml`.** Every line marked `CHANGE`:
- the robot's IP address;
- how the hand is mounted on the flange;
- the camera serials (keep them in quotes), each with a rough tape-measured position and a point it looks at (metres, in the robot base frame);
- the box the flange may move in;
- the table height.

**3. Camera extrinsics** (each camera's pose in the robot base frame).
- If you already have them, write them into `configs/calibration.yaml`: one 4×4 matrix per camera serial, for the depth optical frame.
- If not, run:
  ```bash
  python -m lawm_deploy.calibrate --config configs/real.yaml
  ```
  It switches the arm to freedrive. Move it by hand to 5 different poses, with the hand in view of both cameras and 15 cm or more above the table, pressing Enter after each. It prints, per camera, the share of the robot's points it could match and **OK** (over 70%) or **SUSPECT**. On SUSPECT, re-measure the rough pose (it needs to be within about 3 cm and 5°) and run it again.

**4. Check everything without the model.**
```bash
python -m lawm_deploy --config configs/real.yaml --standin
```
Open `http://localhost:8080`, then:
- **Overlay check:** tick **Show raw camera points**. The red camera points must sit on the blue hand (drawn from the joint angles) in a few different arm poses. If they're offset, fix the extrinsics or the mount before going on.
- **Workspace box:** in **1. Workspace**, set the box to the table area the hand works in and press **Save box to config**. Keep it the same for every trial.

## Each trial
Start the page with `python -m lawm_deploy --config configs/real.yaml` and open `http://localhost:8080`.

1. **Goal:** put the robot and objects where they should end up, then press **Capture goal**. In **Arrange the robot**:
   - **Teleoperation mode:** hands the robot to your teleop system. The page keeps showing it live.
   - **Freedrive and the finger slider:** to place it by hand instead.
2. **Start:** arrange the start the same way, then press **Capture start**. Untick **Teleoperation mode** (and stop your teleop system) before the next step.
3. **Run:** pick the task and press **Run**. The status line shows the replan number, the distance to the goal hand, and why the robot held if it did.

A run ends by itself:
- with **success** when the hand stays within 5 cm of the goal hand for 5 replans in a row;
- when it stalls, after 5 holds in a row, or after 80 replans;
- or when you press **Stop**, which takes effect after the current move.

**For emergencies use the pendant's e-stop.**

**On the first day only:** with a start captured, press **4. Check model frame**. If another spin gives clearly more hand motion and clearly less scene motion than the one marked "now", put it in `model.spin_deg` in `configs/real.yaml` and restart.

**Please send back** the whole `runs/` folder and a phone video of each run.

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
- the UR has no IK solution, or an arm joint would jump more than 30°;
- the wrist would turn more than 30°, or a finger more than 60°;
- the flange would leave the box;
- the hand would go below `floor_z_m`.

Each move is at most 60 mm, in a straight line, with the fingers moving smoothly over the same time.

## Not verified yet
- Nothing has run on the real hardware.
- **Teleoperation mode** assumes the Sharpa SDK can read the joint angles while your teleop system controls the hand. If it can't, say so.
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
   │  executor: flange target (step ≤ 60 mm, rotation smoothed); checks: fit, IK, jumps, box, floor ──> hold if any fails
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
│   ├── cell.py                  real hardware: UR (ur_rtde), Sharpa (SharpaHand: fill in), D435i (pyrealsense2)
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
