# lawm-deployment

Runs the latent action world model (LAWM) on a real **UR7e arm with a right Sharpa Wave hand and two RealSense D435i depth cameras**.

You capture a goal and a start, choose **reach** or **grasp**, and press **Run**. On every replan the robot:

1. reads its joint angles;
2. reads both depth cameras;
3. builds a point cloud;
4. lets the model pick the best next hand motion;
5. moves there;
6. looks again.

It executes only the first planned step each time, which is called a receding horizon. The robot stops by itself when it reaches the goal, or when you press Stop.

## What the model sees

- **Hand:** 1,024 points on the Sharpa mesh, placed from the measured joint angles. The hand is never taken from the cameras.
- **Scene:** whatever the two cameras see, in the robot's base frame, with these steps applied:
  - optionally cropped to a workspace box you set;
  - the arm and hand removed, using their measured joint angles;
  - one point kept per 1.5 cm voxel;
  - brought to exactly 5,120 points: a random subsample if there are more, repeats if there are fewer.

The scene has no object or table labels, so only reach and grasp are supported.

Nothing else is read: no object poses and no simulator state.

## Hardware hookup

| Device | Connection |
|---|---|
| Linux PC with an NVIDIA GPU (11 GB or more) | runs everything |
| UR7e control box | Ethernet. On the pendant: enable **Remote Control** and **RTDE** (Settings → Security → Services), then switch to Remote mode. |
| Sharpa Wave | as the Sharpa Pilot / SDK documentation says |
| 2× D435i | USB 3 cable each, ideally on different USB controllers. `rs-enumerate-devices` lists their serials. |

Keep the pendant's emergency stop within reach. **Stop** in the page stops only after the current move.

## Install

```bash
conda create -n lawm python=3.10 && conda activate lawm
# 1. The model: torch 2.5 with CUDA, spconv, torch_scatter, timm, addict, einops
#    (exact versions: latent-action-world-model/requirements.md, section 2; on x86 use pip install spconv-cu12x)
git clone https://github.com/VinayMohan0409/latent-action-world-model && cd latent-action-world-model
git checkout <COMMIT> && pip install -e . && cd ..
# 2. This repo
git clone https://github.com/VinayMohan0409/lawm-deployment && cd lawm-deployment
pip install -e ".[real,dev]"
```

Copy the model files into `checkpoint/`: `best.pt`, `run_manifest.json` and `prior.npz`.

## Before the first run

### 1. Sharpa hand
Wire `SharpaHand` in `lawm_deploy/cell.py`. It has three methods, each a few lines from the SDK samples in `/opt/sharpa-wave-sdk/python/`:
- `__init__`: connect to the hand;
- `read()`: return the 22 joint angles in radians;
- `write(q)`: send 22 absolute joint-angle targets.

Joint order is the order in `assets/sharpa_wave/right_hand.xml` (thumb CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP, then index, middle, ring and pinky). Map the SDK's order to this order by joint name.

### 2. `configs/real.yaml`
Set every line marked `CHANGE`:
- `robot_ip`: the UR control box's address.
- `mount`: the pose of the hand's base on the UR flange, from the adapter drawing.
- Camera serials, each with a tape-measured `rough_eye` (camera position) and `rough_target` (a point it looks at), in metres in the **robot base frame**. That is the frame the pendant shows TCP poses in, with z up.
- `safety.box_lo` / `box_hi`: the box the flange may be commanded in.
- `safety.floor_z_m`: the table height plus a few mm.

The rough camera pose needs to be within about **3 cm and 5°** for calibration to work.

### 3. Camera extrinsics
Each camera's pose in the robot base frame is needed so the camera points and the hand mesh (from the joint angles) are in the same frame.

**If the lab already has the extrinsics**, write them into `configs/calibration.yaml`: one 4×4 matrix per camera serial, giving the pose of the camera's **depth optical frame** (x right, y down, z forward) in the robot base frame, in metres:
```yaml
"123456789012": [[r11, r12, r13, x], [r21, r22, r23, y], [r31, r32, r33, z], [0, 0, 0, 1]]
```

**If not**, run the calibration tool:
```bash
python -m lawm_deploy.calibrate --config configs/real.yaml --poses 5
```
1. The arm switches to freedrive.
2. Move it by hand to 5 clearly different poses, with the forearm and hand in view of both cameras and the hand at least 15 cm above the table. Press Enter after each.
3. The tool fits each camera's pose by matching the moving robot's depth points to the robot's own surface (at its measured joint angles), so no calibration board is needed.
4. It writes `configs/calibration.yaml` and prints a verdict per camera.
   - **OK:** more than 70% of the robot points matched and the median distance is under 3 mm.
   - Otherwise it prints **SUSPECT**: re-measure the rough pose and run it again.

Calibrate again whenever a camera is moved. Either way, do the overlay check in step 4.

### 4. Check without the model
```bash
python -m lawm_deploy --config configs/real.yaml --standin
```
Open `http://localhost:8080`. With `--standin` a dummy model is used, so the full pipeline can be checked without the GPU.

- **Overlay check:** tick **Show raw camera points**. The red camera points on the hand must sit on the blue mesh hand (from the joint angles), and the red points on the arm on the arm. Move the arm to a few poses and check each. If they are offset, the extrinsics or `mount` are wrong; fix them before any trial.
- With the switch off, no camera points should remain on the arm or hand.
- **Box:** set the workspace box with the sliders. "Crop to box" off shows everything the cameras see. Save it with **Save box to config**.

**Set the box once, as the table area the hand can work in, and do not change it between trials.**

## A trial

Start the page with `python -m lawm_deploy --config configs/real.yaml` (no `--standin`), then:

1. **Arrange the goal**, either way:
   - **Teleoperation:** in **Arrange the robot**, tick **Teleoperation mode**. This program then releases the arm (its RTDE control script stops) and stops commanding the hand, but keeps reading the joint angles and cameras, so the live view follows the robot. Teleoperate it into place.
   - **By hand:** tick **Freedrive** and move the arm, untick it so the arm holds, then set the fingers with the closure slider and **Set fingers**.

   Put the objects where they should be, then press **Capture goal**.
2. **Arrange the start** the same way, then press **Capture start**. If you teleoperated, untick **Teleoperation mode** before Run so this program takes control again (stop your teleop system first).
3. **Choose the task** (`reach` or `grasp`) and press **Run**.

The status line shows each replan:
- the distance of the hand to the goal hand;
- whether the robot moved or held, and why it held;
- the hand-fit error.

A run ends by itself with **success** when the hand stays within 5 cm (50 mm) of the goal hand for 5 replans in a row. It also ends when it stalls (reach), after 5 held replans in a row, or after 80 replans. You can also press **Stop**.

**Check the model frame on the first day.** The model reads raw coordinates, so the cloud is rotated into its training data's frame before the model (`model.training_up`, `model.spin_deg`). The spin (315°) was chosen in simulation, where the table lies in front of the robot along +x. Your layout may differ. With a start captured, open **4. Check model frame** and press the button. If another spin moves the hand clearly more and the scene clearly less than the current one, set it as `model.spin_deg` in `configs/real.yaml` and restart.

Optional checks before Run:
- **Plan only:** plans once without moving. The orange points show where the hand would go.
- **Step:** plans and makes a single move.

Every trial is logged in `runs/<time>/`:
- `goal.npz` and `start.npz`;
- one `replan_###.npz` per replan: depth images, joint angles, the cloud, the 5 predicted steps and the command sent;
- `records.jsonl`, one line per replan;
- `run.json`, with the task, the seed and the full config used.

Please send the whole `runs/` folder back.

## What to tune, and where

| Where | Setting | When |
|---|---|---|
| UI → Save | workspace box, crop on or off | once per setup |
| `cell.motion` | `speed`, `accel`, `rot_speed`, `finger_speed`, `settle_s` | start slow; raise once moves look right |
| `cell.cameras` | `preset`, `n_frames` | noisy depth |
| `perception` | `robot_margin_m` | arm or hand points left in the cloud: raise it; object points next to the fingers lost: lower it |
| `safety` | `floor_z_m`, `box_*`, `max_*` | the robot holds too often (the reason is shown), or is too free |
| `planner.gates.hover.margin_mm` | 200 mm | grasp first aims this far above the goal hand; lower it if the workspace is short |
| `loop` | `success_mm`, `success_replans`, `max_replans` | when a run counts as a success or should end |

The planner's search and cost settings are copied from lawm's `configs/stage3/dexwm.json`. Do not change them without asking.

## Safety

The robot holds instead of moving, and the page shows the reason, when:
- the hand fit is poor;
- the UR has no IK solution, or its solution jumps an arm joint by more than 30°;
- one replan would turn the wrist more than 30° or move a finger joint more than 60°;
- the flange target is outside the safety box;
- a hand point would go below `floor_z_m`.

Each move is limited to 60 mm. The arm moves in a straight line (`moveL`) while the fingers follow a smooth (minimum-jerk) path over the same time. Both the UR and the Sharpa hand keep their own controllers; this code only sends position targets.

## Test in simulation (no hardware)

`configs/sim.yaml` is a MuJoCo copy of the cell: the same robot, a table, two boxes and two simulated depth cameras. Each run is also recorded to `video.mp4` in its run folder, with the goal hand drawn as green dots.

```bash
MUJOCO_GL=osmesa python -m lawm_deploy --config configs/sim.yaml              # the page with the real model (GPU)
MUJOCO_GL=osmesa python -m lawm_deploy.sim_demo --task reach                   # an example goal and start, one Run, no page
MUJOCO_GL=osmesa pytest tests
```

In simulation the page has an extra **Simulation** panel. On the real robot it is absent: you arrange the robot and objects by hand, and film the run yourself.

- **Place the robot:** sliders move the hand from home and set the finger closure. Tick **Arrange directly** to put the robot there at once, as a person would arrange the real cell. **Example reach / grasp goal** fills in the sliders. **Reset scene** puts the robot home with the hand open and the objects back.
- **Video:** shows the latest frame.

For a trial in simulation:
1. Load an example (or set your own) and press **Move**, then **Capture goal**.
2. Press **Reset scene**, then **Capture start**.
3. Press **Run**.

## Not verified yet
- Teleoperation mode assumes the Sharpa SDK lets this program **read** the hand's joint angles while your teleop system commands it. If the SDK allows only one client, the hand angles must come from the teleop system instead: tell Vinay.
- Nothing here has run on the real hardware.
- The Sharpa SDK calls are still a stub (see step 1).
- The UR7e is modelled with the UR5e kinematics from MuJoCo Menagerie. The rotations match UR's base frame exactly and positions are within 2 mm of the UR5e DH table. That the UR7e shares those kinematics is an assumption: the calibration residual and the hand overlay in step 4 will show if it does not.
- The real checkpoint has not yet been run through this pipeline, even in simulation.

## Third-party assets
- `assets/sharpa_wave` (Apache-2.0) and `assets/universal_robots_ur5e` (BSD-3-Clause) are from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), commit `98a70c5`. Their licences are in their folders.
- Voxel sampling follows [PointWorld](https://github.com/NVlabs/PointWorld) (`grid_sample_transform`, 1.5 cm).
