"""The operator's browser page: live cloud and workspace box, goal and start capture, and Plan only / Step / Run / Stop."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import viser

from lawm.sim.costs import SEG_HAND

GRAY, BLUE, GREEN, ORANGE, YELLOW, RED, PURPLE = (150, 150, 150), (30, 110, 220), (40, 170, 70), (240, 140, 20), (230, 200, 0), (220, 40, 40), (150, 60, 200)
LEGEND = ("**Colours:** blue = hand now (joint angles + mesh), green = goal hand, purple = start hand, orange = hand after the planned move, "
          "gray = camera scene (model input), red = raw camera points (overlay check), yellow = workspace box.")
FLEX = ("FE", "PIP", "DIP", "IP")  # Sharpa joints that curl a finger (the sim jog closes these)
SIM_PRESETS = {  # flange offset from home (m), finger closure, arrange directly; the grasp wraps the red cube of configs/sim.yaml
    "reach": ((0.06, -0.06, -0.12), 0.3, False),
    "grasp": ((0.013, 0.033, -0.342), 0.5, True),
}


def box_edges(lo, hi):
    """The 12 edges (12, 2, 3) of an axis-aligned box."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return np.stack([c[[a, b]] for a, b in pairs])


def save_box(path, use_box, lo, hi):
    """Write use_box, box_lo and box_hi of the perception section into the YAML file, keeping its comments."""
    text = Path(path).read_text()
    start = text.index("\nperception:")
    end = re.search(r"\n\S", text[start + 1:]).start() + start + 1
    section = text[start:end]
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"  # noqa: E731
    section = re.sub(r"(\n  use_box: )\S+", rf"\g<1>{str(use_box).lower()}", section)
    section = re.sub(r"(\n  box_lo: )\[[^\]]*\]", rf"\g<1>{fmt(lo)}", section)
    section = re.sub(r"(\n  box_hi: )\[[^\]]*\]", rf"\g<1>{fmt(hi)}", section)
    Path(path).write_text(text[:start] + section + text[end:])


class App:
    """One page per session; every use of the cell goes through one lock, runs happen in a background thread."""

    def __init__(self, session, config_path, port, sim_cell=None):
        """Build the page; sim_cell adds the sim-only jog controls."""
        self.session, self.config_path, self.sim = session, config_path, sim_cell
        self.lock, self.stop_flag, self.task_ready, self.reverting = threading.Lock(), threading.Event(), None, False
        self.worker = None
        self.closing = False
        self.server = viser.ViserServer(port=port)
        self.scene, gui = self.server.scene, self.server.gui
        p = session.perception.s
        with gui.add_folder("1. Workspace"):
            self.use_box = gui.add_checkbox("Crop to box", p.use_box, hint="Off: everything the cameras see between near and far")
            self.xs = gui.add_multi_slider("x (m)", -1.5, 1.5, 0.01, (p.box_lo[0], p.box_hi[0]))
            self.ys = gui.add_multi_slider("y (m)", -1.5, 1.5, 0.01, (p.box_lo[1], p.box_hi[1]))
            self.zs = gui.add_multi_slider("z (m)", -0.5, 1.0, 0.01, (p.box_lo[2], p.box_hi[2]))
            self.live = gui.add_checkbox("Live view", True)
            self.show_raw = gui.add_checkbox("Show raw camera points", False, hint="Overlay check: red camera points on the arm and hand should sit on the blue mesh hand")
            gui.add_button("Save box to config").on_click(lambda _: self.save())
            self.counts = gui.add_markdown("")
        if getattr(session.cell, "read_only", False):
            gui.add_markdown("**Read-only:** Plan only is available. No UR control script or hand targets. Controller IK is not checked. Restart with --execute only after site commissioning.")
        trial_step = getattr(session.cell, 'trial_step', False)
        if trial_step:
            gui.add_markdown('**Single-step trial:** existing setup accepted for this test; physical checks remain unverified. Step permits one move attempt. Run and manual jog are disabled.')
        trial_run = getattr(session.cell, 'trial_run', False)
        if trial_run:
            gui.add_markdown('**Closed-loop trial:** Run observes, plans and executes one move repeatedly. Stop ends the run. Existing setup is accepted for this test; physical checks remain unverified.')
        if hasattr(session.cell, "freedrive") and not getattr(session.cell, "read_only", False) and not (trial_step or trial_run):
            with gui.add_folder("Arrange the robot"):
                gui.add_markdown("**Teleoperation:** release disconnects the UR controller and Sharpa SDK. Stop teleop and take control back before captures; simultaneous Sharpa readers are not assumed.  \n"
                                 "**By hand:** tick Freedrive to move the arm, untick to hold it, and set the fingers with the slider.")
                self.teleop = gui.add_checkbox("Teleoperation mode (release the robot)", False)
                self.teleop.on_update(lambda _: self.toggle(self.teleop, self.set_teleop))
                self.free = gui.add_checkbox("Freedrive (arm)", False)
                self.free.on_update(lambda _: self.toggle(self.free, self.set_freedrive))
                self.real_closure = gui.add_slider("finger closure", 0.0, 1.0, 0.05, 0.0)
                gui.add_button("Set fingers").on_click(lambda _: self.background(self.set_fingers))
        with gui.add_folder("2. Goal, then start"):
            gui.add_button("Capture goal").on_click(lambda _: self.background(lambda: self.capture("goal")))
            gui.add_button("Capture start").on_click(lambda _: self.background(lambda: self.capture("start")))
            self.show_goal = gui.add_checkbox("Show goal hand", True)
            self.show_goal_scene = gui.add_checkbox("Show goal scene", False)
            self.show_start = gui.add_checkbox("Show start hand", True)
            self.captured = gui.add_markdown(
                "Saved goal/start loaded. Start is a reference; every run observes the robot and scene again. No automatic return to start."
                if session.goal is not None and session.start is not None else "Goal: none. Start: none.")
        with gui.add_folder("3. Run"):
            self.task = gui.add_dropdown("Task", ("reach", "grasp"), initial_value=session.cfg.get('experiment', {}).get('task', 'reach'))
            gui.add_button("Plan only (no motion)").on_click(lambda _: self.background(self.plan_only))
            gui.add_button("Step (one move)").on_click(lambda _: self.background(self.one_step))
            gui.add_button("Run", disabled=trial_step).on_click(lambda _: self.background(self.run))
            gui.add_button("Stop", color="red").on_click(lambda _: self.stop())
            self.status = gui.add_markdown("Idle.")
            gui.add_markdown(LEGEND)
            gui.add_markdown("Success measures hand-goal distance only. Object grasp/lift success is evaluated separately from video.")
        if sim_cell is not None:
            with gui.add_folder("Simulation"):
                with gui.add_folder("Place the robot"):
                    self.jog = [gui.add_slider(f"d{a} from home (m)", -0.45, 0.45, 0.001, 0.0) for a in "xyz"]
                    self.closure = gui.add_slider("finger closure", 0.0, 1.0, 0.05, 0.0)
                    self.arrange = gui.add_checkbox("Arrange directly", False, hint="Put the robot there at once (as a person arranges the real cell) instead of driving it there; objects stay")
                    gui.add_button("Move").on_click(lambda _: self.background(self.sim_move))
                    gui.add_button("Reset scene (robot home, hand open, objects back)").on_click(lambda _: self.background(self.sim_reset))
                    for name, preset in SIM_PRESETS.items():
                        gui.add_button(f"Example {name} goal").on_click(lambda _, p=preset: self.sim_preset(p))
                    self.home = sim_cell.state()[2]
                with gui.add_folder("Video"):
                    self.video_frame = gui.add_image(sim_cell.render_frame(), label="latest frame (green dots: goal hand)")
                    self.video_path = gui.add_markdown("Each run is recorded to `video.mp4` in its run folder.")
        for handle in (self.use_box, self.xs, self.ys, self.zs):
            handle.on_update(lambda _: self.box_changed())
        self.show_goal.on_update(lambda _: self.draw_goal())
        self.show_goal_scene.on_update(lambda _: self.draw_goal())
        self.show_start.on_update(lambda _: self.draw_start())
        lo, hi = np.asarray(p.box_lo), np.asarray(p.box_hi)
        self.server.initial_camera.look_at = tuple((lo + hi) / 2)
        self.server.initial_camera.position = tuple((lo + hi) / 2 + [0.9, -0.9, 0.7])
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.task.on_update(lambda _: setattr(self, "task_ready", None))
        self.scene.add_frame("/base", axes_length=0.1, axes_radius=0.003)
        self.box_changed()
        self.draw_goal()
        self.draw_start()

    def box_changed(self):
        """Apply the box settings to perception and redraw the box."""
        lo, hi = (self.xs.value[0], self.ys.value[0], self.zs.value[0]), (self.xs.value[1], self.ys.value[1], self.zs.value[1])
        self.session.perception.s = replace(self.session.perception.s, use_box=self.use_box.value, box_lo=lo, box_hi=hi)
        self.session.cfg["perception"].update(use_box=self.use_box.value, box_lo=list(lo), box_hi=list(hi))
        self.scene.add_line_segments("/box", box_edges(lo, hi), YELLOW, thickness=0.004, visible=self.use_box.value)

    def save(self):
        """Write the box into the config file."""
        s = self.session.perception.s
        save_box(self.config_path, s.use_box, s.box_lo, s.box_hi)
        self.status.content = f"Box saved to `{self.config_path}`."

    def draw(self, prefix, obs, hand_color, scene_color, show_hand=True, show_scene=True):
        """Draw an observation's hand and scene points."""
        hand = obs.segment == SEG_HAND
        self.scene.add_point_cloud(f"{prefix}/hand", obs.coords[hand], hand_color, point_size=0.004, point_shape="rounded", visible=show_hand)
        self.scene.add_point_cloud(f"{prefix}/scene", obs.coords[~hand], scene_color, point_size=0.006, point_shape="rounded", visible=show_scene)

    def draw_goal(self):
        """Draw the goal (green), its hand and scene each shown or hidden."""
        if self.session.goal is not None:
            self.draw("/goal", self.session.goal, GREEN, (120, 200, 140), self.show_goal.value, self.show_goal_scene.value)

    def draw_start(self):
        """Draw the start hand (purple), shown or hidden."""
        if self.session.start is not None:
            hand = self.session.start.segment == SEG_HAND
            self.scene.add_point_cloud("/start/hand", self.session.start.coords[hand], PURPLE, point_size=0.004, point_shape="rounded", visible=self.show_start.value)

    def refresh(self):
        """Observe now and draw it, with the point counts (skipped while an action holds the robot)."""
        if not self.lock.acquire(blocking=False):
            return None
        try:
            obs = self.session.observe()
        except Exception as e:  # show perception problems on the page instead of crashing
            self.counts.content = f"**{type(e).__name__}:** {e}"
            return None
        finally:
            self.lock.release()
        self.draw("/current", obs, BLUE, GRAY)
        raw = self.session.perception.scene_points(obs.views, np.zeros((0, 3)))  # robot not removed
        self.scene.add_point_cloud("/raw", raw, RED, point_size=0.003, point_shape="rounded", visible=self.show_raw.value)
        n = self.session.perception.s.n_scene
        self.counts.content = f"{obs.info['n_scene_seen']} camera voxels, {n} scene points ({obs.info['n_repeated']} repeats) + {int((obs.segment == SEG_HAND).sum())} hand."
        return obs

    def capture(self, which):
        """Capture the goal or the start and show it (called with the lock held)."""
        getattr(self.session, f"capture_{which}")()
        self.task_ready = None
        g, s = self.session.goal is not None, self.session.start is not None
        self.captured.content = f"Goal: {'captured' if g else 'none'}. Start: {'captured' if s else 'none'}. Log: `{self.session.log}`"
        self.draw_goal()
        self.draw_start()
        self.draw("/current", getattr(self.session, which), BLUE, GRAY)

    def background(self, fn, on_reject=None):
        """Run fn in a thread that holds the robot lock; refused (on_reject called) while another action holds it; errors are shown on the page and the robot is stopped."""
        if self.closing or not self.lock.acquire(blocking=False):
            self.status.content = "Busy: wait for the current action to finish, or press Stop."
            if on_reject is not None:
                on_reject()
            return
        self.stop_flag.clear()
        self.session.cell.stop_requested.clear()

        def work():
            """fn, with errors reported and the lock always released."""
            try:
                fn()
            except Exception as e:
                self.status.content = f"**Error ({type(e).__name__}):** {e}"
                try:
                    self.session.cell.stop()
                except Exception:
                    pass
            finally:
                self.lock.release()
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def toggle(self, box, apply):
        """A checkbox that changes the robot's mode: applied in the background, flipped back if refused."""
        if self.reverting:
            return
        value = box.value

        def revert():
            """Show the box as it was."""
            self.reverting = True
            box.value = not value
            self.reverting = False
        self.background(lambda: apply(value), on_reject=revert)

    def ready(self, new_run=False, execute=True):
        """Start a run (new_run, a new task or captures, or the last run ended); False with a message if the goal or start is missing or the robot is released or in freedrive."""
        if execute and getattr(self.session.cell, 'trial_step', False):
            if new_run or self.session.cell.trial_attempted:
                self.status.content = 'Single-step trial: Run is disabled; only one move attempt per launch. Review the saved result before restarting.'
                return False
        if execute and getattr(self.session.cell, "read_only", False):
            self.status.content = "Read-only: use Plan only. Execution requires --execute and completed site checks."
            return False
        if getattr(self.session.cell, "released", False):
            self.status.content = "The robot is released (Teleoperation mode): untick it first."
            return False
        if getattr(self.session.cell, "teach", False):
            self.status.content = "Freedrive is on: untick it first."
            return False
        if self.session.goal is None or self.session.start is None:
            self.status.content = "Capture the goal, then the start."
            return False
        if new_run or self.task_ready != self.task.value or self.session.finished:
            self.session.begin(self.task.value)
            self.task_ready = self.task.value
        return True

    def show(self, record):
        """Draw the latest plan and state and print the record."""
        if self.session.last_plan is not None:
            self.scene.add_point_cloud("/plan/hand", self.session.last_plan["commanded_hand"], ORANGE, point_size=0.004, point_shape="rounded")
        self.draw("/current", self.session.current, BLUE, GRAY)
        if self.sim is not None:
            self.video_frame.image = self.sim.frame
            self.video_path.content = f"Recording to `{self.session.run_dir / 'video.mp4'}` (complete when the run ends)."
        what = "planned only (no motion)" if record.get("plan_only") else "moved" if record["moved"] else "held"
        self.status.content = (f"Replan {record['replan']}: distance {record['distance_mm']:.1f} mm, {what}"
                               f"{' (' + record['hold'] + ')' if record['hold'] else ''}, fit {record['fit_residual_mm']:.1f} mm, within {self.session.loop.success_mm:.0f} mm of the goal for {record['within_success']}/{self.session.loop.success_replans} replans."
                               f"{' **Stop: ' + record['stop'] + '**' if record['stop'] else ''}")

    def plan_only(self):
        """One plan without motion; the orange hand shows where it would move."""
        if self.ready(execute=False):
            self.show(self.session.step(execute=False))

    def one_step(self):
        """One replan with motion."""
        if self.ready():
            self.show(self.session.step(execute=True))
            if getattr(self.session.cell, 'trial_step', False) and self.session.cell.trial_attempted:
                self.session.finish('trial step complete')
                self.status.content += f'  \nSingle move completed. Further motion disabled. Logs: `{self.session.run_dir}`'

    def run(self):
        """A new run: replan and move until a stop condition or Stop."""
        if self.ready(new_run=True):
            reason = self.session.run(execute=True, should_stop=self.stop_flag.is_set, on_step=self.show)
            self.status.content += f"  \nRun ended: **{reason}**. Logs: `{self.session.run_dir}`"

    def stop(self):
        """Stop: the moving thread halts the arm and fingers now, and the run ends after this replan."""
        self.stop_flag.set()
        self.session.cell.stop_requested.set()
        self.status.content = "Stopping."
        # If idle (including freedrive), no worker will consume the stop flag.
        if self.lock.acquire(blocking=False):
            try:
                self.session.cell.stop()
                if self.session.run_dir is not None and not self.session.finished:
                    self.session.finish('stopped by user')
            finally:
                self.lock.release()

    def shutdown(self):
        """Let the command owner stop before the main thread disconnects devices."""
        self.closing = True
        self.stop_flag.set()
        self.session.cell.stop_requested.set()
        if self.worker is not None:
            self.worker.join(timeout=10.0)
            if self.worker.is_alive():
                print("Worker did not exit; avoiding concurrent RTDE calls. The motion watchdog remains the arm fallback; use pendant stop.")
                return False
        if self.session.run_dir is not None and not self.session.finished:
            self.session.finish('application shutdown')
        return True

    def sim_move(self):
        """Sim only: move the flange by the jog offsets from home and close the fingers by the slider."""
        q = self.closure_angles(self.closure.value)
        target = self.home.copy()
        target[:3, 3] += [j.value for j in self.jog]
        if self.arrange.value:
            q_arm = self.sim.ik(target, self.sim.state()[0])
            if q_arm is None:
                self.status.content = "No IK solution for that pose."
                return
            self.sim.set_state(q_arm, q, self.sim.object_positions())
        else:
            self.sim.move(target, q)
        self.video_frame.image = self.sim.render_frame()

    def sim_reset(self):
        """Sim only: robot at home with the hand open, objects back where the config has them."""
        self.sim.set_state(np.radians(self.sim.cfg["home_deg"]), np.zeros(22), [o["pos"] for o in self.sim.cfg["objects"]])
        for j in self.jog:
            j.value = 0.0
        self.closure.value, self.arrange.value = 0.0, False
        self.video_frame.image = self.sim.render_frame()
        self.status.content = "Scene reset: robot at home, hand open, objects at their config positions. Now Capture start."

    def sim_preset(self, preset):
        """Sim only: fill the placement sliders with an example goal (press Move to apply)."""
        offset, closure, arrange = preset
        for j, v in zip(self.jog, offset):
            j.value = v
        self.closure.value, self.arrange.value = closure, arrange
        self.status.content = "Example loaded: press Move, then Capture goal. Then Reset scene and Capture start."

    def closure_angles(self, closure):
        """Sharpa angles with the curling joints at closure of their upper limit, the others at 0."""
        names, limits = self.session.robot.hand_joint_names, self.session.robot.hand_limits
        return np.array([closure * hi if any(n.endswith(f) or f"_{f}" in n for f in FLEX) else 0.0 for n, (lo, hi) in zip(names, limits)])

    def set_teleop(self, on):
        """Real robot: release it to a teleoperation system, or take control back."""
        self.session.cell.release() if on else self.session.cell.take_control()
        self.status.content = "Teleoperation mode: the robot is released; the live view follows it." if on else "Control taken back: Plan, Step and Run are available."

    def set_freedrive(self, on):
        """Real robot: arm teach mode on or off."""
        self.session.cell.freedrive(on)
        self.status.content = "Freedrive on: move the arm by hand." if on else "Freedrive off: the arm holds."

    def set_fingers(self):
        """Move only the fingers to the slider's closure, the arm where it is."""
        if getattr(self.session.cell, "teach", False):
            self.status.content = "Untick Freedrive first."
            return
        self.session.cell.move(self.session.cell.state()[2], self.closure_angles(self.real_closure.value))

    def serve(self):
        """Keep the live view fresh while idle, until Ctrl+C."""
        while True:
            if self.live.value:
                self.refresh()
            time.sleep(1.0)
