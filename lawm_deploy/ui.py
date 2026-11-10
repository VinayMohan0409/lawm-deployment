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

GRAY, BLUE, GREEN, ORANGE, YELLOW, RED = (150, 150, 150), (30, 110, 220), (40, 170, 70), (240, 140, 20), (230, 200, 0), (220, 40, 40)
FLEX = ("FE", "PIP", "DIP", "IP")  # Sharpa joints that curl a finger (the sim jog closes these)


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
        self.lock, self.stop_flag, self.task_ready = threading.Lock(), threading.Event(), None
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
        with gui.add_folder("2. Goal, then start"):
            gui.add_button("Capture goal").on_click(lambda _: self.capture("goal"))
            gui.add_button("Capture start").on_click(lambda _: self.capture("start"))
            self.show_goal = gui.add_checkbox("Show goal hand", True)
            self.show_goal_scene = gui.add_checkbox("Show goal scene", False)
            self.captured = gui.add_markdown("Goal: none. Start: none.")
        with gui.add_folder("3. Run"):
            self.task = gui.add_dropdown("Task", ("reach", "grasp"))
            gui.add_button("Plan only (no motion)").on_click(lambda _: self.background(self.plan_only))
            gui.add_button("Step (one move)").on_click(lambda _: self.background(self.one_step))
            gui.add_button("Run").on_click(lambda _: self.background(self.run))
            gui.add_button("Stop", color="red").on_click(lambda _: self.stop())
            self.status = gui.add_markdown("Idle.")
        if sim_cell is not None:
            with gui.add_folder("Sim only: place the robot"):
                self.jog = [gui.add_slider(f"d{a} (m)", -0.3, 0.3, 0.005, 0.0) for a in "xyz"]
                self.closure = gui.add_slider("finger closure", 0.0, 1.0, 0.05, 0.0)
                gui.add_button("Move").on_click(lambda _: self.background(self.sim_move))
                self.home = sim_cell.state()[2]
        for handle in (self.use_box, self.xs, self.ys, self.zs):
            handle.on_update(lambda _: self.box_changed())
        self.show_goal.on_update(lambda _: self.draw_goal())
        self.show_goal_scene.on_update(lambda _: self.draw_goal())
        lo, hi = np.asarray(p.box_lo), np.asarray(p.box_hi)
        self.server.initial_camera.look_at = tuple((lo + hi) / 2)
        self.server.initial_camera.position = tuple((lo + hi) / 2 + [0.9, -0.9, 0.7])
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.task.on_update(lambda _: setattr(self, "task_ready", None))
        self.scene.add_frame("/base", axes_length=0.1, axes_radius=0.003)
        self.box_changed()

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

    def refresh(self):
        """Observe now and draw it, with the point counts."""
        with self.lock:
            try:
                obs = self.session.observe()
            except Exception as e:  # show perception problems on the page instead of crashing
                self.counts.content = f"**{type(e).__name__}:** {e}"
                return None
        self.draw("/current", obs, BLUE, GRAY)
        raw = self.session.perception.scene_points(obs.views, np.zeros((0, 3)))  # robot not removed
        self.scene.add_point_cloud("/raw", raw, RED, point_size=0.003, point_shape="rounded", visible=self.show_raw.value)
        n = self.session.perception.s.n_scene
        self.counts.content = f"{obs.info['n_scene_seen']} camera voxels, {n} scene points ({obs.info['n_repeated']} repeats) + {int((obs.segment == SEG_HAND).sum())} hand."
        return obs

    def capture(self, which):
        """Capture the goal or the start and show it."""
        with self.lock:
            try:
                getattr(self.session, f"capture_{which}")()
            except Exception as e:
                self.status.content = f"**Capture failed:** {e}"
                return
        self.task_ready = None
        g, s = self.session.goal is not None, self.session.current is not None
        self.captured.content = f"Goal: {'captured' if g else 'none'}. Start: {'captured' if s else 'none'}. Log: `{self.session.log}`"
        self.draw_goal()
        if self.session.current is not None:
            self.draw("/current", self.session.current, BLUE, GRAY)

    def background(self, fn):
        """Run fn in a thread unless something else holds the cell."""
        if self.lock.locked():
            self.status.content = "Busy: wait or press Stop."
            return
        threading.Thread(target=fn, daemon=True).start()

    def ready(self):
        """Set up the planner if the task or the captures changed; False with a message if goal or start is missing."""
        if self.session.goal is None or self.session.current is None:
            self.status.content = "Capture the goal, then the start."
            return False
        if self.task_ready != self.task.value:
            self.session.begin(self.task.value)
            self.task_ready = self.task.value
        return True

    def show(self, record):
        """Draw the latest plan and state and print the record."""
        if self.session.last_plan is not None:
            self.scene.add_point_cloud("/plan/hand", self.session.last_plan["commanded_hand"], ORANGE, point_size=0.004, point_shape="rounded")
        self.draw("/current", self.session.current, BLUE, GRAY)
        self.status.content = (f"Replan {record['replan']}: distance {record['distance_mm']:.1f} mm, {'moved' if record['moved'] else 'held' if record['hold'] else 'planned, no motion'}"
                               f"{' (' + record['hold'] + ')' if record['hold'] else ''}, fit {record['fit_residual_mm']:.1f} mm."
                               f"{' **Stop: ' + record['stop'] + '**' if record['stop'] else ''}")

    def plan_only(self):
        """One replan without motion; the orange hand shows where it would move."""
        with self.lock:
            if self.ready():
                self.show(self.session.step(execute=False))

    def one_step(self):
        """One replan with motion."""
        with self.lock:
            if self.ready():
                self.show(self.session.step(execute=True))

    def run(self):
        """Replan and move until a stop condition or Stop."""
        self.stop_flag.clear()
        with self.lock:
            if self.ready():
                reason = self.session.run(execute=True, should_stop=self.stop_flag.is_set, on_step=self.show)
                self.status.content += f"  \nRun ended: **{reason}**."

    def stop(self):
        """Stop after the current move and hold the robot."""
        self.stop_flag.set()
        self.session.cell.stop()
        self.status.content = "Stopping after the current move."

    def sim_move(self):
        """Sim only: move the flange by the jog offsets from home and close the fingers by the slider."""
        names, limits = self.session.robot.hand_joint_names, self.session.robot.hand_limits
        q = np.array([self.closure.value * hi if any(n.endswith(f) or f"_{f}" in n for f in FLEX) else 0.0 for n, (lo, hi) in zip(names, limits)])
        target = self.home.copy()
        target[:3, 3] += [j.value for j in self.jog]
        with self.lock:
            self.sim.move(target, q)

    def serve(self):
        """Keep the live view fresh while idle, forever."""
        while True:
            if self.live.value and not self.lock.locked():
                self.refresh()
            time.sleep(1.0)
