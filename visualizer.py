import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


SIDEBAR_LEFT = 0.67


# Each example contains joint angles (or a starting guess), a target, and a check.
EXAMPLES = {
    "FK": ((0, 0, 0), None, "Example: straight arm.\nTool (4, 0), direction 0 deg."),
    "Geometric IK": (None, (3, 1, 0), "Example: two solutions.\nTarget (3, 1), direction 0 deg."),
    "Numerical IK": ((30, -60, 30), (3, 1, 0), "Example guess: (30, -60, 30) deg."),
}


def draw_scene(ax, arm, angles, target=None, other=None, title=""):
    """Draw joint positions and the tool frame in base coordinates."""
    ax.clear()
    if other is not None:
        positions, _ = arm.forward_kinematics(*other)
        ax.plot(*np.array(positions)[:, :2].T, "--", color="0.6")
    if angles is not None:
        positions, pose = arm.forward_kinematics(*angles)
        xy = np.array(positions)[:, :2]
        ax.plot(*xy.T, "ko-", linewidth=2, markersize=5)
        for name, point, offset in zip(
            ("Base", "Elbow", "Wrist", "Tool"), xy,
            ((-18, -18), (5, 10), (-20, -15), (15, 12)),
        ):
            ax.annotate(name, point, xytext=offset, textcoords="offset points", fontsize=8)
        for column, name in enumerate(("x", "y")):
            tip = xy[-1] + 0.5 * pose.T[:2, column]
            ax.annotate(name, tip, fontsize=8)
            ax.annotate("", tip, xy[-1], arrowprops={"arrowstyle": "->", "color": "black"})
    if target is not None:
        direction = np.deg2rad(target[2])
        tip = np.array(target[:2]) + 0.7 * np.array([np.cos(direction), np.sin(direction)])
        ax.plot(*target[:2], "kx", markersize=8)
        ax.annotate("target", target[:2], xytext=(5, -15),
                    textcoords="offset points", fontsize=8)
        ax.annotate("", tip, target[:2], arrowprops={
            "arrowstyle": "->", "color": "black", "linestyle": "--"})
    reach = arm.link1_length + arm.link2_length + arm.link3_length
    target_extent = max(abs(value) for value in target[:2]) if target is not None else 0
    limit = max(reach, target_extent) + 1
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Base x", ylabel="Base y")
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=9)


class ArmVisualizer:
    """A plain plot with sidebar controls and one starting example per mode."""

    def __init__(self, arm):
        self.arm = arm
        self.mode, self.frame, self.solution_index = "FK", 0, 0
        self.history, self.solutions, self.sliders = [], [], []
        self.playing, self.changing = False, False
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor="white")
        self.fig.subplots_adjust(left=0.075, right=0.61, bottom=0.12, top=0.93)
        self.info = self.fig.text(SIDEBAR_LEFT, 0.22, "", fontsize=7.5, va="top", linespacing=1.2)
        self.caption = self.fig.text(SIDEBAR_LEFT, 0.06, "", fontsize=7, va="top")
        self.input_title = self.fig.text(SIDEBAR_LEFT, 0.64, "", fontsize=8)
        self.seed_title = self.fig.text(SIDEBAR_LEFT, 0.42, "Initial guess (deg)", fontsize=8)
        self.mode_control = RadioButtons(self.fig.add_axes([SIDEBAR_LEFT, 0.77, 0.31, 0.19]),
                                         tuple(EXAMPLES), activecolor="black")
        for label in self.mode_control.labels:
            label.set_fontsize(8)
        self.reset_button = self._button(SIDEBAR_LEFT, "Reset", self._load_example)
        self.start_button = self._button(SIDEBAR_LEFT + 0.16, "Stop", self._toggle_motion)
        self.seed_sliders = [self._slider(f"q{i+1}", 0.365 - i * 0.055, -180, 180)
                             for i in range(3)]
        self.mode_control.on_clicked(self._switch_mode)
        self._switch_mode("FK")
        self._set_playing(True)
        self._draw()
        self.animation = FuncAnimation(self.fig, self._tick, interval=100,
                                        cache_frame_data=False)
        self.fig.canvas.draw_idle()

    def _button(self, x, label, callback):
        button = Button(self.fig.add_axes([x, 0.69, 0.14, 0.055]), label,
                        color="white", hovercolor="0.9")
        button.label.set_fontsize(8)
        button.on_clicked(callback)
        return button

    def _slider(self, label, y, minimum, maximum):
        slider = Slider(self.fig.add_axes([0.73, y, 0.21, 0.025]),
                        label, minimum, maximum, valinit=0, valfmt="%.2f", color="black",
                        initcolor="black", handle_style={"facecolor": "white", "edgecolor": "black"})
        slider.label.set_fontsize(8)
        slider.label.set_horizontalalignment("left")
        slider.label.set_position(((SIDEBAR_LEFT - 0.73) / 0.21, 0.5))
        slider.valtext.set_fontsize(7)
        slider.drawon = False
        slider.on_changed(self._calculate)
        return slider

    def _switch_mode(self, mode):
        self.mode = mode
        for slider in self.sliders:
            slider.disconnect_events()
            slider.ax.remove()
        labels = ("q1", "q2", "q3") if mode == "FK" else ("x", "y", "phi")
        limits = [(-180, 180)] * 3 if mode == "FK" else [(-5, 5), (-5, 5), (-180, 180)]
        self.sliders = [self._slider(label, 0.585 - i * 0.055, *bounds)
                        for i, (label, bounds) in enumerate(zip(labels, limits))]
        self.input_title.set_text("Joint angles (deg)" if mode == "FK" else "Target (phi in deg)")
        self.info.set_y(0.22 if mode == "Numerical IK" else 0.40)
        self.caption.set_y(0.06 if mode == "Numerical IK" else 0.28)
        for slider in self.seed_sliders:
            slider.ax.set_visible(mode == "Numerical IK")
            slider.set_active(mode == "Numerical IK")
        self.seed_title.set_visible(mode == "Numerical IK")
        self._load_example()

    def _load_example(self, event=None):
        angles, target, caption = EXAMPLES[self.mode]
        self.changing = True
        for slider, value in zip(self.sliders, angles if self.mode == "FK" else target):
            slider.set_val(value)
        if self.mode == "Numerical IK":
            for slider, value in zip(self.seed_sliders, angles):
                slider.set_val(value)
        self.changing = False
        self._calculate()
        self.caption.set_text(caption)
        self.fig.canvas.draw_idle()

    def _calculate(self, value=None):
        if self.changing:
            return
        self._set_playing(False)
        self.frame, self.solution_index = 0, 0
        self.caption.set_text("")
        self.history, self.solutions = [], []
        target = [slider.val for slider in self.sliders]
        if self.mode == "Geometric IK":
            self.solutions = self.arm.geometric_inverse_kinematics(target[:2], target[2])
        elif self.mode == "Numerical IK":
            self.result = self.arm.numerical_inverse_kinematics(
                target[:2], target[2], [slider.val for slider in self.seed_sliders],
                angle_history=self.history)
        self._draw()

    def _set_playing(self, playing):
        self.playing = playing
        for slider in self.sliders + self.seed_sliders:
            slider.set_active(slider.ax.get_visible() and not playing)

    def _toggle_motion(self, event=None):
        if self.mode == "Numerical IK":
            if self.frame == len(self.history) - 1:
                self.frame = 0
        self._set_playing(not self.playing)
        self._draw()

    def _tick(self, frame):
        if not self.playing:
            return
        if self.mode == "FK":
            angles = np.array([slider.val for slider in self.sliders])
            angles = (angles + [1.5, 1.0, -2.0] + 180) % 360 - 180
            self.changing = True
            for slider, angle in zip(self.sliders, angles):
                slider.set_val(angle)
            self.changing = False
            self.caption.set_text("")
        elif self.mode == "Geometric IK":
            if frame % 10 or len(self.solutions) < 2:
                return
            self.solution_index = (self.solution_index + 1) % len(self.solutions)
        else:
            if frame % 3:
                return
            self.frame = min(self.frame + 1, len(self.history) - 1)
            if self.frame == len(self.history) - 1:
                self._set_playing(False)
        self._draw()

    def _draw(self):
        target = None if self.mode == "FK" else [slider.val for slider in self.sliders]
        other, title = None, self.mode
        if self.mode == "FK":
            angles = [slider.val for slider in self.sliders]
        elif self.mode == "Geometric IK":
            angles = self.solutions[self.solution_index] if self.solutions else None
            if len(self.solutions) == 2:
                other = self.solutions[1 - self.solution_index]
            if angles is not None:
                title += f" | solution {self.solution_index + 1} of {len(self.solutions)}"
            else:
                title += " | unreachable"
        else:
            angles = self.history[self.frame]
            title += f" | iteration {self.frame}/{len(self.history) - 1}"
        draw_scene(self.ax, self.arm, angles, target, other, title)
        info = "No solution for this target pose."
        if angles is not None:
            positions, pose = self.arm.forward_kinematics(*angles)
            direction = np.rad2deg(np.arctan2(pose.T[1, 0], pose.T[0, 0]))
            info = ("q = (" + ", ".join(f"{q:.1f}" for q in angles) + ") deg"
                    f"\nTool = ({positions[-1][0]:.3f}, {positions[-1][1]:.3f})"
                    f"\nDirection = {direction:.1f} deg")
            if self.mode == "Numerical IK":
                distance = np.linalg.norm(np.array(target[:2]) - positions[-1][:2])
                angle_error = abs((target[2] - direction + 180) % 360 - 180)
                status = ""
                if self.frame == len(self.history) - 1:
                    status = "Converged" if self.result is not None else "No convergence"
                info += (f"\nPosition error {distance:.2e}"
                         f"\nAngle error {angle_error:.2e} deg\n{status}")
        self.info.set_text(info)
        self.start_button.label.set_text("Stop" if self.playing else "Start")
        can_start = self.mode != "Geometric IK" or len(self.solutions) > 1
        self.start_button.set_active(can_start)
        self.start_button.label.set_color("black" if can_start else "0.5")
        self.fig.canvas.draw_idle()


def animate_arm(arm):
    """Open the interactive arm plot and retain its animation."""
    visualizer = ArmVisualizer(arm)
    plt.show()
    return visualizer
