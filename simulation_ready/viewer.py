"""Static figures and playback for already simulated differential-drive runs.

The estimator reports an *effective* input efficiency.  It should not be read
as a uniquely identified motor, friction, mass, or slope parameter.
"""

from pathlib import Path

import numpy as np


_COLORS = {
    "fixed": "#4c78a8",
    "pi": "#f58518",
    "window_40": "#b279a2",
    "window_80": "#72b7b2",
    "window_160": "#eeca3b",
    "adaptive": "#228b50",
}
_LABELS = {
    "fixed": "Fixed model",
    "pi": "PI baseline",
    "window_40": "Window 0.8 s",
    "window_80": "Window 1.6 s",
    "window_160": "Window 3.2 s",
    "adaptive": "Adaptive window",
}
_ACTIVE_ANIMATIONS = []


def _label(mode, config):
    """Show the actual duration even when the caller changes the sample time."""
    if mode.startswith("window_"):
        try:
            return f"Window {int(mode.split('_')[1]) * config['dt']:.2g} s"
        except (KeyError, ValueError, IndexError):
            pass
    return _LABELS.get(mode, mode.replace("_", " ").title())


def _selected_mode(results):
    if not results:
        raise ValueError("At least one simulation run is needed for plotting.")
    if "adaptive" in results:
        return "adaptive"
    for mode in ("window_160", "window_80", "window_40"):
        if mode in results:
            return mode
    return next(iter(results))


def _position_error(run):
    return np.linalg.norm(
        np.asarray(run["states"])[:, :2] - np.asarray(run["reference"])[:, :2],
        axis=1,
    )


def _changes(ax, results):
    change_times = sorted(
        {
            float(run["change_time"])
            for run in results.values()
            if run.get("change_time") is not None
            and np.isfinite(float(run["change_time"]))
        }
    )
    for index, change_time in enumerate(change_times):
        ax.axvline(
            change_time,
            color="0.5",
            linestyle=":",
            linewidth=1.2,
            label="Dynamics change" if index == 0 else "_nolegend_",
        )


def _time_axis(ax, results, ylabel):
    _changes(ax, results)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _command_limit(config):
    # The supplied lesson uses +/- 6 N per wheel.  The top-level override lets
    # the same viewer describe a subsequently configured actuator limit.
    return float(config.get("command_limit", 6.0))


def save_dashboard(results, config, output_path):
    """Save six diagnostic panels at 150 dpi; no GUI or core import is needed.

    The caller may set ``matplotlib.use('Agg')`` before calling this function.
    Every controller is shown on the shared tracking panels.  Detailed
    estimator, input and window panels use adaptive mode when available.
    """
    import matplotlib.pyplot as plt

    selected = _selected_mode(results)
    detail = results[selected]
    detail_label = _label(selected, config)
    first = next(iter(results.values()))
    times = np.asarray(first["times"])
    reference = np.asarray(first["reference"])
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    path_ax, error_ax, speed_ax, eta_ax, command_ax, window_ax = axes.flat
    scenario = str(config.get("scenario", "Dynamics adaptation"))
    path_kind = str(config.get("reference", {}).get("path", "trajectory"))
    figure.suptitle(
        f"Closed-loop simulation: {scenario} / {path_kind}\n"
        "Effective efficiency is an equivalent model quantity, not a physical diagnosis",
        fontsize=14,
    )

    path_ax.plot(reference[:, 0], reference[:, 1], "k--", label="Reference", lw=1.5)
    speed_ax.plot(times, reference[:, 3], "k--", label="Reference", lw=1.5)
    for mode, run in results.items():
        color = _COLORS.get(mode)
        label = _label(mode, config)
        states = np.asarray(run["states"])
        run_times = np.asarray(run["times"])
        path_ax.plot(states[:, 0], states[:, 1], label=label, color=color, lw=1.5)
        error_ax.plot(run_times, _position_error(run), label=label, color=color, lw=1.2)
        speed_ax.plot(run_times, states[:, 3], label=label, color=color, lw=1.2)
    path_ax.set(title="Robot path (equal distance scales)", xlabel="x (m)", ylabel="y (m)")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(True, alpha=0.25)
    path_ax.legend(fontsize=8, loc="best")
    error_ax.set_title("Position tracking error (magnifies small deviations)")
    _time_axis(error_ax, results, "Position error (m)")
    error_ax.set_ylim(bottom=0)
    error_ax.legend(fontsize=8, loc="best")
    speed_ax.set_title("Actual forward speed")
    _time_axis(speed_ax, results, "Speed (m/s)")
    speed_ax.legend(fontsize=8, loc="best")

    detail_times = np.asarray(detail["times"])
    raw = np.asarray(detail["eta_raw"])
    used = np.asarray(detail["eta_used"])
    for side, color in enumerate(("#4477aa", "#cc6677")):
        wheel = ("Left", "Right")[side]
        eta_ax.plot(
            detail_times, raw[:, side], color=color, alpha=0.35, lw=0.8,
            label=f"{wheel}: raw estimate",
        )
        eta_ax.plot(
            detail_times, used[:, side], color=color, lw=1.6,
            label=f"{wheel}: used by controller",
        )
    eta_ax.set_title(f"{detail_label}: effective efficiency")
    _time_axis(eta_ax, results, "Effective efficiency (-)")
    eta_ax.legend(fontsize=8, loc="best")

    commands = np.asarray(detail["commands"])
    command_times = detail_times[:-1]
    command_ax.plot(command_times, commands[:, 0], label="Left command", color="#4477aa")
    command_ax.plot(command_times, commands[:, 1], label="Right command", color="#cc6677")
    limit = _command_limit(config)
    command_ax.axhline(limit, color="0.45", ls="--", lw=1, label=f"Limit +/- {limit:g} N")
    command_ax.axhline(-limit, color="0.45", ls="--", lw=1)
    command_ax.set_title(f"{detail_label}: wheel commands")
    _time_axis(command_ax, results, "Nominal wheel force (N)")
    command_ax.legend(fontsize=8, loc="best")

    window = np.asarray(detail["window_steps"], dtype=float) * float(config["dt"])
    window_ax.step(detail_times, window, where="post", color="#228b50", label="Window duration")
    alarms = np.asarray(detail["alarm"], dtype=bool)
    window_ax.scatter(
        detail_times[alarms], window[alarms], color="#c83232", marker="v",
        s=34, zorder=4, label="Change detector alarm",
    )
    window_ax.set_title(f"{detail_label}: active history (0 = no window estimate)")
    _time_axis(window_ax, results, "History duration (s)")
    window_ax.set_ylim(bottom=0)
    window_ax.legend(fontsize=8, loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)


def show_animation(results, config):
    """Replay completed runs in a window and return the retained animation.

    This is a playback of precomputed simulation data, not a live physics
    engine.  ``playback_speed=2.0`` runs two simulated seconds per wall second.
    Fixed, PI and adaptive runs are selected if present.  If none is present,
    all supplied window runs are shown.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import FancyArrowPatch

    _selected_mode(results)  # Also validates that the input is non-empty.
    selected = [mode for mode in ("fixed", "pi", "adaptive") if mode in results]
    if not selected:
        selected = list(results)
    runs = {mode: results[mode] for mode in selected}
    first = next(iter(runs.values()))
    reference = np.asarray(first["reference"])
    times = np.asarray(first["times"])
    playback_speed = float(config.get("playback_speed", 2.0))
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("playback_speed must be finite and greater than zero.")
    dt = float(config["dt"])
    stride = max(1, int(round(playback_speed / (30.0 * dt))))
    frames = np.unique(np.r_[np.arange(0, len(times), stride), len(times) - 1])

    figure, (path_ax, error_ax) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    figure.suptitle(
        f"Simulation replay ({playback_speed:g}x): precomputed trajectories", fontsize=14
    )
    path_ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.3, label="Reference")
    reference_marker, = path_ax.plot([], [], "ko", ms=5, label="Reference now")
    all_positions = np.vstack([reference[:, :2]] + [np.asarray(run["states"])[:, :2] for run in runs.values()])
    mins, maxs = all_positions.min(axis=0), all_positions.max(axis=0)
    span = max(float(np.max(maxs - mins)), 1.0)
    margin = max(0.05 * span, 0.3)
    center = (mins + maxs) / 2.0
    half_extent = span / 2.0 + margin
    path_ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    path_ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.set(title="Position and heading (equal scales)", xlabel="x (m)", ylabel="y (m)")
    path_ax.grid(True, alpha=0.25)
    error_ax.set(title="Tracking error during replay", xlabel="Time (s)", ylabel="Position error (m)")
    error_ax.set_xlim(float(times[0]), float(times[-1]))
    errors = {mode: _position_error(run) for mode, run in runs.items()}
    maximum_error = max(float(np.max(error)) for error in errors.values())
    error_ax.set_ylim(0, max(0.02, 1.1 * maximum_error))
    error_ax.grid(True, alpha=0.25)
    _changes(error_ax, runs)
    cursor = error_ax.axvline(float(times[0]), color="0.3", lw=1)
    time_text = path_ax.text(
        0.02, 0.98, "", transform=path_ax.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
    )
    heading_length = max(0.02 * span, 0.2)
    artists = {}
    for mode, run in runs.items():
        color = _COLORS.get(mode)
        trail, = path_ax.plot([], [], color=color, label=_label(mode, config), lw=1.7)
        marker, = path_ax.plot([], [], "o", color=trail.get_color(), ms=7)
        arrow = FancyArrowPatch(
            (0, 0), (heading_length, 0), color=trail.get_color(),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.8,
        )
        path_ax.add_patch(arrow)
        error_line, = error_ax.plot([], [], color=trail.get_color(), label=_label(mode, config), lw=1.5)
        artists[mode] = (trail, marker, arrow, error_line)
    path_ax.legend(fontsize=8, loc="lower left")
    error_ax.legend(fontsize=8, loc="best")

    def update(frame):
        current_time = float(times[int(frame)])
        reference_index = min(int(frame), len(reference) - 1)
        reference_marker.set_data([reference[reference_index, 0]], [reference[reference_index, 1]])
        changed = False
        for mode, run in runs.items():
            run_times = np.asarray(run["times"])
            index = min(int(np.searchsorted(run_times, current_time, side="right") - 1), len(run_times) - 1)
            index = max(0, index)
            states = np.asarray(run["states"])
            x, y, heading = states[index, :3]
            trail, marker, arrow, error_line = artists[mode]
            trail.set_data(states[:index + 1, 0], states[:index + 1, 1])
            marker.set_data([x], [y])
            arrow.set_positions(
                (x, y), (x + heading_length * np.cos(heading), y + heading_length * np.sin(heading))
            )
            error_line.set_data(run_times[:index + 1], errors[mode][:index + 1])
            change_time = run.get("change_time")
            changed = changed or (change_time is not None and current_time >= float(change_time))
        cursor.set_xdata([current_time, current_time])
        has_disturbance = any(run.get("change_time") is not None for run in runs.values())
        status = ("After dynamics change" if changed else "Before dynamics change") if has_disturbance else "No disturbance"
        time_text.set_text(f"t = {current_time:.2f} s\n" + status)
        return []

    # Draw the initial pose immediately, including when the GUI is resized.
    update(0)
    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000.0 * stride * dt / playback_speed,
        repeat=False, blit=False, cache_frame_data=False,
    )
    _ACTIVE_ANIMATIONS[:] = [animation]
    plt.show()
    return animation
