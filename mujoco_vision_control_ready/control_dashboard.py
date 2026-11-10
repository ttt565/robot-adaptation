"""Static comparison of control ablations on the same case and noise seed."""

from pathlib import Path

import numpy as np


_MODES = ("baseline", "motion_only", "vision_motion")
_LABELS = {"baseline": "Baseline", "motion_only": "Motion only", "vision_motion": "Vision + motion"}
_COLORS = {"baseline": "#4c78a8", "motion_only": "#f58518", "vision_motion": "#228b50"}


def save_comparison(runs, output_path):
    """Save a 2x3 comparison for one case and seed, then close its figure.

    ``visible_low_friction`` is selected when present; otherwise the first case
    is selected.  Only the first supplied seed for that case is compared, so
    the plotted differences do not mix scenes or random trials.
    """
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("Provide at least one control simulation run.")
    case = "visible_low_friction" if any(run["case"] == "visible_low_friction" for run in runs) else runs[0]["case"]
    candidates = [run for run in runs if run["case"] == case]
    seed = candidates[0]["seed"]
    selected = [run for run in candidates if run["seed"] == seed]
    selected.sort(key=lambda run: _MODES.index(run["mode"]) if run["mode"] in _MODES else len(_MODES))
    first = selected[0]
    times = np.asarray(first["times"])
    reference = np.asarray(first["reference"])
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    position_ax, speed_ax, cap_ax, slip_ax, error_ax, progress_ax = axes.flat
    figure.suptitle(
        "Vision-assisted speed limiting: matched control ablations\n"
        f"Case: {case.replace('_', ' ')} | seed {seed} | original timed reference",
        fontsize=14,
    )
    position_ax.plot(times, reference[:, 0], "k--", label="Original reference", lw=1.5)
    speed_ax.plot(times, reference[:, 3], "k--", label="Original reference", lw=1.5)
    labels, colors, rmses, end_x = [], [], [], []
    for run in selected:
        mode = run["mode"]
        label = _LABELS.get(mode, mode.replace("_", " "))
        color = _COLORS.get(mode, "#777777")
        current_times = np.asarray(run["times"])
        states = np.asarray(run["states"])
        cap = np.asarray(run["cap"])
        slip = np.asarray(run["measured_slip"])
        maximum_difference = np.max(np.abs(slip), axis=1)
        position_ax.plot(current_times, states[:, 0], label=label, color=color, lw=1.4)
        speed_ax.plot(current_times, states[:, 3], label=label, color=color, lw=1.2)
        cap_ax.step(current_times[:-1], cap, where="post", label=label, color=color, lw=1.3)
        slip_ax.plot(current_times[1:], maximum_difference, label=label, color=color, lw=1.0, alpha=.9)
        labels.append(label)
        colors.append(color)
        rmses.append(float(run["metrics"]["position_rmse"]))
        end_x.append(float(states[-1, 0]))

    for axis, title, ylabel in (
        (position_ax, "Forward position against the original reference", "x (m)"),
        (speed_ax, "Actual forward speed", "Speed (m/s)"),
        (cap_ax, "Controller speed cap", "Speed cap (m/s)"),
        (slip_ax, "Maximum wheel/body speed difference", "max |wheel speed - body speed| (m/s)"),
    ):
        axis.set(title=title, xlabel="Time (s)", ylabel=ylabel)
        axis.grid(True, alpha=.25)
        axis.legend(fontsize=8, loc="best")
    slip_ax.axhline(.2, ls="--", color="0.3", lw=1, label="Evidence threshold 0.2 m/s")
    slip_ax.set_ylim(bottom=0)
    slip_ax.legend(fontsize=8, loc="best")
    cap_ax.set_ylim(bottom=0)
    bars = error_ax.bar(np.arange(len(selected)), rmses, color=colors)
    error_ax.set_xticks(np.arange(len(selected)), [label.replace(" + ", " +\n") for label in labels], fontsize=9)
    error_ax.set(title="Tracking error against the same timed reference", ylabel="Position RMSE (m)")
    error_ax.grid(True, axis="y", alpha=.25)
    error_ax.set_ylim(bottom=0)
    error_ax.margins(y=.18)
    for bar, value in zip(bars, rmses):
        error_ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    y_positions = np.arange(len(selected))[::-1]
    progress_ax.barh(y_positions, end_x, color=colors, height=.65)
    progress_ax.set_yticks(y_positions, labels, fontsize=9)
    original_end = float(reference[-1, 0])
    all_end_x = [0.0, original_end, *end_x]
    minimum, maximum = min(all_end_x), max(all_end_x)
    span = max(maximum-minimum, 1.0)
    progress_ax.set_xlim(minimum-.03*span, maximum+.1*span)
    progress_ax.set_ylim(-3.5, len(selected)-.35)
    progress_ax.axvline(original_end, ymin=.51, ymax=1, color="0.25", ls="--", lw=1,
                        label=f"Reference end x = {original_end:.2f} m")
    for ypos, value in zip(y_positions, end_x):
        progress_ax.annotate(f"{value:.2f} m", (value, ypos), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=9)
    progress_ax.set(title=f"Task progress at t = {times[-1]:.1f} s", xlabel="End x (m)")
    progress_ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, .37))
    progress_ax.text(
        .02, .34,
        "All modes retain the original timed reference.\n"
        "Speed limiting may reduce slip and delay the task.\n"
        "A lower slip score alone is not a tracking win.\n"
        "Isolate visual benefit with the motion-only control.",
        transform=progress_ax.transAxes, va="top", fontsize=9, linespacing=1.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95},
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)
