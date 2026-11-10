"""Offline prediction diagnostics on the same state/input/measurement log."""

from pathlib import Path

import numpy as np


_MODES = ("fixed", "history_long", "history_adaptive", "vision_history")
_LABELS = {
    "fixed": "Fixed model", "history_long": "Long history",
    "history_adaptive": "Adaptive history", "vision_history": "Vision + history",
}
_COLORS = {
    "fixed": "#4c78a8", "history_long": "#f58518",
    "history_adaptive": "#228b50", "vision_history": "#9467bd",
}


def _rolling_rmse(times, errors):
    """Trailing one-second RMSE; missing targets remain missing, never zero."""
    times, errors = np.asarray(times), np.asarray(errors, dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    width = max(1, int(round(1.0 / dt)))
    valid = np.isfinite(errors)
    squares = np.r_[0.0, np.cumsum(np.where(valid, errors**2, 0.0))]
    counts = np.r_[0, np.cumsum(valid)]
    ends = np.arange(1, len(errors) + 1)
    starts = np.maximum(0, ends - width)
    totals = squares[ends] - squares[starts]
    denominators = counts[ends] - counts[starts]
    result = np.full(len(errors), np.nan)
    available = valid & (denominators > 0)
    result[available] = np.sqrt(totals[available] / denominators[available])
    return result


def _event(axis, event_time):
    if event_time is not None and np.isfinite(float(event_time)):
        axis.axvline(float(event_time), color="0.45", ls=":", lw=1.2,
                     label="True event (evaluation only)")


def save_dashboard(runs, output_path):
    """Save a six-panel, same-case/same-seed comparison and close its figure."""
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("Provide at least one prediction run.")
    case = "motor_fault" if any(run["case"] == "motor_fault" for run in runs) else runs[0]["case"]
    candidates = [run for run in runs if run["case"] == case]
    seed = candidates[0]["seed"]
    selected = [run for run in candidates if run["seed"] == seed]
    selected.sort(key=lambda run: _MODES.index(run["mode"]) if run["mode"] in _MODES else len(_MODES))
    first = selected[0]
    event_time = first.get("event_time")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    speed_ax, yaw_ax, gain_ax, input_ax, bars_ax, protocol_ax = axes.flat
    figure.suptitle(
        "Dynamics prediction on a shared motion log\n"
        f"Case: {case.replace('_', ' ')} | seed {seed} | all times denote prediction origins",
        fontsize=14,
    )
    labels, one_step, half_second = [], [], []
    for run in selected:
        mode = run["mode"]
        label = _LABELS.get(mode, mode.replace("_", " "))
        color = _COLORS.get(mode, "#777777")
        times = np.asarray(run["times"])
        errors = np.asarray(run["errors_half"])
        speed_ax.plot(times, _rolling_rmse(times, errors[:, 0]), color=color, label=label, lw=1.3)
        yaw_ax.plot(times, _rolling_rmse(times, errors[:, 1]), color=color, label=label, lw=1.3)
        gain_ax.plot(times, np.asarray(run["eta"])[:, 0], color=color, label=label, lw=1.3)
        labels.append(label)
        one_step.append(float(run["metrics"]["one_speed_rmse_post"]))
        half_second.append(float(run["metrics"]["half_speed_rmse_post"]))
    for axis, title, ylabel in (
        (speed_ax, "0.5 s terminal speed error: trailing 1 s RMSE", "Speed prediction RMSE (m/s)"),
        (yaw_ax, "0.5 s terminal yaw-rate error: trailing 1 s RMSE", "Yaw-rate prediction RMSE (rad/s)"),
        (gain_ax, "Estimated left effective drive gain", "Effective gain (-)"),
    ):
        _event(axis, event_time)
        axis.set(title=title, xlabel="Prediction origin time (s)", ylabel=ylabel)
        axis.grid(True, alpha=.25)
        axis.legend(fontsize=8, loc="best")
    speed_ax.set_ylim(bottom=0)
    yaw_ax.set_ylim(bottom=0)

    times = np.asarray(first["times"])
    commands = np.asarray(first["commands"])
    input_ax.plot(times, commands[:, 0], color="#4477aa", label="Left input")
    input_ax.plot(times, commands[:, 1], color="#cc6677", label="Right input")
    _event(input_ax, event_time)
    input_ax.set(title="Shared input log for every predictor", xlabel="Time (s)", ylabel="Wheel input (N)")
    input_ax.grid(True, alpha=.25)
    input_ax.legend(fontsize=8, loc="best")

    positions = np.arange(len(selected))
    width = .36
    bars_ax.bar(positions-width/2, one_step, width, color="#9ecae1", label="One-step horizon")
    bars_ax.bar(positions+width/2, half_second, width, color="#3182bd", label="0.5 s horizon, frozen gain")
    bars_ax.set_xticks(positions, [label.replace(" ", "\n", 1) for label in labels], fontsize=8)
    bars_ax.set(title="Speed prediction RMSE after the true event", ylabel="Speed prediction RMSE (m/s)")
    bars_ax.set_ylim(bottom=0)
    bars_ax.grid(True, axis="y", alpha=.25)
    bars_ax.legend(fontsize=8, loc="best")
    if event_time is None:
        bars_ax.set_title("Speed prediction RMSE over the scoring interval")

    protocol_ax.set_axis_off()
    protocol_ax.set_title("Prediction protocol")
    protocol_ax.text(
        .02, .96,
        "Every model receives the same observed states, inputs\n"
        "and measurement-noise realization.\n\n"
        "Predict first; observe the transition; then update.\n"
        "The 0.5 s rollout freezes the current gain.\n"
        "A longer horizon is not 25 repeated updates.\n\n"
        "No speed-policy differences between predictors.\n"
        "Visual changes can trigger a shorter history;\n"
        "RGB does not directly supply a friction value.\n\n"
        "All curves use prediction-origin timestamps.\n"
        "A rollout crossing an event can raise error\n"
        "at an origin before the event.\n"
        "Unavailable terminal targets remain NaN.\n"
        "Event lines are used only for evaluation.",
        transform=protocol_ax.transAxes, va="top", fontsize=10, linespacing=1.3,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)
