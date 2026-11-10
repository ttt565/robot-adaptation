"""Static RGB/motion evidence report.  This module does not control the robot."""

from pathlib import Path

import numpy as np


_CASE_LABELS = {
    "normal": "Normal\nbaseline",
    "visible_surface_only": "Visible\nsurface only",
    "hidden_low_friction": "Hidden\nlow friction",
    "visible_low_friction": "Visible +\nlow friction",
    "motor_fault": "Motor\nfault",
}


def _camera_frames(run):
    frames = run.get("preview_frames")
    if frames is None or len(frames) == 0:
        frames = run.get("camera_images", [])
    if len(frames) == 0:
        return None, None
    return np.asarray(frames[0]), np.asarray(frames[-1])


def _event_line(axis, run):
    contact = run.get("contact_time")
    if contact is not None and np.isfinite(float(contact)):
        axis.axvline(float(contact), color="0.45", ls=":", lw=1.3,
                     label="Contact (evaluation only)")


def save_dashboard(runs, model_report, output_path):
    """Save a six-panel figure from RGB images and already computed run logs."""
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("At least one vision/motion run is needed.")
    selected = "visible_low_friction" if "visible_low_friction" in runs else next(iter(runs))
    run = runs[selected]
    rows = run.get("rows", [])
    first_frame, last_frame = _camera_frames(run)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    before_ax, after_ax, scores_ax, flags_ax, errors_ax, report_ax = axes.flat
    figure.suptitle(
        "RGB appearance and motion evidence: simulation validation\n"
        f"Camera and evidence panels: {selected.replace('_', ' ')}",
        fontsize=14,
    )
    for axis, rgb, index, caption in (
        (before_ax, first_frame, 0, "First camera frame"),
        (after_ax, last_frame, -1, "Later camera frame"),
    ):
        if rgb is not None:
            axis.imshow(rgb)
        else:
            axis.text(.5, .5, "No RGB frame", ha="center", va="center", transform=axis.transAxes)
        if rows:
            row = rows[index]
            label = str(row.get("label", "unknown")).replace("_", " ")
            axis.set_title(f"{caption}: t = {float(row.get('time', 0)):.2f} s\nRGB label: {label}")
        else:
            axis.set_title(caption)
        axis.set_axis_off()

    frame_times = np.array([float(row.get("time", 0)) for row in rows])
    motion = np.array([float(row.get("motion_score", 0)) for row in rows])
    slip = np.array([float(row.get("slip_score", 0)) for row in rows])
    scores_ax.plot(frame_times, motion, color="#4477aa", lw=1.4, label="Motion residual score")
    scores_ax.plot(frame_times, slip, color="#cc6677", lw=1.4, label="Wheel/body slip score")
    scores_ax.axhline(1, color="0.25", ls="--", lw=1, label="Evidence threshold = 1")
    _event_line(scores_ax, run)
    scores_ax.set(title="Motion evidence scores", xlabel="Time (s)", ylabel="Normalized score (-)")
    scores_ax.set_ylim(bottom=0)
    scores_ax.grid(True, alpha=.25)
    scores_ax.legend(fontsize=8, loc="best")

    for offset, key, label, color in (
        (0, "visual_change", "Visual change", "#228833"),
        (1.7, "motion_anomaly", "Motion anomaly", "#4477aa"),
        (3.4, "slip_evidence", "Slip evidence", "#cc6677"),
    ):
        flags = np.array([bool(row.get(key, False)) for row in rows], dtype=float)
        flags_ax.step(frame_times, offset + flags, where="post", color=color, lw=1.5, label=label)
        flags_ax.axhline(offset, color="0.85", ls=":", lw=.7)
        flags_ax.axhline(offset + 1, color="0.85", ls=":", lw=.7)
    _event_line(flags_ax, run)
    flags_ax.set(title="Evidence flags (each band: low = 0, high = 1)", xlabel="Time (s)")
    flags_ax.set_yticks([.5, 2.2, 3.9], ["Visual change", "Motion anomaly", "Slip evidence"])
    flags_ax.set_ylim(-.25, 4.7)
    flags_ax.grid(True, axis="x", alpha=.25)

    cases = list(runs)
    errors = [float(runs[case].get("metrics", {}).get("position_rmse", np.nan)) for case in cases]
    bars = errors_ax.bar(
        np.arange(len(cases)), errors,
        color=["#228833" if case == selected else "#8a9aab" for case in cases],
    )
    errors_ax.set_xticks(np.arange(len(cases)), [_CASE_LABELS.get(case, case.replace("_", "\n")) for case in cases], fontsize=8)
    errors_ax.set(title="Position tracking error across cases", ylabel="Position RMSE (m)")
    errors_ax.set_ylim(bottom=0)
    errors_ax.grid(True, axis="y", alpha=.25)
    for bar, value in zip(bars, errors):
        if np.isfinite(value):
            errors_ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value),
                               xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    if any(np.isfinite(errors)):
        errors_ax.margins(y=.18)

    correct = model_report.get("heldout_correct")
    total = model_report.get("heldout_images")
    accuracy = model_report.get("accuracy")
    if accuracy is None and correct is not None and total:
        accuracy = float(correct) / float(total)
    accuracy_text = "unavailable" if accuracy is None else f"{float(accuracy):.1%}"
    counts_text = "unavailable" if correct is None or total is None else f"{correct} / {total}"
    report_ax.set_axis_off()
    report_ax.set_title("What this experiment measures")
    report_ax.text(
        .02, .95,
        f"Held-out rendered RGB: {counts_text}\n"
        f"Appearance accuracy: {accuracy_text}\n\n"
        "RGB predicts procedural appearance labels.\n"
        "It does not measure friction or real materials.\n\n"
        "Fused evidence suggests a condition;\n"
        "it does not uniquely identify its cause.\n\n"
        "match_score is a similarity, not a probability.\n"
        "No control changes use the visual output.\n\n"
        "Contact markers are ground truth for evaluation;\n"
        "the detector does not read them.",
        transform=report_ax.transAxes, va="top", fontsize=10, linespacing=1.3,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)
