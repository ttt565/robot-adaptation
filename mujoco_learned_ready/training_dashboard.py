"""Render saved learning results without fitting or synthesizing any scores."""

import csv
import json
from pathlib import Path

import numpy as np


_MODELS = ("fixed", "history", "vision", "fusion", "fusion_reset_history", "fusion_shuffle_image")
_LABELS = {
    "fixed": "Fixed", "history": "History", "vision": "Vision", "fusion": "Fusion",
    "fusion_reset_history": "Fusion\nreset", "fusion_shuffle_image": "Fusion\nshuffled",
}
_COLORS = {
    "fixed": "#4c78a8", "history": "#f58518", "vision": "#b279a2",
    "fusion": "#228b50", "fusion_reset_history": "#72b7b2", "fusion_shuffle_image": "#9d755d",
}


def _unavailable(axis, message):
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes,
              color="0.4", fontsize=10, wrap=True)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _episode_path(data_root, episode):
    name = episode.get("file")
    if not name:
        return None
    raw = Path(name)
    candidates = [raw] if raw.is_absolute() else [data_root/raw, data_root.parent/raw]
    return next((path for path in candidates if path.exists()), None)


def _choose_episode(manifest):
    episodes = [episode for episode in manifest.get("episodes", []) if episode.get("split") == "test"]
    preferred = [episode for episode in episodes
                 if "correlated" in str(episode.get("condition", "")).lower()
                 and "uncorrelated" not in str(episode.get("condition", "")).lower()]
    return (preferred or episodes or [None])[0]


def _plot_loss(axis, path):
    axis.set(title="Validation loss during training", xlabel="Epoch", ylabel="Validation loss")
    axis.grid(True, alpha=.25)
    if not path.exists():
        _unavailable(axis, "Training history unavailable")
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        _unavailable(axis, "Training history is empty")
        return
    names = {name.lower(): name for name in rows[0]}
    epoch_key = next((names[key] for key in ("epoch", "step", "iteration") if key in names), None)
    loss_key = next((names[key] for key in ("val_loss", "validation_loss", "valid_loss", "loss_val") if key in names), None)
    model_key = next((names[key] for key in ("model", "method", "model_name") if key in names), None)
    if epoch_key is None or loss_key is None:
        _unavailable(axis, "No epoch / validation-loss columns found")
        return
    groups = {}
    for row in rows:
        epoch, loss = _number(row.get(epoch_key)), _number(row.get(loss_key))
        if np.isfinite(epoch) and np.isfinite(loss):
            name = row.get(model_key, "Learned model") if model_key else "Learned model"
            groups.setdefault(name, []).append((epoch, loss))
    for model, pairs in groups.items():
        pairs = np.asarray(sorted(pairs))
        axis.plot(pairs[:, 0], pairs[:, 1], label=_LABELS.get(model, model), color=_COLORS.get(model), lw=1.4)
    if groups:
        axis.legend(fontsize=8)
    else:
        _unavailable(axis, "No finite validation losses")


def _plot_bars(axis, report, split, metric, title, ylabel):
    models = report.get("aggregate", {}).get(split, {})
    positions = np.arange(len(_MODELS))
    values = np.array([_number(models.get(model, {}).get(metric)) for model in _MODELS])
    valid = np.isfinite(values)
    axis.set(title=title, ylabel=ylabel)
    axis.set_xticks(positions, [_LABELS[model] for model in _MODELS], fontsize=8)
    axis.set_xlim(-.6, len(_MODELS)-.4)
    axis.set_ylim(bottom=0)
    axis.grid(True, axis="y", alpha=.25)
    if not valid.any():
        _unavailable(axis, f"{split.upper()} scores unavailable")
        return
    bars = axis.bar(positions[valid], values[valid], color=[_COLORS[model] for model, good in zip(_MODELS, valid) if good])
    axis.set_ylim(0, max(float(np.max(values[valid])) * 1.23, 1e-8))
    for bar, value in zip(bars, values[valid]):
        axis.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)


def _plot_camera(axis, data_root, episode):
    axis.set_axis_off()
    if episode is None:
        axis.set_title("Test RGB and projected wheel locations")
        _unavailable(axis, "Test episode unavailable")
        return
    axis.set_title(f"Test RGB + wheel projections: {episode.get('id', '?')}\n{episode.get('condition', 'condition unavailable')}", fontsize=10)
    path = _episode_path(data_root, episode)
    if path is None:
        _unavailable(axis, "Episode image file unavailable")
        return
    with np.load(path, allow_pickle=False) as data:
        if "rgb" not in data or len(data["rgb"]) == 0:
            _unavailable(axis, "RGB frames unavailable")
            return
        frames = data["rgb"]
        image0, image1 = frames[0], frames[-1]
        h, w = image0.shape[:2]
        gap = np.full((h, 4, 3), 255, dtype=image0.dtype)
        axis.imshow(np.concatenate([image0, gap, image1], axis=1))
        if "wheel_pixels" in data and len(data["wheel_pixels"]):
            wheel_pixels = data["wheel_pixels"]
            for index, offset in ((0, 0), (-1, w+4)):
                for wheel, color in enumerate(("#ffcc00", "#ff3366")):
                    point = np.asarray(wheel_pixels[index, wheel], dtype=float)
                    if point.shape == (2,) and np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h:
                        axis.scatter(point[0]+offset, point[1], s=55, marker="x", linewidth=1.6, color=color)
        axis.set_xlim(-.5, 2*w+3.5)
        axis.set_ylim(h-.5, -.5)
        axis.text(.02, .98, "First", transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axis.text(.55, .98, "Last", transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})


def _event_time(episode):
    metadata = episode.get("meta", episode.get("metadata", {}))
    value = episode.get("event_time", metadata.get("event_time") if isinstance(metadata, dict) else None)
    value = _number(value)
    return value if np.isfinite(value) else None


def _online_report(data_root, output, episode):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    speed_ax, yaw_ax, latent_ax, note_ax = axes.flat
    episode_id = str(episode.get("id", "unavailable")) if episode else "unavailable"
    condition = str(episode.get("condition", "unavailable")) if episode else "unavailable"
    figure.suptitle(f"Online hidden-state adaptation on an independent episode\n{episode_id}: {condition}", fontsize=13)
    loaded = {}
    if episode:
        for model in _MODELS:
            path = output/"predictions"/episode_id/f"{model}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    loaded[model] = {key: data[key].copy() for key in data.files}
    for model, data in loaded.items():
        if "times" not in data or "half_error" not in data:
            continue
        times, errors = np.asarray(data["times"]), np.asarray(data["half_error"])
        if errors.ndim != 2 or errors.shape[1] < 2:
            continue
        length = min(len(times), len(errors))
        speed_ax.plot(times[:length], errors[:length, 0], label=_LABELS[model].replace("\n", " "), color=_COLORS[model], lw=1)
        yaw_ax.plot(times[:length], errors[:length, 1], label=_LABELS[model].replace("\n", " "), color=_COLORS[model], lw=1)
    for axis, title, ylabel in ((speed_ax, "0.5 s terminal speed prediction error", "Speed error (m/s)"),
                               (yaw_ax, "0.5 s terminal yaw-rate prediction error", "Yaw-rate error (rad/s)")):
        axis.set(title=title, xlabel="Prediction origin time (s)", ylabel=ylabel)
        axis.grid(True, alpha=.25)
        if axis.lines:
            axis.legend(fontsize=7, loc="best")
        else:
            _unavailable(axis, "Saved prediction errors unavailable")
    latent_model = next((model for model in ("fusion", "history", "vision")
                         if model in loaded and "latent" in loaded[model] and "times" in loaded[model]), None)
    if latent_model:
        data = loaded[latent_model]
        latent = np.asarray(data["latent"])
        if latent.ndim == 1:
            latent = latent[:, None]
        if latent.ndim == 2 and latent.shape[1] > 0:
            length = min(len(data["times"]), len(latent))
            for coordinate in range(min(4, latent.shape[1])):
                latent_ax.plot(data["times"][:length], latent[:length, coordinate], label=f"Coordinate {coordinate+1}", lw=1.2)
            latent_ax.legend(fontsize=8, loc="best")
    latent_ax.set(title=f"{_LABELS.get(latent_model, 'Learned')} latent coordinates (first four)", xlabel="Time (s)", ylabel="Latent coordinate (arbitrary units)")
    latent_ax.grid(True, alpha=.25)
    if not latent_ax.lines:
        _unavailable(latent_ax, "Saved hidden-state coordinates unavailable")
    event_time = _event_time(episode) if episode else None
    if event_time is not None:
        for axis in (speed_ax, yaw_ax, latent_ax):
            axis.axvline(event_time, ls=":", color="0.4", lw=1, label="Event: evaluation only")
    note_ax.set_axis_off()
    note_ax.set_title("How to interpret these traces")
    note_ax.text(.02, .95,
                 "The episode is separate from training episodes.\n"
                 "Weights remain fixed during evaluation;\n"
                 "the recurrent hidden state updates online.\n\n"
                 "Latent coordinates are learned internal values,\n"
                 "not identified physical friction parameters.\n\n"
                 "Errors use prediction-origin timestamps.\n"
                 "A 0.5 s rollout can cross a later event.\n"
                 "Missing future targets are not plotted as zero.\n\n"
                 "The event marker is read only for scoring;\n"
                 "the model does not receive the event time.",
                 transform=note_ax.transAxes, va="top", fontsize=10, linespacing=1.4)
    try:
        figure.savefig(output/"online_adaptation.png", dpi=150, facecolor="white")
    finally:
        plt.close(figure)


def render_report(data_root, output):
    """Render two PNGs solely from saved metrics, training logs and NPZ arrays."""
    import matplotlib.pyplot as plt

    data_root, output = Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(data_root/"manifest.json")
    report = _load_json(output/"metrics.json")
    episode = _choose_episode(manifest)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    loss_ax, test_speed_ax, test_yaw_ax, ood_ax, camera_ax, protocol_ax = axes.flat
    figure.suptitle("Learning dynamics prediction: validation and independent episode evaluation", fontsize=14)
    _plot_loss(loss_ax, output/"train_history.csv")
    _plot_bars(test_speed_ax, report, "test", "half_speed_rmse", "TEST: 0.5 s speed prediction RMSE", "Speed RMSE (m/s)")
    _plot_bars(test_yaw_ax, report, "test", "half_yaw_rmse", "TEST: 0.5 s yaw-rate prediction RMSE", "Yaw-rate RMSE (rad/s)")
    _plot_bars(ood_ax, report, "ood", "half_speed_rmse", "OOD: 0.5 s speed prediction RMSE", "Speed RMSE (m/s)")
    _plot_camera(camera_ax, data_root, episode)
    protocol_ax.set_axis_off()
    protocol_ax.set_title("Evaluation protocol and scope")
    protocol_ax.text(.02, .96,
                     "Entire episodes are split for train / validation / test.\n"
                     "Reported test and OOD scores use saved predictions.\n"
                     "Targets are scored at the saved future horizons.\n\n"
                     "Learned weights stay fixed during evaluation.\n"
                     "History updates the recurrent hidden state online.\n"
                     "Reset = reset history; shuffled = shuffled images.\n"
                     "Both are ablations of the same fusion model.\n\n"
                     "RGB depicts procedural appearance, not a direct\n"
                     "measurement of physical friction or material.\n"
                     "Wheel projections check image/contact alignment.\n\n"
                     "Missing files or fields are marked unavailable.\n"
                     "No values are fabricated to fill missing results.",
                     transform=protocol_ax.transAxes, va="top", fontsize=10, linespacing=1.35)
    try:
        figure.savefig(output/"training_dashboard.png", dpi=150, facecolor="white")
    finally:
        plt.close(figure)
    _online_report(data_root, output, episode)
