"""Explicit two-parameter estimator vs. the learned GRU on the same held-out logs.

The learned stage (mujoco_learned_ready) scores its neural predictors against the fixed
nominal model only.  This script replays the 24 held-out episodes stored with the
repository through control_core.WheelEstimator and scores its 0.5 s forecasts with the
rule of learning.evaluate_all, so both kinds of adaptation are compared on identical
observations, commands and forecast origins.  Nothing is retrained, and neither MuJoCo
nor PyTorch is needed: the neural forecasts are read from results/trained/predictions.

Reported: pooled test/OOD errors; errors per condition, split into the first 1.5 s
after the change and afterwards; and a per-episode table for the motor-fault cases,
with the share of the GRU's tanh-bounded latent state that sits at its limits.

Only NumPy is required; Matplotlib is used for the optional figure.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEARNED = REPO / "mujoco_learned_ready"
sys.path[:0] = [str(HERE), str(LEARNED)]

import control_core  # noqa: E402

HORIZON_SECONDS = 0.5
BURN_IN_STEPS = 10  # 与 learning.BURN_IN 相同：评分跳过的冷启动步数
SETTLE_SECONDS = 1.5  # 变化之后的预测分成"头 1.5 秒"和"之后"两段
SATURATION = 0.95  # 潜在状态的绝对值超过它，视为贴在 tanh 的上下限上
SAVED_MODELS = ["fixed", "history", "vision", "fusion"]
PRIMARY = "explicit_adaptive"
CHANNELS = ["speed", "yaw"]
# 图中的四组回合：(键, 划分, 场景, 评分通道, 标题)
GROUPS = [
    ("motor_test", "test", ["motor_fault"], 1, "Motor fault, test split"),
    ("motor_ood", "ood", ["severe_motor"], 1, "Motor fault, OOD split (more severe)"),
    ("friction_test", "test", ["correlated", "hidden_friction"], 0, "Low friction, test split"),
    ("friction_ood", "ood", ["severe_hidden"], 0, "Low friction, OOD split (more severe)"),
]
BIN_EDGES = np.arange(-2.0, 5.0 + 1e-9, 0.5)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def explicit_variants(estimator_settings, dt):
    """Window lengths of the 50 Hz stages (0.8 s to 3.2 s, alarm after 0.1 s) re-expressed in steps of ``dt``."""
    steps = lambda seconds: max(1, round(seconds / dt))  # noqa: E731
    common = dict(estimator_settings, min_window=steps(0.8), max_window=steps(3.2), trigger_updates=steps(0.1))
    return {
        PRIMARY: ("adaptive", common),
        "explicit_adaptive_alarm_0.2s": ("adaptive", dict(common, trigger_updates=steps(0.2))),
        "explicit_window_0.8s": (f"window_{steps(0.8)}", common),
        "explicit_window_1.6s": (f"window_{steps(1.6)}", common),
        "explicit_window_3.2s": (f"window_{steps(3.2)}", common),
    }


def explicit_forecasts(observed, commands, physics, dt, horizon, noise_std, mode, settings):
    """Causal replay: forecast from each origin with the gain known then, afterwards reveal the next observation."""
    estimator = control_core.WheelEstimator(mode, physics, dt, settings, noise_std)
    count = len(commands)
    half = np.full((count, 2), np.nan)
    for origin in range(count):
        gain = estimator.used().copy()
        if origin + horizon <= count:
            speed, yaw = float(observed[origin, 0]), float(observed[origin, 1])
            for step in range(horizon):
                left, right = gain * commands[origin + step]
                speed, yaw = (speed + dt * (left + right - physics.linear_drag * speed) / physics.mass,
                              yaw + dt * (physics.half_track * (right - left) - physics.yaw_drag * yaw) / physics.inertia)
            half[origin] = speed, yaw
        if origin + 1 < len(observed):
            estimator.update(observed[origin, :2], commands[origin], observed[origin + 1, :2])
    return half


def score(half, truth, times, event, horizon):
    """Same origins and truth as learning.evaluate_all, plus the split around SETTLE_SECONDS."""
    valid = len(half) - horizon + 1
    error = half[:valid, :2] - truth[horizon:, :2]
    start = min(BURN_IN_STEPS, valid - 1)
    origin_times = times[:valid]
    result = {"sum_squared": np.sum(error[start:] ** 2, axis=0), "count": valid - start,
              "all": np.sqrt(np.mean(error[start:] ** 2, axis=0)), "error": error,
              "scored": origin_times >= times[start]}
    if event is not None:
        post = origin_times >= max(float(event), float(times[start]))
        for name, mask in [("post", post), ("early", post & (origin_times < event + SETTLE_SECONDS)),
                           ("late", post & (origin_times >= event + SETTLE_SECONDS)),
                           ("before", result["scored"] & (origin_times + HORIZON_SECONDS < event))]:
            if mask.any():
                result[name] = np.sqrt(np.mean(error[mask] ** 2, axis=0))
    return result


def evaluate():
    manifest = json.loads((LEARNED / "demo_assets" / "manifest.json").read_text(encoding="utf-8"))
    reported = json.loads((LEARNED / "results" / "trained" / "metrics.json").read_text(encoding="utf-8"))
    reported_rows = {(row["id"], row["model"]): row for row in reported["per_episode"]}
    config = manifest["config"]
    dt = float(config["dt"])
    horizon = round(HORIZON_SECONDS / dt)
    physics = control_core.Physics(**config["physics"])
    noise_std = config["measurement_noise"]["std"][3:5]
    variants = explicit_variants(config["estimator"], dt)
    episodes = [item for item in manifest["episodes"] if item["split"] in ("test", "ood")]
    scores = {name: {} for name in SAVED_MODELS + list(variants)}
    latent_share, checks = {}, {"fixed_rollout_max_difference": 0.0, "metrics_json_max_difference": 0.0}
    for item in episodes:
        directory = LEARNED / "results" / "trained" / "predictions" / item["id"]
        saved = {}
        for name in SAVED_MODELS:
            with np.load(directory / f"{name}.npz", allow_pickle=False) as source:
                saved[name] = {key: np.asarray(source[key]) for key in source.files}
        log = saved["fixed"]
        observed, commands = log["observed"].astype(float), log["commands"].astype(float)
        truth, times, event = log["truth"], log["times"], item["event_time"]
        valid = len(commands) - horizon + 1
        # 核对一：这里的名义模型展开，应当重现仓库保存的固定模型预测。
        nominal = explicit_forecasts(observed, commands, physics, dt, horizon, noise_std, "fixed", variants[PRIMARY][1])
        checks["fixed_rollout_max_difference"] = max(checks["fixed_rollout_max_difference"],
                                                     float(np.max(np.abs(nominal[:valid] - log["half"][:valid, :2]))))
        for name in SAVED_MODELS:
            result = score(saved[name]["half"], truth, times, event, horizon)
            scores[name][item["id"]] = result
            # 核对二：评分应当与 metrics.json 逐回合一致，包括变化之后的部分（它依赖清单里的事件时刻）。
            row = reported_rows[(item["id"], name)]
            if row["split"] != item["split"] or row["condition"] != item["condition"]:
                raise AssertionError(f"Manifest and metrics.json disagree about episode {item['id']}")
            pairs = [(result["all"], (row["half_speed_rmse"], row["half_yaw_rmse"]))]
            if "half_speed_rmse_post" in row:
                pairs.append((result["post"], (row["half_speed_rmse_post"], row["half_yaw_rmse_post"])))
            for mine, theirs in pairs:
                checks["metrics_json_max_difference"] = max(checks["metrics_json_max_difference"],
                                                            float(np.max(np.abs(mine - np.asarray(theirs)))))
        for name, (mode, settings) in variants.items():
            half = explicit_forecasts(observed, commands, physics, dt, horizon, noise_std, mode, settings)
            scores[name][item["id"]] = score(half, truth, times, event, horizon)
        latent = saved["history"]["latent"]
        if event is not None:
            settled, earlier = latent[times >= event + SETTLE_SECONDS], latent[(times >= times[BURN_IN_STEPS]) & (times < event)]
            latent_share[item["id"]] = {"after": float(np.mean(np.abs(settled) > SATURATION)),
                                        "before": float(np.mean(np.abs(earlier) > SATURATION))}
    if checks["fixed_rollout_max_difference"] > 1e-5 or checks["metrics_json_max_difference"] > 1e-9:
        raise AssertionError(f"Replay does not reproduce the stored results: {checks}")
    return manifest, episodes, scores, variants, latent_share, checks, dt


def pooled(scores, model, identifiers):
    total = sum(scores[model][key]["sum_squared"] for key in identifiers)
    count = sum(scores[model][key]["count"] for key in identifiers)
    return np.sqrt(total / count)


def episode_mean(scores, model, identifiers, part):
    values = [scores[model][key][part] for key in identifiers if part in scores[model][key]]
    return np.mean(values, axis=0) if values else None


def group_curve(scores, model, items, channel, dt):
    """RMS error in half-second bins of forecast start time relative to the change, pooled over the group."""
    relative, squared = [], []
    for item in items:
        result = scores[model][item["id"]]
        times = np.arange(len(result["error"])) * dt
        keep = result["scored"]
        relative.append(times[keep] - item["event_time"])
        squared.append(result["error"][keep, channel] ** 2)
    relative, squared = np.concatenate(relative), np.concatenate(squared)
    index = np.digitize(relative, BIN_EDGES) - 1
    return [float(np.sqrt(np.mean(squared[index == position]))) if np.any(index == position) else None
            for position in range(len(BIN_EDGES) - 1)]


def build_summary(manifest, episodes, scores, variants, latent_share, checks, dt):
    models = SAVED_MODELS + list(variants)
    pair = lambda value: None if value is None else [float(value[0]), float(value[1])]  # noqa: E731
    # 训练集中实际出现过的取值（不是采样区间）：故障回合很少，测试值可能已落在它们之外。
    train = [item for item in manifest["episodes"] if item["split"] == "train"]
    efficiency = sorted(item["motor_efficiency"] for item in train if item["condition"] == "motor_fault")
    friction = [item["after_mu"] for item in train if item["after_mu"] < item["before_mu"]]
    summary = {
        "protocol": {"source": "mujoco_learned_ready/results/trained/predictions and demo_assets/manifest.json",
                     "episodes": len(episodes), "dt": dt, "forecast_seconds": HORIZON_SECONDS,
                     "scoring": "learning.evaluate_all re-implemented; skips the first 10 origins of every episode",
                     "units": "speed RMSE in m/s, yaw-rate RMSE in rad/s, listed as [speed, yaw]",
                     "explicit_inputs": "body speed, yaw rate and nominal commands; no wheel rates, no RGB",
                     "explicit_settings": {name: {"mode": mode, "min_window": settings["min_window"],
                                                  "max_window": settings["max_window"],
                                                  "trigger_updates": settings["trigger_updates"]}
                                           for name, (mode, settings) in variants.items()},
                     "primary_explicit_variant": PRIMARY, "settle_seconds": SETTLE_SECONDS,
                     "per_condition_parts": "all = pooled over origins; post/early/late = mean of per-episode RMSE",
                     "training_motor_efficiencies": efficiency,
                     "training_lowered_friction_range": [min(friction), max(friction)]},
        "checks": checks, "pooled": {}, "per_condition": [], "motor_fault_episodes": [], "curves": {},
    }
    for split in ("test", "ood"):
        identifiers = [item["id"] for item in episodes if item["split"] == split]
        summary["pooled"][split] = {model: pair(pooled(scores, model, identifiers)) for model in models}
    conditions = list(dict.fromkeys((item["split"], item["condition"]) for item in episodes))
    for split, condition in conditions:
        identifiers = [item["id"] for item in episodes if (item["split"], item["condition"]) == (split, condition)]
        rows = {model: {"all": pair(pooled(scores, model, identifiers)),
                        **{part: pair(episode_mean(scores, model, identifiers, part)) for part in ("post", "early", "late")}}
                for model in models}
        summary["per_condition"].append({"split": split, "condition": condition, "episodes": len(identifiers), "models": rows})
    for item in sorted(episodes, key=lambda item: -item["motor_efficiency"]):
        if item["condition"] not in ("motor_fault", "severe_motor"):
            continue
        late = lambda model: float(scores[model][item["id"]]["late"][1])  # noqa: E731
        summary["motor_fault_episodes"].append({
            "id": item["id"], "split": item["split"], "left_motor_efficiency": item["motor_efficiency"],
            "settled_yaw_rmse": {model: late(model) for model in ["fixed", "history", "fusion", PRIMARY]},
            "yaw_rmse_before_fault": {model: float(scores[model][item["id"]]["before"][1]) for model in ["history", PRIMARY]},
            "gru_latent_share_at_limit": latent_share[item["id"]]})
    for key, split, conditions_in_group, channel, title in GROUPS:
        items = [item for item in episodes if item["split"] == split and item["condition"] in conditions_in_group]
        summary["curves"][key] = {"title": title, "episodes": len(items), "channel": CHANNELS[channel],
                                  "bin_centers": ((BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2).tolist(),
                                  "models": {model: group_curve(scores, model, items, channel, dt)
                                             for model in ["fixed", "history", PRIMARY]}}
    return summary


def print_report(summary):
    cell = lambda value: "       -       " if value is None else f"{value[0]:.4f}/{value[1]:.4f}"  # noqa: E731
    checks, protocol = summary["checks"], summary["protocol"]
    print(f"Replay check: nominal rollout differs from the stored one by at most {checks['fixed_rollout_max_difference']:.1e}; "
          f"scores differ from metrics.json by at most {checks['metrics_json_max_difference']:.1e}.")
    print("\n0.5 s forecast RMSE, speed [m/s] / yaw rate [rad/s], pooled over every scored origin")
    for split, rows in summary["pooled"].items():
        for model, value in rows.items():
            print(f"  {split:<5}{model:<30}{cell(value)}")
    print(f"\nPer condition: all origins | after the change | first {SETTLE_SECONDS:g} s after it | later")
    for entry in summary["per_condition"]:
        print(f"-- {entry['split']} / {entry['condition']} ({entry['episodes']} episodes)")
        for model, parts in entry["models"].items():
            print(f"  {model:<30}{cell(parts['all'])} | {cell(parts['post'])} | {cell(parts['early'])} | {cell(parts['late'])}")
    seen = ", ".join(f"{value:.2f}" for value in protocol["training_motor_efficiencies"])
    print(f"\nMotor-fault episodes. Left-motor efficiency of the {len(protocol['training_motor_efficiencies'])} "
          f"training faults: {seen}.")
    print(f"Yaw-rate RMSE of forecasts starting at least {SETTLE_SECONDS:g} s after the fault [rad/s]; "
          f"latent share = part of the GRU latent values beyond +-{SATURATION:g}.")
    print(f"  {'efficiency':>10} {'episode':<9} {'split':<8} {'fixed':>7} {'GRU':>7} {'fusion':>7} {'explicit':>8}   "
          f"{'before fault: GRU':>17} {'explicit':>8}   {'latent share after':>18} {'before':>6}")
    for row in summary["motor_fault_episodes"]:
        settled, before, share = row["settled_yaw_rmse"], row["yaw_rmse_before_fault"], row["gru_latent_share_at_limit"]
        print(f"  {row['left_motor_efficiency']:>10.3f} {row['id']:<9} {row['split']:<8} "
              f"{settled['fixed']:>7.4f} {settled['history']:>7.4f} {settled['fusion']:>7.4f} {settled[PRIMARY]:>8.4f}   "
              f"{before['history']:>17.4f} {before[PRIMARY]:>8.4f}   {share['after']:>18.2f} {share['before']:>6.2f}")


def save_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import plot_style as style

    series = [("fixed", style.CONTEXT, "Fixed model"), ("history", style.AQUA, "GRU (motion history)"),
              (PRIMARY, style.BLUE, "Explicit estimator")]
    ticks = {"yaw": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], "speed": [0.003, 0.01, 0.03, 0.1]}
    limits = {"yaw": (0.0008, 0.6), "speed": (0.0025, 0.25)}
    units = {"yaw": "yaw-rate RMSE (rad/s)", "speed": "speed RMSE (m/s)"}
    with plt.rc_context(style.RC):
        figure, axes = plt.subplots(2, 2, figsize=(10.6, 6.6), sharex=True)
        for axis, (key, *_rest) in zip(axes.flat, GROUPS):
            curve = summary["curves"][key]
            centers = np.asarray(curve["bin_centers"])
            axis.axvline(0.0, color=style.AXIS, linewidth=0.8, zorder=1)
            for model, color, label in series:
                values = np.array([np.nan if value is None else value for value in curve["models"][model]])
                axis.plot(centers, values, color=color, zorder=3)
                if key.endswith("_ood"):  # 右列的三条线分得开，在线尾直接标名字
                    axis.annotate(label, (centers[-1], values[-1]), xytext=(7, 0), textcoords="offset points",
                                  ha="left", va="center", color=style.INK_SECONDARY, fontsize=9, annotation_clip=False)
            style.plain_log_axis(axis, "y", ticks[curve["channel"]])
            axis.set_ylim(*limits[curve["channel"]])
            axis.set_xlim(BIN_EDGES[0], BIN_EDGES[-1])
            axis.set_title(f"{curve['title']} ({curve['episodes']} episodes)")
            style.style_axes(axis, "y")
        for row, channel in enumerate(["yaw", "speed"]):
            axes[row, 0].set_ylabel(f"0.5 s forecast, {units[channel]}")
            axes[row, 1].tick_params(labelleft=False)
        for axis in axes[1]:
            axis.set_xlabel("Forecast start, seconds after the change")
        figure.legend(handles=[Line2D([0], [0], color=color) for _, color, _ in series], labels=[label for *_, label in series],
                      loc="lower center", ncol=3, handlelength=1.8, columnspacing=2.4)
        figure.subplots_adjust(left=0.085, right=0.865, top=0.935, bottom=0.135, wspace=0.09, hspace=0.3)
        figure.savefig(path, dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-plot", action="store_true", help="skip the Matplotlib figure")
    args = parser.parse_args()
    summary = build_summary(*evaluate())
    print_report(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "explicit_vs_learned.json", summary)
    if not args.no_plot:
        save_figure(summary, args.output_dir / "explicit_vs_learned.png")
    print(f"\nResults saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
