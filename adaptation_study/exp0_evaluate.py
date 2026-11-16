"""Experiment 0, step 3: score every predictor on the held-out episodes, with episode-level statistics.

Predictors: the fixed nominal model; the explicit two-gain estimator (five window settings,
the adaptive one fixed in advance as primary); the motion-history GRU of every training
seed; and the checkpoint stored with the learned stage.  Scoring follows
learning.evaluate_all, as re-implemented and checked in followup_checks.

The new held-out episodes are the confirmatory set: the statements tested here were written
in followup_checks/README.md before these episodes existed.  The original 24 episodes are
reported next to them for continuity.

Needs NumPy; Matplotlib for the optional figure.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "followup_checks")]

import study  # noqa: E402
import explicit_vs_learned as base  # noqa: E402  (also puts mujoco_learned_ready on the path)
import control_core  # noqa: E402

EXPLICIT = base.PRIMARY
CONDITIONS = [("test", "normal"), ("test", "low_friction"), ("test", "medium_friction"), ("test", "motor_fault"),
              ("ood", "reversed"), ("ood", "severe_hidden"), ("ood", "severe_motor")]
LABELS = {"normal": "No change", "low_friction": "Low friction", "medium_friction": "Medium friction",
          "motor_fault": "Motor fault", "reversed": "Low to normal friction", "severe_hidden": "Friction 0.015",
          "severe_motor": "Motor fault to 0.25"}
CHANNELS = ("speed", "yaw")


def score_everything(manifest, arrays, predictions):
    config = manifest["config"]
    dt, horizon = float(config["dt"]), study.horizon_steps(manifest)
    physics = control_core.Physics(**config["physics"])
    noise_std = config["measurement_noise"]["std"][3:5]
    variants = base.explicit_variants(config["estimator"], dt)
    held = [(index, item) for index, item in enumerate(manifest["episodes"]) if item["split"] in study.HELD_OUT]
    if [item["id"] for _, item in held] != [str(value) for value in predictions["ids"]]:
        raise ValueError("Forecasts and dataset list different held-out episodes; rerun exp0_train.py on this dataset.")
    seeds = [int(seed) for seed in predictions["seeds"]]
    times = np.arange(arrays["commands"].shape[1]) * dt
    scores = {}

    def add(model, identifier, half, truth, event):
        scores.setdefault(model, {})[identifier] = base.score(half, truth, times, event, horizon)

    for position, (index, item) in enumerate(held):
        observed, commands, truth = arrays["observed"][index], arrays["commands"][index], arrays["truth"][index]
        event = item["event_time"]
        forecast = lambda mode, settings: base.explicit_forecasts(  # noqa: E731
            observed, commands, physics, dt, horizon, noise_std, mode, settings)
        add("fixed", item["id"], forecast("fixed", variants[EXPLICIT][1]), truth, event)
        for name, (mode, settings) in variants.items():
            add(name, item["id"], forecast(mode, settings), truth, event)
        for number, seed in enumerate(seeds):
            add(f"gru_seed{seed}", item["id"], predictions["half"][number, position], truth, event)
        if "stored_half" in predictions:
            add("gru_stored", item["id"], predictions["stored_half"][position], truth, event)
    # 潜在状态贴近 tanh 上下限的比例：变化 1.5 秒之后，各种子取平均。
    saturation = {}
    for position, (_, item) in enumerate(held):
        if item["event_time"] is not None:
            settled = times >= item["event_time"] + base.SETTLE_SECONDS
            share = np.mean(np.abs(predictions["latent"][:, position][:, settled]) > base.SATURATION, axis=(1, 2))
            saturation[item["id"]] = {"mean": float(share.mean()), "min": float(share.min()), "max": float(share.max())}
    return [item for _, item in held], seeds, scores, saturation, list(variants)


def part(scores, model, identifier, name):
    return scores[model][identifier].get(name)


def gru_mean(scores, seeds, identifier, name):
    values = [part(scores, f"gru_seed{seed}", identifier, name) for seed in seeds]
    return None if values[0] is None else np.mean(values, axis=0)


def ratio_summary(numerator, denominator):
    """Geometric-mean ratio over episodes with an interval from resampling episodes; arrays are (episode, channel)."""
    logs = np.log(np.asarray(numerator) / np.asarray(denominator))
    result = {}
    for channel, name in enumerate(CHANNELS):
        low, high = study.bootstrap_interval(logs[:, channel], np.mean)
        result[name] = {"ratio": float(np.exp(logs[:, channel].mean())), "interval": [float(np.exp(low)), float(np.exp(high))],
                        "episodes_with_lower_numerator": int(np.sum(logs[:, channel] < 0)), "episodes": int(len(logs))}
    return result


def condition_summary(items, seeds, scores, parts):
    identifiers = [item["id"] for item in items]
    summary = {"episodes": len(identifiers), "errors": {}, "gru_over_explicit": {}}
    for name in parts:
        rows = {}
        for model in ["fixed", EXPLICIT] + (["gru_stored"] if "gru_stored" in scores else []):
            rows[model] = np.mean([part(scores, model, key, name) for key in identifiers], axis=0).tolist()
        per_seed = np.array([np.mean([part(scores, f"gru_seed{seed}", key, name) for key in identifiers], axis=0)
                             for seed in seeds])
        rows["gru"] = {"mean": per_seed.mean(axis=0).tolist(), "sd_across_seeds": per_seed.std(axis=0).tolist(),
                       "min": per_seed.min(axis=0).tolist(), "max": per_seed.max(axis=0).tolist()}
        summary["errors"][name] = rows
        explicit = [part(scores, EXPLICIT, key, name) for key in identifiers]
        comparison = ratio_summary([gru_mean(scores, seeds, key, name) for key in identifiers], explicit)
        seed_ratios = np.array([np.exp(np.mean(np.log(np.array([part(scores, f"gru_seed{seed}", key, name)
                                                                 for key in identifiers]) / np.array(explicit)), axis=0))
                                for seed in seeds])
        for channel, channel_name in enumerate(CHANNELS):
            comparison[channel_name]["range_across_seeds"] = [float(seed_ratios[:, channel].min()),
                                                              float(seed_ratios[:, channel].max())]
        summary["gru_over_explicit"][name] = comparison
        if name == "all":
            fixed = [part(scores, "fixed", key, name) for key in identifiers]
            summary["over_fixed"] = {"explicit": ratio_summary(explicit, fixed),
                                     "gru": ratio_summary([gru_mean(scores, seeds, key, name) for key in identifiers], fixed)}
    return summary


def verdict(interval, expected_below_one, threshold=1.0):
    low, high = interval
    if expected_below_one:
        return "holds" if high < threshold else "opposite" if low > threshold else "inconclusive"
    return "holds" if low > threshold else "opposite" if high < threshold else "inconclusive"


def expectations(new, saturation_by_group):
    """Statements made in followup_checks/README.md from the original 24 episodes, checked on the new ones."""
    def ratio(group, name, channel):
        return new[group]["gru_over_explicit"][name][channel]

    def entry(statement, measure, value, expected_below_one, threshold=1.0):
        return {"statement": statement, "measure": measure, "ratio": value["ratio"], "interval": value["interval"],
                "range_across_seeds": value.get("range_across_seeds"),
                "verdict": verdict(value["interval"], expected_below_one, threshold)}

    measure = "GRU error / explicit error, geometric mean over episodes"
    rows = [
        entry("After a motor fault the explicit estimator settles at a lower yaw-rate error than the GRU",
              measure + ", motor fault, later than 1.5 s, yaw rate", ratio("motor_fault", "late", "yaw"), False),
        entry("For the fault to 0.25, outside the training faults, the gap is at least tenfold",
              measure + ", motor fault to 0.25, later than 1.5 s, yaw rate", ratio("severe_motor", "late", "yaw"), False, 10.0),
        entry("In the first 1.5 s after a motor fault the GRU has the lower yaw-rate error",
              measure + ", motor fault, first 1.5 s, yaw rate", ratio("motor_fault", "early", "yaw"), True),
        entry("On low friction the GRU has the lower speed error",
              measure + ", low friction, later than 1.5 s, speed", ratio("low_friction", "late", "speed"), True),
        entry("At friction 0.015 the explicit estimator settles at the lower speed error",
              measure + ", friction 0.015, later than 1.5 s, speed", ratio("severe_hidden", "late", "speed"), False),
        entry("At friction 0.015 the GRU settles at the lower yaw-rate error",
              measure + ", friction 0.015, later than 1.5 s, yaw rate", ratio("severe_hidden", "late", "yaw"), True),
        entry("Without any change the explicit estimator is worse than the fixed model in yaw rate",
              "explicit error / fixed error, no change, all origins, yaw rate", new["normal"]["over_fixed"]["explicit"]["yaw"], False),
        entry("Without any change the GRU is worse than the explicit estimator in yaw rate",
              measure + ", no change, all origins, yaw rate", ratio("normal", "all", "yaw"), False),
    ]
    severe, usual = saturation_by_group["severe_motor"], saturation_by_group["motor_fault"]
    rows.append({"statement": "The GRU latent state sits at its limits after the fault to 0.25 but not after faults like the training ones",
                 "measure": "share of latent values beyond +-0.95 later than 1.5 s, mean over episodes and seeds",
                 "motor_fault_to_0.25": severe, "motor_fault": usual,
                 # 判据：严重故障下超过四分之一，且至少是测试集故障的三倍。
                 "verdict": "holds" if severe > 0.25 and usual < severe / 3 else "does not hold"})
    return rows


def build_summary(manifest, items, seeds, scores, saturation, variant_names, training):
    train_faults = sorted(item["motor_efficiency"] for item in manifest["episodes"]
                          if item["split"] == "train" and item["group"] == "motor_fault")
    summary = {"protocol": {
        "confirmatory_set": "new held-out episodes (set = new); statements fixed beforehand in followup_checks/README.md",
        "primary_explicit_variant": EXPLICIT, "explicit_variants": variant_names, "gru_seeds": seeds,
        "errors": "mean over episodes of the per-episode 0.5 s forecast RMSE, [speed m/s, yaw rate rad/s]",
        "parts": {"all": "every scored origin", "early": f"first {base.SETTLE_SECONDS:g} s after the change",
                  "late": f"later than {base.SETTLE_SECONDS:g} s after the change"},
        "ratios": "geometric mean over episodes; interval = 95% percentile bootstrap over episodes, GRU averaged over seeds; "
                  "range_across_seeds = the same ratio for each training seed separately",
        "training_motor_efficiencies": train_faults,
        "mujoco_version": manifest["mujoco_version"], "reference_check": manifest["reference_check"]},
        "training": training, "sets": {}, "continuity": {}, "motor_faults": [], "sensitivity": {}}
    for name in ("new", "original"):
        groups = {}
        for split, group in CONDITIONS:
            chosen = [item for item in items if item["set"] == name and item["split"] == split and item["group"] == group]
            if chosen:
                groups[group] = condition_summary(chosen, seeds, scores, ("all",) if group == "normal" else ("all", "early", "late"))
        summary["sets"][name] = groups
    # 与学习阶段 README 里的汇总数字对照：原有 12+12 个回合，所有起点合并的 RMSE。
    for split in study.HELD_OUT:
        identifiers = [item["id"] for item in items if item["set"] == "original" and item["split"] == split]
        pooled = lambda model: base.pooled(scores, model, identifiers).tolist()  # noqa: E731
        per_seed = np.array([pooled(f"gru_seed{seed}") for seed in seeds])
        summary["continuity"][split] = {
            "episodes": len(identifiers), "fixed": pooled("fixed"), "explicit": pooled(EXPLICIT),
            "gru_stored_40_epochs": pooled("gru_stored") if "gru_stored" in scores else None,
            "gru_converged": {"mean": per_seed.mean(axis=0).tolist(), "sd_across_seeds": per_seed.std(axis=0).tolist(),
                              "min": per_seed.min(axis=0).tolist(), "max": per_seed.max(axis=0).tolist()}}
    for item in sorted(items, key=lambda item: (-item["motor_efficiency"], item["id"])):
        if item["group"] not in ("motor_fault", "severe_motor"):
            continue
        late = lambda model: float(part(scores, model, item["id"], "late")[1])  # noqa: E731
        per_seed = [late(f"gru_seed{seed}") for seed in seeds]
        summary["motor_faults"].append({
            "id": item["id"], "set": item["set"], "split": item["split"], "left_motor_efficiency": item["motor_efficiency"],
            "within_training_faults": bool(train_faults[0] <= item["motor_efficiency"] <= train_faults[-1]),
            "settled_yaw_rmse": {"fixed": late("fixed"), "explicit": late(EXPLICIT),
                                 "gru_mean": float(np.mean(per_seed)), "gru_min": float(min(per_seed)), "gru_max": float(max(per_seed)),
                                 "gru_stored": late("gru_stored") if "gru_stored" in scores else None},
            "gru_latent_share_at_limit": saturation[item["id"]]})
    new_items = [item for item in items if item["set"] == "new"]
    by_group = {group: float(np.mean([saturation[item["id"]]["mean"] for item in new_items if item["group"] == group]))
                for group in ("motor_fault", "severe_motor")}
    summary["expectations"] = expectations(summary["sets"]["new"], by_group)
    # 显式估计器五组窗口设置的敏感性：新回合上稳定后的误差。
    for group in ("motor_fault", "severe_motor", "low_friction", "severe_hidden"):
        identifiers = [item["id"] for item in new_items if item["group"] == group]
        summary["sensitivity"][group] = {name: np.mean([part(scores, name, key, "late") for key in identifiers], axis=0).tolist()
                                         for name in variant_names}
    return summary


def print_report(summary):
    pair = lambda value: f"{value[0]:.4f}/{value[1]:.4f}"  # noqa: E731
    protocol, training = summary["protocol"], summary["training"]
    check = protocol["reference_check"]
    print(f"Data: MuJoCo {protocol['mujoco_version']}; regenerated held-out truth matches the stored learned stage for "
          f"{check['held_out_episodes_matching']} of {check['held_out_episodes_compared']} episodes.")
    print(f"GRU: {len(protocol['gru_seeds'])} training seeds; best epoch " +
          ", ".join(str(row["best_epoch"]) for row in training["seeds"]) +
          f" (limit {training['protocol']['max_epochs']}); stopped by patience: "
          f"{sum(row['stopped_by_patience'] for row in training['seeds'])} of {len(training['seeds'])}.")
    print("\nContinuity with the learned-stage README (original episodes, all origins pooled, speed/yaw RMSE):")
    for split, row in summary["continuity"].items():
        converged = row["gru_converged"]
        print(f"  {split:<5} fixed {pair(row['fixed'])} | explicit {pair(row['explicit'])} | GRU stored (40 epochs) "
              f"{pair(row['gru_stored_40_epochs']) if row['gru_stored_40_epochs'] else '-'} | GRU converged {pair(converged['mean'])} "
              f"(seed range {pair(converged['min'])} to {pair(converged['max'])})")
    for name, title in [("new", "New held-out episodes (confirmatory)"), ("original", "Original 24 held-out episodes")]:
        print(f"\n{title}: mean per-episode RMSE, speed/yaw; GRU = mean over seeds (sd across seeds)")
        for group, entry in summary["sets"][name].items():
            print(f"-- {LABELS[group]} ({entry['episodes']} episodes)")
            for part_name, rows in entry["errors"].items():
                gru = rows["gru"]
                print(f"   {part_name:<6} fixed {pair(rows['fixed'])} | explicit {pair(rows[EXPLICIT])} | "
                      f"GRU {pair(gru['mean'])} ({gru['sd_across_seeds'][0]:.4f}/{gru['sd_across_seeds'][1]:.4f})")
            for part_name, comparison in entry["gru_over_explicit"].items():
                cells = "  ".join(f"{channel} {value['ratio']:.2f} [{value['interval'][0]:.2f}, {value['interval'][1]:.2f}]"
                                  f" seeds {value['range_across_seeds'][0]:.2f}-{value['range_across_seeds'][1]:.2f}"
                                  for channel, value in comparison.items())
                print(f"   GRU/explicit, {part_name:<6} {cells}")
    print("\nStatements from followup_checks/README.md, checked on the new episodes:")
    for row in summary["expectations"]:
        if "ratio" in row:
            print(f"  [{row['verdict']:<12}] {row['statement']}: ratio {row['ratio']:.2f} "
                  f"[{row['interval'][0]:.2f}, {row['interval'][1]:.2f}]")
        else:
            print(f"  [{row['verdict']:<12}] {row['statement']}: {row['motor_fault_to_0.25']:.2f} vs {row['motor_fault']:.2f}")
    faults = protocol["training_motor_efficiencies"]
    print(f"\nMotor faults, settled yaw-rate RMSE [rad/s]; training faults had efficiency {faults[0]:.2f} to {faults[-1]:.2f}:")
    print(f"  {'eff.':>5} {'episode':<9} {'set':<9} {'in range':<8} {'fixed':>7} {'explicit':>8} {'GRU mean':>8} "
          f"{'GRU min-max':>15} {'stored':>7} {'latent':>6}")
    for row in summary["motor_faults"]:
        error = row["settled_yaw_rmse"]
        stored = f"{error['gru_stored']:.4f}" if error["gru_stored"] is not None else "-"
        print(f"  {row['left_motor_efficiency']:>5.3f} {row['id']:<9} {row['set']:<9} "
              f"{'yes' if row['within_training_faults'] else 'no':<8} {error['fixed']:>7.4f} {error['explicit']:>8.4f} "
              f"{error['gru_mean']:>8.4f} {error['gru_min']:>7.4f}-{error['gru_max']:<7.4f} {stored:>7} "
              f"{row['gru_latent_share_at_limit']['mean']:>6.2f}")


def save_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import plot_style as style

    with plt.rc_context(style.RC):
        figure = plt.figure(figsize=(11.6, 4.9))
        # 左图与右边两图之间要给场景名留出位置；右边两图共用场景名，挨在一起。
        outer = figure.add_gridspec(1, 2, width_ratios=[1.25, 2.0], wspace=0.46,
                                    left=0.07, right=0.985, top=0.86, bottom=0.225)
        inner = outer[1].subgridspec(1, 2, wspace=0.07)
        scatter, early, late = figure.add_subplot(outer[0]), figure.add_subplot(inner[0]), figure.add_subplot(inner[1])

        # 左：电机故障回合稳定后的角速度误差，按故障后的电机效率排开。
        faults = summary["protocol"]["training_motor_efficiencies"]
        scatter.axvspan(faults[0], faults[-1], color=style.GRID, alpha=0.6, linewidth=0, zorder=0)
        scatter.annotate("training faults", ((faults[0] + faults[-1]) / 2, 0.985), xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=9, color=style.INK_SECONDARY)
        rows = summary["motor_faults"]
        efficiency = [row["left_motor_efficiency"] for row in rows]
        for row in rows:
            error = row["settled_yaw_rmse"]
            scatter.plot([row["left_motor_efficiency"]] * 2, [error["gru_min"], error["gru_max"]], color=style.AQUA,
                         linewidth=1.2, zorder=2)
        for key, color in [("fixed", style.CONTEXT), ("gru_mean", style.AQUA), ("explicit", style.BLUE)]:
            scatter.plot(efficiency, [row["settled_yaw_rmse"][key] for row in rows], color=color, marker="o",
                         linestyle="none", zorder=3, **dict(style.MARKER, markersize=7))
        style.plain_log_axis(scatter, "y", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        scatter.set_ylim(0.0008, 0.6)
        scatter.set_xlim(0.2, 0.8)
        scatter.set_xlabel("Left-motor efficiency after the fault")
        scatter.set_ylabel("0.5 s yaw-rate forecast RMSE, later than 1.5 s (rad/s)")
        scatter.set_title(f"Motor faults ({len(rows)} episodes)", pad=24)
        style.style_axes(scatter, "y")

        # 中、右：新回合上 GRU 误差与显式估计器误差之比，1 表示相同。
        groups = [group for _, group in CONDITIONS if group != "normal"]
        new = summary["sets"]["new"]
        for axis, name, title in [(early, "early", "First 1.5 s after the change"), (late, "late", "Later than 1.5 s")]:
            axis.axvline(1.0, color=style.AXIS, linewidth=0.8, zorder=1)
            for index, group in enumerate(groups):
                y = len(groups) - 1 - index
                for channel, marker, offset in [("speed", "o", 0.16), ("yaw", "D", -0.16)]:
                    value = new[group]["gru_over_explicit"][name][channel]
                    axis.plot(value["interval"], [y + offset] * 2, color=style.INK_SECONDARY, linewidth=1.4, zorder=2)
                    axis.plot(value["ratio"], y + offset, color=style.INK_SECONDARY, marker=marker, linestyle="none",
                              zorder=3, **dict(style.MARKER, markersize=7))
            style.plain_log_axis(axis, "x", [0.1, 0.3, 1, 3, 10, 30, 100])
            axis.set_xlim(0.08, 130)
            axis.set_ylim(-0.6, len(groups) - 0.4)
            axis.set_yticks(range(len(groups)))
            axis.set_yticklabels([LABELS[group] for group in reversed(groups)] if axis is early else [])
            axis.tick_params(axis="y", length=0)
            axis.set_title(title, pad=24)
            axis.set_xlabel("GRU error / explicit error (log scale)")
            style.style_axes(axis, "x")
            axis.annotate("GRU lower", (0.0, 1.015), xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5,
                          color=style.INK_SECONDARY)
            axis.annotate("explicit lower", (1.0, 1.015), xycoords="axes fraction", ha="right", va="bottom", fontsize=8.5,
                          color=style.INK_SECONDARY)

        def handle(color, marker):
            return Line2D([0], [0], color="none", marker=marker, markerfacecolor=color, **dict(style.MARKER, markersize=7))

        figure.legend(handles=[handle(style.CONTEXT, "o"), handle(style.AQUA, "o"), handle(style.BLUE, "o"),
                               handle(style.INK_SECONDARY, "o"), handle(style.INK_SECONDARY, "D")],
                      labels=["Fixed model", "GRU, mean and range over seeds", "Explicit estimator",
                              "Ratio for speed", "Ratio for yaw rate"],
                      loc="lower center", ncol=5, handlelength=1.2, columnspacing=2.0)
        figure.savefig(path, dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=HERE / "runs" / "exp0" / "data")
    parser.add_argument("--training-dir", type=Path, default=HERE / "runs" / "exp0" / "training")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "exp0" / "evaluation")
    parser.add_argument("--no-plot", action="store_true", help="skip the Matplotlib figure")
    args = parser.parse_args()
    manifest, arrays = study.load_dataset(args.data_dir)
    with np.load(args.training_dir / "predictions.npz", allow_pickle=False) as source:
        predictions = {key: np.asarray(source[key]) for key in source.files}
    training = study.read_json(args.training_dir / "training_report.json")
    items, seeds, scores, saturation, variant_names = score_everything(manifest, arrays, predictions)
    summary = build_summary(manifest, items, seeds, scores, saturation, variant_names, training)
    print_report(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study.write_json(args.output_dir / "exp0_summary.json", summary)
    if not args.no_plot:
        save_figure(summary, args.output_dir / "exp0_summary.png")
    print(f"\nResults saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
