"""Lesson 9: verify the noise-versus-delay tradeoff of history averaging.

Keep compare_noisy_identification.py and compare_varying_inputs.py beside this file.
Run: python compare_window_lengths.py
All windows update every 0.02 s. Use --no-show for a headless run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import compare_noisy_identification as baseline
import compare_varying_inputs as inputs


WINDOWS = [40, 80, 160]  # 0.8、1.6、3.2秒；只改变历史长度。
WINDOW_METHODS = [f"window_{size}" for size in WINDOWS]
METHODS = ["fixed", "error_update"] + WINDOW_METHODS
COLORS = {"fixed": "tab:blue", "error_update": "tab:orange",
          "window_40": "tab:green", "window_80": "tab:purple", "window_160": "tab:red"}
LABELS = {"fixed": "Fixed", "error_update": "EMA (alpha=0.05)"}
LABELS.update({f"window_{size}": f"LS {size * baseline.DT:.1f} s" for size in WINDOWS})


def run_window_estimators(observations, commands):
    """只读观测和输入；每次用最近W个已完成的状态转移估计效率。"""
    steps = len(commands)
    estimates = {method: np.ones((steps + 1, 2)) for method in WINDOW_METHODS}
    sum_uy = np.zeros((steps + 1, 2))
    sum_uu = np.zeros((steps + 1, 2))

    for step in range(steps):
        current, following = observations[step], observations[step + 1]
        # y是从观测反推的车轮等效力，满足 y_i = eta_i * u_i + 观测误差。
        force_sum = (baseline.MASS / baseline.DT * (following[0] - current[0])
                     + baseline.LINEAR_DRAG * current[0])
        force_difference = (
            baseline.INERTIA / (baseline.HALF_TRACK * baseline.DT)
            * (following[1] - current[1])
            + baseline.YAW_DRAG / baseline.HALF_TRACK * current[1]
        )
        observed_forces = np.array([(force_sum - force_difference) / 2.0,
                                    (force_sum + force_difference) / 2.0])
        # y_k必须与产生这一状态转移的u_k配对。
        sum_uy[step + 1] = sum_uy[step] + commands[step] * observed_forces
        sum_uu[step + 1] = sum_uu[step] + commands[step] ** 2

        for size, method in zip(WINDOWS, WINDOW_METHODS):
            next_estimate = estimates[method][step].copy()
            if step + 1 >= size:
                start = step + 1 - size
                numerator = sum_uy[step + 1] - sum_uy[start]
                denominator = sum_uu[step + 1] - sum_uu[start]
                identifiable = denominator > 1e-10
                next_estimate[identifiable] = numerator[identifiable] / denominator[identifiable]
            # 收集满窗口前保持初值；故障时不重置，也不裁剪估计来掩盖噪声。
            estimates[method][step + 1] = next_estimate
    return estimates


def predict_log(observations, commands, estimates):
    """预测k+1时使用eta_hat[k]；eta_hat[k+1]只能用于之后的预测。"""
    forces = commands * estimates[:-1]
    predicted = np.empty_like(commands)
    predicted[:, 0] = observations[:-1, 0] + baseline.DT / baseline.MASS * (
        forces[:, 0] + forces[:, 1] - baseline.LINEAR_DRAG * observations[:-1, 0]
    )
    predicted[:, 1] = observations[:-1, 1] + baseline.DT / baseline.INERTIA * (
        baseline.HALF_TRACK * (forces[:, 1] - forces[:, 0])
        - baseline.YAW_DRAG * observations[:-1, 1]
    )
    return predicted


def score_run(times, truth, efficiencies, estimates, predictions, seed, phase):
    steady = times >= baseline.STEADY_START
    errors = estimates[steady] - efficiencies[steady]
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    post_fault = times[1:] > baseline.CHANGE_TIME
    prediction_rmse = np.sqrt(np.mean((truth[1:] - predictions)[post_fault] ** 2, axis=0))
    return {
        "seed": seed, "phase_degrees": phase,
        "left_rmse": float(rmse[0]), "right_rmse": float(rmse[1]),
        # 同一段内围绕其均值的标准差=抖动；同时报告偏差，避免把平稳偏差当成功。
        "steady_left_jitter_sd": float(np.std(errors[:, 0])),
        "steady_left_bias": float(np.mean(errors[:, 0])),
        "steady_left_in_band_fraction": float(np.mean(
            np.abs(errors[:, 0]) <= baseline.PARAMETER_TOLERANCE + 1e-12
        )),
        "next_true_speed_rmse": float(prediction_rmse[0]),
        "next_true_yaw_rmse": float(prediction_rmse[1]),
        "recovery": baseline.confirmed_left_recovery(times, estimates[:, 0]),
    }


def summarize(runs):
    record = inputs.summarize(runs)
    for metric in ["steady_left_jitter_sd", "steady_left_bias"]:
        values = [run[metric] for run in runs]
        record[metric + "_mean"] = float(np.mean(values))
        record[metric + "_sd"] = float(np.std(values))
    return record


def sample_age_statistics(times, commands, size):
    ages = np.arange(size - 1, -1, -1) * baseline.DT
    weighted_ages = []
    for end in range(size, len(commands) + 1):
        if times[end] >= baseline.STEADY_START:
            energy = commands[end - size:end] ** 2
            weighted_ages.append(np.sum(energy * ages[:, None], axis=0) / np.sum(energy, axis=0))
    weighted_ages = np.asarray(weighted_ages)
    return {"mean_seconds": np.mean(weighted_ages, axis=0).tolist(),
            "min_seconds": np.min(weighted_ages, axis=0).tolist(),
            "max_seconds": np.max(weighted_ages, axis=0).tolist()}


def run_study(seeds):
    profiles, examples = [], {}
    for configuration in inputs.CONFIGURATIONS:
        times, commands, truth, efficiencies = inputs.generate_motion(configuration)
        levels = []
        for sigma in baseline.NOISE_LEVELS:
            measurements = {method: [] for method in METHODS}
            for seed in range(seeds):
                noise = np.random.default_rng(seed).normal(size=truth.shape) * sigma
                observations = truth + noise
                reference_estimates, reference_predictions = baseline.run_estimators(observations, commands)
                estimates = {method: reference_estimates[method] for method in ["fixed", "error_update"]}
                estimates.update(run_window_estimators(observations, commands))
                # 40步实现应重现上一课；也能及时发现索引或输入对齐错误。
                if not np.allclose(estimates["window_40"], reference_estimates["window_ls"],
                                   rtol=0.0, atol=1e-12):
                    raise AssertionError("The 0.8 s window differs from the previous lesson.")
                for method in METHODS:
                    predictions = (reference_predictions[method] if method in reference_predictions
                                   else predict_log(observations, commands, estimates[method]))
                    measurements[method].append(score_run(
                        times, truth, efficiencies, estimates[method], predictions,
                        seed, configuration["phase_degrees"]
                    ))
                if (configuration["id"] == "varying_000" and seed == baseline.EXAMPLE_SEED
                        and sigma in [0.0, baseline.EXAMPLE_SIGMA]):
                    examples[sigma] = {"times": times, "estimates": estimates,
                                       "efficiencies": efficiencies}
            levels.append({"sigma_speed": sigma, "sigma_yaw": sigma,
                           "methods": {method: summarize(measurements[method]) for method in METHODS}})
        profiles.append({"configuration": configuration,
                         "weighted_sample_ages": {f"window_{size}": sample_age_statistics(times, commands, size)
                                                   for size in WINDOWS},
                         "noise_levels": levels})
        print(f"Completed {configuration['id']}: {len(baseline.NOISE_LEVELS)} noise levels x {seeds} seeds.",
              flush=True)

    aggregate = []
    for level, sigma in enumerate(baseline.NOISE_LEVELS):
        combined = {}
        for method in METHODS:
            records = [profile["noise_levels"][level]["methods"][method] for profile in profiles[1:]]
            combined[method] = summarize([run for record in records for run in record["runs"]])
            combined[method]["phase_mean_left_rmse_range"] = [
                min(record["left_rmse_mean"] for record in records),
                max(record["left_rmse_mean"] for record in records),
            ]
        aggregate.append({"sigma_speed": sigma, "sigma_yaw": sigma, "methods": combined})
    return profiles, aggregate, examples


def save_figure(profiles, aggregate, examples, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, sigma in zip(axes[0, :2], [0.0, baseline.EXAMPLE_SIGMA]):
        example = examples[sigma]
        for method in ["error_update"] + WINDOW_METHODS:
            axis.plot(example["times"], example["estimates"][method][:, 0],
                      color=COLORS[method], label=LABELS[method], linewidth=0.9)
        axis.step(example["times"], example["efficiencies"][:, 0], where="post",
                  color="black", linestyle="--", label="Actual (evaluation only)")
        axis.axvline(baseline.CHANGE_TIME, color="gray", linestyle=":")
        axis.axhspan(0.45, 0.55, color="gray", alpha=0.15)
        axis.set_xlim((4.5, 9.0) if sigma == 0 else (4.5, 20.0))
        axis.set_title(f"Phase 0, seed 0: noise std {sigma:g}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Left efficiency estimate")
        axis.legend(fontsize=7)

    example_level = baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)
    durations = np.array(WINDOWS) * baseline.DT
    for profile, label, marker in [(profiles[0]["noise_levels"], "Constant", "o"),
                                    (aggregate, "Varying: 4 phases", "s")]:
        records = [profile[example_level]["methods"][method] for method in WINDOW_METHODS]
        axes[0, 2].errorbar(durations, [r["left_rmse_mean"] for r in records],
                            yerr=[r["left_rmse_sd"] for r in records], marker=marker, capsize=3,
                            label=label)
    axes[0, 2].set_title("Steady accuracy: noise std 0.01")
    axes[0, 2].set_xlabel("Window duration (s)")
    axes[0, 2].set_xticks(durations)
    axes[0, 2].set_ylabel("Left RMSE (10-20 s), mean +/- SD")
    axes[0, 2].legend(fontsize=8)

    sigmas = baseline.NOISE_LEVELS
    for method in WINDOW_METHODS:
        records = [level["methods"][method] for level in aggregate]
        axes[1, 0].plot(sigmas, [100 * r["steady_left_in_band_fraction_mean"] for r in records],
                        color=COLORS[method], marker="o", label=LABELS[method])
        axes[1, 1].plot(sigmas, [100 * r["recovery_rate"] for r in records],
                        color=COLORS[method], marker="o", label=LABELS[method])
    for axis in axes[1, :2]:
        axis.set_xticks(sigmas)
        axis.set_xlabel("Noise std: speed (m/s), yaw rate (rad/s)")
        axis.set_ylim(-5, 105)
        axis.legend(fontsize=8)
    axes[1, 0].set_title("Varying inputs: steady in-band time")
    axes[1, 0].set_ylabel("Time with left error <= 0.05 (%)")
    axes[1, 1].set_title("Varying inputs: at least one 0.5 s interval")
    axes[1, 1].set_ylabel("Run-level confirmation rate (%)")

    # 用无噪声结果单独展示历史混合造成的延迟，避免被噪声越阈掩盖。
    for profile in profiles:
        records = [profile["noise_levels"][0]["methods"][method] for method in WINDOW_METHODS]
        label = ("Constant" if profile["configuration"]["kind"] == "constant"
                 else f"Varying {profile['configuration']['phase_degrees']} deg")
        axes[1, 2].plot(durations, [r["median_confirmed_recovery"] for r in records],
                        marker="o", label=label)
    axes[1, 2].set_title("Noiseless: adaptation delay from averaging")
    axes[1, 2].set_xlabel("Window duration (s)")
    axes[1, 2].set_xticks(durations)
    axes[1, 2].set_ylabel("Confirmation after fault (s; includes 0.5 s)")
    axes[1, 2].legend(fontsize=7)
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Window-length ablation: every window updates every 0.02 s")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles, aggregate, examples = run_study(args.seeds)
    dependencies = [Path(baseline.__file__), Path(inputs.__file__)]
    protocol = {
        "dependencies_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        "dt": baseline.DT, "duration": baseline.DURATION, "change_time": baseline.CHANGE_TIME,
        "window_steps": WINDOWS, "window_seconds": [w * baseline.DT for w in WINDOWS],
        "update_period_seconds": baseline.DT,
        "startup": "Hold eta=[1,1] until each full window is available. All are full before the fault.",
        "estimator": "Per-wheel LS: eta=sum(u*y)/sum(u^2); causal sliding window; no reset or clipping.",
        "reference": "Unchanged fixed model and EMA alpha=0.05. Window lengths intentionally have different mean sample ages.",
        "steady_interval": [baseline.STEADY_START, baseline.DURATION],
        "steady_metrics": "RMSE, within-run jitter SD, bias, and fraction of samples within left error 0.05.",
        "recovery": "Left error <=0.05 for 0.5 s; delay includes hold time. First qualified interval does not imply permanent stability.",
        "failure_handling": "Failed runs retained; infinite recovery for overall median (stored as null).",
        "noise_seeds_per_phase": args.seeds,
        "input_profiles": inputs.CONFIGURATIONS,
        "aggregation": "Descriptive mean over four preset phases x paired seeds; phases are not independent noise trials.",
        "noise": "Same Gaussian measurement samples across methods, windows and profiles; same numeric sigma for v and omega in their own units.",
        "causality": "eta[k] predicts k+1. Observing k+1 updates eta[k+1]. Truth and change time used only for plant and scoring.",
        "scope": "Known other dynamics, smooth positive exogenous commands, one sustained efficiency change, no process noise or closed-loop control.",
    }
    json_path = args.output_dir / "window_lengths_summary.json"
    json_path.write_text(json.dumps({"protocol": protocol, "profiles": profiles,
                                    "varying_aggregate": aggregate}, indent=2, allow_nan=False), encoding="utf-8")
    level = baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)
    print("\nVarying inputs, four phases: noise std 0.01. Recovery includes a 0.5 s hold.")
    for method in METHODS:
        record = aggregate[level]["methods"][method]
        median = record["median_confirmed_recovery"]
        median_text = "not confirmed" if median is None else f"{median:.2f} s"
        print(f"  {LABELS[method]}: RMSE={record['left_rmse_mean']:.5f}; "
              f"jitter SD={record['steady_left_jitter_sd_mean']:.5f}; "
              f"in-band={100 * record['steady_left_in_band_fraction_mean']:.1f}%; "
              f"confirmed={record['recovered_runs']}/{record['total_runs']}; median={median_text}")
    print("\nNoiseless constant input: confirmation delay after the fault")
    for method in WINDOW_METHODS:
        delay = profiles[0]["noise_levels"][0]["methods"][method]["median_confirmed_recovery"]
        print(f"  {LABELS[method]}: {delay:.2f} s")
    figure_path = args.output_dir / "window_lengths_result.png"
    save_figure(profiles, aggregate, examples, figure_path, show=not args.no_show)
    print(f"\nSaved plot: {figure_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
