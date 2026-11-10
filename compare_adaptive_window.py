"""Lesson 10: causal, noise-aware adjustment of a sliding LS window.

Keep the previous three compare_*.py lessons in the same folder.
Run: python compare_adaptive_window.py
This is a rule-based prototype, with supplied sensor noise specifications.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import compare_noisy_identification as baseline
import compare_varying_inputs as inputs
import compare_window_lengths as windows


MIN_WINDOW = 40
MAX_WINDOW = 160
DIFFERENCE_FLOOR = 0.10
NOISE_MULTIPLIER = 4.0
TRIGGER_UPDATES = 5
METHODS = windows.WINDOW_METHODS + ["adaptive"]
LABELS = {**windows.LABELS, "adaptive": "Adaptive 0.8-3.2 s"}
COLORS = {**windows.COLORS, "adaptive": "black"}


def force_prefixes(observations, commands):
    """每个前缀只含已经完成的转移；保留差分噪声的相邻相关性。"""
    current, following = observations[:-1], observations[1:]
    force_sum = (baseline.MASS / baseline.DT * (following[:, 0] - current[:, 0])
                 + baseline.LINEAR_DRAG * current[:, 0])
    force_difference = (
        baseline.INERTIA / (baseline.HALF_TRACK * baseline.DT)
        * (following[:, 1] - current[:, 1])
        + baseline.YAW_DRAG / baseline.HALF_TRACK * current[:, 1]
    )
    observed_forces = np.column_stack([(force_sum - force_difference) / 2,
                                      (force_sum + force_difference) / 2])
    zeros = np.zeros((1, 2))
    return (np.vstack([zeros, np.cumsum(commands * observed_forces, axis=0)]),
            np.vstack([zeros, np.cumsum(commands ** 2, axis=0)]))


def ls_estimate(sum_uy, sum_uu, end, size):
    numerator = sum_uy[end] - sum_uy[end - size]
    denominator = sum_uu[end] - sum_uu[end - size]
    if np.any(denominator <= 1e-10):
        return None
    return numerator / denominator


def difference_noise_sd(command_window, sigma):
    """短窗减当前窗的精确噪声标准差，假设v/omega观测独立高斯。

    sigma=[速度噪声标准差,角速度噪声标准差]来自传感器标定。
    y_k=A*n[k+1]+B*n[k]，按观测时刻合并系数，不能把相邻y当独立。
    """
    size = len(command_window)
    weights = -command_window / np.sum(command_window ** 2, axis=0)
    weights[-MIN_WINDOW:] += (command_window[-MIN_WINDOW:]
                             / np.sum(command_window[-MIN_WINDOW:] ** 2, axis=0))
    speed_next = baseline.MASS / (2 * baseline.DT)
    speed_current = baseline.LINEAR_DRAG / 2 - speed_next
    yaw_next = np.array([-1.0, 1.0]) * baseline.INERTIA / (2 * baseline.HALF_TRACK * baseline.DT)
    yaw_current = np.array([-1.0, 1.0]) * baseline.YAW_DRAG / (2 * baseline.HALF_TRACK) - yaw_next
    speed_coefficients = np.zeros((size + 1, 2))
    yaw_coefficients = np.zeros((size + 1, 2))
    speed_coefficients[:-1] += weights * speed_current
    speed_coefficients[1:] += weights * speed_next
    yaw_coefficients[:-1] += weights * yaw_current
    yaw_coefficients[1:] += weights * yaw_next
    variance = (sigma[0] ** 2 * np.sum(speed_coefficients ** 2, axis=0)
                + sigma[1] ** 2 * np.sum(yaw_coefficients ** 2, axis=0))
    return np.sqrt(variance)


def run_adaptive(observations, commands, sensor_noise_std):
    """估计器只接收观测、输入和噪声标定值，不接收真值或故障时刻。"""
    sigma = np.asarray(sensor_noise_std, dtype=float)
    if sigma.shape != (2,) or np.any(sigma < 0):
        raise ValueError("Provide nonnegative [speed noise std, yaw noise std].")
    steps = len(commands)
    sum_uy, sum_uu = force_prefixes(observations, commands)
    estimates = np.ones((steps + 1, 2))
    window_sizes = np.zeros(steps + 1, dtype=int)
    difference = np.full((steps + 1, 2), np.nan)
    thresholds = np.full((steps + 1, 2), np.nan)
    alarms = []
    # 正常启动与最长固定窗相同：收满3.2秒，再开始更新和检测。
    current_size = MAX_WINDOW
    last_alarm = None
    consecutive = 0

    for end in range(1, steps + 1):
        estimates[end] = estimates[end - 1]
        if end < MAX_WINDOW:
            continue
        if last_alarm is not None:
            # 先保持短窗0.8秒，再每步增加一个样本。
            # 扩窗后窗口起点>=上次报警时刻，不会重新捞回报警前的数据。
            current_size = min(MAX_WINDOW, max(MIN_WINDOW, end - last_alarm))
        short = ls_estimate(sum_uy, sum_uu, end, MIN_WINDOW)
        current = ls_estimate(sum_uy, sum_uu, end, current_size)
        if short is None or current is None:
            consecutive = 0
            continue
        sd = difference_noise_sd(commands[end - current_size:end], sigma)
        threshold = np.maximum(DIFFERENCE_FLOOR, NOISE_MULTIPLIER * sd)
        difference[end] = np.abs(short - current)
        thresholds[end] = threshold
        different = bool(np.any(difference[end] > threshold))
        consecutive = consecutive + 1 if different else 0
        if consecutive >= TRIGGER_UPDATES:
            alarms.append(end)
            last_alarm = end
            current_size = MIN_WINDOW
            current = short
            consecutive = 0
        estimates[end] = current
        window_sizes[end] = current_size
    return {"estimates": estimates, "window_sizes": window_sizes,
            "alarms": alarms, "difference": difference, "thresholds": thresholds}


def healthy_motion(configuration):
    times, commands, truth, efficiencies = inputs.generate_motion(configuration)
    efficiencies[:] = 1.0
    for step in range(len(commands)):
        truth[step + 1] = baseline.predict_motion(truth[step], commands[step], efficiencies[step])
    return times, commands, truth, efficiencies


def summarize(runs):
    record = windows.summarize(runs)
    for metric in ["mean_window_seconds", "short_window_time_fraction", "pre_fault_alarm_count"]:
        record[metric + "_mean"] = float(np.mean([run[metric] for run in runs]))
    detection_delays = [np.inf if run["detection_delay"] is None else run["detection_delay"] for run in runs]
    median = float(np.median(detection_delays))
    record["median_detection_delay"] = median if math.isfinite(median) else None
    record["detected_runs"] = int(np.sum(np.isfinite(detection_delays)))
    return record


def run_study(seeds):
    profiles, examples, healthy_runs = [], {}, []
    for configuration in inputs.CONFIGURATIONS:
        times, commands, truth, efficiencies = inputs.generate_motion(configuration)
        steady = times >= baseline.STEADY_START
        levels = []
        for sigma in baseline.NOISE_LEVELS:
            measurements = {method: [] for method in METHODS}
            for seed in range(seeds):
                observations = truth + np.random.default_rng(seed).normal(size=truth.shape) * sigma
                estimates = windows.run_window_estimators(observations, commands)
                adaptive = run_adaptive(observations, commands, [sigma, sigma])
                estimates["adaptive"] = adaptive["estimates"]
                alarm_times = times[np.asarray(adaptive["alarms"], dtype=int)]
                detections = alarm_times[alarm_times > baseline.CHANGE_TIME]
                for method in METHODS:
                    record = windows.score_run(times, truth, efficiencies, estimates[method],
                                               windows.predict_log(observations, commands, estimates[method]),
                                               seed, configuration["phase_degrees"])
                    sizes = (adaptive["window_sizes"] if method == "adaptive"
                             else np.full(len(times), int(method.split("_")[1])))
                    record.update({
                        "mean_window_seconds": float(np.mean(sizes[steady]) * baseline.DT),
                        "short_window_time_fraction": float(np.mean(sizes[steady] == MIN_WINDOW)),
                        "pre_fault_alarm_count": int(np.sum(alarm_times <= baseline.CHANGE_TIME)) if method == "adaptive" else 0,
                        "detection_delay": float(detections[0] - baseline.CHANGE_TIME) if method == "adaptive" and len(detections) else None,
                        "alarm_times": alarm_times.tolist() if method == "adaptive" else [],
                    })
                    measurements[method].append(record)
                if configuration["id"] == "varying_000" and seed == 0 and sigma in [0, baseline.EXAMPLE_SIGMA]:
                    examples[sigma] = {"times": times, "truth": efficiencies,
                                       "estimates": estimates, "adaptive": adaptive}
            levels.append({"sigma_speed": sigma, "sigma_yaw": sigma,
                           "methods": {method: summarize(measurements[method]) for method in METHODS}})

        # 独立的无故障日志：检查抖动是否会误触发缩窗；不拿它调节规则。
        healthy_times, healthy_commands, healthy_truth, _ = healthy_motion(configuration)
        for seed in range(seeds):
            observations = healthy_truth + np.random.default_rng(seed).normal(size=healthy_truth.shape) * baseline.EXAMPLE_SIGMA
            adaptive = run_adaptive(observations, healthy_commands, [baseline.EXAMPLE_SIGMA] * 2)
            healthy_runs.append({"configuration": configuration["id"], "seed": seed,
                                 "alarm_times": healthy_times[np.asarray(adaptive["alarms"], dtype=int)].tolist(),
                                 "steady_left_rmse": float(np.sqrt(np.mean((adaptive["estimates"][steady, 0] - 1) ** 2)))})
        profiles.append({"configuration": configuration, "noise_levels": levels})
        print(f"Completed {configuration['id']}: fault trials plus {seeds} healthy trials.", flush=True)

    aggregate = []
    for level, sigma in enumerate(baseline.NOISE_LEVELS):
        aggregate.append({"sigma_speed": sigma, "sigma_yaw": sigma,
                          "methods": {method: summarize([
                              run for profile in profiles[1:]
                              for run in profile["noise_levels"][level]["methods"][method]["runs"]
                          ]) for method in METHODS}})
    healthy = {"sigma_speed": baseline.EXAMPLE_SIGMA, "sigma_yaw": baseline.EXAMPLE_SIGMA,
               "total_runs": len(healthy_runs),
               "runs_with_false_alarms": sum(bool(run["alarm_times"]) for run in healthy_runs),
               "total_false_alarms": sum(len(run["alarm_times"]) for run in healthy_runs),
               "monitored_seconds_per_run": baseline.DURATION - MAX_WINDOW * baseline.DT,
               "runs": healthy_runs}
    return profiles, aggregate, healthy, examples


def save_figure(aggregate, healthy, examples, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    example = examples[baseline.EXAMPLE_SIGMA]
    times = example["times"]
    for method in METHODS:
        axes[0, 0].plot(times, example["estimates"][method][:, 0], label=LABELS[method],
                        color=COLORS[method], linewidth=1.5 if method == "adaptive" else 0.8)
    axes[0, 0].step(times, example["truth"][:, 0], where="post", linestyle="--", color="gray", label="Actual")
    axes[0, 0].axhspan(0.45, 0.55, color="gray", alpha=0.15)
    axes[0, 0].set_xlim(4, 20)
    axes[0, 0].set_title("Phase 0, noise std 0.01, seed 0")
    axes[0, 0].set_ylabel("Left efficiency estimate")
    axes[0, 0].legend(fontsize=8)

    for sigma, style in [(0.0, "--"), (baseline.EXAMPLE_SIGMA, "-")]:
        example = examples[sigma]
        duration = example["adaptive"]["window_sizes"].astype(float) * baseline.DT
        duration[duration == 0] = np.nan
        axes[0, 1].plot(times, duration, linestyle=style, label=f"Noise std {sigma:g}")
        for index in example["adaptive"]["alarms"]:
            axes[0, 1].scatter(times[index], MIN_WINDOW * baseline.DT, marker="x", color="red", zorder=5)
    axes[0, 1].axvline(baseline.CHANGE_TIME, color="gray", linestyle=":", label="Actual change (evaluation)")
    axes[0, 1].set_title("Window changes from observations; red x = alarm")
    axes[0, 1].set_ylabel("Window duration (s)")
    axes[0, 1].set_ylim(0.5, 3.5)
    axes[0, 1].legend(fontsize=8)
    for axis in axes[0]:
        axis.set_xlabel("Time (s)")

    level = aggregate[baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)]
    positions = np.arange(len(METHODS))
    records = [level["methods"][method] for method in METHODS]
    axes[1, 0].bar(positions, [r["left_rmse_mean"] for r in records],
                    yerr=[r["left_rmse_sd"] for r in records], capsize=3,
                    color=[COLORS[method] for method in METHODS])
    axes[1, 0].set_xticks(positions, [LABELS[method] for method in METHODS], fontsize=8)
    axes[1, 0].set_title("Varying inputs: 4 phases x paired seeds; noise 0.01")
    axes[1, 0].set_ylabel("Steady left RMSE (10-20 s), mean +/- SD")

    for method in METHODS:
        records = [level["methods"][method] for level in aggregate]
        medians = [np.nan if r["median_confirmed_recovery"] is None else r["median_confirmed_recovery"] for r in records]
        axes[1, 1].plot(baseline.NOISE_LEVELS, medians, marker="o", color=COLORS[method], label=LABELS[method])
    axes[1, 1].set_xticks(baseline.NOISE_LEVELS)
    axes[1, 1].set_xlabel("Noise std: speed (m/s), yaw rate (rad/s)")
    axes[1, 1].set_ylabel("Median confirmation after change (s)")
    axes[1, 1].set_title("Includes 0.5 s hold; missing point = infinite median")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.suptitle("Adaptive window: supplied noise specification; all methods update every 0.02 s")
    figure.text(0.5, 0.01, f"Healthy controls (noise 0.01): {healthy['runs_with_false_alarms']}/{healthy['total_runs']} runs had a false alarm; prototype, not an optimality claim.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
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
    profiles, aggregate, healthy, examples = run_study(args.seeds)
    dependencies = [Path(module.__file__) for module in [baseline, inputs, windows]]
    protocol = {
        "dependencies_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        "window_bounds_steps": [MIN_WINDOW, MAX_WINDOW], "dt": baseline.DT,
        "difference_floor": DIFFERENCE_FLOOR, "noise_multiplier": NOISE_MULTIPLIER,
        "consecutive_trigger_updates": TRIGGER_UPDATES,
        "noise_information": "Detector receives calibrated speed/yaw sensor noise SD; here supplied from simulator settings, not estimated online.",
        "noise_model": "Independent Gaussian state-observation samples; exact short-minus-current-window noise variance retains MA(1) force-noise correlation.",
        "detection": "Compare per-wheel LS over last 40 transitions with current window; any side above max(0.10, 4*noise_SD) for five updates triggers shrinkage.",
        "regrowth": "Hold W=40 for 40 updates after alarm, then add one sample each update to W=160; growing window starts no earlier than last alarm.",
        "startup": "Adaptive estimator holds nominal eta until 160 transitions collected; no monitoring before 3.2 s. Fixed short windows start earlier.",
        "truth_isolation": "Adaptive API takes observations, commands, supplied noise SD only; it has no true efficiency or fault time.",
        "primary_metrics": "Unchanged steady 10-20 s RMSE and in-band fraction; first left error<=0.05 for 0.5 s confirms recovery, including hold time.",
        "failure_handling": "Failures count as infinite for overall median; stored as null. Missing plotted medians remain gaps.",
        "noise_seeds_per_phase": args.seeds,
        "aggregation": "Four preset varying-input phases share paired noise seeds; descriptive comparisons, not independent-trial significance claims. Noiseless seeds repeat identical trajectories.",
        "healthy_control": "No efficiency change, same five input profiles, noise 0.01; count false alarms across full monitored interval 3.2-20 s.",
        "scope": "Rule-based illustrative prototype; known remaining dynamics, supplied correct noise specification, one sustained change, no slip/process noise or closed-loop control. Frequent changes and noise miscalibration untested.",
    }
    json_path = args.output_dir / "adaptive_window_summary.json"
    json_path.write_text(json.dumps({"protocol": protocol, "profiles": profiles,
                                    "varying_aggregate": aggregate, "healthy_control": healthy},
                                   indent=2, allow_nan=False), encoding="utf-8")
    print("\nVarying inputs, noise std 0.01; four phases x paired seeds:")
    for method, record in aggregate[baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)]["methods"].items():
        median = record["median_confirmed_recovery"]
        median_text = "not confirmed" if median is None else f"{median:.2f} s"
        print(f"  {LABELS[method]}: RMSE={record['left_rmse_mean']:.5f}; "
              f"in-band={100*record['steady_left_in_band_fraction_mean']:.1f}%; "
              f"confirmed={record['recovered_runs']}/{record['total_runs']}; median={median_text}")
    print(f"Healthy controls: {healthy['runs_with_false_alarms']}/{healthy['total_runs']} runs with false alarms.")
    path = args.output_dir / "adaptive_window_result.png"
    save_figure(aggregate, healthy, examples, path, show=not args.no_show)
    print(f"Saved plot: {path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
