"""Lesson 8: check noisy identification with changing drive commands.

Keep compare_noisy_identification.py in the same folder.
Run: python compare_varying_inputs.py
Use --no-show for headless execution, or --seeds 10 for a shorter run.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import compare_noisy_identification as baseline


# 输入预先确定，与观测噪声和实际故障无关。
# 两侧分别归一化到与恒输入相同的RMS=2 N。
PHASES = [0, 90, 180, 270]
CONFIGURATIONS = [{"id": "constant", "kind": "constant", "phase_degrees": None}]
CONFIGURATIONS += [
    {"id": f"varying_{phase:03d}", "kind": "varying", "phase_degrees": phase}
    for phase in PHASES
]


def generate_motion(configuration):
    times, commands, motion, efficiencies = baseline.generate_motion()
    if configuration["kind"] == "constant":
        return times, commands, motion, efficiencies

    command_times = times[:-1]
    phase = math.radians(configuration["phase_degrees"])
    raw = np.column_stack([
        1.0 + 0.4 * np.sin(2.0 * math.pi * 0.4 * command_times + phase),
        1.0 + 0.35 * np.cos(2.0 * math.pi * 0.6 * command_times + 0.4 + phase),
    ])
    commands = raw * (2.0 / np.sqrt(np.mean(raw ** 2, axis=0)))
    motion = np.zeros_like(motion)
    motion[0] = [1.0, 0.0]
    for step in range(len(commands)):
        # u_k必须与状态转移k -> k+1配对。
        motion[step + 1] = baseline.predict_motion(
            motion[step], commands[step], efficiencies[step]
        )
    return times, commands, motion, efficiencies


def command_statistics(times, commands):
    # LS按u^2加权；变输入时，实际有效样本龄随波形变化。
    window_ages = []
    ages = np.arange(baseline.WINDOW_STEPS - 1, -1, -1) * baseline.DT
    for end in range(baseline.WINDOW_STEPS, len(commands) + 1):
        if times[end] < baseline.STEADY_START:
            continue
        energy = commands[end - baseline.WINDOW_STEPS:end] ** 2
        window_ages.append(np.sum(energy * ages[:, None], axis=0) / np.sum(energy, axis=0))
    window_ages = np.asarray(window_ages)
    return {
        "rms": np.sqrt(np.mean(commands ** 2, axis=0)).tolist(),
        "minimum": np.min(commands, axis=0).tolist(),
        "maximum": np.max(commands, axis=0).tolist(),
        "mean": np.mean(commands, axis=0).tolist(),
        "largest_step_change": np.max(np.abs(np.diff(commands, axis=0)), axis=0).tolist(),
        "ls_weighted_sample_age_mean_seconds": np.mean(window_ages, axis=0).tolist(),
        "ls_weighted_sample_age_min_seconds": np.min(window_ages, axis=0).tolist(),
        "ls_weighted_sample_age_max_seconds": np.max(window_ages, axis=0).tolist(),
    }


def known_efficiency_prediction(observations, commands, efficiencies):
    """仅作事后诊断：已知真实效率，但当前速度观测仍含噪声。"""
    forces = commands * efficiencies[:-1]
    predicted = np.empty_like(commands)
    predicted[:, 0] = observations[:-1, 0] + (
        forces[:, 0] + forces[:, 1] - baseline.LINEAR_DRAG * observations[:-1, 0]
    ) / baseline.MASS * baseline.DT
    predicted[:, 1] = observations[:-1, 1] + (
        baseline.HALF_TRACK * (forces[:, 1] - forces[:, 0])
        - baseline.YAW_DRAG * observations[:-1, 1]
    ) / baseline.INERTIA * baseline.DT
    return predicted


def summarize(runs):
    record = baseline.summarize_runs(runs)
    for metric in ["next_true_speed_rmse", "next_true_yaw_rmse"]:
        record[metric + "_mean"] = float(np.mean([run[metric] for run in runs]))
    fractions = [run["steady_left_in_band_fraction"] for run in runs]
    record["steady_left_in_band_fraction_mean"] = float(np.mean(fractions))
    record["steady_left_in_band_fraction_sd"] = float(np.std(fractions))
    return record


def run_study(seeds):
    profile_results = []
    example = None
    for configuration in CONFIGURATIONS:
        times, commands, truth, true_efficiencies = generate_motion(configuration)
        steady = times >= baseline.STEADY_START
        post_fault = times[1:] > baseline.CHANGE_TIME
        noise_results = []

        for sigma in baseline.NOISE_LEVELS:
            measurements = {method: [] for method in baseline.METHODS}
            known_efficiency_scores = []
            for seed in range(seeds):
                # 不同方法、输入档和相位使用配对的标准噪声序列。
                noise = np.random.default_rng(seed).normal(size=truth.shape) * sigma
                observations = truth + noise
                estimates, predictions = baseline.run_estimators(observations, commands)
                for method in baseline.METHODS:
                    parameter_error = estimates[method][steady] - true_efficiencies[steady]
                    rmse = np.sqrt(np.mean(parameter_error ** 2, axis=0))
                    prediction_error = truth[1:] - predictions[method]
                    prediction_rmse = np.sqrt(np.mean(prediction_error[post_fault] ** 2, axis=0))
                    measurements[method].append({
                        "seed": seed,
                        "phase_degrees": configuration["phase_degrees"],
                        "left_rmse": float(rmse[0]),
                        "right_rmse": float(rmse[1]),
                        "next_true_speed_rmse": float(prediction_rmse[0]),
                        "next_true_yaw_rmse": float(prediction_rmse[1]),
                        "steady_left_in_band_fraction": float(np.mean(
                            np.abs(parameter_error[:, 0]) <= baseline.PARAMETER_TOLERANCE + 1e-12
                        )),
                        "recovery": baseline.confirmed_left_recovery(times, estimates[method][:, 0]),
                    })

                diagnostic_prediction = known_efficiency_prediction(observations, commands, true_efficiencies)
                diagnostic_error = truth[1:] - diagnostic_prediction
                known_efficiency_scores.append(np.sqrt(np.mean(diagnostic_error[post_fault] ** 2, axis=0)))
                if (configuration["id"] == "varying_000"
                        and sigma == baseline.EXAMPLE_SIGMA and seed == baseline.EXAMPLE_SEED):
                    example = {"times": times, "commands": commands, "estimates": estimates,
                               "true_efficiencies": true_efficiencies}

            diagnostic_mean = np.mean(known_efficiency_scores, axis=0)
            noise_results.append({
                "sigma_speed": sigma, "sigma_yaw": sigma,
                "methods": {method: summarize(measurements[method]) for method in baseline.METHODS},
                "known_efficiency_speed_rmse": float(diagnostic_mean[0]),
                "known_efficiency_yaw_rmse": float(diagnostic_mean[1]),
            })

        profile_results.append({"configuration": configuration,
                                "input_statistics": command_statistics(times, commands),
                                "noise_levels": noise_results})
        print(f"Completed {configuration['id']}: {len(baseline.NOISE_LEVELS)} noise levels x {seeds} seeds.", flush=True)

    # 相位-噪声案例的描述性汇总。各相位重复使用配对噪声，不把它们声称为独立样本。
    varying_summary = []
    varying_profiles = [item for item in profile_results if item["configuration"]["kind"] == "varying"]
    for level, sigma in enumerate(baseline.NOISE_LEVELS):
        methods = {}
        for method in baseline.METHODS:
            per_phase = [item["noise_levels"][level]["methods"][method] for item in varying_profiles]
            combined_runs = [run for phase in per_phase for run in phase["runs"]]
            record = summarize(combined_runs)
            record["phase_mean_left_rmse_range"] = [min(phase["left_rmse_mean"] for phase in per_phase),
                                                     max(phase["left_rmse_mean"] for phase in per_phase)]
            record["phase_recovery_rate_range"] = [min(phase["recovery_rate"] for phase in per_phase),
                                                   max(phase["recovery_rate"] for phase in per_phase)]
            methods[method] = record
        varying_summary.append({"sigma_speed": sigma, "sigma_yaw": sigma, "methods": methods})
    return profile_results, varying_summary, example


def save_figure(profiles, varying_summary, example, output_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    input_times = example["times"][:-1]
    axes[0, 0].plot(input_times, example["commands"][:, 0], color="tab:orange", label="Left: varying phase 0")
    axes[0, 0].plot(input_times, example["commands"][:, 1], color="tab:purple", label="Right: varying phase 0")
    axes[0, 0].axhline(2.0, color="black", linestyle="--", label="Constant input")
    axes[0, 0].axvline(baseline.CHANGE_TIME, color="gray", linestyle=":")
    axes[0, 0].set_xlim(0.0, 10.0)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Nominal-force command (N)")
    axes[0, 0].set_title("Exogenous inputs; RMS = 2 N per side")
    axes[0, 0].legend(fontsize=8, loc="best")

    for method in baseline.METHODS:
        axes[0, 1].plot(example["times"], example["estimates"][method][:, 0],
                        color=baseline.COLORS[method], label=baseline.LABELS[method], linewidth=1.0)
    axes[0, 1].step(example["times"], example["true_efficiencies"][:, 0], where="post",
                    color="black", linestyle="--", label="Actual left efficiency")
    axes[0, 1].axhspan(0.5 - baseline.PARAMETER_TOLERANCE, 0.5 + baseline.PARAMETER_TOLERANCE,
                       color="gray", alpha=0.15, label="Left parameter band")
    axes[0, 1].axvline(baseline.CHANGE_TIME, color="gray", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Left efficiency estimate")
    axes[0, 1].set_title("Example: phase 0, noise std 0.01, seed 0")
    axes[0, 1].legend(fontsize=7, loc="best")

    example_level = baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)
    positions = np.arange(len(profiles))
    for method, offset in [("error_update", -0.16), ("window_ls", 0.16)]:
        records = [item["noise_levels"][example_level]["methods"][method] for item in profiles]
        axes[1, 0].bar(positions + offset, [record["left_rmse_mean"] for record in records],
                       width=0.30, yerr=[record["left_rmse_sd"] for record in records], capsize=3,
                       color=baseline.COLORS[method], label=baseline.LABELS[method])
    axes[1, 0].set_xticks(positions, ["Constant", "Var 0 deg", "Var 90 deg", "Var 180 deg", "Var 270 deg"])
    axes[1, 0].tick_params(axis="x", labelsize=8)
    axes[1, 0].set_ylabel("Steady left-efficiency RMSE")
    axes[1, 0].set_title("Noise std 0.01: mean +/- 1 SD across seeds")
    axes[1, 0].legend(fontsize=8, loc="best")

    sigmas = baseline.NOISE_LEVELS
    constant = profiles[0]["noise_levels"]
    for method in ["error_update", "window_ls"]:
        axes[1, 1].plot(sigmas, [100 * level["methods"][method]["recovery_rate"] for level in constant],
                        color=baseline.COLORS[method], marker="o", label=baseline.LABELS[method] + ": constant")
        axes[1, 1].plot(sigmas, [100 * level["methods"][method]["recovery_rate"] for level in varying_summary],
                        color=baseline.COLORS[method], marker="s", linestyle="--",
                        label=baseline.LABELS[method] + ": varying, 4 phases")
    axes[1, 1].axhline(0.0, color="tab:blue", linestyle=":", label="Fixed model: 0%")
    axes[1, 1].set_xticks(sigmas)
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].set_xlabel("Noise std: speed (m/s), yaw rate (rad/s)")
    axes[1, 1].set_ylabel("Run-level confirmation rate (%)")
    axes[1, 1].set_title("At least one 0.5 s in-band interval")
    axes[1, 1].legend(fontsize=7, loc="best")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    seeds = profiles[0]["noise_levels"][0]["methods"]["fixed"]["total_runs"]
    figure.suptitle(f"Constant versus smooth varying inputs: 4 preset phases, {seeds} paired noise seeds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
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

    profiles, varying_summary, example = run_study(args.seeds)
    baseline_path = Path(baseline.__file__).resolve()
    protocol = {
        "baseline_file": baseline_path.name,
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "dt": baseline.DT, "duration": baseline.DURATION, "change_time": baseline.CHANGE_TIME,
        "alpha": baseline.ALPHA, "window_steps": baseline.WINDOW_STEPS,
        "steady_interval": [baseline.STEADY_START, baseline.DURATION],
        "input_design": "Left raw=1+0.4*sin(2*pi*0.4*t+phase); right raw=1+0.35*cos(2*pi*0.6*t+0.4+phase). Each scaled to RMS=2 N.",
        "phases_degrees": PHASES, "noise_seeds_per_phase": args.seeds,
        "recovery": "Left eta within 0.05 of 0.5 for 0.5 s; confirmation includes hold time.",
        "failure_handling": "Failures remain in rates and count as infinity for the overall median; null median means infinity.",
        "aggregation": "Varying results describe 4 preset phases x paired noise seeds, not independent-sample significance tests.",
        "supplementary_stability_metric": "Fraction of steady samples (10-20 s) with left parameter error <=0.05. Added to distinguish a first confirmed interval from longer-term stability; the original recovery criterion is unchanged.",
        "sample_age": "LS window weights are u^2; weighted sample ages are reported separately for each input profile.",
        "diagnostic_reference": "True efficiency with noisy current state is evaluation-only, without a state filter; not a universal lower bound.",
        "scope": "Known other parameters, smooth positive exogenous inputs, measurement noise only; no feedback-generated inputs, slip or process noise.",
    }
    json_path = args.output_dir / "varying_inputs_summary.json"
    json_path.write_text(json.dumps({"protocol": protocol, "profiles": profiles,
                                    "varying_aggregate": varying_summary}, indent=2, allow_nan=False), encoding="utf-8")

    level = baseline.NOISE_LEVELS.index(baseline.EXAMPLE_SIGMA)
    print("\nNoise std 0.01: all entries are paired descriptive comparisons.")
    for profile in profiles:
        print(f"\n{profile['configuration']['id']}; input RMS={profile['input_statistics']['rms']}")
        for method in baseline.METHODS:
            record = profile["noise_levels"][level]["methods"][method]
            print(f"  {method}: left RMSE={record['left_rmse_mean']:.5f}; right RMSE={record['right_rmse_mean']:.5f}; recovery={record['recovered_runs']}/{record['total_runs']}; steady in-band time={100 * record['steady_left_in_band_fraction_mean']:.1f}%")
    print("\nVarying-input aggregate over four preset phases:")
    for noise_level in varying_summary:
        print(f"  Noise std={noise_level['sigma_speed']:.3f}")
        for method in ["error_update", "window_ls"]:
            record = noise_level["methods"][method]
            print(f"    {method}: left RMSE={record['left_rmse_mean']:.5f}; recovery={record['recovered_runs']}/{record['total_runs']}; steady in-band time={100 * record['steady_left_in_band_fraction_mean']:.1f}%")

    figure_path = args.output_dir / "varying_inputs_result.png"
    save_figure(profiles, varying_summary, example, figure_path, show=not args.no_show)
    print(f"\nSaved plot: {figure_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
