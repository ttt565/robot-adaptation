"""Lesson 7: noisy online identification on exactly the same motion log.

Run: python compare_noisy_identification.py
Use --no-show for a headless run, or --seeds 10 for a shorter experiment.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# 与前几课相同的理想动力学。本课只给速度观测添加噪声。
MASS = 10.0
INERTIA = 0.5
HALF_TRACK = 0.2
LINEAR_DRAG = 4.0
YAW_DRAG = 0.8
DT = 0.02
DURATION = 20.0
CHANGE_TIME = 5.0
STEADY_START = 10.0
ALPHA = 0.05
WINDOW_STEPS = 40  # 0.8秒；平均样本龄0.39秒，EMA约为0.38秒。
PARAMETER_TOLERANCE = 0.05
HOLD_DURATION = 0.5
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02]
EXAMPLE_SIGMA = 0.01
EXAMPLE_SEED = 0
METHODS = ["fixed", "error_update", "window_ls"]
LABELS = {"fixed": "Fixed model", "error_update": "Error update (alpha=0.05)",
          "window_ls": "Window LS (0.8 s)"}
COLORS = {"fixed": "tab:blue", "error_update": "tab:orange", "window_ls": "tab:green"}


def generate_motion():
    """生成真实运动；真实效率只由仿真器和事后评价使用。"""
    steps = round(DURATION / DT)
    times = np.arange(steps + 1) * DT
    commands = np.full((steps, 2), 2.0)
    motion = np.zeros((steps + 1, 2))  # [线速度, 角速度]
    motion[0] = [1.0, 0.0]
    efficiencies = np.ones((steps + 1, 2))
    efficiencies[times >= CHANGE_TIME, 0] = 0.5
    for step in range(steps):
        left_force, right_force = efficiencies[step] * commands[step]
        speed, yaw_rate = motion[step]
        motion[step + 1] = [
            speed + (left_force + right_force - LINEAR_DRAG * speed) / MASS * DT,
            yaw_rate + (HALF_TRACK * (right_force - left_force)
                        - YAW_DRAG * yaw_rate) / INERTIA * DT,
        ]
    return times, commands, motion, efficiencies


def predict_motion(observation, command, efficiency_estimate):
    left_force, right_force = command * efficiency_estimate
    speed, yaw_rate = observation
    return np.array([
        speed + (left_force + right_force - LINEAR_DRAG * speed) / MASS * DT,
        yaw_rate + (HALF_TRACK * (right_force - left_force)
                    - YAW_DRAG * yaw_rate) / INERTIA * DT,
    ])


def run_estimators(observations, commands):
    """只接收带噪观测和输入；不接收真实状态、真实效率或故障时刻。"""
    steps = len(commands)
    estimates = {method: np.ones((steps + 1, 2)) for method in METHODS}
    predictions = {method: np.zeros((steps, 2)) for method in METHODS}
    sum_input_output = np.zeros((steps + 1, 2))
    sum_input_squared = np.zeros((steps + 1, 2))

    for step in range(steps):
        command = commands[step]
        current = observations[step]

        # 1. 使用尚未读取下一步观测的估计值，先作预测。
        for method in METHODS:
            predictions[method][step] = predict_motion(
                current, command, estimates[method][step]
            )

        # 2. 下一步观测到达后，才更新误差法的估计值。
        following = observations[step + 1]
        innovation = following - predictions["error_update"][step]
        sum_force_error = MASS / DT * innovation[0]
        difference_force_error = INERTIA / (HALF_TRACK * DT) * innovation[1]
        wheel_force_errors = np.array([
            (sum_force_error - difference_force_error) / 2.0,
            (sum_force_error + difference_force_error) / 2.0,
        ])
        next_estimate = estimates["error_update"][step].copy()
        excited = np.abs(command) >= 0.1
        next_estimate[excited] += ALPHA * wheel_force_errors[excited] / command[excited]
        estimates["error_update"][step + 1] = next_estimate

        # 3. 从观测计算每侧的等效力观测y，模型为 y_i = eta_i * u_i。
        # 差分会放大测量噪声；相邻y的噪声不是独立白噪声。
        observed_sum_force = MASS / DT * (following[0] - current[0]) + LINEAR_DRAG * current[0]
        observed_difference_force = (
            INERTIA / (HALF_TRACK * DT) * (following[1] - current[1])
            + YAW_DRAG / HALF_TRACK * current[1]
        )
        observed_wheel_forces = np.array([
            (observed_sum_force - observed_difference_force) / 2.0,
            (observed_sum_force + observed_difference_force) / 2.0,
        ])
        sum_input_output[step + 1] = sum_input_output[step] + command * observed_wheel_forces
        sum_input_squared[step + 1] = sum_input_squared[step] + command ** 2

        # 4. 滑窗最小二乘只使用截至当前已到达的最近40个转移样本。
        # 启动时先保持初值；故障时不清空历史，也不重置估计。
        window_estimate = estimates["window_ls"][step].copy()
        if step + 1 >= WINDOW_STEPS:
            start = step + 1 - WINDOW_STEPS
            numerator = sum_input_output[step + 1] - sum_input_output[start]
            denominator = sum_input_squared[step + 1] - sum_input_squared[start]
            identifiable = denominator > 1e-10
            window_estimate[identifiable] = numerator[identifiable] / denominator[identifiable]
        estimates["window_ls"][step + 1] = window_estimate

        # 本轮只做辨识，不把估计用于控制；保留未裁剪估计以看清噪声波动。
        # 固定模型的整段估计始终为[1, 1]。

    return estimates, predictions


def confirmed_left_recovery(times, left_estimates):
    """确认时间包括保持期：左侧误差<=0.05持续0.5秒后才确认。"""
    needed_samples = math.ceil(HOLD_DURATION / DT) + 1
    consecutive = 0
    for index in range(len(times)):
        if times[index] <= CHANGE_TIME:
            continue
        if abs(left_estimates[index] - 0.5) <= PARAMETER_TOLERANCE + 1e-12:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= needed_samples:
            return float(times[index] - CHANGE_TIME)
    return None


def summarize_runs(runs):
    """未恢复样本保留为无穷；总体中位数不只统计成功样本。"""
    left_rmse = np.array([run["left_rmse"] for run in runs])
    right_rmse = np.array([run["right_rmse"] for run in runs])
    recovery = np.array([np.inf if run["recovery"] is None else run["recovery"] for run in runs])
    median = float(np.median(recovery))
    return {
        "left_rmse_mean": float(np.mean(left_rmse)),
        "left_rmse_sd": float(np.std(left_rmse)),
        "right_rmse_mean": float(np.mean(right_rmse)),
        "right_rmse_sd": float(np.std(right_rmse)),
        "recovered_runs": int(np.sum(np.isfinite(recovery))),
        "total_runs": len(runs),
        "recovery_rate": float(np.mean(np.isfinite(recovery))),
        "median_confirmed_recovery": median if math.isfinite(median) else None,
        "runs": runs,
    }


def run_study(seeds):
    times, commands, truth, true_efficiencies = generate_motion()
    steady = times >= STEADY_START
    after_fault = times[1:] > CHANGE_TIME
    summaries = []
    example = None

    for sigma in NOISE_LEVELS:
        measurements = {method: [] for method in METHODS}
        for seed in range(seeds):
            # 同一seed对各方法使用同一噪声；不同噪声档缩放同一标准噪声。
            noise = np.random.default_rng(seed).normal(size=truth.shape) * sigma
            observations = truth + noise
            estimates, predictions = run_estimators(observations, commands)
            for method in METHODS:
                parameter_error = estimates[method][steady] - true_efficiencies[steady]
                parameter_rmse = np.sqrt(np.mean(parameter_error ** 2, axis=0))
                # 诊断评分：输入是带噪当前状态，真实下一状态只作事后评分目标。
                prediction_error = truth[1:] - predictions[method]
                prediction_rmse = np.sqrt(np.mean(prediction_error[after_fault] ** 2, axis=0))
                measurements[method].append({
                    "seed": seed,
                    "left_rmse": float(parameter_rmse[0]),
                    "right_rmse": float(parameter_rmse[1]),
                    "next_true_speed_rmse": float(prediction_rmse[0]),
                    "next_true_yaw_rmse": float(prediction_rmse[1]),
                    "recovery": confirmed_left_recovery(times, estimates[method][:, 0]),
                })
            if sigma == EXAMPLE_SIGMA and seed == EXAMPLE_SEED:
                example = {"times": times, "estimates": estimates,
                           "true_efficiencies": true_efficiencies}

        summaries.append({"sigma_speed": sigma, "sigma_yaw": sigma,
                          "methods": {method: summarize_runs(measurements[method]) for method in METHODS}})

    return summaries, example


def save_figure(summaries, example, output_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    times = example["times"]
    for side, title in [(0, "Left efficiency: fault adaptation"), (1, "Right efficiency: false adaptation")]:
        axis = axes[0, side]
        for method in METHODS:
            axis.plot(times, example["estimates"][method][:, side], color=COLORS[method],
                      label=LABELS[method], linewidth=1.0)
        axis.step(times, example["true_efficiencies"][:, side], where="post", color="black",
                  linestyle="--", label="Actual (evaluation only)")
        axis.axvline(CHANGE_TIME, color="gray", linestyle=":")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Efficiency estimate")
        axis.set_title(title)
        axis.legend(fontsize=7, loc="best")

    sigmas = [item["sigma_speed"] for item in summaries]
    for method in ["error_update", "window_ls"]:
        means = [item["methods"][method]["left_rmse_mean"] for item in summaries]
        spreads = [item["methods"][method]["left_rmse_sd"] for item in summaries]
        axes[1, 0].errorbar(sigmas, means, yerr=spreads, color=COLORS[method],
                           marker="o", capsize=3, label=LABELS[method])
    axes[1, 0].set_title("Steady left-efficiency RMSE: mean +/- 1 SD")
    axes[1, 0].text(0.03, 0.95, "Fixed model: left RMSE = 0.5", transform=axes[1, 0].transAxes,
                    va="top", fontsize=8)
    axes[1, 0].set_ylabel("Efficiency RMSE (10-20 s)")
    axes[1, 0].legend(fontsize=8, loc="lower right")

    for method in METHODS:
        rates = [100.0 * item["methods"][method]["recovery_rate"] for item in summaries]
        axes[1, 1].plot(sigmas, rates, color=COLORS[method], marker="o", label=LABELS[method])
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].set_ylabel("Confirmed left-recovery rate (%)")
    axes[1, 1].set_title("Within 15 s after fault; failures retained")
    axes[1, 1].legend(fontsize=8, loc="best")

    for axis in axes[1]:
        axis.set_xlabel("Noise std: speed (m/s), yaw rate (rad/s)")
        axis.set_xticks(sigmas)
    for axis in axes.flat:
        axis.grid(True, alpha=0.35)

    seeds = summaries[0]["methods"]["fixed"]["total_runs"]
    figure.suptitle(f"Noisy identification: example sigma=0.01, seed=0; {seeds} seeds per noise level")
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
    summaries, example = run_study(args.seeds)

    protocol = {
        "dt": DT, "duration": DURATION, "change_time": CHANGE_TIME,
        "steady_interval": [STEADY_START, DURATION], "alpha": ALPHA,
        "window_steps": WINDOW_STEPS, "window_seconds": WINDOW_STEPS * DT,
        "ema_mean_sample_age_seconds": (1.0 - ALPHA) / ALPHA * DT,
        "window_mean_sample_age_seconds": (WINDOW_STEPS - 1) / 2.0 * DT,
        "parameter_tolerance": PARAMETER_TOLERANCE, "hold_duration": HOLD_DURATION,
        "recovery_definition": "Left efficiency within 0.05 of 0.5 for 0.5 s; delay includes hold time.",
        "median_definition": "Unrecovered runs count as infinity; null means infinite median.",
        "noise": "Independent Gaussian speed and yaw-rate measurement noise; same numeric std, different units.",
        "prediction_score": "Noisy current state as input; true next state only as evaluation target.",
        "assumptions": "Known mass, inertia and drags; constant noiseless commands; no process noise or slip; no clipping.",
    }
    json_path = args.output_dir / "noisy_identification_summary.json"
    json_path.write_text(json.dumps({"protocol": protocol, "results": summaries}, indent=2,
                                    allow_nan=False), encoding="utf-8")

    print("Same motion and noisy observations for all methods; no fault-triggered reset.")
    print("EMA mean sample age: 0.38 s; window LS mean sample age: 0.39 s.")
    print("Recovery is a LEFT-parameter metric; confirmation includes the 0.5 s hold time.")
    print("Steady RMSE is evaluated over 10-20 s; SD is across seeds.")
    for item in summaries:
        print(f"\nNoise std: speed={item['sigma_speed']:.3f} m/s, yaw={item['sigma_yaw']:.3f} rad/s")
        for method in METHODS:
            record = item["methods"][method]
            median = record["median_confirmed_recovery"]
            median_text = "not recovered" if median is None else f"{median:.2f} s"
            print(f"  {LABELS[method]}:")
            print(f"    Left RMSE: {record['left_rmse_mean']:.5f} +/- {record['left_rmse_sd']:.5f}")
            print(f"    Right RMSE: {record['right_rmse_mean']:.5f} +/- {record['right_rmse_sd']:.5f}")
            print(f"    Confirmed recovery: {record['recovered_runs']}/{record['total_runs']}; median={median_text}")

    figure_path = args.output_dir / "noisy_identification_result.png"
    save_figure(summaries, example, figure_path, show=not args.no_show)
    print(f"\nSaved plot: {figure_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
