"""Compare fixed and adaptive one-step predictions on the same robot motion."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 示例参数：质量、阻力已知，无观测噪声，只估计有效驱动力。
mass = 10.0
drag = 4.0
dt = 0.02
duration = 10.0
change_time = 5.0
command_force = 4.0
learning_rate = 0.05
estimated_force = 4.0
velocity = 0.0

# 评价规则预先固定；不参与预测模型的计算。
recovery_threshold = 0.0004  # 一步速度预测误差的绝对值，单位m/s
hold_duration = 0.5         # 必须连续满足阈值至少0.5秒

times = [0.0]
speeds = [velocity]
fixed_predictions = [velocity]
adaptive_predictions = [velocity]
fixed_errors = [0.0]
adaptive_errors = [0.0]
force_estimates = [estimated_force]

for step in range(round(duration / dt)):
    t = step * dt

    # 1. 两个模型使用完全相同的当前速度和驱动指令。
    fixed_next = velocity + (
        command_force - drag * velocity
    ) / mass * dt
    adaptive_next = velocity + (
        estimated_force - drag * velocity
    ) / mass * dt

    # 2. 只有仿真器知道变化时刻及实际驱动力。
    if t < change_time:
        actual_force = command_force
    else:
        actual_force = command_force * 0.5
    actual_next = velocity + (
        actual_force - drag * velocity
    ) / mass * dt

    # 3. 获得实际下一刻速度后，先确定本次误差。
    fixed_error = actual_next - fixed_next
    adaptive_error = actual_next - adaptive_next

    # 4. 更新估计值；更新结果只能用于下一次预测。
    estimated_force = (
        estimated_force + learning_rate * mass / dt * adaptive_error
    )

    velocity = actual_next
    times.append((step + 1) * dt)
    speeds.append(actual_next)
    fixed_predictions.append(fixed_next)
    adaptive_predictions.append(adaptive_next)
    fixed_errors.append(fixed_error)
    adaptive_errors.append(adaptive_error)
    force_estimates.append(estimated_force)


def find_recovery_time(errors):
    """事后评价：返回连续满足阈值的起点相对变化时刻的时间。"""
    # 0.5秒在本例对应25个采样间隔，因此检查26个采样点。
    hold_samples = math.ceil(hold_duration / dt) + 1
    for start in range(1, len(times) - hold_samples + 1):
        if times[start] <= change_time:
            continue
        window = errors[start:start + hold_samples]
        if all(abs(error) <= recovery_threshold for error in window):
            return times[start] - change_time
    return None


# 只统计变化后的预测误差；初始记录不是预测样本。
post_change = np.asarray(times) > change_time
fixed_rmse = float(np.sqrt(np.mean(np.asarray(fixed_errors)[post_change] ** 2)))
adaptive_rmse = float(np.sqrt(np.mean(np.asarray(adaptive_errors)[post_change] ** 2)))
fixed_recovery = find_recovery_time(fixed_errors)
adaptive_recovery = find_recovery_time(adaptive_errors)

print(f"Post-change fixed RMSE:    {fixed_rmse:.8f} m/s")
print(f"Post-change adaptive RMSE: {adaptive_rmse:.8f} m/s")
print(f"Recovery threshold: {recovery_threshold:.4f} m/s")
print(f"Required hold duration: {hold_duration:.2f} s")
if fixed_recovery is None:
    print(f"Fixed recovery: not recovered within {duration - change_time:.2f} s")
else:
    print(f"Fixed recovery time: {fixed_recovery:.2f} s")
if adaptive_recovery is None:
    print(f"Adaptive recovery: not recovered within {duration - change_time:.2f} s")
else:
    print(f"Adaptive recovery time: {adaptive_recovery:.2f} s")
    print("Recovery time is the start of the interval; it is confirmed 0.5 s later.")
print(f"Final estimated force: {estimated_force:.6f} N")

figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(times, speeds, color="black", label="Actual speed")
axes[0].plot(times, fixed_predictions, "--", label="Fixed one-step prediction")
axes[0].plot(times, adaptive_predictions, ":", label="Adaptive one-step prediction")
axes[0].set_ylabel("Speed (m/s)")
axes[0].set_title("Same motion and input for both predictors")
axes[0].legend(loc="lower right")

axes[1].plot(times, fixed_errors, label="Fixed error")
axes[1].plot(times, adaptive_errors, label="Adaptive error")
axes[1].axhspan(-recovery_threshold, recovery_threshold,
               color="green", alpha=0.12, label="Recovery band")
if adaptive_recovery is not None:
    axes[1].axvline(change_time + adaptive_recovery, color="green",
                   linestyle="--", label=f"Recovery: {adaptive_recovery:.2f} s")
axes[1].set_ylabel("Prediction error (m/s)")
axes[1].legend(loc="lower left")

# 实际驱动力仅用于事后画图，不提供给预测器或更新公式。
axes[2].step([0.0, change_time, duration],
             [command_force, command_force * 0.5, command_force * 0.5],
             where="post", color="black", linestyle="--", label="Actual force")
axes[2].plot(times, force_estimates, label="Estimated force (after update)")
axes[2].set_ylabel("Driving force (N)")
axes[2].set_xlabel("Time (s)")
axes[2].legend(loc="upper right")

for axis in axes:
    axis.axvline(change_time, color="gray", linestyle=":")
    axis.grid(True, alpha=0.4)

figure.tight_layout()
output_path = Path(__file__).resolve().with_name("compare_models_result.png")
figure.savefig(output_path, dpi=150)
print(f"Saved plot: {output_path}")
plt.show()
