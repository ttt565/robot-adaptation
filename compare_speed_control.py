"""Lesson 3: use prediction-error adaptation for closed-loop speed tracking."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 本例只研究直线速度：质量、阻力已知，速度观测无噪声。
mass = 10.0
drag = 4.0
dt = 0.02
duration = 20.0
change_time = 5.0
target_speed = 1.0

# 从正常稳定行驶开始，单独观察故障后的恢复过程。
initial_speed = target_speed
speed_gain = 1.5       # 速度误差转换为期望加速度的增益，单位1/s
integral_gain = 10.0   # 积分补偿增益，单位N/m
learning_rate = 0.05
min_command = 0.0
max_command = 10.0     # 正常状态下的力指令上限，单位N
min_identification_command = 0.1

# 恢复指标是速度跟踪误差，不是上一课的预测误差。
speed_tolerance = 0.02  # 目标速度的正负0.02 m/s
hold_duration = 0.5
steps = round(duration / dt)
times = np.arange(steps + 1) * dt


def plant_step(speed, command, time):
    """仿真器：只有这里读取真实效率和变化时刻。"""
    if time < change_time:
        actual_efficiency = 1.0
    else:
        actual_efficiency = 0.5
    actual_force = actual_efficiency * command
    return speed + (actual_force - drag * speed) / mass * dt


def simulate_controller(mode):
    """每个控制器独立闭环运行，初始状态和物理变化完全相同。"""
    speed = initial_speed
    efficiency_estimate = 1.0
    integral_error = 0.0

    speeds = [speed]
    commands = []
    estimates = [efficiency_estimate]
    prediction_errors = [0.0]

    for step in range(steps):
        time = step * dt
        speed_error = target_speed - speed

        # 1. 相同的基础控制器：希望速度误差逐渐减小。
        desired_acceleration = speed_gain * speed_error
        required_force = mass * desired_acceleration + drag * speed

        if mode == "adaptive":
            # 效率降低时，提高指令以产生所需的实际力。
            raw_command = required_force / efficiency_estimate
        elif mode == "pi":
            # 对照：使用速度误差的积分进行补偿，不更新动力学模型。
            raw_command = required_force + integral_gain * integral_error
        elif mode == "fixed":
            raw_command = required_force
        else:
            raise ValueError(f"Unknown controller: {mode}")

        command = float(np.clip(raw_command, min_command, max_command))

        # 2. 执行之前预测；固定模型和PI都假设效率为1。
        predicted_next = speed + (
            efficiency_estimate * command - drag * speed
        ) / mass * dt

        # 3. 执行同一个驱动指令，获得实际下一刻速度。
        actual_next = plant_step(speed, command, time)
        prediction_error = actual_next - predicted_next

        # 4. 用本次误差更新效率，供下一周期使用。
        # 指令接近零时，运动数据不足以辨识效率，因此暂停更新。
        if mode == "adaptive" and command >= min_identification_command:
            efficiency_estimate += (
                learning_rate * mass / (dt * command) * prediction_error
            )
            efficiency_estimate = float(np.clip(efficiency_estimate, 0.1, 1.5))

        # PI的积分只使用速度误差。达到指令上限时避免继续积累饱和。
        if mode == "pi":
            if (min_command <= raw_command <= max_command
                    or (raw_command > max_command and speed_error < 0.0)
                    or (raw_command < min_command and speed_error > 0.0)):
                integral_error += speed_error * dt

        speed = actual_next
        speeds.append(speed)
        commands.append(command)
        estimates.append(efficiency_estimate)
        prediction_errors.append(prediction_error)

    return {
        "speeds": np.asarray(speeds),
        "commands": np.asarray(commands),
        "estimates": np.asarray(estimates),
        "prediction_errors": np.asarray(prediction_errors),
    }


def find_speed_recovery(speeds):
    """先检测离开容差带，再检测重新进入且保持足够长时间。"""
    errors = np.abs(target_speed - speeds)
    outside = np.flatnonzero((times > change_time) & (errors > speed_tolerance))
    if len(outside) == 0:
        return 0.0  # 始终留在容差带内。

    first_outside = int(outside[0])
    hold_samples = math.ceil(hold_duration / dt) + 1
    for start in range(first_outside + 1, len(times) - hold_samples + 1):
        if np.all(errors[start:start + hold_samples] <= speed_tolerance):
            return float(times[start] - change_time)
    return None


results = {
    "fixed": simulate_controller("fixed"),
    "pi": simulate_controller("pi"),
    "adaptive": simulate_controller("adaptive"),
}
labels = {
    "fixed": "Fixed model + feedback",
    "pi": "Fixed model + PI compensation",
    "adaptive": "Adaptive model + feedback",
}
colors = {"fixed": "tab:blue", "pi": "tab:green", "adaptive": "tab:orange"}

post_change = times > change_time
metrics = {}
print("Target speed: 1.00 m/s; actual efficiency changes from 1.0 to 0.5 at 5 s.")
print(f"Command range: {min_command:.1f} to {max_command:.1f} N (nominal-force command).")
print(f"Speed recovery band: +/-{speed_tolerance:.2f} m/s, held for {hold_duration:.2f} s.")
print("Recovery time is measured from the fault to the start of a qualifying interval.")
for mode, data in results.items():
    tracking_errors = target_speed - data["speeds"][post_change]
    peak_error = float(np.max(np.abs(tracking_errors)))
    rmse = float(np.sqrt(np.mean(tracking_errors ** 2)))
    recovery = find_speed_recovery(data["speeds"])
    metrics[mode] = {"peak_error": peak_error, "rmse": rmse, "recovery": recovery}
    recovery_text = "not recovered" if recovery is None else f"{recovery:.2f} s"
    print(f"\n{labels[mode]}")
    print(f"  Peak speed error: {peak_error:.4f} m/s")
    print(f"  Post-change speed RMSE: {rmse:.4f} m/s")
    print(f"  Speed recovery: {recovery_text}")
    print(f"  Final speed: {data['speeds'][-1]:.4f} m/s")
    print(f"  Largest command: {np.max(data['commands']):.4f} N")
print(f"\nFinal adaptive efficiency estimate: {results['adaptive']['estimates'][-1]:.6f}")

figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].axhspan(target_speed - speed_tolerance, target_speed + speed_tolerance,
               color="gray", alpha=0.15, label="Speed recovery band")
axes[0].axhline(target_speed, color="black", linestyle="--", label="Target speed")
for mode, data in results.items():
    axes[0].plot(times, data["speeds"], color=colors[mode], label=labels[mode])
axes[0].set_ylabel("Actual speed (m/s)")
axes[0].set_title("Closed-loop speed tracking: independent run for each controller")
axes[0].legend(loc="lower left", fontsize=8)

for mode, data in results.items():
    axes[1].step(times[:-1], data["commands"], where="post",
                 color=colors[mode], label=labels[mode])
axes[1].axhline(max_command, color="black", linestyle="--", label="Command limit")
axes[1].set_ylim(0.0, max_command + 0.6)
axes[1].set_ylabel("Drive command (N nominal)")
axes[1].legend(loc="lower right", fontsize=8)

# 真实效率仅用于事后画图，不提供给控制器或估计器。
axes[2].step([0.0, change_time, duration], [1.0, 0.5, 0.5],
             where="post", color="black", linestyle="--", label="Actual efficiency")
axes[2].plot(times, results["adaptive"]["estimates"], color=colors["adaptive"],
             label="Adaptive efficiency estimate (after update)")
axes[2].set_ylabel("Drive efficiency")
axes[2].set_xlabel("Time (s)")
axes[2].legend(loc="upper right", fontsize=8)

for axis in axes:
    axis.axvline(change_time, color="gray", linestyle=":")
    axis.grid(True, alpha=0.4)

figure.tight_layout()
output_path = Path(__file__).resolve().with_name("compare_speed_control_result.png")
figure.savefig(output_path, dpi=150)
print(f"Saved plot: {output_path}")
plt.show()
