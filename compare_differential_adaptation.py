"""Lesson 5: estimate both wheel efficiencies from prediction errors."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 与上一课相同的二维模型：平地、无侧向滑移、无观测噪声。
# 本课只让电机效率变化，质量、惯量和阻力均为已知。
mass = 10.0
yaw_inertia = 0.5
half_track = 0.2
linear_drag = 4.0
yaw_drag = 0.8
dt = 0.02
duration = 12.0
change_time = 5.0
left_efficiency_after_change = 0.5
right_efficiency_after_change = 1.0
left_command = 2.0
right_command = 2.0
learning_rate = 0.05
min_identification_command = 0.1

# 联合恢复标准：两种预测误差同时满足阈值并保持0.5秒。
speed_error_tolerance = 0.0002  # m/s
yaw_error_tolerance = 0.0008    # rad/s
hold_duration = 0.5
steps = round(duration / dt)
times = np.arange(steps + 1) * dt


def predict_motion(state, left_estimate, right_estimate):
    """只用当前状态、驱动指令和效率估计，预测下一步[v, omega]。"""
    speed, yaw_rate = state[3], state[4]
    estimated_left_force = left_estimate * left_command
    estimated_right_force = right_estimate * right_command
    next_speed = speed + (
        estimated_left_force + estimated_right_force - linear_drag * speed
    ) / mass * dt
    next_yaw_rate = yaw_rate + (
        half_track * (estimated_right_force - estimated_left_force)
        - yaw_drag * yaw_rate
    ) / yaw_inertia * dt
    return np.array([next_speed, next_yaw_rate])


def plant_step(state, time):
    """仿真器：真实效率和变化时刻只在这里用于生成运动。"""
    if time < change_time:
        actual_left_efficiency = 1.0
        actual_right_efficiency = 1.0
    else:
        actual_left_efficiency = left_efficiency_after_change
        actual_right_efficiency = right_efficiency_after_change

    left_force = actual_left_efficiency * left_command
    right_force = actual_right_efficiency * right_command
    x, y, heading, speed, yaw_rate = state
    acceleration = (left_force + right_force - linear_drag * speed) / mass
    yaw_acceleration = (
        half_track * (right_force - left_force) - yaw_drag * yaw_rate
    ) / yaw_inertia
    return np.array([
        x + speed * math.cos(heading) * dt,
        y + speed * math.sin(heading) * dt,
        heading + yaw_rate * dt,
        speed + acceleration * dt,
        yaw_rate + yaw_acceleration * dt,
    ])


states = np.zeros((steps + 1, 5))
states[0] = [0.0, 0.0, 0.0, 1.0, 0.0]
fixed_errors = np.zeros((steps + 1, 2))
adaptive_errors = np.zeros((steps + 1, 2))
efficiency_estimates = np.ones((steps + 1, 2))
left_estimate = 1.0
right_estimate = 1.0

for step in range(steps):
    time = step * dt
    current_state = states[step]

    # 1. 两个预测器读取完全相同的当前状态和驱动指令。
    fixed_prediction = predict_motion(current_state, 1.0, 1.0)
    adaptive_prediction = predict_motion(current_state, left_estimate, right_estimate)

    # 2. 执行之后才能获得真实下一步状态，再确定本次误差。
    next_state = plant_step(current_state, time)
    actual_motion = next_state[3:5]
    fixed_error = actual_motion - fixed_prediction
    adaptive_error = actual_motion - adaptive_prediction

    # 3. 线速度误差对应左右力偏差之和，角速度误差对应力偏差之差。
    sum_force_error = mass / dt * adaptive_error[0]
    difference_force_error = yaw_inertia / (half_track * dt) * adaptive_error[1]
    left_force_error = 0.5 * (sum_force_error - difference_force_error)
    right_force_error = 0.5 * (sum_force_error + difference_force_error)

    # 4. 分别更新两侧效率；新估计供下一次预测使用。
    # 指令接近零时无法可靠辨识该侧效率，因此暂停该侧更新。
    if abs(left_command) >= min_identification_command:
        left_estimate += learning_rate * left_force_error / left_command
        left_estimate = float(np.clip(left_estimate, 0.1, 1.5))
    if abs(right_command) >= min_identification_command:
        right_estimate += learning_rate * right_force_error / right_command
        right_estimate = float(np.clip(right_estimate, 0.1, 1.5))

    states[step + 1] = next_state
    fixed_errors[step + 1] = fixed_error
    adaptive_errors[step + 1] = adaptive_error
    efficiency_estimates[step + 1] = [left_estimate, right_estimate]


def find_joint_recovery(errors):
    """事后评价：两种误差同时进入容差带，并连续保持至少0.5秒。"""
    acceptable = (
        (np.abs(errors[:, 0]) <= speed_error_tolerance)
        & (np.abs(errors[:, 1]) <= yaw_error_tolerance)
    )
    hold_samples = math.ceil(hold_duration / dt) + 1
    for start in range(1, len(times) - hold_samples + 1):
        if times[start] <= change_time:
            continue
        if np.all(acceptable[start:start + hold_samples]):
            return float(times[start] - change_time)
    return None


post_change = times > change_time
fixed_rmse = np.sqrt(np.mean(fixed_errors[post_change] ** 2, axis=0))
adaptive_rmse = np.sqrt(np.mean(adaptive_errors[post_change] ** 2, axis=0))
fixed_recovery = find_joint_recovery(fixed_errors)
adaptive_recovery = find_joint_recovery(adaptive_errors)

print("Both predictors read the same state/input history. Commands stay unchanged.")
print("Actual left efficiency changes from 1.0 to 0.5 at 5 s; right stays at 1.0.")
print(f"Post-change fixed speed RMSE:    {fixed_rmse[0]:.8f} m/s")
print(f"Post-change adaptive speed RMSE: {adaptive_rmse[0]:.8f} m/s")
print(f"Post-change fixed yaw RMSE:      {fixed_rmse[1]:.8f} rad/s")
print(f"Post-change adaptive yaw RMSE:   {adaptive_rmse[1]:.8f} rad/s")
if fixed_recovery is None:
    print(f"Fixed joint recovery: not recovered within {duration - change_time:.2f} s")
else:
    print(f"Fixed joint recovery: {fixed_recovery:.2f} s")
if adaptive_recovery is None:
    print(f"Adaptive joint recovery: not recovered within {duration - change_time:.2f} s")
else:
    print(f"Adaptive joint recovery: {adaptive_recovery:.2f} s")
print("Recovery is measured to the interval start and confirmed 0.5 s later.")
print(f"Final estimated left efficiency:  {left_estimate:.6f}")
print(f"Final estimated right efficiency: {right_estimate:.6f}")
print("The controller is unchanged: this experiment evaluates prediction adaptation.")

figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
error_settings = [
    (axes[0, 0], 0, speed_error_tolerance, "Speed prediction error (m/s)", "Forward-speed prediction"),
    (axes[0, 1], 1, yaw_error_tolerance, "Yaw-rate prediction error (rad/s)", "Turning-rate prediction"),
]
for axis, index, tolerance, ylabel, title in error_settings:
    axis.plot(times, fixed_errors[:, index], color="tab:blue", label="Fixed model")
    axis.plot(times, adaptive_errors[:, index], color="tab:orange", label="Adaptive model")
    axis.axhspan(-tolerance, tolerance, color="green", alpha=0.12, label="Recovery band")
    if adaptive_recovery is not None:
        axis.axvline(change_time + adaptive_recovery, color="green", linestyle="--",
                     label=f"Joint recovery: {adaptive_recovery:.2f} s")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(fontsize=8, loc="best")

# 真实效率只用于事后画图，不提供给预测器或更新公式。
for axis, index, actual_after, title in [
        (axes[1, 0], 0, left_efficiency_after_change, "Left motor efficiency"),
        (axes[1, 1], 1, right_efficiency_after_change, "Right motor efficiency")]:
    axis.step([0.0, change_time, duration], [1.0, actual_after, actual_after],
              where="post", color="black", linestyle="--", label="Actual efficiency")
    axis.plot(times, efficiency_estimates[:, index], color="tab:orange",
              label="Estimated efficiency (after update)")
    axis.set_ylim(0.4, 1.1)
    axis.set_ylabel("Efficiency")
    axis.set_xlabel("Time (s)")
    axis.set_title(title)
    axis.legend(fontsize=8, loc="best")

for axis in axes.flat:
    axis.axvline(change_time, color="gray", linestyle=":")
    axis.grid(True, alpha=0.4)

figure.suptitle("Two motor-efficiency estimates from speed and yaw prediction errors")
figure.tight_layout()
output_path = Path(__file__).resolve().with_name("differential_adaptation_result.png")
figure.savefig(output_path, dpi=150)
print(f"Saved plot: {output_path}")
plt.show()
