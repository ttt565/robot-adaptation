"""Lesson 6: use two efficiency estimates for planar trajectory tracking."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 简化条件：平地、无侧向滑移、无观测噪声；已知质量、惯量和阻力。
mass = 10.0
yaw_inertia = 0.5
half_track = 0.2
linear_drag = 4.0
yaw_drag = 0.8
dt = 0.02
duration = 25.0
change_time = 5.0

# 参考轨迹：x=t、y=0，车头朝向x轴正方向，速度保持1 m/s。
reference_speed = 1.0
position_gain = 1.0
lateral_gain = 1.5
heading_gain = 2.0
speed_gain = 2.0
yaw_rate_gain = 5.0
speed_integral_gain = 10.0
yaw_integral_gain = 1.0
learning_rate = 0.05
max_command = 6.0  # 每侧正常力指令限制为[-6, 6] N，可正反向驱动。
min_identification_command = 0.1

# 事后评价：位置、朝向、速度同时达标，并持续1秒。
position_tolerance = 0.02
heading_tolerance = math.radians(1.0)
speed_tolerance = 0.02
hold_duration = 1.0

steps = round(duration / dt)
times = np.arange(steps + 1) * dt
references = np.zeros((steps + 1, 5))
references[:, 0] = reference_speed * times
references[:, 3] = reference_speed


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def dynamics_step(state, forces):
    """已知动力学结构；forces是实际作用在车体上的左右驱动力。"""
    x, y, heading, speed, yaw_rate = state
    left_force, right_force = forces
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


def plant_step(state, command, time):
    """真实效率及故障时刻只用于仿真器生成运动。"""
    if time < change_time:
        actual_efficiencies = np.array([1.0, 1.0])
    else:
        actual_efficiencies = np.array([0.5, 1.0])
    return dynamics_step(state, actual_efficiencies * command)


def tracking_targets(state, reference):
    """所有控制器使用相同的位置/朝向反馈，生成期望线速度和角速度。"""
    heading = state[2]
    dx, dy = reference[:2] - state[:2]
    forward_error = math.cos(heading) * dx + math.sin(heading) * dy
    lateral_error = -math.sin(heading) * dx + math.cos(heading) * dy
    heading_error = wrap_angle(reference[2] - heading)

    desired_speed = reference[3] * math.cos(heading_error) + position_gain * forward_error
    desired_yaw_rate = (
        reference[4] + reference[3] * lateral_gain * lateral_error
        + reference[3] * heading_gain * math.sin(heading_error)
    )
    return float(np.clip(desired_speed, 0.0, 1.5)), float(np.clip(desired_yaw_rate, -1.0, 1.0))


def simulate_controller(mode):
    """各控制器分别闭环运行；初态、目标和实际故障相同。"""
    states = np.zeros((steps + 1, 5))
    states[0] = references[0]
    commands = np.zeros((steps, 2))
    estimates = np.ones((steps + 1, 2))
    prediction_errors = np.zeros((steps + 1, 2))
    efficiency_estimate = np.ones(2)
    integrated_speed_error = 0.0
    integrated_yaw_error = 0.0

    for step in range(steps):
        state = states[step]
        time = step * dt
        desired_speed, desired_yaw_rate = tracking_targets(state, references[step])
        speed_error = desired_speed - state[3]
        yaw_error = desired_yaw_rate - state[4]

        # 1. 期望实际合力和转矩，三种方法使用相同基础反馈增益。
        required_sum_force = mass * speed_gain * speed_error + linear_drag * state[3]
        required_torque = yaw_inertia * yaw_rate_gain * yaw_error + yaw_drag * state[4]
        if mode == "pi":
            required_sum_force += speed_integral_gain * integrated_speed_error
            required_torque += yaw_integral_gain * integrated_yaw_error

        # 2. 合力决定前进，力差决定转弯，转换成左右实际力需求。
        required_difference_force = required_torque / half_track
        required_forces = np.array([
            (required_sum_force - required_difference_force) / 2.0,
            (required_sum_force + required_difference_force) / 2.0,
        ])

        # 自适应控制使用当前效率估计；固定和PI的效率假设保持为1。
        raw_command = required_forces / efficiency_estimate
        command = np.clip(raw_command, -max_command, max_command)

        # 3. 执行前预测，获得观测后记录误差。
        predicted_motion = dynamics_step(state, efficiency_estimate * command)[3:5]
        next_state = plant_step(state, command, time)
        prediction_error = next_state[3:5] - predicted_motion

        # 4. 只使用预测误差及实际发出的指令，更新两侧效率。
        if mode == "adaptive":
            sum_force_error = mass / dt * prediction_error[0]
            difference_force_error = yaw_inertia / (half_track * dt) * prediction_error[1]
            wheel_force_errors = np.array([
                (sum_force_error - difference_force_error) / 2.0,
                (sum_force_error + difference_force_error) / 2.0,
            ])
            for side in range(2):
                if abs(command[side]) >= min_identification_command:
                    efficiency_estimate[side] += (
                        learning_rate * wheel_force_errors[side] / command[side]
                    )
            efficiency_estimate = np.clip(efficiency_estimate, 0.1, 1.5)

        # PI只积累跟踪偏差，不更新模型。指令饱和时暂停积分以避免累积。
        if mode == "pi" and np.all(np.abs(raw_command) <= max_command):
            integrated_speed_error += speed_error * dt
            integrated_yaw_error += yaw_error * dt

        states[step + 1] = next_state
        commands[step] = command
        estimates[step + 1] = efficiency_estimate
        prediction_errors[step + 1] = prediction_error

    return {"states": states, "commands": commands, "estimates": estimates,
            "prediction_errors": prediction_errors}


def tracking_errors(states):
    positions = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)
    headings = np.arctan2(np.sin(states[:, 2] - references[:, 2]),
                          np.cos(states[:, 2] - references[:, 2]))
    speeds = states[:, 3] - references[:, 3]
    return positions, headings, speeds


def find_tracking_recovery(states):
    position_errors, heading_errors, speed_errors = tracking_errors(states)
    acceptable = (
        (position_errors <= position_tolerance)
        & (np.abs(heading_errors) <= heading_tolerance)
        & (np.abs(speed_errors) <= speed_tolerance)
    )
    outside = np.flatnonzero((times > change_time) & ~acceptable)
    if len(outside) == 0:
        return 0.0  # 故障后始终保持在容差内。
    hold_samples = math.ceil(hold_duration / dt) + 1
    for start in range(int(outside[0]) + 1, len(times) - hold_samples + 1):
        if np.all(acceptable[start:start + hold_samples]):
            return float(times[start] - change_time)
    return None


results = {mode: simulate_controller(mode) for mode in ["fixed", "pi", "adaptive"]}
labels = {"fixed": "Fixed model + feedback", "pi": "Fixed model + PI compensation",
          "adaptive": "Adaptive model + feedback"}
colors = {"fixed": "tab:blue", "pi": "tab:green", "adaptive": "tab:orange"}
post_change = times > change_time
metrics = {}

print("Reference: x=t, y=0, heading=0, speed=1 m/s.")
print("Left efficiency drops to 0.5 at 5 s; right stays at 1.0.")
print(f"Each wheel command is limited to +/-{max_command:.1f} N nominal.")
print("Recovery: position error <= 0.02 m, heading error <= 1 degree,")
print("          speed error <= 0.02 m/s, all held for 1 s.")
print("Each controller runs independently; this comparison evaluates tracking, not same-data prediction.")
for mode, data in results.items():
    position_errors, heading_errors, speed_errors = tracking_errors(data["states"])
    rmse = float(np.sqrt(np.mean(position_errors[post_change] ** 2)))
    peak_position = float(np.max(position_errors[post_change]))
    peak_heading = float(np.degrees(np.max(np.abs(heading_errors[post_change]))))
    recovery = find_tracking_recovery(data["states"])
    metrics[mode] = {"position_rmse": rmse, "peak_position": peak_position,
                     "peak_heading_degrees": peak_heading, "recovery": recovery}
    recovery_text = "not recovered" if recovery is None else f"{recovery:.2f} s"
    print(f"\n{labels[mode]}")
    print(f"  Position RMSE after fault: {rmse:.5f} m")
    print(f"  Peak position error: {peak_position:.5f} m")
    print(f"  Peak heading error: {peak_heading:.3f} degrees")
    print(f"  Tracking recovery: {recovery_text}")
    print(f"  Final position error: {position_errors[-1]:.6f} m")
    print(f"  Largest absolute wheel command: {np.max(np.abs(data['commands'])):.4f} N")
print(f"\nFinal adaptive efficiencies: {results['adaptive']['estimates'][-1]}")

figure, axes = plt.subplots(3, 2, figsize=(12, 10))
axes[0, 0].plot(references[:, 0], references[:, 1], "k--", label="Reference")
for mode, data in results.items():
    states = data["states"]
    position_errors, heading_errors, speed_errors = tracking_errors(states)
    axes[0, 0].plot(states[:, 0], states[:, 1], color=colors[mode], label=labels[mode])
    axes[0, 1].plot(times, position_errors, color=colors[mode], label=labels[mode])
    axes[1, 0].plot(times, np.degrees(heading_errors), color=colors[mode], label=labels[mode])
    axes[1, 1].plot(times, states[:, 3], color=colors[mode], label=labels[mode])

axes[0, 0].scatter([reference_speed * change_time], [0.0], color="red", marker="x", s=45)
axes[0, 0].set_xlabel("World x (m)")
axes[0, 0].set_ylabel("World y (m)")
axes[0, 0].set_title("Path view (vertical scale enlarged)")
axes[0, 1].axhspan(0.0, position_tolerance, color="gray", alpha=0.15)
axes[0, 1].set_ylabel("Position error norm (m)")
axes[0, 1].set_title("Error relative to x=t, y=0")
axes[1, 0].axhspan(-math.degrees(heading_tolerance), math.degrees(heading_tolerance),
                  color="gray", alpha=0.15)
axes[1, 0].set_ylabel("Heading error (degrees)")
axes[1, 0].set_title("Orientation tracking")
axes[1, 1].axhspan(reference_speed - speed_tolerance, reference_speed + speed_tolerance,
                  color="gray", alpha=0.15)
axes[1, 1].axhline(reference_speed, color="black", linestyle="--")
axes[1, 1].set_ylabel("Forward speed (m/s)")
axes[1, 1].set_title("Speed tracking")

adaptive = results["adaptive"]
axes[2, 0].step(times[:-1], adaptive["commands"][:, 0], where="post",
                color="tab:orange", label="Left command")
axes[2, 0].step(times[:-1], adaptive["commands"][:, 1], where="post",
                color="tab:purple", label="Right command")
axes[2, 0].axhline(max_command, color="black", linestyle="--", label="Command limits")
axes[2, 0].axhline(-max_command, color="black", linestyle="--")
axes[2, 0].set_ylabel("Wheel command (N nominal)")
axes[2, 0].set_title("Adaptive controller inputs")

axes[2, 1].step([0.0, change_time, duration], [1.0, 0.5, 0.5], where="post",
                color="black", linestyle="--", label="Actual left efficiency")
axes[2, 1].axhline(1.0, color="gray", linestyle="--", label="Actual right efficiency")
axes[2, 1].plot(times, adaptive["estimates"][:, 0], color="tab:orange", label="Estimated left")
axes[2, 1].plot(times, adaptive["estimates"][:, 1], color="tab:purple", label="Estimated right")
axes[2, 1].set_ylim(0.4, 1.1)
axes[2, 1].set_ylabel("Efficiency")
axes[2, 1].set_title("Online estimates used by the controller")

for row, column in [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
    axes[row, column].axvline(change_time, color="gray", linestyle=":")
    axes[row, column].set_xlabel("Time (s)")
for axis in axes.flat:
    axis.grid(True, alpha=0.4)
    axis.legend(fontsize=7, loc="best")

figure.suptitle("Planar trajectory tracking after a left-motor efficiency drop")
figure.tight_layout()
output_path = Path(__file__).resolve().with_name("trajectory_control_result.png")
figure.savefig(output_path, dpi=150)
print(f"Saved plot: {output_path}")
plt.show()
