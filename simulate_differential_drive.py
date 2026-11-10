"""Lesson 4: planar differential-drive dynamics with a left-motor fault."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 平地、无侧向滑移的简化模型，尚未加入低摩擦和轮胎滑移。
mass = 10.0               # 质量，kg
yaw_inertia = 0.5         # 绕竖直轴转动的惯量，kg*m^2
half_track = 0.2          # 轮距的一半，m；左右轮间距为0.4 m
linear_drag = 4.0         # 直线运动的阻力系数，N*s/m
yaw_drag = 0.8            # 转动阻力系数，N*m*s/rad
dt = 0.02
duration = 12.0
change_time = 5.0

# 正常情况下，每侧驱动指令对应2 N，合计4 N，可维持1 m/s。
left_command = 2.0
right_command = 2.0
initial_speed = 1.0
steps = round(duration / dt)
times = np.arange(steps + 1) * dt


def simulate(left_fault):
    """状态顺序：[x, y, 朝向角, 线速度, 角速度]。"""
    states = np.zeros((steps + 1, 5))
    states[0] = [0.0, 0.0, 0.0, initial_speed, 0.0]

    for step in range(steps):
        time = step * dt
        x, y, heading, speed, yaw_rate = states[step]

        # 1. 仿真器设置电机效率；两个场景的驱动指令完全相同。
        left_efficiency = 1.0
        right_efficiency = 1.0
        if left_fault and time >= change_time:
            left_efficiency = 0.5

        left_force = left_efficiency * left_command
        right_force = right_efficiency * right_command

        # 2. 两侧驱动力之和决定直线加速度。
        acceleration = (
            left_force + right_force - linear_drag * speed
        ) / mass

        # 3. 两侧驱动力之差产生转动力矩。
        # 右侧力更大时，角加速度为正，车头逆时针向左转。
        yaw_acceleration = (
            half_track * (right_force - left_force) - yaw_drag * yaw_rate
        ) / yaw_inertia

        # 4. 当前速度决定位置变化，当前角速度决定朝向变化。
        # 所有状态都由同一个当前时刻出发，以显式欧拉法更新。
        states[step + 1] = [
            x + speed * math.cos(heading) * dt,
            y + speed * math.sin(heading) * dt,
            heading + yaw_rate * dt,
            speed + acceleration * dt,
            yaw_rate + yaw_acceleration * dt,
        ]

    return states


normal = simulate(left_fault=False)
fault = simulate(left_fault=True)
change_index = round(change_time / dt)

print("Initial state: x=0 m, y=0 m, heading=0 rad, speed=1 m/s, yaw rate=0 rad/s.")
print("The drive commands stay at 2 N nominal per side in both runs.")
print("Fault run: left efficiency drops from 1.0 to 0.5 at 5 s; right stays at 1.0.")
for label, states in [("Normal", normal), ("Left-motor fault", fault)]:
    x, y, heading, speed, yaw_rate = states[-1]
    print(f"\n{label} at {duration:.1f} s:")
    print(f"  Position: x={x:.3f} m, y={y:.3f} m")
    print(f"  Heading: {math.degrees(heading):.2f} degrees")
    print(f"  Forward speed: {speed:.4f} m/s")
    print(f"  Yaw rate: {yaw_rate:.4f} rad/s")

expected_speed = (left_command * 0.5 + right_command) / linear_drag
expected_yaw_rate = half_track * (right_command - left_command * 0.5) / yaw_drag
print(f"\nFault steady-speed limit: {expected_speed:.2f} m/s")
print(f"Fault steady-yaw-rate limit: {expected_yaw_rate:.2f} rad/s")
print("Positive heading / yaw rate means a counterclockwise left turn.")

figure = plt.figure(figsize=(12, 8))
grid = figure.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
trajectory_axis = figure.add_subplot(grid[0, :])
heading_axis = figure.add_subplot(grid[1, 0])
speed_axis = figure.add_subplot(grid[1, 1])
yaw_axis = figure.add_subplot(grid[1, 2])

trajectory_axis.plot(normal[:, 0], normal[:, 1], color="tab:blue", label="Normal motors")
trajectory_axis.plot(fault[:, 0], fault[:, 1], color="tab:orange", label="Left motor at 50% after 5 s")
trajectory_axis.scatter([0.0], [0.0], color="black", marker="o", s=25, label="Start")
trajectory_axis.scatter([fault[change_index, 0]], [fault[change_index, 1]],
                        color="red", marker="x", s=60, label="Fault occurs")
trajectory_axis.set_aspect("equal", adjustable="box")
trajectory_axis.set_xlabel("World x (m)")
trajectory_axis.set_ylabel("World y (m)")
trajectory_axis.set_title("Same wheel commands: a left-motor fault causes a left turn")
trajectory_axis.legend(loc="upper left", fontsize=9)
trajectory_axis.grid(True, alpha=0.4)

for states, label, color in [
        (normal, "Normal", "tab:blue"),
        (fault, "Left-motor fault", "tab:orange")]:
    heading_axis.plot(times, np.degrees(states[:, 2]), color=color, label=label)
    speed_axis.plot(times, states[:, 3], color=color, label=label)
    yaw_axis.plot(times, states[:, 4], color=color, label=label)

heading_axis.set_ylabel("Heading (degrees)")
heading_axis.set_title("Orientation")
speed_axis.set_ylabel("Forward speed (m/s)")
speed_axis.set_title("Linear motion")
yaw_axis.set_ylabel("Yaw rate (rad/s)")
yaw_axis.set_title("Turning motion")
for axis in [heading_axis, speed_axis, yaw_axis]:
    axis.axvline(change_time, color="gray", linestyle=":")
    axis.set_xlabel("Time (s)")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.4)

figure.tight_layout()
output_path = Path(__file__).resolve().with_name("differential_drive_result.png")
figure.savefig(output_path, dpi=150)
print(f"Saved plot: {output_path}")
plt.show()
