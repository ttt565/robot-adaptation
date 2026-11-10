import matplotlib.pyplot as plt

mass = 10.0
drag = 4.0
dt = 0.02
velocity = 0.0

# 这个驱动指令在正常状态下应产生4 N的力
command_force = 4.0

times = [0.0]
speeds = [velocity]
predicted_speeds = [velocity]
errors = [0.0]

for step in range(500):
    t = step * dt

    # 1. 固定模型：用当前实际速度，预测下一刻速度
    predicted_acceleration = (
        command_force - drag * velocity
    ) / mass

    predicted_next = velocity + predicted_acceleration * dt

    # 2. 仿真器：计算小车实际的运动
    if t < 5.0:
        actual_force = command_force
    else:
        actual_force = command_force * 0.5

    actual_acceleration = (
        actual_force - drag * velocity
    ) / mass

    actual_next = velocity + actual_acceleration * dt

    # 3. 获得实际下一刻速度后，计算预测误差
    error = actual_next - predicted_next

    # 4. 更新当前速度，保存本次结果
    velocity = actual_next

    times.append((step + 1) * dt)
    speeds.append(actual_next)
    predicted_speeds.append(predicted_next)
    errors.append(error)

plt.figure(figsize=(9, 6))

plt.subplot(2, 1, 1)
plt.plot(times, speeds, label="Actual speed")
plt.plot(times, predicted_speeds, "--", label="One-step prediction")
plt.axvline(5.0, color="gray", linestyle=":")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(times, errors)
plt.axvline(5.0, color="gray", linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Prediction error (m/s)")
plt.grid(True)

plt.tight_layout()
plt.show()