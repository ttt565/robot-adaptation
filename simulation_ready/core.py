"""Causal planar robot simulation and streaming effective-wheel identification."""

import copy
import math
from collections import deque
from dataclasses import dataclass

import numpy as np


SCENARIOS = ["none", "left_motor_drop", "payload", "slope", "traction_limit"]
MODES = ["fixed", "pi", "window_40", "window_80", "window_160", "adaptive"]


@dataclass(frozen=True)
class Physics:
    mass: float = 10.0
    inertia: float = 0.5
    half_track: float = 0.2
    linear_drag: float = 4.0
    yaw_drag: float = 0.8
    gravity: float = 9.81


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def integrate(state, forces, physics, dt, grade=0.0):
    x, y, heading, speed, yaw_rate = state
    left, right = forces
    # 坡度朝世界+x上升；仅保留沿车身前向的重力分量，假定无侧滑。
    acceleration = ((left + right - physics.linear_drag * speed) / physics.mass
                    - physics.gravity * math.sin(grade) * math.cos(heading))
    angular_acceleration = (physics.half_track * (right - left) - physics.yaw_drag * yaw_rate) / physics.inertia
    return np.array([x + dt * speed * math.cos(heading),
                     y + dt * speed * math.sin(heading), heading + dt * yaw_rate,
                     speed + dt * acceleration, yaw_rate + dt * angular_acceleration])


def plant_step(state, command, time, config):
    """只有仿真器读取真实扰动；控制器和辨识器没有这个接口。"""
    nominal = Physics(**config["physics"])
    physics = nominal
    efficiency = np.ones(2)
    grade = 0.0
    scenario = config["scenario"]
    disturbed = time >= config["change_time"]
    if disturbed and scenario == "left_motor_drop":
        efficiency[0] = config["disturbance"]["left_efficiency"]
    if disturbed and scenario == "payload":
        multiplier = config["disturbance"]["payload_multiplier"]
        physics = Physics(**{**config["physics"], "mass": nominal.mass * multiplier,
                             "inertia": nominal.inertia * multiplier})
    if disturbed and scenario == "slope":
        grade = math.radians(config["disturbance"]["slope_degrees"])
    forces = efficiency * command
    if disturbed and scenario == "traction_limit":
        # 简化纵向牵引力上限，未模拟轮胎转速、轮胎侧偏或真实滑移率。
        cap = config["disturbance"]["friction_coefficient"] * physics.mass * physics.gravity / 2
        forces = np.clip(forces, -cap, cap)
    return integrate(state, forces, physics, config["dt"], grade)


def make_reference(config):
    dt = config["dt"]
    times = np.arange(round(config["duration"] / dt) + 1) * dt
    settings = config["reference"]
    speed = settings["speed"] + settings["speed_amplitude"] * np.sin(settings["speed_frequency"] * times)
    yaw_rate = speed / settings["radius"] if settings["path"] == "circle" else np.zeros_like(speed)
    reference = np.zeros((len(times), 5))
    reference[:, 3] = speed
    reference[:, 4] = yaw_rate
    for index in range(len(times) - 1):
        reference[index + 1, 2] = reference[index, 2] + yaw_rate[index] * dt
        reference[index + 1, 0] = reference[index, 0] + speed[index] * math.cos(reference[index, 2]) * dt
        reference[index + 1, 1] = reference[index, 1] + speed[index] * math.sin(reference[index, 2]) * dt
    accelerations = np.diff(reference[:, 3:5], axis=0) / dt
    return times, reference, accelerations


class WheelEstimator:
    """显式参数状态z=[etaL,etaR]；update只接收已发生的观测转移。"""

    def __init__(self, mode, physics, dt, settings, sensor_noise_std):
        self.mode, self.physics, self.dt = mode, physics, dt
        self.settings = settings
        self.sigma = np.asarray(sensor_noise_std)
        self.minimum, self.maximum = settings["min_window"], settings["max_window"]
        self.fixed_size = int(mode.split("_")[1]) if mode.startswith("window_") else self.maximum
        self.commands = deque(maxlen=self.maximum)
        self.forces = deque(maxlen=self.maximum)
        self.raw = np.ones(2)
        self.window = 0
        self.updates = 0
        self.last_alarm = None
        self.consecutive = 0
        self.alarm = False
        self.skipped_side_updates = np.zeros(2, dtype=int)

    def ls(self, commands, forces, size):
        command, force = commands[-size:], forces[-size:]
        energy = np.sum(command ** 2, axis=0)
        valid = energy >= size * self.settings["min_command_rms"] ** 2
        estimate = self.raw.copy()
        estimate[valid] = np.sum(command * force, axis=0)[valid] / energy[valid]
        return estimate, valid

    def difference_sd(self, commands):
        # 闭环u受观测噪声影响；这是阈值尺度的启发式，不提供4sigma误报保证。
        weights = -commands / np.sum(commands ** 2, axis=0)
        weights[-self.minimum:] += commands[-self.minimum:] / np.sum(commands[-self.minimum:] ** 2, axis=0)
        p = self.physics
        next_v = p.mass / (2 * self.dt)
        current_v = p.linear_drag / 2 - next_v
        next_w = np.array([-1, 1]) * p.inertia / (2 * p.half_track * self.dt)
        current_w = np.array([-1, 1]) * p.yaw_drag / (2 * p.half_track) - next_w
        cv, cw = np.zeros((len(commands) + 1, 2)), np.zeros((len(commands) + 1, 2))
        cv[:-1] += weights * current_v
        cv[1:] += weights * next_v
        cw[:-1] += weights * current_w
        cw[1:] += weights * next_w
        return np.sqrt(self.sigma[0] ** 2 * np.sum(cv ** 2, axis=0)
                       + self.sigma[1] ** 2 * np.sum(cw ** 2, axis=0))

    def update(self, current_motion, command, next_motion):
        self.alarm = False
        self.updates += 1
        if self.mode in ["fixed", "pi"]:
            return
        p = self.physics
        force_sum = p.mass / self.dt * (next_motion[0] - current_motion[0]) + p.linear_drag * current_motion[0]
        force_difference = (p.inertia / (p.half_track * self.dt) * (next_motion[1] - current_motion[1])
                            + p.yaw_drag / p.half_track * current_motion[1])
        self.commands.append(np.asarray(command).copy())
        self.forces.append(np.array([(force_sum - force_difference) / 2, (force_sum + force_difference) / 2]))
        needed = self.maximum if self.mode == "adaptive" else self.fixed_size
        if len(self.commands) < needed:
            return
        commands, forces = np.asarray(self.commands), np.asarray(self.forces)
        size = self.fixed_size
        if self.mode == "adaptive":
            size = self.maximum if self.last_alarm is None else min(self.maximum, max(self.minimum, self.updates - self.last_alarm))
        current, valid = self.ls(commands, forces, size)
        self.skipped_side_updates += ~valid
        if self.mode == "adaptive":
            short, short_valid = self.ls(commands, forces, self.minimum)
            if np.all(valid & short_valid):
                threshold = np.maximum(self.settings["difference_floor"],
                                       self.settings["noise_multiplier"] * self.difference_sd(commands[-size:]))
                different = np.any(np.abs(short - current) > threshold)
                self.consecutive = self.consecutive + 1 if different else 0
            else:
                self.consecutive = 0
            if self.consecutive >= self.settings["trigger_updates"]:
                current = short
                self.last_alarm, self.consecutive = self.updates, 0
                size, self.alarm = self.minimum, True
        self.raw, self.window = current, size

    def used(self):
        return np.clip(self.raw, self.settings["eta_min"], self.settings["eta_max"])


def controller_command(observation, reference, acceleration, eta, integrals, mode, config):
    gains, p = config["controller"], Physics(**config["physics"])
    heading = observation[2]
    dx, dy = reference[:2] - observation[:2]
    forward = math.cos(heading) * dx + math.sin(heading) * dy
    lateral = -math.sin(heading) * dx + math.cos(heading) * dy
    heading_error = wrap(reference[2] - heading)
    desired_speed = np.clip(reference[3] * math.cos(heading_error) + gains["position_gain"] * forward, 0, gains["max_speed"])
    desired_yaw = np.clip(reference[4] + reference[3] * gains["lateral_gain"] * lateral
                          + reference[3] * gains["heading_gain"] * math.sin(heading_error),
                          -gains["max_yaw_rate"], gains["max_yaw_rate"])
    errors = np.array([desired_speed - observation[3], desired_yaw - observation[4]])
    force_sum = p.mass * (acceleration[0] + gains["speed_gain"] * errors[0]) + p.linear_drag * observation[3]
    torque = p.inertia * (acceleration[1] + gains["yaw_rate_gain"] * errors[1]) + p.yaw_drag * observation[4]
    if mode == "pi":
        force_sum += gains["speed_integral_gain"] * integrals[0]
        torque += gains["yaw_integral_gain"] * integrals[1]
    difference = torque / p.half_track
    raw_command = np.array([(force_sum - difference) / 2, (force_sum + difference) / 2]) / eta
    command = np.clip(raw_command, -gains["max_command"], gains["max_command"])
    if mode == "pi" and np.all(np.abs(raw_command) <= gains["max_command"]):
        integrals += errors * config["dt"]
    return command, raw_command


def tracking_metrics(data, config):
    times, states, reference = data["times"], data["states"], data["reference"]
    position = np.linalg.norm(states[:, :2] - reference[:, :2], axis=1)
    heading = np.arctan2(np.sin(states[:, 2] - reference[:, 2]), np.cos(states[:, 2] - reference[:, 2]))
    speed = states[:, 3] - reference[:, 3]
    score = config["evaluation"]
    acceptable = ((position <= score["position_tolerance"])
                  & (np.abs(heading) <= math.radians(score["heading_tolerance_degrees"]))
                  & (np.abs(speed) <= score["speed_tolerance"]))
    after = times > config["change_time"]
    outside = np.flatnonzero(after & ~acceptable)
    confirmation, status = None, "not_recovered"
    if not len(outside):
        confirmation, status = 0.0, "remained_in_band"
    else:
        needed = math.ceil(score["hold_duration"] / config["dt"]) + 1
        consecutive = 0
        for end in range(int(outside[0]) + 1, len(times)):
            consecutive = consecutive + 1 if acceptable[end] else 0
            if consecutive >= needed:
                confirmation, status = float(times[end] - config["change_time"]), "confirmed"
                break
    post_prediction = times[1:] > config["change_time"]
    prediction_rmse = np.sqrt(np.mean(data["prediction_error"][post_prediction] ** 2, axis=0))
    nominal_rmse = np.sqrt(np.mean(data["nominal_prediction_error"][post_prediction] ** 2, axis=0))
    alarm_times = times[data["alarm"]]
    raw_eta = data["eta_raw"]
    used_eta = data["eta_used"]
    return {
        "position_rmse": float(np.sqrt(np.mean(position[after] ** 2))),
        "steady_position_rmse": float(np.sqrt(np.mean(position[times >= max(config["change_time"] + 5, config["duration"] - 10)] ** 2))),
        "peak_position_error": float(np.max(position[after])),
        "heading_rmse_degrees": float(np.degrees(np.sqrt(np.mean(heading[after] ** 2)))),
        "speed_rmse": float(np.sqrt(np.mean(speed[after] ** 2))),
        "tracking_confirmation_delay": confirmation, "tracking_status": status,
        "in_band_fraction_after_change": float(np.mean(acceptable[after])),
        "prediction_speed_rmse": float(prediction_rmse[0]), "prediction_yaw_rmse": float(prediction_rmse[1]),
        "same_log_nominal_speed_rmse": float(nominal_rmse[0]), "same_log_nominal_yaw_rmse": float(nominal_rmse[1]),
        "command_saturation_fraction": float(np.mean(data["saturated"])),
        "max_absolute_command": float(np.max(np.abs(data["commands"]))),
        "eta_protection_fraction": float(np.mean(np.any(np.abs(raw_eta - used_eta) > 1e-12, axis=1))),
        "final_eta_raw": raw_eta[-1].tolist(), "final_eta_used": used_eta[-1].tolist(),
        "alarm_times": alarm_times.tolist(),
        "pre_change_alarm_count": int(np.sum(alarm_times <= config["change_time"])),
        "skipped_side_updates": data["skipped_side_updates"].tolist(),
        "finite": bool(all(np.all(np.isfinite(data[name])) for name in ["states", "commands", "eta_raw", "eta_used"])),
    }


def simulate(config, mode, seed=0):
    times, reference, accelerations = make_reference(config)
    steps, p = len(times) - 1, Physics(**config["physics"])
    states, observations = np.zeros_like(reference), np.zeros_like(reference)
    states[0] = reference[0]
    noise = np.random.default_rng(seed).normal(size=reference.shape) * np.array(config["measurement_noise"]["std"])
    observations[0] = states[0] + noise[0]
    estimator = WheelEstimator(mode, p, config["dt"], config["estimator"], config["measurement_noise"]["std"][3:5])
    data = {"times": times, "reference": reference, "states": states, "observations": observations,
            "commands": np.zeros((steps, 2)), "eta_raw": np.ones((steps + 1, 2)),
            "eta_used": np.ones((steps + 1, 2)), "window_steps": np.zeros(steps + 1, dtype=int),
            "prediction_error": np.zeros((steps, 2)), "nominal_prediction_error": np.zeros((steps, 2)),
            "alarm": np.zeros(steps + 1, dtype=bool), "saturated": np.zeros(steps, dtype=bool),
            "change_time": None if config["scenario"] == "none" else config["change_time"], "mode": mode, "seed": seed}
    integrals = np.zeros(2)
    for step in range(steps):
        eta = estimator.used()
        command, raw_command = controller_command(observations[step], reference[step], accelerations[step], eta, integrals, mode, config)
        # 在看到下一步观测前，先完成本步控制和预测。
        predicted = integrate(observations[step], eta * command, p, config["dt"])[3:5]
        nominal_prediction = integrate(observations[step], command, p, config["dt"])[3:5]
        states[step + 1] = plant_step(states[step], command, times[step], config)
        observations[step + 1] = states[step + 1] + noise[step + 1]
        estimator.update(observations[step, 3:5], command, observations[step + 1, 3:5])
        data["commands"][step] = command
        data["saturated"][step] = np.any(np.abs(raw_command) > config["controller"]["max_command"] + 1e-12)
        data["eta_raw"][step + 1], data["eta_used"][step + 1] = estimator.raw, estimator.used()
        data["window_steps"][step + 1], data["alarm"][step + 1] = estimator.window, estimator.alarm
        data["prediction_error"][step] = states[step + 1, 3:5] - predicted
        data["nominal_prediction_error"][step] = states[step + 1, 3:5] - nominal_prediction
    data["skipped_side_updates"] = estimator.skipped_side_updates
    data["metrics"] = tracking_metrics(data, config)
    if not data["metrics"]["finite"]:
        raise FloatingPointError(f"Non-finite simulation: {config['scenario']}, {mode}, seed {seed}")
    return data


def validate_config(config):
    if config["scenario"] not in SCENARIOS or config["reference"]["path"] not in ["straight", "circle"]:
        raise ValueError("Unsupported scenario or path.")
    dt, duration, change = config["dt"], config["duration"], config["change_time"]
    if not np.all(np.isfinite([dt, duration, change, config["playback_speed"]])) or config["playback_speed"] <= 0:
        raise ValueError("Times and playback_speed must be finite, with positive playback_speed.")
    if not 0 < dt <= 0.05 or not 0 < change < duration - config["evaluation"]["hold_duration"]:
        raise ValueError("Use 0<dt<=0.05 and a change time inside the simulation with time to evaluate.")
    if abs(round(duration / dt) * dt - duration) > 1e-9:
        raise ValueError("duration must be an integer multiple of dt.")
    if config["estimator"]["max_window"] * dt >= change:
        raise ValueError("Allow a complete initial estimation window before the disturbance.")
    if config["duration"] <= max(change + 5, config["duration"] - 10):
        raise ValueError("Allow at least five seconds after the change for steady scoring.")
    for name in ["mass", "inertia", "half_track", "gravity"]:
        if not math.isfinite(config["physics"][name]) or config["physics"][name] <= 0:
            raise ValueError(f"physics.{name} must be positive")
    for name in ["linear_drag", "yaw_drag"]:
        if not math.isfinite(config["physics"][name]) or config["physics"][name] < 0:
            raise ValueError(f"physics.{name} must be finite and nonnegative")
    noise = np.asarray(config["measurement_noise"]["std"])
    if noise.shape != (5,) or not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise ValueError("measurement_noise.std must have five finite nonnegative values.")
    settings = config["estimator"]
    for name in ["min_window", "max_window", "trigger_updates"]:
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"estimator.{name} must be a positive integer")
    if not 2 <= settings["min_window"] <= settings["max_window"] or settings["max_window"] < 160:
        raise ValueError("Use min_window>=2, max_window>=160 and min_window<=max_window.")
    if not 0 < settings["eta_min"] < settings["eta_max"] or settings["min_command_rms"] <= 0:
        raise ValueError("Positive efficiency protection and identification excitation limits are required.")
    for name in ["eta_min", "eta_max", "min_command_rms", "difference_floor", "noise_multiplier"]:
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"estimator.{name} must be finite and nonnegative")
    disturbance = config["disturbance"]
    if not all(math.isfinite(value) for value in disturbance.values()):
        raise ValueError("Disturbance settings must be finite.")
    if disturbance["payload_multiplier"] <= 0 or disturbance["friction_coefficient"] < 0 or disturbance["left_efficiency"] < 0:
        raise ValueError("Use positive payload multiplier and nonnegative friction/efficiency.")
    for name, value in config["controller"].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"controller.{name} must be finite and nonnegative")
    for name in ["max_speed", "max_yaw_rate", "max_command"]:
        if config["controller"][name] <= 0:
            raise ValueError(f"controller.{name} must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in config["evaluation"].values()):
        raise ValueError("Evaluation tolerances and hold duration must be finite and positive.")
    if any(not math.isfinite(value) for key, value in config["reference"].items() if key != "path"):
        raise ValueError("Reference settings must be finite.")
    if config["reference"]["speed"] <= abs(config["reference"]["speed_amplitude"]) or config["reference"]["radius"] <= 0:
        raise ValueError("Reference speed must remain positive, with positive turn radius.")


def self_check(config):
    """验证物理一致性、控制约束、在线时序；不要求算法在所有扰动中获胜。"""
    normal = copy.deepcopy(config)
    normal["scenario"] = "none"
    normal["measurement_noise"]["std"] = [0] * 5
    for mode in MODES:
        run = simulate(normal, mode)
        np.testing.assert_allclose(run["states"], run["reference"], atol=2e-10, rtol=0)
        assert run["metrics"]["max_absolute_command"] <= config["controller"]["max_command"]
    # 低输入时保持参数，不能通过除以小数制造巨大估计。
    estimator = WheelEstimator("adaptive", Physics(**config["physics"]), config["dt"], config["estimator"], [.01, .01])
    for _ in range(config["estimator"]["max_window"] + 10):
        estimator.update(np.zeros(2), np.zeros(2), np.array([.01, -.01]))
    np.testing.assert_array_equal(estimator.raw, [1, 1])
    assert np.all(estimator.skipped_side_updates > 0)
    # 相同前缀在后续输入/观测不同时仍给出完全相同的前缀输出。
    rng = np.random.default_rng(123)
    motions = rng.normal(size=(301, 2)) * .01
    commands = np.full((300, 2), 2.)
    first = WheelEstimator("adaptive", Physics(**config["physics"]), config["dt"], config["estimator"], [.01, .01])
    second = WheelEstimator("adaptive", Physics(**config["physics"]), config["dt"], config["estimator"], [.01, .01])
    for step in range(220):
        first.update(motions[step], commands[step], motions[step + 1])
        second.update(motions[step], commands[step], motions[step + 1])
        np.testing.assert_array_equal(first.raw, second.raw)
    for step in range(220, 300):
        first.update(motions[step], commands[step], motions[step + 1])
        second.update(motions[step] + 1, commands[step] * 2, motions[step + 1] + 1)
    for scenario in SCENARIOS:
        case = copy.deepcopy(config)
        case["scenario"] = scenario
        for mode in ["fixed", "pi", "adaptive"]:
            run = simulate(case, mode, 999)
            assert run["metrics"]["finite"]
            assert run["metrics"]["max_absolute_command"] <= case["controller"]["max_command"] + 1e-12
            assert np.all(run["eta_used"] >= case["estimator"]["eta_min"])
            assert np.all(run["eta_used"] <= case["estimator"]["eta_max"])
    return "PASS: exact nominal tracking, low-input guard, streaming prefix agreement, five disturbance scenarios, finite states, actuator and estimator limits."
