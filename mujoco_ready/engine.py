"""Real MuJoCo wheel/contact backend; observations and plant internals are separate."""

import copy
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from control_core import Physics, WheelEstimator, controller_command, integrate, tracking_metrics


ROOT = Path(__file__).resolve().parent
SCENARIOS = ["none", "left_motor_drop", "low_friction"]
MODES = ["fixed", "pi", "window_40", "window_80", "window_160", "adaptive"]


class ContactPlant:
    def __init__(self, config, scenario=None):
        self.config = config
        self.scenario = scenario or config["scenario"]
        xml = ET.fromstring((ROOT / "robot.xml").read_text(encoding="utf-8"))
        target = ET.SubElement(xml.find("worldbody"), "body", name="reference_target", mocap="true", pos="0 0 0.03")
        ET.SubElement(target, "geom", type="sphere", size="0.045", mass="0", rgba="0.1 0.9 0.25 0.7", contype="0", conaffinity="0")
        # from_xml_string also works when the workspace path contains Chinese characters.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.model.opt.timestep = config["physics_dt"]
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body("chassis").id
        self.tires = [self.model.geom(name).id for name in ["left_tire", "right_tire"]]
        self.floor_id = self.model.geom("floor").id
        self.target_id = int(self.model.body_mocapid[self.model.body("reference_target").id])
        self.sensor_slices = {}
        for name in ["chassis_position", "chassis_quaternion", "imu_velocity", "imu_gyro"]:
            sensor_id = self.model.sensor(name).id
            start, length = int(self.model.sensor_adr[sensor_id]), int(self.model.sensor_dim[sensor_id])
            self.sensor_slices[name] = slice(start, start + length)
        self.wheel_dofs = [int(self.model.jnt_dofadr[self.model.joint(name).id]) for name in ["left_wheel", "right_wheel"]]
        self.substeps = round(config["dt"] / config["physics_dt"])
        self.elapsed_steps = 0
        self.actual_torque = np.zeros(2)
        self.changed = False
        for _ in range(round(config["settle_seconds"] / config["physics_dt"])):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def true_state(self):
        read = lambda name: self.data.sensordata[self.sensor_slices[name]]
        position = read("chassis_position")
        w, x, y, z = read("chassis_quaternion")
        heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([position[0], position[1], heading, read("imu_velocity")[0], read("imu_gyro")[2]])

    def audit(self):
        state = self.true_state()
        wheel_surface_speed = self.config["wheel_radius"] * self.data.qvel[self.wheel_dofs]
        expected = state[3] + np.array([-1, 1]) * self.config["physics"]["half_track"] * state[4]
        coefficients = []
        for contact in self.data.contact:
            if ((contact.geom1 == self.floor_id and contact.geom2 in self.tires)
                    or (contact.geom2 == self.floor_id and contact.geom1 in self.tires)):
                coefficients.append(contact.friction[0])
        return {"height": float(self.data.xpos[self.body_id, 2]),
                "contact_mu": float(np.mean(coefficients)) if coefficients else np.nan,
                "slip_proxy": wheel_surface_speed - expected,
                "actual_torque": self.actual_torque.copy(),
                "engine_actuator_force": self.data.actuator_force.copy()}

    def step(self, nominal_force_command):
        # 估计器始终接收未乘故障效率的指令；实际扭矩只由plant和审计日志使用。
        for _ in range(self.substeps):
            elapsed = self.elapsed_steps * self.config["physics_dt"]
            efficiency = np.ones(2)
            if elapsed + 1e-10 >= self.config["change_time"]:
                if self.scenario == "left_motor_drop":
                    efficiency[0] = self.config["disturbance"]["left_efficiency"]
                    self.model.geom_rgba[self.tires[0]] = [0.85, 0.18, 0.1, 1]
                elif self.scenario == "low_friction" and not self.changed:
                    # Tire priority > floor priority, so this changes actual mixed contact friction.
                    self.model.geom_friction[self.tires, 0] = self.config["low_friction"]
                    self.model.geom_rgba[self.tires] = [0.1, 0.45, 0.9, 1]
                    self.changed = True
            self.actual_torque = self.config["wheel_radius"] * np.asarray(nominal_force_command) * efficiency
            self.data.ctrl[:] = self.actual_torque
            # 已知的粘性环境阻力；驱动只来自轮轴电机和轮地接触。
            self.data.xfrc_applied[:] = 0
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, velocity, 0)
            self.data.xfrc_applied[self.body_id, :2] = -self.config["plant_drag"][0] * velocity[3:5]
            self.data.xfrc_applied[self.body_id, 5] = -self.config["plant_drag"][1] * velocity[2]
            mujoco.mj_step(self.model, self.data)
            self.elapsed_steps += 1
        mujoco.mj_forward(self.model, self.data)
        return self.true_state()


def make_reference(config):
    dt = config["dt"]
    times = np.arange(round(config["duration"] / dt) + 1) * dt
    settings = config["reference"]
    ramp = (1 - np.cos(np.minimum(times / config["ramp_seconds"], 1) * math.pi)) / 2
    speed = ramp * (settings["speed"] + settings["speed_amplitude"] * np.sin(settings["speed_frequency"] * times))
    reference = np.zeros((len(times), 5))
    reference[:, 3] = speed
    reference[:, 4] = speed / settings["radius"] if settings["path"] == "circle" else 0
    for step in range(len(times) - 1):
        x, y, heading, v, omega = reference[step]
        reference[step + 1, :3] = [x + dt * v * math.cos(heading), y + dt * v * math.sin(heading), heading + dt * omega]
    return times, reference, np.diff(reference[:, 3:5], axis=0) / dt


def calibrate(config):
    """独立正常运行、无测量噪声的离线标定，不包含任何故障数据。"""
    plant = ContactPlant(config, "none")
    current = plant.true_state()
    features_v, features_w, targets_v, targets_w = [], [], [], []
    dt = config["dt"]
    for step in range(round(config["calibration_seconds"] / dt)):
        time_value = step * dt
        total = 3.2 + 1.0 * math.sin(0.8 * time_value) + 1.2 * math.sin(1.7 * time_value)
        difference = 0.75 * math.sin(1.1 * time_value)
        command = np.array([(total - difference) / 2, (total + difference) / 2])
        following = plant.step(command)
        if time_value >= 1:
            features_v.append([total, -current[3]])
            features_w.append([config["physics"]["half_track"] * difference, -current[4]])
            targets_v.append((following[3] - current[3]) / dt)
            targets_w.append((following[4] - current[4]) / dt)
        current = following
    beta_v, _, rank_v, _ = np.linalg.lstsq(features_v, targets_v, rcond=None)
    beta_w, _, rank_w, _ = np.linalg.lstsq(features_w, targets_w, rcond=None)
    if rank_v < 2 or rank_w < 2 or np.any(beta_v <= 0) or np.any(beta_w <= 0):
        raise RuntimeError(f"Nominal calibration failed: v={beta_v}, yaw={beta_w}")
    parameters = {**config["physics"], "mass": float(1 / beta_v[0]), "linear_drag": float(beta_v[1] / beta_v[0]),
                  "inertia": float(1 / beta_w[0]), "yaw_drag": float(beta_w[1] / beta_w[0])}
    return {"physics": parameters, "engine_body_mass": float(plant.model.body_subtreemass[plant.body_id]),
            "linear_acceleration_fit_rmse": float(np.sqrt(np.mean((np.asarray(features_v) @ beta_v - targets_v) ** 2))),
            "yaw_acceleration_fit_rmse": float(np.sqrt(np.mean((np.asarray(features_w) @ beta_w - targets_w) ** 2))),
            "samples": len(targets_v), "source": "separate healthy contact run; noiseless calibration observables; no fault data",
            "interpretation": "Effective planar approximation, not exact physical mass or inertia."}


def simulate(config, mode, seed=0, live=False):
    times, reference, accelerations = make_reference(config)
    steps = len(times) - 1
    plant = ContactPlant(config)
    p = Physics(**config["physics"])
    estimator = WheelEstimator(mode, p, config["dt"], config["estimator"], config["measurement_noise"]["std"][3:5])
    noise = np.random.default_rng(seed).normal(size=reference.shape) * config["measurement_noise"]["std"]
    states, observations = np.zeros_like(reference), np.zeros_like(reference)
    states[0] = plant.true_state()
    observations[0] = states[0] + noise[0]
    data = {"times": times, "reference": reference, "states": states, "observations": observations,
            "commands": np.zeros((steps, 2)), "eta_raw": np.ones((steps + 1, 2)), "eta_used": np.ones((steps + 1, 2)),
            "window_steps": np.zeros(steps + 1, dtype=int), "alarm": np.zeros(steps + 1, dtype=bool),
            "prediction_error": np.zeros((steps, 2)), "nominal_prediction_error": np.zeros((steps, 2)),
            "actual_torque": np.zeros((steps, 2)), "engine_actuator_force": np.zeros((steps, 2)),
            "slip_proxy": np.zeros((steps, 2)), "contact_mu": np.full(steps, np.nan), "height": np.zeros(steps),
            "saturated": np.zeros(steps, dtype=bool), "change_time": None if config["scenario"] == "none" else config["change_time"]}
    integrals = np.zeros(2)
    viewer = None
    viewer_threads = []
    if live:
        from mujoco import viewer as mujoco_viewer
        previous_threads = set(threading.enumerate())
        viewer = mujoco_viewer.launch_passive(plant.model, plant.data, show_left_ui=False, show_right_ui=False)
        viewer_threads = [thread for thread in threading.enumerate() if thread not in previous_threads and thread.daemon]
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = plant.body_id
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -28
    completed = 0
    try:
        for step in range(steps):
            start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            eta = estimator.used()
            command, raw_command = controller_command(observations[step], reference[step], accelerations[step], eta, integrals, mode, config)
            prediction = integrate(observations[step], eta * command, p, config["dt"])[3:5]
            nominal_prediction = integrate(observations[step], command, p, config["dt"])[3:5]
            # 以下step是实际MuJoCo子步；没有把简化模型的下一状态写入MuJoCo。
            states[step + 1] = plant.step(command)
            observations[step + 1] = states[step + 1] + noise[step + 1]
            estimator.update(observations[step, 3:5], command, observations[step + 1, 3:5])
            audit = plant.audit()
            for key in ["actual_torque", "engine_actuator_force", "slip_proxy", "contact_mu", "height"]:
                data[key][step] = audit[key]
            data["commands"][step] = command
            data["saturated"][step] = np.any(np.abs(raw_command) > config["controller"]["max_command"] + 1e-12)
            data["eta_raw"][step + 1], data["eta_used"][step + 1] = estimator.raw, estimator.used()
            data["window_steps"][step + 1], data["alarm"][step + 1] = estimator.window, estimator.alarm
            data["prediction_error"][step] = states[step + 1, 3:5] - prediction
            data["nominal_prediction_error"][step] = states[step + 1, 3:5] - nominal_prediction
            plant.data.mocap_pos[plant.target_id] = [reference[step + 1, 0], reference[step + 1, 1], 0.04]
            completed = step + 1
            if viewer is not None:
                viewer.sync()
                remaining = config["dt"] / config["playback_speed"] - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()
            # launch_passive在后台daemon线程绘制；等待退出，避免Windows在解释器结束时中断OpenGL清理。
            for thread in viewer_threads:
                thread.join(timeout=3.0)
                if thread.is_alive():
                    raise RuntimeError("MuJoCo viewer thread did not finish closing.")
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value[:completed + 1] if len(value) == steps + 1 else value[:completed]
    data["skipped_side_updates"] = estimator.skipped_side_updates
    complete = completed == steps
    if complete:
        data["metrics"] = tracking_metrics(data, config)
    else:
        position_error = np.linalg.norm(data["states"][:, :2] - data["reference"][:, :2], axis=1)
        data["metrics"] = {"partial_position_rmse": float(np.sqrt(np.mean(position_error ** 2))),
                           "tracking_confirmation_delay": None, "tracking_status": "partial",
                           "command_saturation_fraction": float(np.mean(data["saturated"])) if completed else 0}
    data["metrics"].update({"complete": complete, "completed_seconds": completed * config["dt"],
                            "score_interval_seconds": [config["change_time"], config["duration"]] if complete else [0, completed * config["dt"]],
                            "physics_backend": "MuJoCo", "mujoco_version": mujoco.__version__,
                            "minimum_chassis_height": float(np.min(data["height"])) if completed else None,
                            "maximum_abs_slip_proxy": float(np.max(np.abs(data["slip_proxy"]))) if completed else None})
    if not np.all(np.isfinite(data["states"])) or not np.all(np.isfinite(data["eta_raw"])):
        raise FloatingPointError("MuJoCo simulation produced non-finite states or estimates.")
    return data, plant


def save_snapshot(plant, path):
    from matplotlib import pyplot as plt
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = plant.data.xpos[plant.body_id]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -25
    with mujoco.Renderer(plant.model, height=640, width=960) as renderer:
        renderer.update_scene(plant.data, camera=camera)
        plt.imsave(path, renderer.render())
