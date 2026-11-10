"""Live MuJoCo differential-drive simulation with causal online adaptation."""

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

try:
    import mujoco
    import numpy as np
    import engine
    from control_core import validate_config
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error}\nRun: python -m pip install -r requirements.txt") from error


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def self_check(config):
    normal = engine.ContactPlant(config, "none")
    initial = normal.true_state()
    assert abs(initial[2]) < .01 and abs(initial[3]) < .01
    for _ in range(50):
        state = normal.step(np.array([2., 2.]))
    assert state[0] > .1 and abs(state[1]) < .01 and abs(state[2]) < .01
    for _ in range(50):
        turning = normal.step(np.array([1., 3.]))
    assert turning[4] > .02
    assert np.all(np.isfinite(normal.data.qpos)) and np.all(np.isfinite(normal.data.qvel))
    assert normal.audit()["height"] > .05
    # 只由plant缩放故障；核对未缩放指令对应的实际执行器扭矩。
    fault = copy.deepcopy(config)
    fault["change_time"] = 0
    plant = engine.ContactPlant(fault, "left_motor_drop")
    plant.step(np.array([2., 2.]))
    np.testing.assert_allclose(plant.data.ctrl, [.1, .2], atol=1e-12)
    np.testing.assert_allclose(plant.data.actuator_force, [.1, .2], atol=1e-12)
    friction = engine.ContactPlant(fault, "low_friction")
    friction.step(np.zeros(2))
    np.testing.assert_allclose(friction.audit()["contact_mu"], config["low_friction"], atol=1e-12)
    print("PASS: real contacts, forward/yaw signs, finite free-body states, torque fault isolation, actual mixed contact friction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--mode", choices=engine.MODES, default="adaptive")
    parser.add_argument("--scenario", choices=engine.SCENARIOS)
    parser.add_argument("--path", choices=["straight", "circle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--snapshot", action="store_true", help="Also save a rendered 3D image; requires OpenGL")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.scenario:
        config["scenario"] = args.scenario
    if args.path:
        config["reference"]["path"] = args.path
    if args.duration is not None:
        config["duration"] = args.duration
    checked = copy.deepcopy(config)
    checked["scenario"] = "traction_limit" if checked["scenario"] == "low_friction" else checked["scenario"]
    try:
        validate_config(checked)
        if config["scenario"] not in engine.SCENARIOS:
            raise ValueError("This backend currently supports none, left_motor_drop and low_friction.")
        if not all(math.isfinite(config[name]) for name in ["physics_dt", "wheel_radius", "low_friction", "ramp_seconds", "calibration_seconds", "settle_seconds"]):
            raise ValueError("Engine settings must be finite.")
        if not 0 < config["physics_dt"] <= .005:
            raise ValueError("Use 0<physics_dt<=0.005.")
        ratio = config["dt"] / config["physics_dt"]
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("physics_dt must be <=0.005 and divide dt exactly.")
        if abs(config["wheel_radius"] - .1) > 1e-12 or abs(config["physics"]["half_track"] - .2) > 1e-12:
            raise ValueError("robot.xml currently fixes wheel radius=0.1 and half_track=0.2.")
        if config["controller"]["max_command"] > 6 or args.seed < 0:
            raise ValueError("Use command limit<=6 N and a nonnegative seed.")
        if config["low_friction"] < 0 or config["ramp_seconds"] <= 0 or config["calibration_seconds"] < 4:
            raise ValueError("Invalid friction, ramp or calibration duration.")
        if config["settle_seconds"] < .5 or len(config["plant_drag"]) != 2 or any(not math.isfinite(value) or value < 0 for value in config["plant_drag"]):
            raise ValueError("Use settle_seconds>=0.5 and two finite nonnegative plant drag values.")
    except ValueError as error:
        parser.error(str(error))
    if args.self_check:
        self_check(config)
        return 0

    import matplotlib
    matplotlib.use("Agg")
    from plotting import save_dashboard

    output = args.output_dir.resolve() if args.output_dir else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; select a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    print(f"MuJoCo {mujoco.__version__}; actual contact physics dt={config['physics_dt']} s.")
    print("Calibrating nominal planar predictor on a separate healthy MuJoCo run ...", flush=True)
    calibration = engine.calibrate(config)
    write_json(output / "calibration.json", calibration)
    config["physics"] = calibration["physics"]
    write_json(output / "config_used.json", config)
    print(f"Effective nominal mass={config['physics']['mass']:.3f}; inertia={config['physics']['inertia']:.4f}.")
    results = {}
    modes = engine.MODES if args.compare else [args.mode]
    if args.compare and not args.headless:
        print("Comparison runs headlessly; use a single --mode for the live viewer.")
    for mode in modes:
        data, plant = engine.simulate(config, mode, args.seed, live=not args.headless and not args.compare)
        results[mode] = data
        np.savez_compressed(output / f"{mode}.npz", **{key: value for key, value in data.items() if isinstance(value, np.ndarray)})
        metric = data["metrics"]
        delay = metric["tracking_confirmation_delay"]
        text = "not confirmed" if delay is None else f"{delay:.2f} s"
        score_name = "position_rmse" if metric["complete"] else "partial_position_rmse"
        print(f"{mode}: {score_name}={metric[score_name]:.4f} m; status={metric['tracking_status']}; confirmation={text}", flush=True)
    write_json(output / "metrics.json", {mode: data["metrics"] for mode, data in results.items()})
    save_dashboard(results, {**config, "command_limit": config["controller"]["max_command"]}, output / "dashboard.png")
    if args.snapshot or (not args.headless and not args.compare):
        try:
            engine.save_snapshot(plant, output / "mujoco_scene.png")
        except Exception as error:
            print(f"Snapshot unavailable; physics logs were saved: {error}")
    source_names = ["robot.xml", "engine.py", "run_mujoco.py", "control_core.py", "plotting.py"]
    write_json(output / "protocol.json", {
        "backend": "MuJoCo", "version": mujoco.__version__, "seed": args.seed,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
        "controls": "Logged nominal wheel force command in N; actual motor torque=r*command*hidden efficiency, gear=1.",
        "plant": "Free base, wheel hinges, gravity and frictional contact; explicit known viscous drag; no commanded base velocity or driving base force.",
        "calibration": "Separate healthy noiseless nominal simulation; approximate effective planar mass/inertia/drag fit before evaluation.",
        "observation": "Synthetic pose/IMU observations from MuJoCo sensors plus configured Gaussian noise; not a real localization stack.",
        "eta": "Explicit effective parameter state; not physical-cause identification or neural latent learning.",
        "scores": "Independent closed-loop trajectories; each run also has same-log nominal shadow prediction errors. Partial runs not scored as full experiments.",
        "limits": "Feedback-dependent noise threshold is heuristic; low friction is an actual contact-coefficient change but not a calibrated tire model.",
    })
    print(f"Results saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
