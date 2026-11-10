"""MuJoCo onboard RGB appearance recognition plus time-ordered movement evidence."""

import argparse
import csv
import hashlib
import json
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import engine
from appearance import AppearanceClassifier, KINDS
from control_core import Physics, WheelEstimator, controller_command, integrate
from diagnostics import EvidenceMonitor
from surfaces import CASES, GroundCamera, SurfacePlant, collect_dataset


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_model(retrain=False):
    folder = ROOT / "prepared_model"
    folder.mkdir(exist_ok=True)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ["appearance.py", "surfaces.py", "engine.py", "robot.xml"]}
    if not retrain and (folder / "model.npz").exists() and (folder / "heldout_test.json").exists():
        report = json.loads((folder / "heldout_test.json").read_text())
        dataset_path = folder / "appearance_dataset.npz"
        dataset_matches = dataset_path.exists() and report.get("dataset_sha256") == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        model_matches = report.get("model_sha256") == hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest()
        if report.get("source_sha256") != sources or not dataset_matches or not model_matches:
            raise RuntimeError("Appearance cache no longer matches its camera/features/data. Run with --retrain.")
        return AppearanceClassifier.load(folder / "model.npz"), report
    images, labels, splits = collect_dataset(folder)
    train = [index for index, split in enumerate(splits) if split == "train"]
    test = [index for index, split in enumerate(splits) if split == "test"]
    classifier = AppearanceClassifier().fit([images[i] for i in train], [labels[i] for i in train])
    classifier.save(folder / "model.npz")
    predictions = [{"truth": labels[index], **classifier.predict(images[index])} for index in test]
    correct = sum(row["truth"] == row["label"] for row in predictions)
    report = {"train_images": len(train), "heldout_images": len(test), "heldout_correct": correct,
              "heldout_accuracy": correct / len(test), "rows": predictions,
              "scope": "Synthetic texture appearances rendered by MuJoCo, not real-world material recognition.",
              "split": "Train texture seeds 0..19 per class; test seeds 200..209 per class; runtime seeds >=1000.",
              "appearance_sha256": hashlib.sha256((ROOT / "appearance.py").read_bytes()).hexdigest()}
    report.update({"source_sha256": sources,
                   "dataset_sha256": hashlib.sha256((folder / "appearance_dataset.npz").read_bytes()).hexdigest(),
                   "model_sha256": hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest()})
    write_json(folder / "heldout_test.json", report)
    return classifier, report


def run_case(config, case, classifier, output, seed=0, live=False):
    output.mkdir(parents=True, exist_ok=True)
    (output / "camera").mkdir(exist_ok=True)
    plant = SurfacePlant(config, case, texture_seed=1000 + seed)
    camera = GroundCamera(plant)
    times, reference, accelerations = engine.make_reference(config)
    steps = len(times) - 1
    states, observations = np.zeros_like(reference), np.zeros_like(reference)
    states[0] = plant.true_state()
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=reference.shape) * config["measurement_noise"]["std"]
    encoder_noise = rng.normal(size=(steps, 2)) * .05
    observations[0] = states[0] + noise[0]
    p = Physics(**config["physics"])
    estimator = WheelEstimator("adaptive", p, config["dt"], config["estimator"], config["measurement_noise"]["std"][3:5])
    monitor = EvidenceMonitor(config["wheel_radius"], config["physics"]["half_track"])
    commands, innovations = np.zeros((steps, 2)), np.zeros((steps, 2))
    estimates = np.ones((steps + 1, 2))
    contact_mu = np.full(steps, np.nan)
    rows, images = [], []
    first_contact, viewer, viewer_threads, figure = None, None, [], None
    if live:
        import matplotlib.pyplot as plt
        from mujoco import viewer as native_viewer
        existing_threads = set(threading.enumerate())
        viewer = native_viewer.launch_passive(plant.model, plant.data, show_left_ui=False, show_right_ui=False)
        viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads and thread.daemon]
        with viewer.lock():
            viewer.cam.type, viewer.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, plant.body_id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -28
        plt.ion()
        figure, (image_axis, text_axis) = plt.subplots(1, 2, figsize=(10, 4))
        image_artist = image_axis.imshow(camera.read_rgb())
        image_axis.axis("off")
        text_axis.axis("off")
        status_text = text_axis.text(.02, .96, "Starting camera...", va="top", fontsize=10)
        figure.suptitle("Onboard RGB appearance and movement evidence")
        figure.tight_layout()
    completed = 0
    integrals = np.zeros(2)
    try:
        for step in range(steps):
            start = time.perf_counter()
            if live and (not viewer.is_running() or not plt.fignum_exists(figure.number)):
                break
            guard = viewer.lock() if viewer is not None else nullcontext()
            with guard:
                eta = estimator.used()
                command, _ = controller_command(observations[step], reference[step], accelerations[step], eta, integrals, "adaptive", config)
                nominal_prediction = integrate(observations[step], command, p, config["dt"])[3:5]
                states[step + 1] = plant.step(command)
                observations[step + 1] = states[step + 1] + noise[step + 1]
                estimator.update(observations[step, 3:5], command, observations[step + 1, 3:5])
                commands[step], estimates[step + 1] = command, estimator.raw
                innovations[step] = observations[step + 1, 3:5] - nominal_prediction
                wheel_rates = plant.data.qvel[plant.wheel_dofs].copy() + encoder_noise[step]
                monitor.observe_motion(observations[step], observations[step + 1], nominal_prediction, wheel_rates)
                audit = plant.audit()  # 只用于事后评分，不传给视觉分类器或EvidenceMonitor。
                contact_mu[step] = audit["contact_mu"]
                if first_contact is None and audit["after_surface_contact"]:
                    first_contact = float(times[step + 1])
                completed = step + 1
                if step % 10 == 0:
                    rgb = camera.read_rgb()
                    prediction = classifier.predict(rgb)  # 唯一视觉输入为RGB像素。
                    evidence = monitor.observe_image(prediction)
                    image_file = f"camera/frame_{len(rows):04d}.png"
                    Image.fromarray(rgb).save(output / image_file)
                    rows.append({"time": float(times[step + 1]), "label": prediction["label"],
                                 "match_score": prediction["match_score"], **evidence, "image_file": image_file})
                    images.append(rgb)
                    if live:
                        image_artist.set_data(rgb)
                        image_axis.set_title(f"{prediction['label']} | match {prediction['match_score']:.2f}")
                        status_text.set_text(f"t = {times[step+1]:.2f} s\n\n{evidence['status'].replace('_',' ')}\n\n"
                                             f"Motion score: {monitor.motion_score:.2f}\nSlip evidence score: {monitor.slip_score:.2f}\n\n"
                                             "Appearance is not friction.\nCause remains a hypothesis.")
            if live:
                # 显示同步与50Hz物理控制分开，避免每个控制步都等待窗口绘制。
                if step % 2 == 0:
                    viewer.sync()
                if step % 10 == 0:
                    plt.pause(.001)
                remaining = config["dt"] - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        camera.close()
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=3)
            plt.close(figure)
            plt.ioff()
    states, reference, times = states[:completed + 1], reference[:completed + 1], times[:completed + 1]
    def first_event(key):
        return next((row["time"] for row in rows if row[key]), None)
    physical_change = config["change_time"] if case == "motor_fault" else first_contact if plant.mu_change else None
    metric = {"position_rmse": float(np.sqrt(np.mean(np.sum((states[:, :2] - reference[:, :2]) ** 2, axis=1)))),
              "score_interval_seconds": [0, completed * config["dt"]],
              "first_visual_change": first_event("visual_change"), "first_motion_anomaly": first_event("motion_anomaly"),
              "first_slip_evidence": first_event("slip_evidence"), "post_patch_first_contact": first_contact,
              "physical_change_time_evaluation_only": physical_change,
              "complete": completed == steps, "camera_frames": len(rows),
              "any_motion_anomaly": any(row["motion_anomaly"] for row in rows),
              "any_slip_evidence": any(row["slip_evidence"] for row in rows)}
    np.savez_compressed(output / "robot_log.npz", times=times, states=states, reference=reference,
                        observations=observations[:completed + 1], commands=commands[:completed],
                        eta_raw=estimates[:completed + 1], measured_innovations=innovations[:completed], contact_mu=contact_mu[:completed])
    if rows:
        with (output / "camera_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "vision_report.json", metric)
    return {"times": times, "states": states, "reference": reference, "rows": rows,
            "camera_images": images, "contact_time": first_contact, "metrics": metric}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--case", choices=CASES, default="visible_low_friction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrain", action="store_true", help="Refit using only the saved training split")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Use a nonnegative seed.")
    import matplotlib
    if args.headless or args.all_cases:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    output = args.output_dir.resolve() if args.output_dir else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    classifier, model_report = prepare_model(args.retrain)
    print(f"Synthetic held-out appearance test: {model_report['heldout_correct']}/{model_report['heldout_images']}.", flush=True)
    config = json.loads((ROOT / "config.json").read_text())
    calibration = engine.calibrate(config, plant_factory=lambda settings: SurfacePlant(settings, "normal", texture_seed=4000))
    config["physics"] = calibration["physics"]
    write_json(output / "config_used.json", config)
    write_json(output / "calibration.json", calibration)
    runs = {}
    for case in CASES if args.all_cases else [args.case]:
        print(f"Running {case} ...", flush=True)
        runs[case] = run_case(config, case, classifier, output / case, args.seed, live=not args.headless and not args.all_cases)
        print(runs[case]["metrics"], flush=True)
    from vision_dashboard import save_dashboard
    save_dashboard(runs, model_report, output / "vision_dashboard.png")
    write_json(output / "vision_report.json", {
        "model_test": model_report, "cases": {case: run["metrics"] for case, run in runs.items()},
        "protocol": {"camera_hz": 5, "controller_hz": 50, "vision_controls_motion": False,
                     "appearance_only": list(KINDS), "runtime_classifier_input": "RGB pixels only",
                     "no_truth_input": "EvidenceMonitor sees image predictions, noisy innovations and noisy wheel/IMU/pose observations only.",
                     "scope": "Synthetic appearance plus correlational motion evidence, not real material/causal fault identification.",
                     "headless": "No display windows, but OpenGL offscreen rendering is needed for camera images."}})
    print(f"Results saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
