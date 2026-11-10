"""Compare unchanged tracking with movement-only and RGB-plus-movement speed caps."""

import argparse
import csv
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
from control_core import Physics, WheelEstimator, controller_command, integrate, motion_targets
from diagnostics import EvidenceMonitor
from run_vision import prepare_model, write_json
from speed_governor import MODES, SETTINGS, SpeedGovernor
from surfaces import CASES, GroundCamera, SurfacePlant

ROOT = Path(__file__).resolve().parent


def score_run(run, dt):
    states, reference = run['states'], run['reference']
    position_error = np.linalg.norm(states[:, :2] - reference[:, :2], axis=1)
    # 使用完整预定几何路径评分；提前关闭时仍保留完整任务的目标距离。
    path = run['planned_path']
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-12
    starts, vectors, segment_lengths = path[:-1][valid], segments[valid], lengths[valid]
    offsets = np.concatenate([[0.], np.cumsum(lengths)])[:-1][valid]
    path_distances, progress = [], []
    for point in states[:, :2]:
        fraction = np.clip(np.sum((point - starts) * vectors, axis=1) / segment_lengths ** 2, 0, 1)
        distances = np.linalg.norm(point - starts - fraction[:, None] * vectors, axis=1)
        nearest = int(np.argmin(distances))
        path_distances.append(distances[nearest])
        progress.append(offsets[nearest] + fraction[nearest] * segment_lengths[nearest])
    run['path_progress'] = np.asarray(progress)
    slips = run['measured_slip']
    limited = run['cap_limited']
    rows = run['rows']
    def first_row(key):
        return next((row['time'] for row in rows if row[key]), None)
    metrics = {
        'position_rmse': float(np.sqrt(np.mean(position_error ** 2))),
        'peak_position_error': float(np.max(position_error)),
        'path_distance_rmse': float(np.sqrt(np.mean(np.asarray(path_distances) ** 2))),
        'speed_rmse_original_reference': float(np.sqrt(np.mean((states[:, 3] - reference[:, 3]) ** 2))),
        'end_x': float(states[-1, 0]),
        'end_path_progress': float(progress[-1]),
        'planned_path_distance': float(np.sum(lengths)),
        'progress_fraction': float(progress[-1] / np.sum(lengths)),
        'wheel_body_mismatch_mean': float(np.mean(np.abs(slips))),
        'wheel_body_mismatch_integral': float(np.sum(np.mean(np.abs(slips), axis=1)) * dt),
        'wheel_body_threshold_fraction': float(np.mean(np.max(np.abs(slips), axis=1) > .2)),
        'wheel_body_mismatch_max': float(np.max(np.abs(slips))),
        'command_saturation_fraction': float(np.mean(run['saturated'])),
        'absolute_command_integral': float(np.sum(np.abs(run['commands'])) * dt),
        'cap_limited_seconds': float(np.sum(limited) * dt),
        'first_control_action_time': float(run['times'][:-1][np.flatnonzero(limited)[0]]) if np.any(limited) else None,
        'first_visual_change_report': first_row('visual_change'),
        'first_motion_anomaly_report': first_row('motion_anomaly'),
        'first_slip_report': first_row('slip_evidence'),
        'first_post_patch_contact_evaluation_only': run['contact_time'],
        'score_interval_seconds': [0., float(run['times'][-1])],
        'complete': run['complete'], 'camera_frames': len(rows),
        'finite': bool(all(np.isfinite(run[name]).all() for name in
                           ['states', 'observations', 'commands', 'cap', 'measured_slip', 'eta_raw'])),
    }
    return metrics


def run_case(config, case, mode, classifier, output, seed=0, live=False):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'camera').mkdir(exist_ok=True)
    times, reference, accelerations = engine.make_reference(config)
    steps, dt = len(times) - 1, config['dt']
    plant = SurfacePlant(config, case, texture_seed=1000 + seed)
    camera = GroundCamera(plant)
    p = Physics(**config['physics'])
    estimator = WheelEstimator('adaptive', p, dt, config['estimator'], config['measurement_noise']['std'][3:5])
    monitor = EvidenceMonitor(config['wheel_radius'], config['physics']['half_track'])
    governor = SpeedGovernor(mode, config['controller']['max_speed'])
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=reference.shape) * config['measurement_noise']['std']
    encoder_noise = rng.normal(size=(steps, 2)) * .05
    states, observations = np.zeros_like(reference), np.zeros_like(reference)
    states[0] = plant.true_state()
    observations[0] = states[0] + noise[0]
    commands, eta, measured_slip, innovations = (np.zeros((steps, 2)) for _ in range(4))
    caps, targets, requested = (np.zeros(steps) for _ in range(3))
    limited, saturated = np.zeros(steps, dtype=bool), np.zeros(steps, dtype=bool)
    contact_mu = np.full(steps, np.nan)
    rows, policy_rows, preview = [], [], []
    contact_time, viewer, figure, viewer_threads = None, None, None, []
    if live:
        import matplotlib.pyplot as plt
        from mujoco import viewer as native_viewer
        before = set(threading.enumerate())
        viewer = native_viewer.launch_passive(plant.model, plant.data, show_left_ui=False, show_right_ui=False)
        viewer_threads = [thread for thread in threading.enumerate() if thread not in before and thread.daemon]
        with viewer.lock():
            viewer.cam.type, viewer.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, plant.body_id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -28
        plt.ion()
        figure, (image_axis, text_axis) = plt.subplots(1, 2, figsize=(10, 4))
        artist = image_axis.imshow(camera.read_rgb())
        image_axis.axis('off')
        text_axis.axis('off')
        status_text = text_axis.text(.02, .96, 'Starting...', va='top', fontsize=10)
        figure.suptitle('RGB and movement evidence used for a cautious speed cap')
        figure.tight_layout()
    integrals, completed = np.zeros(2), 0
    try:
        for step in range(steps):
            start_time = time.perf_counter()
            if live and (not viewer.is_running() or not plt.fignum_exists(figure.number)):
                break
            with viewer.lock() if viewer is not None else nullcontext():
                # 只使用截至本步开始时已观测的线索；本步新图像最早影响下一步。
                decision = governor.update(dt, observations[step, 3], monitor.visual_change,
                                           monitor.motion_anomaly, monitor.slip_evidence)
                caps[step], targets[step] = decision['cap'], decision['target']
                requested[step] = motion_targets(observations[step], reference[step], config)[0]
                limited[step] = caps[step] < requested[step] - 1e-12
                command, raw = controller_command(observations[step], reference[step], accelerations[step],
                                                  estimator.used(), integrals, 'adaptive', config, speed_limit=caps[step])
                nominal_prediction = integrate(observations[step], command, p, dt)[3:5]
                states[step + 1] = plant.step(command)
                observations[step + 1] = states[step + 1] + noise[step + 1]
                estimator.update(observations[step, 3:5], command, observations[step + 1, 3:5])
                wheel_rates = plant.data.qvel[plant.wheel_dofs].copy() + encoder_noise[step]
                monitor.observe_motion(observations[step], observations[step + 1], nominal_prediction, wheel_rates)
                side_speed = observations[step + 1, 3] + np.array([-1, 1]) * config['physics']['half_track'] * observations[step + 1, 4]
                measured_slip[step] = config['wheel_radius'] * wheel_rates - side_speed
                commands[step], eta[step] = command, estimator.raw
                innovations[step] = observations[step + 1, 3:5] - nominal_prediction
                saturated[step] = np.any(np.abs(raw) > config['controller']['max_command'] + 1e-12)
                policy_rows.append({'time': float(times[step]), **decision,
                                    'requested_speed': float(requested[step]),
                                    'capped_speed_target': float(min(caps[step], requested[step])),
                                    'cap_limited': bool(limited[step])})
                # 真值和接触信息只用于事后评分，与 governor 接口隔离。
                audit = plant.audit()
                contact_mu[step] = audit['contact_mu']
                if contact_time is None and audit['after_surface_contact']:
                    contact_time = float(times[step + 1])
                completed = step + 1
                if step % 10 == 0:
                    rgb = camera.read_rgb()
                    prediction = classifier.predict(rgb)
                    evidence = monitor.observe_image(prediction)
                    rows.append({'time': float(times[step + 1]), 'label': prediction['label'],
                                 'match_score': prediction['match_score'], **evidence})
                    if not preview:
                        preview.append(rgb)
                    elif len(preview) == 1:
                        preview.append(rgb)
                    else:
                        preview[-1] = rgb
                    if live:
                        Image.fromarray(rgb).save(output / 'camera' / f'frame_{len(rows)-1:04d}.png')
                        artist.set_data(rgb)
                        image_axis.set_title(f"{prediction['label']} | match {prediction['match_score']:.2f}")
                        status_text.set_text(f"Mode: {mode}\nt = {times[step+1]:.2f} s\n\n"
                                             f"Speed cap: {caps[step]:.2f} m/s\n{decision['reason'].replace('_', ' ')}\n\n"
                                             f"Motion score: {monitor.motion_score:.2f}\nSlip score: {monitor.slip_score:.2f}\n\n"
                                             'Original timed reference is unchanged.\nA speed cap can delay the task.')
            if live:
                if step % 2 == 0:
                    viewer.sync()
                if step % 10 == 0:
                    plt.pause(.001)
                remaining = dt - (time.perf_counter() - start_time)
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
    run = {'case': case, 'mode': mode, 'seed': seed, 'times': times[:completed+1],
           'states': states[:completed+1], 'reference': reference[:completed+1],
           'planned_path': reference[:, :2].copy(),
           'observations': observations[:completed+1], 'commands': commands[:completed],
           'eta_raw': eta[:completed], 'cap': caps[:completed], 'cap_target': targets[:completed],
           'requested_speed': requested[:completed], 'cap_limited': limited[:completed],
           'saturated': saturated[:completed], 'measured_slip': measured_slip[:completed],
           'innovations': innovations[:completed], 'contact_mu': contact_mu[:completed],
           'rows': rows, 'contact_time': contact_time, 'complete': completed == steps}
    if not completed:
        raise RuntimeError('Simulation was closed before any control step completed.')
    run['metrics'] = score_run(run, dt)
    if not run['metrics']['finite']:
        raise FloatingPointError('Non-finite result; inspect the log.')
    run['preview_frames'] = preview
    for caption, rgb in zip(['first', 'last'], preview):
        Image.fromarray(rgb).save(output / 'camera' / f'{caption}.png')
    for name, records in [('camera_predictions.csv', rows), ('policy.csv', policy_rows)]:
        if records:
            with (output/name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    np.savez_compressed(output/'robot_log.npz', **{key: value for key, value in run.items() if isinstance(value, np.ndarray)})
    write_json(output/'metrics.json', run['metrics'])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--compare', action='store_true', help='Run all three policies without interactive windows')
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--case', choices=CASES, default='visible_low_friction')
    parser.add_argument('--mode', choices=MODES, default='vision_motion')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use distinct nonnegative seeds.')
    import matplotlib
    live = not (args.headless or args.compare or args.all_cases or len(args.seeds) > 1)
    matplotlib.use('TkAgg' if live else 'Agg')
    output = args.output_dir.resolve() if args.output_dir else ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    classifier, model_report = prepare_model()
    config = json.loads((ROOT/'config.json').read_text())
    calibration = engine.calibrate(config, plant_factory=lambda settings: SurfacePlant(settings, 'normal', texture_seed=4000))
    config['physics'] = calibration['physics']
    write_json(output/'config_used.json', config)
    write_json(output/'calibration.json', calibration)
    runs = []
    for seed in args.seeds:
        for case in CASES if args.all_cases else [args.case]:
            for mode in MODES if args.compare else [args.mode]:
                print(f'Running {case} / {mode} / seed {seed}', flush=True)
                run = run_case(config, case, mode, classifier, output/f'seed_{seed}'/case/mode, seed, live)
                runs.append(run)
                brief = {key: run['metrics'][key] for key in ['position_rmse', 'end_path_progress', 'wheel_body_mismatch_mean', 'first_control_action_time', 'complete']}
                print(brief, flush=True)
    from control_dashboard import save_comparison
    save_comparison(runs, output/'control_comparison.png')
    write_json(output/'comparison.json', {
        'runs': [{'case': run['case'], 'mode': run['mode'], 'seed': run['seed'], 'metrics': run['metrics']} for run in runs],
        'protocol': {'reference': 'identical original timed reference for all policies; no time warping',
                     'policy_settings': SETTINGS, 'camera_hz': 5, 'control_hz': 50,
                     'governor_inputs': 'past image-change flag, past motion/slip flags, observed body speed, dt',
                     'evaluation_truth': 'contact information and true states are offline scores only',
                     'scope': 'heuristic bounded speed cap, not causal terrain diagnosis or guaranteed tracking improvement',
                     'headless_images': 'all images classified at 5Hz; first/last RGB saved in headless mode'},
        'appearance_test': {'correct': model_report['heldout_correct'], 'images': model_report['heldout_images']}})
    print(f'Results saved: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
