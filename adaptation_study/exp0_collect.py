"""Experiment 0, step 1: motion data for a repeated comparison, collected without rendering.

Rebuilds the 84 episodes of the learned stage from their original plans (same seeds,
same MuJoCo plant, no camera) and adds new held-out episodes that nobody has looked at:
twelve per physical condition.  Only body speed, yaw rate, wheel rates and commands are
stored, which is all the motion-history model and the explicit estimator use.

Before writing anything the script compares what it regenerated with what the repository
stores for the original episodes: nominal calibration, event times and held-out truth.

Needs MuJoCo, NumPy and Pillow.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mujoco_learned_ready")]

import study  # noqa: E402

try:
    import mujoco
    import collect_data
    import engine
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error}\nInstall mujoco_learned_ready/requirements.txt first.") from error

NEW_PER_CONDITION = 12
NEW_CONDITIONS = {"test": ["normal", "low_friction", "medium_friction", "motor_fault"],
                  "ood": ["reversed", "severe_motor", "severe_hidden"]}
FIRST_NEW_INDEX = 100  # 原有回合的编号都小于 48，新回合从 100 起，随机种子不会重合
SPLIT_BASE = {"train": 1000, "val": 100000, "test": 200000, "ood": 300000}  # 与 collect_data.episode_plan 相同
OBSERVATION_NOISE = np.array([.01, .01, .05, .05])  # 与 collect_data.collect_episode 相同
MATCH_TOLERANCE = 1e-4


def plan_new(split, index, condition):
    """Same draws, in the same order, as collect_data.episode_plan; the condition is given instead of derived."""
    base = SPLIT_BASE[split] + index
    rng = np.random.default_rng(base)
    high, low, medium = float(rng.uniform(.55, .9)), float(rng.uniform(.025, .06)), float(rng.uniform(.07, .13))
    before_mu, after_mu = high, high
    if condition == "low_friction":
        after_mu = low
    elif condition == "medium_friction":
        after_mu = medium
    elif condition == "severe_hidden":
        after_mu = .015
    elif condition == "reversed":
        before_mu, after_mu = low, high
    return {"id": f"{split}_{index:03d}", "split": split, "condition": condition,
            "texture_seed": base * 2, "input_seed": base + 900000, "noise_seed": base + 1900000,
            "before_kind": "tile_grid", "after_kind": "tile_grid", "before_mu": before_mu, "after_mu": after_mu,
            "boundary": float(rng.uniform(1.4, 2.8)), "motor_event": float(rng.uniform(3., 7.)),
            "motor_efficiency": .25 if condition == "severe_motor" else float(rng.uniform(.45, .75)),
            "light": float(rng.uniform(.65, 1.05)), "ambient": float(rng.uniform(.15, .35))}


def physical_group(plan):
    """What happens to the motion; appearance does not matter without a camera."""
    condition = plan["condition"]
    if condition in ("normal", "appearance_only"):
        return "normal"
    if condition in ("correlated", "hidden_friction", "low_friction", "medium_friction"):
        return "medium_friction" if plan["after_mu"] >= .065 else "low_friction"
    return condition


def collect_motion(config, plan):
    """collect_data.collect_episode without the renderer and the camera audit."""
    plant = collect_data.LearningPlant(config, plan)
    commands = collect_data.planned_commands(config, plan["input_seed"])
    count = len(commands)
    truth, positions, first_contact = np.zeros((count + 1, 4)), np.zeros((count + 1, 3)), None
    for step in range(count + 1):
        truth[step] = plant.motion()
        positions[step] = plant.data.xpos[plant.body_id]
        after, _ = plant.terrain_audit()
        if first_contact is None and after:
            first_contact = step * config["dt"]
        if step < count:
            plant.step(commands[step])
    observed = truth + np.random.default_rng(plan["noise_seed"]).normal(size=truth.shape) * OBSERVATION_NOISE
    group = physical_group(plan)
    motor = plan["condition"] in ("motor_fault", "severe_motor")
    event = plan["motor_event"] if motor else first_contact if plan["before_mu"] != plan["after_mu"] else None
    valid = bool(np.isfinite(truth).all() and np.isfinite(observed).all()
                 and positions[:, 2].min() > .045 and positions[:, 2].max() < .3
                 and (np.abs(positions[:, 1]) < 19).all() and (positions[:, 0] > -7).all() and (positions[:, 0] < 31).all()
                 and (event is not None or group == "normal"))
    metadata = {key: plan[key] for key in ("id", "split", "condition", "before_mu", "after_mu", "boundary", "motor_event",
                                           "motor_efficiency", "input_seed", "noise_seed")}
    metadata.update(group=group, event_time=event, first_post_boundary_contact=first_contact, valid=valid,
                    min_body_height=float(positions[:, 2].min()), max_body_height=float(positions[:, 2].max()))
    return truth, observed, commands, metadata


def reference_check(config, calibration, episodes, truth):
    """Compare the regenerated original episodes with what the repository stores."""
    stored = study.read_json(study.LEARNED / "demo_assets" / "manifest.json")
    stored_episodes = {item["id"]: item for item in stored["episodes"]}
    physics = {key: abs(calibration["physics"][key] - stored["config"]["physics"][key])
               for key in ("mass", "inertia", "linear_drag", "yaw_drag")}
    event_difference, diverged, compared = 0.0, [], 0
    for index, item in enumerate(episodes):
        if item["set"] != "original":
            continue
        reference = stored_episodes[item["id"]]
        if (item["event_time"] is None) != (reference["event_time"] is None):
            raise AssertionError(f"Event presence differs for {item['id']}")
        if item["event_time"] is not None:
            event_difference = max(event_difference, abs(item["event_time"] - reference["event_time"]))
        file = study.LEARNED / "results" / "trained" / "predictions" / item["id"] / "fixed.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as source:
                difference = np.abs(truth[index][:, :2] - source["truth"][:, :2]).max(axis=1)
            compared += 1
            if difference.max() > MATCH_TOLERANCE:
                first = float(np.argmax(difference > MATCH_TOLERANCE) * config["dt"])
                diverged.append({"id": item["id"], "max_difference": float(difference.max()), "first_seconds": first})
    return {"stored_mujoco_version": "3.16.0 (mujoco_ready/ready_check.json)", "calibration_abs_difference": physics,
            "event_time_max_difference": event_difference, "held_out_episodes_compared": compared,
            "held_out_episodes_matching": compared - len(diverged), "match_tolerance": MATCH_TOLERANCE,
            "diverged": diverged}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "exp0" / "data")
    args = parser.parse_args()
    if (args.output_dir / "manifest.json").exists():
        parser.error("This directory already holds a dataset; choose another --output-dir.")
    started = time.perf_counter()
    config = study.read_json(study.LEARNED / "config.json")
    config.update({"dt": .1, "duration": 12., "physics_dt": .002, "visualize_hidden_disturbance": False})
    # 名义模型的标定方式与 collect_data.collect_dataset 完全相同。
    calibration_plan = collect_data.episode_plan("train", 0)
    calibration_plan.update(condition="normal", texture_seed=9000001, before_kind="tile_grid", after_kind="tile_grid",
                            before_mu=.8, after_mu=.8, boundary=20.)
    calibration = engine.calibrate(config, plant_factory=lambda settings: collect_data.LearningPlant(settings, calibration_plan))
    config["physics"] = calibration["physics"]
    plans = [("original", collect_data.episode_plan(split, index))
             for split, count in collect_data.COUNTS.items() for index in range(count)]
    for split, conditions in NEW_CONDITIONS.items():
        for position, condition in enumerate(conditions):
            plans += [("new", plan_new(split, FIRST_NEW_INDEX + position * NEW_PER_CONDITION + number, condition))
                      for number in range(NEW_PER_CONDITION)]
    for name in ("input_seed", "noise_seed", "id"):
        if len({plan[name] for _, plan in plans}) != len(plans):
            raise AssertionError(f"{name} is not unique across episodes")
    truth, observed, commands, episodes = [], [], [], []
    for number, (origin, plan) in enumerate(plans, 1):
        result = collect_motion(config, plan)
        truth.append(result[0]), observed.append(result[1]), commands.append(result[2])
        episodes.append(dict(result[3], set=origin))
        if number % 40 == 0 or number == len(plans):
            print(f"Collected {number}/{len(plans)} episodes, {time.perf_counter() - started:.0f} s", flush=True)
    truth, observed, commands = np.asarray(truth), np.asarray(observed), np.asarray(commands)
    invalid = [item["id"] for item in episodes if not item["valid"]]
    check = reference_check(config, calibration, episodes, truth)
    counts = {}
    for item in episodes:
        key = f"{item['set']}/{item['split']}/{item['group']}"
        counts[key] = counts.get(key, 0) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "dataset.npz", truth=truth, observed=observed, commands=commands)
    study.write_json(args.output_dir / "manifest.json", {
        "config": config, "mujoco_version": mujoco.__version__, "episodes": episodes, "episode_counts": counts,
        "invalid_episodes": invalid, "reference_check": check,
        "protocol": {"dt": .1, "physics_dt": .002, "rendering": "none; motion and commands only",
                     "arrays": "truth/observed are (episode, step, [v, yaw rate, left wheel rate, right wheel rate]); "
                               "commands are (episode, step, [left, right]) in N",
                     "observation_noise_std": OBSERVATION_NOISE.tolist(),
                     "original": "plans from collect_data.episode_plan, same seeds as the learned stage",
                     "new": f"{NEW_PER_CONDITION} held-out episodes per physical condition, indices from {FIRST_NEW_INDEX}",
                     "group": "physical condition used for reporting; appearance-only episodes count as normal"}})
    print(f"\nMuJoCo {mujoco.__version__}. Reference check against the stored learned stage:")
    print("  calibration differs by at most "
          f"{max(check['calibration_abs_difference'].values()):.1e}; event times by {check['event_time_max_difference']:.1e} s")
    print(f"  held-out truth matches within {MATCH_TOLERANCE:g} for {check['held_out_episodes_matching']} of "
          f"{check['held_out_episodes_compared']} episodes" + "".join(
              f"\n    {item['id']}: differs by up to {item['max_difference']:.4f} from {item['first_seconds']:.1f} s"
              for item in check["diverged"]))
    print("Episodes per set/split/group:")
    for key, value in counts.items():
        print(f"  {key:<34}{value:>4}")
    if invalid:
        print(f"Episodes that failed the physical audit (kept and flagged): {invalid}")
    print(f"\nFinished in {time.perf_counter() - started:.0f} s. Dataset saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
