"""One-command entry point for the planar robot adaptation simulator."""

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from core import MODES, SCENARIOS, self_check, simulate, validate_config
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error}\nRun: python -m pip install -r requirements.txt") from error


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def aggregate(records):
    summaries = []
    scenarios = list(dict.fromkeys(record["scenario"] for record in records))
    for scenario in scenarios:
        methods = {}
        for mode in MODES:
            group = [record["metrics"] for record in records if record["scenario"] == scenario and record["mode"] == mode]
            delays = np.array([np.inf if record["tracking_confirmation_delay"] is None
                               else record["tracking_confirmation_delay"] for record in group])
            median = float(np.median(delays))
            methods[mode] = {
                "runs": len(group), "confirmed_or_remained_runs": int(np.sum(np.isfinite(delays))),
                "confirmation_rate": float(np.mean(np.isfinite(delays))),
                "median_confirmation_delay": median if math.isfinite(median) else None,
                "position_rmse_mean": float(np.mean([record["position_rmse"] for record in group])),
                "position_rmse_sd": float(np.std([record["position_rmse"] for record in group])),
                "saturation_fraction_mean": float(np.mean([record["command_saturation_fraction"] for record in group])),
                "eta_protection_fraction_mean": float(np.mean([record["eta_protection_fraction"] for record in group])),
                "pre_change_alarm_count_mean": float(np.mean([record["pre_change_alarm_count"] for record in group])),
            }
        summaries.append({"scenario": scenario, "methods": methods})
    return summaries


def save_case(results, config, path, save_dashboard):
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config_used.json", config)
    write_json(path / "metrics.json", {mode: run["metrics"] for mode, run in results.items()})
    for mode, run in results.items():
        arrays = {key: value for key, value in run.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(path / f"{mode}.npz", **arrays)
    data = results["adaptive"]
    csv_data = np.column_stack([data["times"], data["states"], data["reference"],
                                data["eta_raw"], data["eta_used"], data["window_steps"] * config["dt"]])
    header = "time,x,y,heading,v,omega,ref_x,ref_y,ref_heading,ref_v,ref_omega,eta_raw_L,eta_raw_R,eta_used_L,eta_used_R,window_seconds"
    np.savetxt(path / "adaptive_trajectory.csv", csv_data, delimiter=",", header=header, comments="", fmt="%.9g")
    # 兼容独立viewer的上限配置。
    plot_config = {**config, "command_limit": config["controller"]["max_command"]}
    save_dashboard(results, plot_config, path / "dashboard.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--path", choices=["straight", "circle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="Number of paired seeds, starting from --seed")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true", help="Run readiness checks and exit")
    args = parser.parse_args()
    if args.seed < 0 or args.seeds < 1:
        parser.error("Use --seed>=0 and --seeds>=1.")
    if args.scenario and args.all_scenarios:
        parser.error("Choose --scenario or --all-scenarios.")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.scenario:
            config["scenario"] = args.scenario
        if args.path:
            config["reference"]["path"] = args.path
        validate_config(config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid configuration: {error}")
    if args.self_check:
        print(self_check(config))
        return 0

    import matplotlib
    if args.no_animation:
        matplotlib.use("Agg")
    from viewer import save_dashboard, show_animation

    if args.output_dir:
        output = args.output_dir.resolve()
        # 防止无意覆盖先前实验；每次运行的目录必须是新目录或空目录。
        if output.exists() and any(output.iterdir()):
            parser.error("--output-dir is not empty. Choose a new directory to preserve previous results.")
    else:
        output = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "base_config.json", config)
    scenarios = SCENARIOS if args.all_scenarios else [config["scenario"]]
    records, replay = [], None
    print(f"Output directory: {output}")
    print("Each controller runs independently with paired measurement-noise samples.")
    print("Confirmation includes a continuous 1 s hold using the configured tolerances.")
    for scenario in scenarios:
        case = copy.deepcopy(config)
        case["scenario"] = scenario
        for seed in range(args.seed, args.seed + args.seeds):
            results = {mode: simulate(case, mode, seed) for mode in MODES}
            folder = output / f"{scenario}_seed_{seed:03d}"
            save_case(results, case, folder, save_dashboard)
            print(f"\n{scenario}, seed {seed}:", flush=True)
            for mode, run in results.items():
                metric = run["metrics"]
                delay = metric["tracking_confirmation_delay"]
                delay_text = "not confirmed" if delay is None else f"{delay:.2f} s ({metric['tracking_status']})"
                print(f"  {mode:12s} position RMSE={metric['position_rmse']:.4f} m; "
                      f"confirmation={delay_text}; saturation={100*metric['command_saturation_fraction']:.1f}%")
                records.append({"scenario": scenario, "seed": seed, "mode": mode,
                                "folder": folder.name, "metrics": metric})
            if replay is None:
                replay = (results, case)
    source_names = ["core.py", "viewer.py", "run_simulation.py"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
        "protocol": {
            "state": "[x,y,heading,v,omega]", "effective_parameter_state": "[eta_L,eta_R]",
            "recovery": "True position/heading/speed within configured bands for hold_duration; confirmation includes hold; 0 means remained in band throughout.",
            "failure_median": "Failed runs count as infinity; null median means infinity.",
            "noise_pairing": "Controllers share standard noise draws but run different closed-loop trajectories.",
            "same_log_prediction": "Within each run, nominal shadow predictions use the same observations and controls as that run's adaptive predictor.",
            "estimator_scope": "Explicit effective efficiency, not learned neural latent state or unique physical-cause identification.",
            "statistics_scope": "Feedback-generated commands correlate with observations; noise-aware detection is heuristic.",
            "traction_scope": "Longitudinal force cap only, no wheel-spin/lateral-slip physics.",
            "animation": "Precomputed simulation replay, not wall-clock physics integration.",
        },
        "aggregates": aggregate(records), "runs": records,
    }
    write_json(output / "summary.json", summary)
    print(f"\nCompleted {len(records)} controller runs. Saved summary.json, dashboards and raw logs.")
    print(f"Results: {output}")
    if not args.no_animation:
        if len(scenarios) * args.seeds > 1:
            print("Animation replays the first case; all other cases are saved on disk.")
        show_animation(replay[0], {**replay[1], "command_limit": replay[1]["controller"]["max_command"]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
