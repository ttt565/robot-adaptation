"""Closed-loop bias of the wheel-efficiency estimate, and an instrumental-variable remedy.

Runs the repository's planar simulator (simulation_ready/core.py) without changing it.

1. Healthy robot, measurement noise scaled by 0, 0.5, 1 and 2: the least-squares
   estimate drifts away from the true efficiency 1.0 as noise grows, while the
   instrumental-variable estimate stays near it.
2. Four scenarios with the same noise seeds for every method: fixed model, PI
   feedback, the adaptive window as shipped, and the adaptive window with the
   instrumental-variable regression.

Only NumPy is required; Matplotlib is used for the optional figure.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "simulation_ready")]

import core  # noqa: E402
from iv_estimator import estimator_swapped, instrumental_variable_estimator  # noqa: E402

IVEstimator = instrumental_variable_estimator(core.WheelEstimator)

NOISE_SCALES = [0.0, 0.5, 1.0, 2.0]
SWEEP_MODE = "window_160"
SCENARIOS = ["none", "left_motor_drop", "slope", "payload"]
# 名称 -> (core 中的控制模式, 是否改用工具变量回归)
METHODS = {
    "fixed": ("fixed", False),
    "pi": ("pi", False),
    "adaptive_least_squares": ("adaptive", False),
    "adaptive_instrumental": ("adaptive", True),
}
WARMUP_SECONDS = 4.0  # 3.2 秒的最长窗口填满之后才统计估计值
SETTLE_SECONDS = 5.0  # 变化后留出的收敛时间，与 core.tracking_metrics 的稳态区间一致


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(config, scenario, mode, seed, instrumental, noise_scale=1.0):
    case = copy.deepcopy(config)
    case["scenario"] = scenario
    case["measurement_noise"]["std"] = [noise_scale * value for value in case["measurement_noise"]["std"]]
    if instrumental:
        with estimator_swapped(core, IVEstimator):
            data = core.simulate(case, mode, seed)
    else:
        data = core.simulate(case, mode, seed)
    times, change = data["times"], case["change_time"]
    before = (times >= WARMUP_SECONDS) & (times < change)
    after = times >= change + SETTLE_SECONDS
    metrics = data["metrics"]
    return {"eta_before": data["eta_raw"][before].mean(axis=0), "eta_after": data["eta_raw"][after].mean(axis=0),
            "position_rmse": metrics["position_rmse"], "delay": metrics["tracking_confirmation_delay"],
            "pre_change_alarms": metrics["pre_change_alarm_count"]}


def summarize(rows):
    rmse = np.array([row["position_rmse"] for row in rows]) * 1000
    delays = np.array([math.inf if row["delay"] is None else row["delay"] for row in rows])
    median = float(np.median(delays))
    return {"runs": len(rows),
            "eta_before_change_mean": np.mean([row["eta_before"] for row in rows], axis=0).tolist(),
            "eta_after_change_mean": np.mean([row["eta_after"] for row in rows], axis=0).tolist(),
            "position_rmse_mm_mean": float(rmse.mean()), "position_rmse_mm_sd": float(rmse.std()),
            "confirmed_or_remained_runs": int(np.sum(np.isfinite(delays))),
            "median_confirmation_delay": median if math.isfinite(median) else None,
            "pre_change_alarm_count_mean": float(np.mean([row["pre_change_alarms"] for row in rows]))}


def noise_sweep(config, seeds):
    rows = []
    for scale in NOISE_SCALES:
        entry = {"noise_scale": scale}
        for name, instrumental in [("least_squares", False), ("instrumental_variable", True)]:
            entry[name] = summarize([run(config, "none", SWEEP_MODE, seed, instrumental, scale) for seed in seeds])
        rows.append(entry)
    for name in ["least_squares", "instrumental_variable"]:
        noiseless = np.array(rows[0][name]["eta_before_change_mean"])
        if not np.allclose(noiseless, 1.0, atol=1e-9):
            raise AssertionError(f"{name}: a noise-free healthy run must return efficiency 1, got {noiseless}")
    return rows


def scenario_comparison(config, seeds):
    table = []
    for scenario in SCENARIOS:
        methods = {name: summarize([run(config, scenario, mode, seed, instrumental) for seed in seeds])
                   for name, (mode, instrumental) in METHODS.items()}
        table.append({"scenario": scenario, "methods": methods})
    return table


def print_report(summary):
    count = summary["protocol"]["seeds"]
    print(f"Reference path: {summary['protocol']['path']}")
    print(f"\nHealthy robot, {SWEEP_MODE} ({count} noise seeds). True efficiency is 1.0 on both wheels.")
    print(f"{'noise':>6}  {'estimator':<22} {'eta left':>9} {'eta right':>9}  {'position RMSE [mm]':>18}")
    for entry in summary["noise_sweep"]:
        for name in ["least_squares", "instrumental_variable"]:
            row = entry[name]
            left, right = row["eta_before_change_mean"]
            print(f"{'x' + format(entry['noise_scale'], 'g'):>6}  {name:<22} {left:9.3f} {right:9.3f}  "
                  f"{row['position_rmse_mm_mean']:10.2f} +- {row['position_rmse_mm_sd']:.2f}")
    change = summary["protocol"]["change_time"]
    print(f"\nPosition RMSE after t = {change:g} s [mm], mean +- sd over {count} seeds; "
          "eta is the raw estimate once settled after the change.")
    header = f"{'scenario':<16}" + "".join(f"{name:>26}" for name in METHODS)
    print(header)
    for entry in summary["scenarios"]:
        cells = "".join(f"{row['position_rmse_mm_mean']:19.2f} +- {row['position_rmse_mm_sd']:<4.2f}"
                        for row in entry["methods"].values())
        print(f"{entry['scenario']:<16}{cells}")
    print(f"\n{'scenario':<16} {'eta, least squares':>22} {'eta, instrumental':>22} {'confirmed runs LS / IV':>24}")
    for entry in summary["scenarios"]:
        ls, iv = entry["methods"]["adaptive_least_squares"], entry["methods"]["adaptive_instrumental"]
        fmt = lambda pair: f"[{pair[0]:.3f}, {pair[1]:.3f}]"  # noqa: E731
        print(f"{entry['scenario']:<16} {fmt(ls['eta_after_change_mean']):>22} {fmt(iv['eta_after_change_mean']):>22} "
              f"{ls['confirmed_or_remained_runs']:>11} / {iv['confirmed_or_remained_runs']}")


def save_figure(summary, config, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import plot_style as style

    with plt.rc_context(style.RC):
        figure, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.7), gridspec_kw={"width_ratios": [1, 1.3]})

        # 左：正常工况下的估计值随噪声的变化（两轮取平均）。
        scales = [entry["noise_scale"] for entry in summary["noise_sweep"]]
        left.axhline(1.0, color=style.AXIS, linewidth=0.8, zorder=1)
        for name, color in [("least_squares", style.BLUE), ("instrumental_variable", style.ORANGE)]:
            values = [float(np.mean(entry[name]["eta_before_change_mean"])) for entry in summary["noise_sweep"]]
            left.plot(scales, values, color=color, marker="o", zorder=3, **style.MARKER)
            # 只标终点，以及现有估计器在默认噪声下的值。
            labelled = [len(scales) - 1] + ([scales.index(1.0)] if name == "least_squares" else [])
            for index in labelled:
                left.annotate(f"{values[index]:.2f}", (scales[index], values[index]), xytext=(-9, 7),
                              textcoords="offset points", ha="right", va="bottom", color=style.INK_SECONDARY, fontsize=9)
        left.set_xticks(scales)
        left.set_xticklabels([f"×{scale:g}" + ("\n(default)" if scale == 1.0 else "") for scale in scales])
        left.set_xlim(-0.12, 2.12)
        left.set_ylim(0.95, 1.48)
        left.set_xlabel("Measurement noise, relative to the default setting")
        left.set_ylabel("Estimated wheel efficiency (true value 1.0)")
        left.set_title("Healthy robot: efficiency estimate vs. noise")
        style.style_axes(left, "y")

        # 右：各场景的跟踪误差。每个场景两行：上面是自适应控制器换回归方式前后的变化，
        # 下面是作为参照的固定模型和 PI，分开画以免数值接近时互相遮挡。
        disturbance = config["disturbance"]
        labels = {"none": "No change", "left_motor_drop": f"Left motor at {disturbance['left_efficiency']:.0%}",
                  "slope": f"{disturbance['slope_degrees']:g}° slope",
                  "payload": f"Payload ×{disturbance['payload_multiplier']:g}"}
        rows = summary["scenarios"]
        for index, entry in enumerate(rows):
            y = len(rows) - 1 - index
            value = {name: row["position_rmse_mm_mean"] for name, row in entry["methods"].items()}
            upper, lower = y + 0.17, y - 0.17
            shipped, remedy = value["adaptive_least_squares"], value["adaptive_instrumental"]
            right.plot([shipped, remedy], [upper, upper], color=style.AXIS, linewidth=1.4, zorder=2)
            right.plot(shipped, upper, color=style.BLUE, marker="o", linestyle="none", zorder=3, **style.MARKER)
            right.plot(remedy, upper, color=style.ORANGE, marker="o", linestyle="none", zorder=4, **style.MARKER)
            # 箭头用 mathtext，避免部分系统字体缺少该字符。
            right.annotate(f"{shipped:.1f} $\\rightarrow$ {remedy:.1f}", (max(shipped, remedy), upper), xytext=(9, 0),
                           textcoords="offset points", ha="left", va="center", color=style.INK_SECONDARY, fontsize=9)
            right.plot(value["fixed"], lower, color=style.CONTEXT, marker="o", linestyle="none", zorder=3, **style.MARKER)
            right.plot(value["pi"], lower, color=style.CONTEXT, marker="D", linestyle="none", zorder=3, **style.MARKER)
        right.set_yticks(range(len(rows)))
        right.set_yticklabels([labels[entry["scenario"]] for entry in reversed(rows)])
        right.set_ylim(-0.55, len(rows) - 0.45)
        style.plain_log_axis(right, "x", [1, 3, 10, 30, 100, 300])
        right.set_xlim(0.6, 300)
        right.set_xlabel(f"Position RMSE after t = {config['change_time']:g} s (mm, log scale)")
        right.set_title("Tracking error by scenario")
        style.style_axes(right, "x")
        right.tick_params(axis="y", length=0)

        def handle(color, marker, line):
            return Line2D([0], [0], color=color if line else "none", marker=marker, markerfacecolor=color, **style.MARKER)

        figure.legend(handles=[handle(style.BLUE, "o", True), handle(style.ORANGE, "o", True),
                               handle(style.CONTEXT, "o", False), handle(style.CONTEXT, "D", False)],
                      labels=["Adaptive, least squares (as shipped)", "Adaptive, instrumental variable",
                              "Fixed model", "PI feedback"],
                      loc="lower center", ncol=4, handlelength=1.6, columnspacing=2.2)
        figure.subplots_adjust(left=0.075, right=0.975, top=0.9, bottom=0.235, wspace=0.42)
        figure.savefig(path, dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10, help="number of paired noise seeds, starting at 0")
    parser.add_argument("--path", choices=["straight", "circle"], default="straight", help="reference path")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-plot", action="store_true", help="skip the Matplotlib figure")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    config = json.loads((REPO / "simulation_ready" / "config.json").read_text(encoding="utf-8"))
    config["reference"]["path"] = args.path
    core.validate_config(config)
    seeds = range(args.seeds)
    started = time.perf_counter()
    summary = {
        "protocol": {"simulator": "simulation_ready/core.py, unmodified", "seeds": args.seeds, "path": args.path,
                     "change_time": config["change_time"], "noise_sweep_mode": SWEEP_MODE,
                     "eta_before_change": f"mean raw estimate for {WARMUP_SECONDS:g} s <= t < change",
                     "eta_after_change": f"mean raw estimate for t >= change + {SETTLE_SECONDS:g} s",
                     "position_rmse": "core.tracking_metrics position_rmse (t > change), in millimetres",
                     "instrument": "previous command u[t-1]; see iv_estimator.py",
                     "paired_noise": "every method sees the same measurement-noise seeds"},
        "noise_sweep": noise_sweep(config, seeds),
        "scenarios": scenario_comparison(config, seeds),
    }
    print_report(summary)
    # 默认的直线轨迹用不带后缀的文件名，其他轨迹另存，避免互相覆盖。
    stem = "closed_loop_bias" if args.path == "straight" else f"closed_loop_bias_{args.path}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{stem}.json", summary)
    if not args.no_plot:
        save_figure(summary, config, args.output_dir / f"{stem}.png")
    print(f"\nFinished in {time.perf_counter() - started:.0f} s. Results saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
