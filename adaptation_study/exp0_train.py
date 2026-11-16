"""Experiment 0, step 2: train the motion-history GRU several times, to convergence, and forecast the held-out episodes.

The model, loss, optimiser, data windows and normalisation come from
mujoco_learned_ready/learning.py unchanged.  Compared with learning.train_all:

- only the history model is trained (no images are needed);
- the epoch budget is larger, and a run stops once the validation loss has not improved
  for --patience epochs; the best validation epoch is kept;
- several training seeds are run, one process each.

The stored learned-stage checkpoint is also replayed, as a reference and as a check that
inference here reproduces the stored forecasts.

Needs PyTorch and NumPy.
"""

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mujoco_learned_ready")]

import study  # noqa: E402

try:
    import torch
    from torch.utils.data import DataLoader
    import learning
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error}\nInstall mujoco_learned_ready/requirements.txt first.") from error

MODE = "history"
STORED_CHECKPOINT = study.LEARNED / "results" / "trained" / "checkpoints" / "history.pt"
STORED_TOLERANCE = 1e-5


def build_episodes(manifest, arrays, splits):
    """Episode dictionaries in the layout learning.py expects, in manifest order."""
    episodes = []
    for index, item in enumerate(manifest["episodes"]):
        if item["split"] not in splits:
            continue
        observed = arrays["observed"][index].astype(np.float32)
        commands = arrays["commands"][index].astype(np.float32)
        episodes.append({"id": item["id"], "observed": observed, "commands": commands,
                         # 历史模型不读图像；放一个 1×1 的占位，以便原样复用 learning 的数据窗口。
                         "rgb": np.zeros((len(observed), 1, 1, 3), dtype=np.uint8),
                         "features": learning.causal_features(observed, commands, manifest["config"])})
    return episodes


def forecast(stream, episodes, horizon):
    half, latent = [], []
    for episode in episodes:
        _, half_step, context = learning._neural_predictions(stream, episode, horizon)
        half.append(half_step[:, :2])
        latent.append(context)
    return np.asarray(half, dtype=np.float32), np.asarray(latent, dtype=np.float32)


def train_seed(job):
    seed, data_dir, max_epochs, patience, min_epochs = job
    manifest, arrays = study.load_dataset(data_dir)
    config, horizon = manifest["config"], study.horizon_steps(manifest)
    learning._seed(seed)
    torch.set_num_threads(1)  # 每个种子一个进程，固定单线程，结果不随并行数变化
    train, val = build_episodes(manifest, arrays, ("train",)), build_episodes(manifest, arrays, ("val",))
    statistics = learning._statistics(train, config)
    train_set, val_set = learning.Sequences(train, horizon), learning.Sequences(val, horizon)
    validation_loader = DataLoader(val_set, batch_size=learning.BATCH_SIZE, shuffle=False, num_workers=0)
    training_loader = DataLoader(train_set, batch_size=learning.BATCH_SIZE, shuffle=True,
                                 generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = learning.ContextModel(MODE, config, statistics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale, rows = math.inf, 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = learning._loss(model, batch, horizon, "cpu")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: seed {seed}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total += float(loss.detach()) * len(batch["q"])
            count += len(batch["q"])
        validation_loss = learning._validate(model, validation_loader, horizon, "cpu")
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        rows.append({"seed": seed, "epoch": epoch, "train_loss": total / count, "validation_loss": validation_loss,
                     "best_epoch": best_epoch})
        if epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    held_out = build_episodes(manifest, arrays, study.HELD_OUT)
    half, latent = forecast(learning.StreamModel(model, {"mode": MODE}, "cpu"), held_out, horizon)
    return {"seed": seed, "rows": rows, "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "epochs_run": len(rows), "stopped_by_patience": len(rows) < max_epochs,
            "validation_loss_at_epoch_40": rows[39]["validation_loss"] if len(rows) >= 40 else None,
            "seconds": time.perf_counter() - started, "statistics": statistics, "state": best_state,
            "half": half, "latent": latent}


def stored_reference(manifest, arrays, horizon):
    """Replay the checkpoint shipped with the learned stage and compare with its stored forecasts."""
    if not STORED_CHECKPOINT.exists():
        return None, None, None
    stream = learning.load_model(STORED_CHECKPOINT)
    held_out = build_episodes(manifest, arrays, study.HELD_OUT)
    half, latent = forecast(stream, held_out, horizon)
    valid = half.shape[1] - horizon + 1
    compared, different = 0, []
    for index, episode in enumerate(held_out):
        file = study.LEARNED / "results" / "trained" / "predictions" / episode["id"] / "history.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as source:
                difference = float(np.max(np.abs(half[index, :valid] - source["half"][:valid, :2])))
            compared += 1
            # 重新仿真后轨迹有出入的回合（见数据清单的 reference_check），预测自然也不同。
            if difference > STORED_TOLERANCE:
                different.append({"id": episode["id"], "max_difference": difference})
    return half, latent, {"checkpoint": "mujoco_learned_ready/results/trained/checkpoints/history.pt",
                          "best_epoch": stream.checkpoint["best_epoch"], "episodes_compared": compared,
                          "episodes_matching_stored_forecasts": compared - len(different),
                          "tolerance": STORED_TOLERANCE, "different": different}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=HERE / "runs" / "exp0" / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "exp0" / "training")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=50, help="epochs without a better validation loss before stopping")
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)), help="seeds trained in parallel")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use distinct nonnegative seeds.")
    if (args.output_dir / "predictions.npz").exists():
        parser.error("This directory already holds a training run; choose another --output-dir.")
    manifest, arrays = study.load_dataset(args.data_dir)
    horizon = study.horizon_steps(manifest)
    started = time.perf_counter()
    jobs = [(seed, str(args.data_dir), args.max_epochs, args.patience, args.min_epochs) for seed in args.seeds]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(train_seed, jobs):
            results.append(result)
            print(f"seed {result['seed']}: {result['epochs_run']} epochs, best at {result['best_epoch']} "
                  f"(validation loss {result['best_validation_loss']:.5f}), "
                  f"{'stopped by patience' if result['stopped_by_patience'] else 'reached the epoch limit'}, "
                  f"{result['seconds']:.0f} s", flush=True)
    stored_half, stored_latent, stored_report = stored_reference(manifest, arrays, horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    for result in results:
        torch.save({"format_version": 1, "mode": MODE,
                    "config": {"dt": float(manifest["config"]["dt"]), "physics": manifest["config"]["physics"]},
                    "statistics": result["statistics"], "state_dict": result["state"],
                    "best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"],
                    "training_seed": result["seed"], "epochs_run": result["epochs_run"],
                    "training_scope": "train observations only; val for checkpoint selection; no test/ood/truth"},
                   args.output_dir / "checkpoints" / f"history_seed{result['seed']}.pt")
    with (args.output_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "epoch", "train_loss", "validation_loss", "best_epoch"])
        writer.writeheader()
        for result in results:
            writer.writerows(result["rows"])
    held_out_ids = [item["id"] for item in manifest["episodes"] if item["split"] in study.HELD_OUT]
    payload = {"ids": np.asarray(held_out_ids), "seeds": np.asarray(args.seeds),
               "half": np.stack([result["half"] for result in results]),
               "latent": np.stack([result["latent"] for result in results])}
    if stored_half is not None:
        payload.update(stored_half=stored_half, stored_latent=stored_latent)
    np.savez_compressed(args.output_dir / "predictions.npz", **payload)
    study.write_json(args.output_dir / "training_report.json", {
        "model": MODE, "parameter_count": int(sum(value.numel() for value in results[0]["state"].values())),
        "torch_version": str(torch.__version__), "threads_per_seed": 1,
        "protocol": {"source": "learning.py model, loss, optimiser, windows and normalisation, unchanged",
                     "max_epochs": args.max_epochs, "patience": args.patience, "min_epochs": args.min_epochs,
                     "selection": "epoch with the lowest validation loss", "improvement_threshold": 1e-6,
                     "training_episodes": sum(item["split"] == "train" for item in manifest["episodes"]),
                     "validation_episodes": sum(item["split"] == "val" for item in manifest["episodes"])},
        "seeds": [{key: result[key] for key in ("seed", "epochs_run", "best_epoch", "best_validation_loss",
                                                "validation_loss_at_epoch_40", "stopped_by_patience", "seconds")}
                  for result in results],
        "stored_model": stored_report})
    if stored_report:
        print(f"Stored checkpoint replayed: forecasts match the stored ones within {STORED_TOLERANCE:g} for "
              f"{stored_report['episodes_matching_stored_forecasts']} of {stored_report['episodes_compared']} episodes"
              + "".join(f"\n  {item['id']}: differs by up to {item['max_difference']:.4f}" for item in stored_report["different"]))
    print(f"\nFinished in {time.perf_counter() - started:.0f} s. Training saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
