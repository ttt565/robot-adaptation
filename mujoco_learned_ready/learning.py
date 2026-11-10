"""Causal, episode-split PyTorch system-identification experiments.

Only ``observed``, nominal ``commands`` and current RGB enter a predictor.
Physical conditions, event annotations and ``truth`` are evaluation-only data.
The learned eight-dimensional context is not a friction estimate.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


MODES = ("history", "vision", "fusion")
SEQUENCE = 20
BURN_IN = 10
STRIDE = 10
BATCH_SIZE = 16
MULTI_WEIGHT = .20
WHEEL_WEIGHT = .08


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    # All operations used here have deterministic CPU implementations.
    torch.use_deterministic_algorithms(True)


def _manifest(data_root):
    data_root = Path(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if not 0 < float(config["dt"]) <= .1 + 1e-8:
        raise ValueError("Expected macro-step dt <= 0.1 seconds.")
    horizon = round(.5 / float(config["dt"]))
    if not math.isclose(horizon * float(config["dt"]), .5, abs_tol=1e-7):
        raise ValueError("The half-second horizon must contain an integer number of macro steps.")
    ids = [str(item["id"]) for item in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must be unique across train/val/test/ood.")
    if any(item["split"] not in ("train", "val", "test", "ood") for item in manifest["episodes"]):
        raise ValueError("Unknown episode split.")
    return manifest, config, horizon


def _read_episode(data_root, item, need_truth=False):
    path = (Path(data_root) / item["file"]).resolve()
    if not path.is_relative_to(Path(data_root).resolve()):
        raise ValueError("Episode files must stay inside data_root.")
    with np.load(path, allow_pickle=False) as source:
        # Deliberately do not even load the truth array during training.
        episode = {key: np.asarray(source[key]) for key in ("times", "observed", "commands", "rgb")}
        if need_truth:
            episode["truth"] = np.asarray(source["truth"])
    count = len(episode["commands"])
    if episode["observed"].shape != (count + 1, 4):
        raise ValueError(f"Expected observed(T+1,4): {path}")
    if episode["commands"].shape != (count, 2):
        raise ValueError(f"Expected commands(T,2): {path}")
    if episode["rgb"].shape != (count + 1, 64, 64, 3):
        raise ValueError(f"Expected rgb(T+1,64,64,3): {path}")
    if episode["rgb"].dtype != np.uint8:
        raise ValueError("RGB must use uint8 pixels, without hidden metadata channels.")
    for key in ("observed", "commands", "times"):
        if not np.isfinite(episode[key]).all():
            raise ValueError(f"Nonfinite {key}: {path}")
    if need_truth and (episode["truth"].shape != (count + 1, 4)
                       or not np.isfinite(episode["truth"]).all()):
        raise ValueError(f"Invalid evaluation truth: {path}")
    episode["observed"] = episode["observed"].astype(np.float32)
    episode["commands"] = episode["commands"].astype(np.float32)
    return episode


def physical_numpy(q, command, config):
    """Normal planar body model and wheel-rate persistence, no hidden parameters."""
    q, command = np.asarray(q), np.asarray(command)
    p, dt = config["physics"], float(config["dt"])
    result = q.copy()
    result[..., 0] = q[..., 0] + dt * (command[..., 0] + command[..., 1]
                                      - p["linear_drag"] * q[..., 0]) / p["mass"]
    result[..., 1] = q[..., 1] + dt * (p["half_track"] * (command[..., 1] - command[..., 0])
                                      - p["yaw_drag"] * q[..., 1]) / p["inertia"]
    return result


def causal_features(observed, commands, config):
    """At k: q[k], u[k-1], q[k]-normal(q[k-1],u[k-1]); no future target."""
    observed = np.asarray(observed, dtype=np.float32)
    count = len(commands)
    past = np.zeros((count, 2), dtype=np.float32)
    residual = np.zeros((count, 2), dtype=np.float32)
    if count > 1:
        past[1:] = commands[:-1]
        normal = physical_numpy(observed[:-2], commands[:-1], config)
        residual[1:] = observed[1:-1, :2] - normal[:, :2]
    return np.concatenate([observed[:-1], past, residual], axis=-1)


def _statistics(episodes, config):
    q = np.concatenate([ep["observed"][:-1] for ep in episodes])
    u = np.concatenate([ep["commands"] for ep in episodes])
    features = np.concatenate([ep["features"] for ep in episodes])
    corrections = np.concatenate([ep["observed"][1:] - physical_numpy(ep["observed"][:-1], ep["commands"], config)
                                  for ep in episodes])
    q_std = np.maximum(np.std(q, axis=0), [.10, .10, 1., 1.])
    u_std = np.maximum(np.std(u, axis=0), [.25, .25])
    residual_std = np.maximum(np.std(features[:, 6:8], axis=0), [.01, .02])
    # Change scales put v and yaw on separate dimensionless training scales.
    # Reported metrics remain in m/s and rad/s, never in this mixed loss unit.
    correction_std = np.maximum(np.std(corrections, axis=0), [.01, .02, .20, .20])
    return {"q_mean": q.mean(axis=0).tolist(), "q_std": q_std.tolist(),
            "u_mean": u.mean(axis=0).tolist(), "u_std": u_std.tolist(),
            "residual_mean": features[:, 6:8].mean(axis=0).tolist(),
            "residual_std": residual_std.tolist(), "correction_std": correction_std.tolist(),
            "fitted_split": "train", "fitted_observation_rows": int(len(q))}


class Sequences(Dataset):
    def __init__(self, episodes, horizon):
        self.episodes, self.horizon = episodes, horizon
        self.origins = []
        for episode_index, episode in enumerate(episodes):
            last = len(episode["commands"]) - SEQUENCE - horizon + 1
            self.origins.extend((episode_index, start) for start in range(0, last + 1, STRIDE))
        if not self.origins:
            raise ValueError("Episodes are too short for 20-step sequences and the forecast horizon.")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        episode_index, start = self.origins[index]
        ep = self.episodes[episode_index]
        return {"q": torch.from_numpy(ep["observed"][start:start + SEQUENCE + self.horizon]),
                "u": torch.from_numpy(ep["commands"][start:start + SEQUENCE + self.horizon - 1]),
                "features": torch.from_numpy(ep["features"][start:start + SEQUENCE]),
                "rgb": torch.from_numpy(ep["rgb"][start:start + SEQUENCE])}


class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 5, stride=2, padding=2), nn.ReLU(),
                                    nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                                    nn.Linear(128, 16), nn.ReLU())

    def forward(self, rgb):
        shape = rgb.shape[:-3]
        # Raw images are the sole vision input: no class, condition, mu or event.
        image = rgb.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2).float() / 255.
        return self.layers(image).reshape(*shape, 16)


class ContextModel(nn.Module):
    def __init__(self, mode, config, statistics):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        # Only the known nominal model is retained, not case metadata.
        self.config = {"dt": float(config["dt"]), "physics": copy.deepcopy(config["physics"])}
        for name in ("q_mean", "q_std", "u_mean", "u_std", "residual_mean", "residual_std", "correction_std"):
            self.register_buffer(name, torch.tensor(statistics[name], dtype=torch.float32))
        self.rgb_encoder = RGBEncoder() if mode in ("vision", "fusion") else None
        if mode in ("history", "fusion"):
            self.gru = nn.GRU(8 + (16 if mode == "fusion" else 0), 32, batch_first=True)
            self.to_latent = nn.Sequential(nn.Linear(32, 8), nn.Tanh())
        else:
            self.gru = None
            self.to_latent = nn.Sequential(nn.Linear(16, 8), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(14, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, features, rgb=None, hidden=None, reset_history=False):
        if self.mode == "vision":
            return self.to_latent(self.rgb_encoder(rgb)), None
        normalized = torch.cat([(features[..., :4] - self.q_mean) / self.q_std,
                                (features[..., 4:6] - self.u_mean) / self.u_std,
                                (features[..., 6:8] - self.residual_mean) / self.residual_std], dim=-1)
        if self.mode == "fusion":
            normalized = torch.cat([normalized, self.rgb_encoder(rgb)], dim=-1)
        if reset_history:
            # Each origin receives its current causal features but no recurrent memory.
            shape = normalized.shape
            output, _ = self.gru(normalized.reshape(-1, 1, shape[-1]), None)
            return self.to_latent(output.reshape(shape[0], shape[1], 32)), None
        encoded, hidden = self.gru(normalized, hidden)
        return self.to_latent(encoded), hidden

    def nominal(self, q, command):
        p, dt = self.config["physics"], self.config["dt"]
        speed = q[..., 0] + dt * (command[..., 0] + command[..., 1] - p["linear_drag"] * q[..., 0]) / p["mass"]
        yaw = q[..., 1] + dt * (p["half_track"] * (command[..., 1] - command[..., 0])
                               - p["yaw_drag"] * q[..., 1]) / p["inertia"]
        return torch.stack([speed, yaw, q[..., 2], q[..., 3]], dim=-1)

    def step(self, q, command, latent):
        normalized = torch.cat([(q - self.q_mean) / self.q_std,
                                (command - self.u_mean) / self.u_std, latent], dim=-1)
        return self.nominal(q, command) + self.decoder(normalized) * self.correction_std


def _loss(model, batch, horizon, device):
    batch = {key: tensor.to(device) for key, tensor in batch.items()}
    latent, _ = model.encode(batch["features"], batch["rgb"] if model.rgb_encoder is not None else None)
    q, commands = batch["q"], batch["u"]
    one = model.step(q[:, :SEQUENCE], commands[:, :SEQUENCE], latent)
    rollout = q[:, :SEQUENCE]
    # The same origin latent is frozen across every step, also while training.
    for offset in range(horizon):
        rollout = model.step(rollout, commands[:, offset:offset + SEQUENCE], latent)
    weights = torch.tensor([1., 1., WHEEL_WEIGHT, WHEEL_WEIGHT], device=device)
    one_error = (one[:, BURN_IN:] - q[:, 1:SEQUENCE + 1][:, BURN_IN:]) / model.correction_std
    half_error = (rollout[:, BURN_IN:] - q[:, horizon:horizon + SEQUENCE][:, BURN_IN:]) / model.correction_std
    return (one_error.square() * weights).mean() + MULTI_WEIGHT * (half_error.square() * weights).mean()


def _validate(model, loader, horizon, device):
    model.eval()
    loss_sum, count = 0., 0
    with torch.inference_mode():
        for batch in loader:
            loss = _loss(model, batch, horizon, device)
            loss_sum += float(loss) * len(batch["q"])
            count += len(batch["q"])
    return loss_sum / count


def train_all(data_root: Path, output: Path, epochs=40, seed=0, device="cpu"):
    """Train actual neural models; only validation selects epochs and checkpoints."""
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    _seed(seed)
    data_root, output = Path(data_root), Path(output)
    manifest, config, horizon = _manifest(data_root)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episodes = {"train": [], "val": []}
    for item in manifest["episodes"]:
        if item["split"] not in episodes:
            continue
        ep = _read_episode(data_root, item, need_truth=False)
        ep["features"] = causal_features(ep["observed"], ep["commands"], config)
        episodes[item["split"]].append(ep)
    if not episodes["train"] or not episodes["val"]:
        raise ValueError("Separate train and val episodes are required.")
    statistics = _statistics(episodes["train"], config)
    _json(output / "normalization.json", statistics)
    train_set, val_set = Sequences(episodes["train"], horizon), Sequences(episodes["val"], horizon)
    validation_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest()
    rows, reports = [], {}
    history_path = output / "train_history.csv"
    fields = ["model", "epoch", "train_loss", "validation_loss", "best_epoch", "seconds"]
    for mode in MODES:
        # Equal initial seed, batch order, epoch budget, loss and early-stop rule.
        _seed(seed)
        generator = torch.Generator().manual_seed(seed)
        training_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                     generator=generator, num_workers=0)
        model = ContextModel(mode, config, statistics).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss, best_epoch, stale = math.inf, 0, 0
        began = time.perf_counter()
        for epoch in range(1, int(epochs) + 1):
            model.train()
            total, count = 0., 0
            for batch in training_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, batch, horizon, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss: {mode}, epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                total += float(loss.detach()) * len(batch["q"])
                count += len(batch["q"])
            val_loss = _validate(model, validation_loader, horizon, device)
            if not math.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation loss")
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                checkpoint = {"format_version": 1, "mode": mode,
                              "config": {"dt": float(config["dt"]), "physics": copy.deepcopy(config["physics"])},
                              "statistics": statistics, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              "best_epoch": epoch, "best_validation_loss": val_loss,
                              "training_seed": int(seed), "training_manifest_sha256": manifest_sha,
                              "learning_source_sha256": source_sha,
                              "training_scope": "train observations/RGB only; val for checkpoint selection; no test/ood/truth"}
                torch.save(checkpoint, checkpoint_dir / f"{mode}.pt")
            else:
                stale += 1
            row = {"model": mode, "epoch": epoch, "train_loss": total / count,
                   "validation_loss": val_loss, "best_epoch": best_epoch,
                   "seconds": time.perf_counter() - began}
            rows.append(row)
            with history_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            print(f"{mode}: epoch {epoch}/{epochs}; train {row['train_loss']:.5f}; val {val_loss:.5f}; best {best_epoch}", flush=True)
            if epoch >= 12 and stale >= 8:
                break
        reports[mode] = {"best_epoch": best_epoch, "epochs_run": epoch,
                         "best_validation_loss": best_loss,
                         "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                         "training_seconds": time.perf_counter() - began,
                         "checkpoint": f"checkpoints/{mode}.pt"}
    report = {"models": reports, "seed": int(seed), "device": str(device), "torch_version": str(torch.__version__),
              "maximum_epochs": int(epochs), "batch_size": BATCH_SIZE, "sequence_length": SEQUENCE,
              "burn_in_steps": BURN_IN, "future_steps": horizon,
              "training_episode_count": len(episodes["train"]), "validation_episode_count": len(episodes["val"]),
              "training_sequences": len(train_set), "validation_sequences": len(val_set),
              "normalization_split": "train", "checkpoint_selection_split": "val", "loss_uses_truth": False,
              "learning_source_sha256": source_sha, "training_manifest_sha256": manifest_sha,
              "early_stop": "at least 12 epochs, patience 8; same maximum epochs and batches for all models",
              "loss": "separate normalized channels; wheel weight .08; one-step + .20 * frozen-context half-second loss"}
    _json(output / "training_report.json", report)
    return report


class StreamModel:
    """Frozen neural weights with causal recurrent-state updates at observation time."""
    def __init__(self, model, checkpoint, device="cpu"):
        self.model, self.checkpoint, self.device = model.eval(), checkpoint, torch.device(device)
        self.config = model.config
        self.reset()

    def reset(self):
        self.hidden, self.previous_observation = None, None

    @torch.inference_mode()
    def stream_predict(self, observed4, current_rgb, previous_command2=None, future_commands=None,
                       reset_history=False):
        q = np.asarray(observed4, dtype=np.float32)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("observed4 must be a finite [v,w,wheelL,wheelR] vector")
        past, residual = np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
        if self.previous_observation is not None:
            if previous_command2 is None:
                raise ValueError("Subsequent observations require the actually applied previous nominal command")
            past = np.asarray(previous_command2, dtype=np.float32)
            if past.shape != (2,) or not np.isfinite(past).all():
                raise ValueError("Invalid previous command")
            residual = q[:2] - physical_numpy(self.previous_observation, past, self.config)[:2]
        feature = np.concatenate([q, past, residual])
        tensor = torch.tensor(feature, device=self.device).reshape(1, 1, 8)
        rgb = None
        if self.model.rgb_encoder is not None:
            image = np.asarray(current_rgb)
            if image.shape != (64, 64, 3) or image.dtype != np.uint8:
                raise ValueError("current_rgb must be uint8(64,64,3)")
            rgb = torch.from_numpy(image.copy()).to(self.device).reshape(1, 1, 64, 64, 3)
        latent, self.hidden = self.model.encode(tensor, rgb, None if reset_history else self.hidden)
        self.previous_observation = q.copy()
        result = {"latent": latent[0, 0].cpu().numpy().copy(), "one": None, "half": None}
        if future_commands is None:
            return result
        commands = np.asarray(future_commands, dtype=np.float32)
        if commands.ndim == 1:
            commands = commands.reshape(1, 2)
        if commands.ndim != 2 or commands.shape[1] != 2 or not np.isfinite(commands).all():
            raise ValueError("future_commands must be a finite (H,2) planned command sequence")
        state = torch.tensor(q, device=self.device).reshape(1, 1, 4)
        horizon = round(.5 / self.config["dt"])
        for index, command in enumerate(commands[:horizon]):
            state = self.model.step(state, torch.tensor(command, device=self.device).reshape(1, 1, 2), latent)
            if index == 0:
                result["one"] = state[0, 0].cpu().numpy().copy()
            if index + 1 == horizon:
                result["half"] = state[0, 0].cpu().numpy().copy()
        return result


def load_model(checkpoint: Path, device="cpu") -> StreamModel:
    """Load only safe tensor/primitive checkpoint content, never an arbitrary module pickle."""
    torch.set_num_threads(4)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint")
    model = ContextModel(payload["mode"], payload["config"], payload["statistics"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return StreamModel(model.to(device), payload, device)


def stream_predict(model: StreamModel, observed4, current_rgb, previous_command2=None,
                   future_commands=None, **kwargs):
    return model.stream_predict(observed4, current_rgb, previous_command2, future_commands, **kwargs)


@torch.inference_mode()
def _neural_predictions(stream_model, episode, horizon, replacement_rgb=None, reset_history=False):
    model, device = stream_model.model, stream_model.device
    observed, commands = episode["observed"], episode["commands"]
    count = len(commands)
    features = causal_features(observed, commands, model.config)
    images = episode["rgb"][:-1] if replacement_rgb is None else replacement_rgb
    latent_parts, hidden = [], None
    # Chunking limits image activations; the forward-only GRU is causal within chunks.
    for start in range(0, count, 32):
        features_tensor = torch.from_numpy(features[start:start + 32]).to(device).unsqueeze(0)
        rgb = torch.from_numpy(images[start:start + 32].copy()).to(device).unsqueeze(0) if model.rgb_encoder is not None else None
        latent, hidden = model.encode(features_tensor, rgb, hidden, reset_history=reset_history)
        latent_parts.append(latent[0])
    latent = torch.cat(latent_parts, dim=0)
    q = torch.from_numpy(observed[:-1]).to(device)
    u = torch.from_numpy(commands).to(device)
    one = model.step(q, u, latent).cpu().numpy()
    valid = count - horizon + 1
    half = np.full((count, 4), np.nan, dtype=np.float32)
    rollout = q[:valid]
    frozen = latent[:valid]
    for offset in range(horizon):
        rollout = model.step(rollout, u[offset:offset + valid], frozen)
    half[:valid] = rollout.cpu().numpy()
    return one, half, latent.cpu().numpy()


def _fixed_predictions(episode, config, horizon):
    observed, commands = episode["observed"], episode["commands"]
    count = len(commands)
    one = physical_numpy(observed[:-1], commands, config)
    valid = count - horizon + 1
    half = np.full((count, 4), np.nan, dtype=np.float32)
    rollout = observed[:valid].copy()
    for offset in range(horizon):
        rollout = physical_numpy(rollout, commands[offset:offset + valid], config)
    half[:valid] = rollout
    return one, half, np.zeros((count, 8), dtype=np.float32)


def evaluate_all(data_root: Path, output: Path, checkpoint_root=None):
    """Evaluate held-out episodes once, with frozen weights and identical inputs."""
    torch.set_num_threads(4)
    data_root, output = Path(data_root), Path(output)
    manifest, config, horizon = _manifest(data_root)
    checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else output
    checkpoint_dir = checkpoint_root / "checkpoints" if (checkpoint_root / "checkpoints").is_dir() else checkpoint_root
    models = {mode: load_model(checkpoint_dir / f"{mode}.pt") for mode in MODES}
    current_manifest_hash = hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest()
    for model in models.values():
        if model.checkpoint["training_manifest_sha256"] != current_manifest_hash:
            raise ValueError("Manifest differs from the one used for training; regenerate explicit matching evaluation data/checkpoints.")
        if model.config != {"dt": float(config["dt"]), "physics": config["physics"]}:
            raise ValueError("Nominal model/config mismatch between dataset and checkpoint.")
    # The independent image replacement pool is built ONLY from training images.
    # It is fixed before evaluation; no future held-out RGB can enter an ablation.
    pool = []
    for item in manifest["episodes"]:
        if item["split"] == "train":
            with np.load(data_root / item["file"], allow_pickle=False) as source:
                pool.extend(np.asarray(source["rgb"])[::10])
    if not pool:
        raise ValueError("A training-only image pool is required for the image ablation.")
    image_pool = np.asarray(pool, dtype=np.uint8)
    per_episode, sums = [], defaultdict(lambda: {"one_sum_squared": np.zeros(2), "half_sum_squared": np.zeros(2),
                                               "one_count": 0, "half_count": 0, "episodes": 0})
    evaluation_modes = ["fixed", "history", "vision", "fusion", "fusion_reset_history", "fusion_shuffle_image"]
    for episode_index, item in enumerate(manifest["episodes"]):
        if item["split"] not in ("test", "ood"):
            continue
        episode = _read_episode(data_root, item, need_truth=True)
        count, valid = len(episode["commands"]), len(episode["commands"]) - horizon + 1
        if valid < 1:
            raise ValueError("Evaluation episode shorter than forecast horizon.")
        # Identical predetermined replacement sequence for a fixed episode ID.
        shuffle_seed = int.from_bytes(hashlib.sha256(str(item["id"]).encode()).digest()[:4], "little")
        replacement = image_pool[np.random.default_rng(shuffle_seed).integers(0, len(image_pool), size=count)]
        score_start = min(BURN_IN, valid - 1)
        for mode in evaluation_modes:
            if mode == "fixed":
                one, half, latent = _fixed_predictions(episode, config, horizon)
            else:
                base_mode = "fusion" if mode.startswith("fusion_") else mode
                one, half, latent = _neural_predictions(models[base_mode], episode, horizon,
                    replacement_rgb=replacement if mode == "fusion_shuffle_image" else None,
                    reset_history=mode == "fusion_reset_history")
            if not np.isfinite(one).all() or not np.isfinite(half[:valid]).all():
                raise FloatingPointError(f"Nonfinite prediction: {item['id']} {mode}")
            one_error = one[:, :2] - episode["truth"][1:, :2]
            half_error = np.full((count, 2), np.nan)
            half_error[:valid] = half[:valid, :2] - episode["truth"][horizon:, :2]
            scored_one, scored_half = one_error[score_start:], half_error[score_start:valid]
            one_squared, half_squared = np.sum(scored_one**2, axis=0), np.sum(scored_half**2, axis=0)
            one_rmse = np.sqrt(one_squared / len(scored_one))
            half_rmse = np.sqrt(half_squared / len(scored_half))
            row = {"id": str(item["id"]), "split": item["split"], "condition": item.get("condition", "unspecified"),
                   "model": mode, "one_speed_rmse": float(one_rmse[0]), "one_yaw_rmse": float(one_rmse[1]),
                   "half_speed_rmse": float(half_rmse[0]), "half_yaw_rmse": float(half_rmse[1]),
                   "one_origins": len(scored_one), "half_origins": len(scored_half),
                   "score_start_origin_seconds": float(episode["times"][score_start]),
                   "missing_half_targets": horizon - 1}
            event = item.get("event_time")
            if event is not None:
                origin_times = episode["times"][:-1]
                mask = (origin_times[:valid] >= max(float(event), float(episode["times"][score_start])))
                if mask.any():
                    post = np.sqrt(np.mean(half_error[:valid][mask]**2, axis=0))
                    row["half_speed_rmse_post"] = float(post[0])
                    row["half_yaw_rmse_post"] = float(post[1])
            per_episode.append(row)
            total = sums[(item["split"], mode)]
            total["one_sum_squared"] += one_squared
            total["half_sum_squared"] += half_squared
            total["one_count"] += len(scored_one)
            total["half_count"] += len(scored_half)
            total["episodes"] += 1
            # Preserve origins and unscored early predictions to allow inspection.
            prediction_dir = output / "predictions" / str(item["id"])
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_dir / f"{mode}.npz", times=episode["times"][:-1],
                                one=one, half=half, one_error=one_error, half_error=half_error,
                                latent=latent, observed=episode["observed"][:-1],
                                commands=episode["commands"], truth=episode["truth"])
        print(f"Evaluated held-out episode {item['id']} ({item['split']})", flush=True)
    aggregate = {}
    for (split, mode), total in sums.items():
        one = np.sqrt(total["one_sum_squared"] / total["one_count"])
        half = np.sqrt(total["half_sum_squared"] / total["half_count"])
        aggregate.setdefault(split, {})[mode] = {"one_speed_rmse": float(one[0]), "one_yaw_rmse": float(one[1]),
            "half_speed_rmse": float(half[0]), "half_yaw_rmse": float(half[1]),
            "one_origins": total["one_count"], "half_origins": total["half_count"], "episodes": total["episodes"]}
    if not per_episode:
        raise ValueError("No held-out test/ood episodes found.")
    report = {"aggregate": aggregate, "per_episode": per_episode,
              "protocol": {"dt": float(config["dt"]), "forecast_seconds": [float(config["dt"]), .5], "half_horizon_steps": horizon,
                           "score_burn_in_steps": BURN_IN, "weights_frozen_during_evaluation": True,
                           "online_updates": "GRU hidden state only, using current observation and past command/residual",
                           "future_inputs": "preplanned nominal commands, known at forecast origin",
                           "rollout": "freeze eight-dimensional latent at origin; do not read future RGB/observations",
                           "normalization": "fitted only on train episodes",
                           "model_selection": "validation episodes only",
                           "targets": "noisy observed for training; simulator truth only for held-out scoring",
                           "metrics": "pooled body v RMSE in m/s and yaw-rate RMSE in rad/s, reported separately",
                           "reset_history": "same fusion weights; hidden reset each origin; retain current causal q/u_previous/residual",
                           "shuffle_image": "same fusion weights; unrelated image drawn from fixed TRAIN-only RGB pool, no future held-out image",
                           "latent_semantics": "learned predictive context; no calibrated physical/friction meaning",
                           "statistics_warning": "overlapping horizons are correlated; episode is the repetition unit"},
              "manifest_sha256": current_manifest_hash,
              "checkpoints": {mode: {"best_epoch": model.checkpoint["best_epoch"],
                                      "sha256": hashlib.sha256((checkpoint_dir / f"{mode}.pt").read_bytes()).hexdigest()}
                              for mode, model in models.items()}}
    _json(output / "metrics.json", report)
    return report
