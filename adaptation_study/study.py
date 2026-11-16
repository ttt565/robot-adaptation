"""Shared pieces of the adaptation study: locations, dataset access and episode-level statistics.

Kept free of MuJoCo, PyTorch and Matplotlib so that every step can import it.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEARNED = REPO / "mujoco_learned_ready"
HELD_OUT = ("test", "ood")
HORIZON_SECONDS = 0.5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dataset(directory):
    """Return (manifest, arrays); every array is stacked over episodes in manifest order."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    with np.load(directory / "dataset.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    count = len(manifest["episodes"])
    if any(len(arrays[key]) != count for key in ("truth", "observed", "commands")):
        raise ValueError("dataset.npz and manifest.json describe different numbers of episodes")
    return manifest, arrays


def horizon_steps(manifest):
    return int(round(HORIZON_SECONDS / float(manifest["config"]["dt"])))


def geometric_mean(values):
    return float(np.exp(np.mean(np.log(values))))


def bootstrap_interval(values, statistic, draws=10000, seed=0, level=0.95):
    """Percentile interval for ``statistic`` when whole episodes are resampled with replacement."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    samples = np.array([statistic(values[row]) for row in index])
    tail = (1 - level) / 2
    return [float(np.quantile(samples, tail)), float(np.quantile(samples, 1 - tail))]
