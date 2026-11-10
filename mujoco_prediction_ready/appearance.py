"""Procedural appearance labels and a small RGB-only texture classifier.

Labels describe image patterns, not materials, friction coefficients, or robot
dynamics.  match_score is a distance-derived similarity, not a probability.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


KINDS = ("tile_grid", "speckled", "parallel_stripes")


def make_texture(kind, seed, size=256):
    """Create an RGB pattern with randomized brightness, tint, scale and pose."""
    if kind not in KINDS:
        raise ValueError(f"Unknown appearance {kind!r}; choose one of {KINDS}.")
    size = int(size)
    if size < 32:
        raise ValueError("Texture size must be at least 32 pixels.")
    random = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(float)
    angle = random.uniform(-np.pi, np.pi)
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    period = size * random.uniform(.075, .19)
    phase = random.uniform(0, period, size=2)
    if kind == "tile_grid":
        # A two-axis mortar grid.  Cell brightness is deliberately not a label.
        distance_u = np.abs((u + phase[0] + period / 2) % period - period / 2)
        distance_v = np.abs((v + phase[1] + period / 2) % period - period / 2)
        width = period * random.uniform(.035, .085)
        pattern = 1 - 2 * np.maximum(np.exp(-(distance_u / width)**2), np.exp(-(distance_v / width)**2))
    elif kind == "parallel_stripes":
        width = random.uniform(.7, 1.4)
        wave = np.cos(2 * np.pi * (u + phase[0]) / period)
        pattern = np.tanh(wave * width * 2)
    else:
        # Smooth random grains remain isotropic; they are not simulated gravel.
        coarse_size = int(random.integers(16, 45))
        coarse = random.integers(0, 256, (coarse_size, coarse_size), dtype=np.uint8)
        grains = Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)
        grains = grains.filter(ImageFilter.GaussianBlur(random.uniform(.25, 1.0) * size / 256))
        pattern = np.asarray(grains, dtype=float) / 127.5 - 1
        pattern = np.tanh(pattern * random.uniform(1.6, 2.7))
    pattern -= np.mean(pattern)
    pattern /= max(np.std(pattern), .05)
    brightness = random.uniform(105, 165)
    contrast = random.uniform(24, 44)
    grain = random.normal(0, random.uniform(.5, 2.5), (size, size))
    grayscale = brightness + contrast * pattern + grain
    tint = random.uniform(.91, 1.09, 3)
    rgb = grayscale[:, :, None] * tint[None, None, :]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _gray(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3 or min(array.shape[:2]) < 24:
        raise ValueError("Expected a height x width x RGB image of at least 24 pixels per side.")
    array = array[:, :, :3].astype(float)
    if not np.isfinite(array).all():
        raise ValueError("RGB image contains non-finite values.")
    if array.max() <= 1.0 and array.min() >= 0:
        array *= 255.0
    return .299 * array[:, :, 0] + .587 * array[:, :, 1] + .114 * array[:, :, 2]


def _region_features(gray):
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize((128, 128), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=float)
    low, high = np.percentile(values, [5, 95])
    values = np.clip((values - np.median(values)) / max(high - low, 3.0), -1.5, 1.5)
    # Edge directions are used only through rotation-invariant summaries.
    gx = (values[1:-1, 2:] - values[1:-1, :-2]) / 2
    gy = (values[2:, 1:-1] - values[:-2, 1:-1]) / 2
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    weight = magnitude.ravel()
    total = max(float(weight.sum()), 1e-10)
    features = [abs(np.sum(weight * np.exp(1j * order * direction.ravel()))) / total for order in (2, 4, 6, 8)]
    histogram, _ = np.histogram(direction, bins=24, range=(-np.pi, np.pi), weights=magnitude)
    histogram = histogram / max(histogram.sum(), 1e-10)
    features.extend([-(histogram * np.log(histogram + 1e-12)).sum() / np.log(24), np.sort(histogram)[-3:].sum()])
    normalized_mag = magnitude / max(float(magnitude.mean()), 1e-8)
    features.extend(np.percentile(normalized_mag, [25, 50, 75, 90, 98]) / 5)
    features.extend([np.mean(normalized_mag > 2), np.mean(normalized_mag < .25)])
    centered = values - values.mean()
    apodization = np.outer(np.hanning(128), np.hanning(128))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * apodization)))**2
    y, x = np.mgrid[-64:64, -64:64]
    radius, orientation = np.hypot(x, y), np.arctan2(y, x)
    active = (radius >= 3) & (radius <= 52)
    energy = max(float(spectrum[active].sum()), 1e-12)
    for lower, upper in ((3, 14), (14, 52)):
        mask = (radius >= lower) & (radius < upper)
        power = spectrum[mask]
        power = power / max(float(power.sum()), 1e-12)
        features.extend([abs(np.sum(power * np.exp(1j * order * orientation[mask]))) for order in (2, 4)])
    for lower, upper in zip((3, 5, 8, 13, 21, 34), (5, 8, 13, 21, 34, 52)):
        features.append(float(spectrum[(radius >= lower) & (radius < upper)].sum()) / energy)
    probabilities = spectrum[active] / energy
    features.append(float(-(probabilities * np.log(probabilities + 1e-12)).sum() / np.log(len(probabilities))))
    return np.asarray(features, dtype=float)


def _features(rgb):
    gray = _gray(rgb)
    h, w = gray.shape
    # Two overlapping floor ROIs reduce sensitivity to a single horizon/shadow.
    regions = ((.22, .94, .06, .94), (.46, .97, .12, .88))
    parts = [_region_features(gray[int(top*h):int(bottom*h), int(left*w):int(right*w)])
             for top, bottom, left, right in regions]
    return np.concatenate(parts)


class AppearanceClassifier:
    """Standardized nearest-neighbor texture appearance classification."""

    def fit(self, images, labels):
        if len(images) != len(labels) or len(images) < 6:
            raise ValueError("Provide matching images and labels, with at least two per class.")
        if set(labels) != set(KINDS) or any(labels.count(kind) < 2 for kind in KINDS):
            raise ValueError(f"Training requires at least two examples of each class: {KINDS}.")
        features = np.vstack([_features(image) for image in images])
        self.mean = features.mean(axis=0)
        # A floor avoids turning near-constant training dimensions into large weights.
        self.scale = np.maximum(features.std(axis=0), .06)
        self.train = (features - self.mean) / self.scale
        self.labels = np.asarray([KINDS.index(label) for label in labels], dtype=np.int64)
        self.thresholds = np.empty(len(KINDS), dtype=float)
        for class_index in range(len(KINDS)):
            members = self.train[self.labels == class_index]
            distances = np.sqrt(np.mean((members[:, None] - members[None, :])**2, axis=2))
            np.fill_diagonal(distances, np.inf)
            k = min(3, len(members) - 1)
            leave_one_out = np.mean(np.sort(distances, axis=1)[:, :k], axis=1)
            q25, q75, q98 = np.percentile(leave_one_out, [25, 75, 98])
            # Both margins scale with the training distribution; no test-label
            # threshold selection and no fixed class-dependent rejection cutoffs.
            self.thresholds[class_index] = max(float(q98 + 2 * (q75 - q25)), float(1.5 * q98), 1e-8)
        return self

    def predict(self, rgb):
        if not hasattr(self, "train"):
            raise RuntimeError("Fit or load the classifier before prediction.")
        standardized = (_features(rgb) - self.mean) / self.scale
        distances = np.sqrt(np.mean((self.train - standardized)**2, axis=1))
        per_class = np.asarray([np.sort(distances[self.labels == i])[:min(3, np.sum(self.labels == i))].mean()
                                for i in range(len(KINDS))])
        best = int(np.argmin(per_class))
        # A common scale preserves the nearest-class ordering in the score map.
        similarities = np.exp(-per_class / max(float(np.median(self.thresholds)), 1e-8))
        # Absolute match rejection is calibrated only from training distances.
        label = KINDS[best] if per_class[best] <= self.thresholds[best] else "unknown"
        return {"label": label, "match_score": float(similarities[best]),
                "scores": {kind: float(value) for kind, value in zip(KINDS, similarities)}}

    def save(self, path):
        if not hasattr(self, "train"):
            raise RuntimeError("Fit the classifier before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, version=np.array([1]), kinds=np.array(KINDS),
                                mean=self.mean, scale=self.scale, train=self.train,
                                labels=self.labels, thresholds=self.thresholds)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            if tuple(data["kinds"].tolist()) != KINDS or data["version"].tolist() != [1]:
                raise ValueError("Unsupported appearance classifier file.")
            result = cls()
            for key in ("mean", "scale", "train", "labels", "thresholds"):
                setattr(result, key, data[key].copy())
        return result
