"""Shared data loading, preprocessing, features, and evaluation helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def load_labels(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No labels found in {data_dir / 'labels.csv'}")
    return rows


def load_array(data_dir: Path, run_id: int, processed_dir: Path | None = None) -> np.ndarray:
    directory = processed_dir if processed_dir is not None else data_dir / "bscans"
    path = directory / f"run_{run_id:04d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing array: {path}")
    array = np.load(path)
    if array.ndim != 2 or array.dtype != np.float32:
        raise ValueError(f"Unexpected array at {path}: {array.shape}, {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {path}")
    return array


def background_remove(array: np.ndarray) -> np.ndarray:
    """Remove each B-scan's trace-mean background without fitting across runs."""
    return (array - array.mean(axis=1, keepdims=True)).astype(np.float32)


def robust_normalize(array: np.ndarray) -> np.ndarray:
    """Scale one B-scan by its 99.5th absolute-amplitude percentile."""
    scale = float(np.percentile(np.abs(array), 99.5))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        raise ValueError("Cannot normalize a zero or non-finite B-scan")
    return (array / scale).astype(np.float32)


def preprocess_array(array: np.ndarray) -> np.ndarray:
    return robust_normalize(background_remove(array))


def _pool(values: np.ndarray, bins: int) -> np.ndarray:
    if values.shape[0] < bins:
        positions = np.linspace(0, values.shape[0] - 1, bins)
        return np.interp(positions, np.arange(values.shape[0]), values).astype(np.float64)
    edges = np.linspace(0, values.shape[0], bins + 1, dtype=int)
    return np.array(
        [values[edges[i]:edges[i + 1]].mean() for i in range(bins)],
        dtype=np.float64,
    )


def extract_features(array: np.ndarray) -> np.ndarray:
    """Extract fixed-size features from a processed B-scan."""
    magnitude = np.abs(array).astype(np.float64)
    time_profile = magnitude.mean(axis=1)
    trace_profile = magnitude.mean(axis=0)
    features = np.concatenate(
        (
            _pool(time_profile, 32),
            _pool(time_profile.max(axis=0, keepdims=True), 1),
            _pool(trace_profile, 16),
            np.array(
                [
                    float(magnitude.mean()),
                    float(magnitude.std()),
                    float(np.percentile(magnitude, 99.5)),
                    float(np.argmax(time_profile) / max(1, array.shape[0] - 1)),
                ],
                dtype=np.float64,
            ),
        )
    )
    return features


def fit_centroid_classifier(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    classes = np.array([0, 1], dtype=np.int64)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    centroids = np.vstack([standardized[labels == value].mean(axis=0) for value in classes])
    return {"feature_mean": means, "feature_scale": scales, "centroids": centroids, "classes": classes}


def predict_centroid(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["feature_mean"]) / model["feature_scale"]
    distances = ((standardized[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2)
    return model["classes"][np.argmin(distances, axis=1)]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    classes = [0, 1]
    matrix = [[int(((y_true == actual) & (y_pred == predicted)).sum()) for predicted in classes] for actual in classes]
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for value in classes:
        tp = matrix[value][value]
        fp = matrix[1 - value][value]
        fn = matrix[value][1 - value]
        support = int((y_true == value).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(value)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    return {
        "confusion_matrix": matrix,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def majority_prediction(y_train: np.ndarray, count: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    return np.full(count, values[np.argmax(counts)], dtype=np.int64)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
