"""Run a dependency-free, leakage-safe target-presence baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research_pipeline import (
    classification_metrics,
    extract_features,
    fit_centroid_classifier,
    load_array,
    load_labels,
    majority_prediction,
    predict_centroid,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic binary target-presence baseline.")
    parser.add_argument("--data-dir", default="data_lean")
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    processed_dir = Path(args.processed_dir) if args.processed_dir else data_dir / "processed" / "background_removed_normalized"
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    rows = load_labels(data_dir)
    features, labels, splits, run_ids = [], [], [], []
    for row in rows:
        run_id = int(row["run_id"])
        features.append(extract_features(load_array(data_dir, run_id, processed_dir)))
        labels.append(int(row["target_type"] != "none"))
        splits.append(row["split"])
        run_ids.append(run_id)
    x = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    splits_array = np.array(splits)
    train, val, test = (splits_array == name for name in ("train", "val", "test"))
    model = fit_centroid_classifier(x[train], y[train])
    results = {
        "task": "binary target presence: target_type != none",
        "features": "32 time-bin mean absolute amplitudes, 1 max time-bin amplitude, 16 trace-bin mean absolute amplitudes, and 4 summary features",
        "preprocessing": "per-B-scan trace-mean background removal and per-B-scan 99.5th-percentile absolute normalization",
        "split_counts": {name: int(mask.sum()) for name, mask in (("train", train), ("val", val), ("test", test))},
        "class_counts": {str(value): int((y == value).sum()) for value in (0, 1)},
        "validation": {},
        "test": {},
        "majority_test": {},
    }
    for name, mask in (("validation", val), ("test", test)):
        predictions = predict_centroid(model, x[mask])
        results[name] = classification_metrics(y[mask], predictions)
    results["majority_test"] = classification_metrics(y[test], majority_prediction(y[train], int(test.sum())))
    results["test_run_ids"] = [run_ids[index] for index in np.flatnonzero(test)]
    output_dir = data_dir.parent / "results" / "baseline"
    write_json(output_dir / "target_presence_metrics.json", results)
    figure_dir = data_dir.parent / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array(results["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["none", "present"], yticklabels=["none", "present"], xlabel="Predicted", ylabel="Actual", title="Target-presence baseline: test confusion matrix")
    fig.colorbar(image, ax=ax, label="Runs")
    fig.savefig(figure_dir / "baseline_target_presence_confusion.png", dpi=180)
    plt.close(fig)
    print(f"Validation metrics: {results['validation']}")
    print(f"Test metrics: {results['test']}")
    print(f"Majority test baseline: {results['majority_test']}")
    print(f"Saved metrics to {output_dir / 'target_presence_metrics.json'}")


if __name__ == "__main__":
    main()
