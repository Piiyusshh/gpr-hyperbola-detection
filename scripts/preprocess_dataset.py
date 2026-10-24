"""Create reproducible processed copies of the completed lean B-scans."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research_pipeline import load_array, load_labels, preprocess_array, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess saved B-scans without changing raw data.")
    parser.add_argument("--data-dir", default="data_lean")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed" / "background_removed_normalized"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    rows = load_labels(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[tuple[np.ndarray, np.ndarray, str]] = []
    for row in rows:
        run_id = int(row["run_id"])
        raw = load_array(data_dir, run_id)
        processed = preprocess_array(raw)
        np.save(output_dir / f"run_{run_id:04d}.npy", processed)
        if len(examples) < 3:
            examples.append((raw, processed, f"run {run_id} ({row['target_type']})"))
    write_json(
        output_dir / "manifest.json",
        {
            "method": "per_bscan_trace_mean_background_removal_then_99.5_percentile_absolute_normalization",
            "source": str(data_dir / "bscans"),
            "run_ids": sorted(int(row["run_id"]) for row in rows),
            "raw_data_preserved": True,
        },
    )
    figure_dir = data_dir.parent / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3 * len(examples)), squeeze=False, constrained_layout=True)
    for row_index, (raw, processed, title) in enumerate(examples):
        for column, (array, label) in enumerate(((raw, "Raw Ez"), (processed, "Processed Ez"))):
            image = axes[row_index, column].imshow(array, aspect="auto", cmap="seismic")
            axes[row_index, column].set(title=f"{title}: {label}", xlabel="Trace index", ylabel="Sample index")
            fig.colorbar(image, ax=axes[row_index, column])
    fig.savefig(figure_dir / "preprocessing_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Processed {len(rows)} B-scans into {output_dir}")
    print(f"Saved comparison figure to {figure_dir / 'preprocessing_comparison.png'}")


if __name__ == "__main__":
    main()
