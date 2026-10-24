"""Verify saved B-scans and plot a labelled, approximate hyperbola overlay.

This script never launches gprMax. It distinguishes missing/incomplete data
from a successful dataset check so an empty output directory cannot pass.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import all_plans, completed, load_config, metadata_matches_plan

C = 299_792_458.0
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"


def status(name: str, state: str, detail: str) -> dict[str, str]:
    print(f"{state}: {name} — {detail}")
    return {"check": name, "status": state, "detail": detail}


def curve(times_zero: float, x: np.ndarray, x0: float, centre_depth: float, radius: float, eps_r: float) -> np.ndarray:
    """Return recorded-time hyperbola in seconds using SI units throughout."""
    velocity = C / np.sqrt(eps_r)
    return times_zero + 2 * (np.hypot(x - x0, centre_depth) - radius) / velocity


def step1_overlay(save: bool = True) -> dict[str, str]:
    model = ROOT / "configs" / "first_pipe_bscan_2d.in"
    bscan_file = ROOT / "results" / "first_pipe_bscan_merged.h5"
    summary_file = ROOT / "results" / "step1_summary.json"
    if not (model.exists() and bscan_file.exists() and summary_file.exists()):
        return status("Step 1 overlay", "NOT RUN", "Required Step 1 input, merged HDF5, or summary is missing")
    with h5py.File(bscan_file, "r") as h5:
        data = h5["/rxs/rx1/Ez"][...]
        dt = float(h5.attrs["dt"])
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if data.shape != (425, 63) or data.dtype != np.float32:
        return status("Step 1 overlay", "FAIL", f"Unexpected Step 1 array {data.shape}, {data.dtype}")
    # Values below are checked against the actual .in file before plotting.
    text = model.read_text(encoding="utf-8")
    required = ("#material: 6 0.01 1 0 soil", "#hertzian_dipole: z 0.052 0.260", "#src_steps: 0.008", "#cylinder: 0.300 0.140 0 0.300 0.140 inf 0.032 pec")
    if not all(item in text for item in required):
        return status("Step 1 overlay", "FAIL", "Input file differs from parameters required for the checked overlay")
    x = 0.052 + np.arange(data.shape[1]) * 0.008
    t = np.arange(data.shape[0]) * dt
    predicted = curve(float(summary["time_zero_reference_s"]), x, 0.300, 0.120, 0.032, 6.0)
    picked = float(summary["picked_echo_recorded_time_s"])
    apex_error = float(predicted[np.argmin(abs(x - 0.300))] - picked)
    FIGURES.mkdir(parents=True, exist_ok=True)
    vmax = float(np.percentile(abs(data), 99))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = ax.imshow(data, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, extent=[x[0], x[-1], t[-1] * 1e9, 0])
    # Do not let a label-only curve expand the plot beyond recorded samples.
    visible = (predicted >= 0) & (predicted <= t[-1])
    ax.plot(x[visible], predicted[visible] * 1e9, color="lime", lw=1.8, label="label-derived geometric reference")
    ax.set(xlabel="Scan position (m)", ylabel="Recorded time (ns)", title="Step 1: data with theoretical pipe overlay")
    ax.legend(loc="lower right")
    fig.colorbar(image, ax=ax, label="Ez (V/m)")
    if save:
        fig.savefig(FIGURES / "step1_hyperbola_overlay.png", dpi=200)
    plt.close(fig)
    criterion = dt
    state = "PASS" if abs(apex_error) <= criterion else "FAIL"
    detail = f"Predicted apex minus saved picked apex = {apex_error:.12e} s; criterion is one sample ({criterion:.12e} s); full arms remain separately unassessed"
    return status("Step 1 overlay", state, detail)


def background_removed_peak(array: np.ndarray, predicted_time_s: float, dt_s: float, trace_index: int) -> tuple[float, float]:
    """Diagnostic peak: subtract trace mean, then search ±0.5 ns and ±2 traces."""
    residual = array - array.mean(axis=1, keepdims=True)
    centre = round(predicted_time_s / dt_s)
    half_time = max(1, round(0.5e-9 / dt_s))
    left, right = max(0, centre - half_time), min(array.shape[0], centre + half_time + 1)
    x0, x1 = max(0, trace_index - 2), min(array.shape[1], trace_index + 3)
    local = abs(residual[left:right, x0:x1])
    offset = np.unravel_index(np.argmax(local), local.shape)
    return (left + offset[0]) * dt_s, float(residual[left + offset[0], x0 + offset[1]])


def geometry_matches_metadata_and_label(plan, info: dict, row: dict[str, str]) -> bool:
    """Check the documented JSON/CSV representation of target geometry."""
    fields = ("top_depth_m", "centre_depth_m", "radius_m", "target_x_m")
    if not plan.target_present:
        return all(info.get(field) is None and row.get(field) == "" for field in fields)
    for field in fields:
        try:
            if not (isinstance(info.get(field), (int, float)) and not isinstance(info[field], bool) and float(row[field]) == float(info[field])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def verify_dataset(data_dir: Path, cfg: dict) -> list[dict[str, str]]:
    labels = data_dir / "labels.csv"
    bscans = data_dir / "bscans"
    metadata = data_dir / "metadata"
    plans = all_plans(cfg)
    complete_ids = [plan.run_id for plan in plans if completed(plan, cfg)]
    total = int(cfg["total_runs"])
    print(f"Dataset status: {'COMPLETE' if len(complete_ids) == total else 'INCOMPLETE'} — {len(complete_ids)} of {total} completed")
    if not labels.exists():
        status("Dataset overlays", "NOT RUN", "Fewer than three completed runs")
        status("Direct-event consistency", "NOT RUN", "Fewer than three completed runs")
        return [status("Dataset integrity", "INSUFFICIENT DATA", f"No labels.csv in {data_dir}; {len(complete_ids)} completed runs")]
    with labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [status("Dataset integrity", "INSUFFICIENT DATA", "labels.csv has no data rows")]
    seen, failures = set(), []
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id in seen:
            failures.append(f"duplicate run_id {run_id}")
            continue
        seen.add(run_id)
        npy, meta = bscans / f"run_{int(run_id):04d}.npy", metadata / f"run_{int(run_id):04d}.json"
        if not npy.exists() or not meta.exists():
            failures.append(f"run {run_id} missing npy or metadata")
            continue
        array = np.load(npy, mmap_mode="r")
        info = json.loads(meta.read_text(encoding="utf-8"))
        if array.dtype != np.float32 or tuple(array.shape) != tuple(info.get("shape", [])) or array.ndim != 2:
            failures.append(f"run {run_id} has invalid shape/dtype")
        elif not np.isfinite(array).all() or np.ptp(array) == 0:
            failures.append(f"run {run_id} contains invalid or constant values")
        plan = plans[int(run_id)] if run_id.isdigit() and int(run_id) < len(plans) else None
        if row.get("target_type") not in {"metal_pipe", "air_void", "none"}:
            failures.append(f"run {run_id} has invalid target_type")
        elif plan is None or not metadata_matches_plan(info, plan):
            failures.append(f"run {run_id} metadata does not match its deterministic plan")
        elif row.get("split") != plan.split:
            failures.append(f"run {run_id} has an invalid deterministic split")
        elif not (cfg["soil_eps_r"][0] <= float(row["eps_r"]) <= cfg["soil_eps_r"][1]):
            failures.append(f"run {run_id} eps_r outside configured range")
        for field in ("run_id", "seed", "target_present", "target_type", "eps_r", "sigma", "n_clutter", "split"):
            if field not in info or str(info[field]) != str(row[field]):
                failures.append(f"run {run_id} metadata/label mismatch for {field}")
        if plan is not None and not geometry_matches_metadata_and_label(plan, info, row):
            if plan.target_present:
                failures.append(f"run {run_id} target geometry metadata/label mismatch")
            else:
                failures.append(f"run {run_id} no-target geometry must be metadata null and CSV empty")
    if set(int(row["run_id"]) for row in rows) != set(complete_ids):
        failures.append("labels do not correspond exactly to completed runs")
    if failures:
        return [status("Dataset integrity", "FAIL", "; ".join(failures))]
    if len(complete_ids) >= 3:
        make_dataset_figures(rows, bscans, metadata, cfg)
        dataset_overlays(rows, bscans, metadata, cfg)
        direct_event_consistency(rows, bscans, metadata)
    else:
        status("Dataset overlays", "NOT RUN", "Fewer than three completed runs")
        status("Direct-event consistency", "NOT RUN", "Fewer than three completed runs")
    state = "PASS" if len(complete_ids) == total else "INSUFFICIENT DATA"
    detail = f"Verified {len(rows)} labelled B-scans"
    if len(complete_ids) >= 3:
        detail += "; saved actual-data figures"
    return [status("Dataset integrity", state, detail)]


def make_dataset_figures(rows: list[dict[str, str]], bscans: Path, metadata: Path, cfg: dict) -> None:
    """Plot only existing arrays and labels; no planned samples fill gaps."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    chosen = rows[:12]
    arrays = [np.load(bscans / f"run_{int(row['run_id']):04d}.npy") for row in chosen]
    # One scale within this figure makes colour comparable across displayed runs.
    scale = max(float(np.percentile(abs(array), 99)) for array in arrays)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for ax, row, array in zip(axes.flat, chosen, arrays):
        run_id = int(row["run_id"])
        dt = float(json.loads((metadata / f"run_{run_id:04d}.json").read_text(encoding="utf-8"))["dt_s"])
        x = cfg["scan_start_m"] + np.arange(array.shape[1]) * cfg["scan_step_m"]
        image = ax.imshow(array, aspect="auto", cmap="seismic", vmin=-scale, vmax=scale,
                          extent=[x[0], x[-1], array.shape[0] * dt * 1e9, 0])
        ax.set_title(f"{row['run_id']}: {row['target_type']}\nz={row['top_depth_m']}, r={row['radius_m']}", fontsize=8)
        ax.set(xlabel="Scan position (m)", ylabel="Recorded time (ns)")
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes.ravel()[:len(chosen)], label="Ez (V/m)")
    fig.savefig(FIGURES / "dataset_samples.png", dpi=200)
    plt.close(fig)
    present = [row for row in rows if row["target_present"].lower() == "true"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
    axes[0].hist([float(row["top_depth_m"]) for row in present]); axes[0].set_title("Target top depth (m)")
    axes[1].hist([float(row["radius_m"]) for row in present]); axes[1].set_title("Target radius (m)")
    axes[2].hist([float(row["eps_r"]) for row in rows]); axes[2].set_title("Soil relative permittivity")
    classes = [row["target_type"] for row in rows]
    axes[3].bar(sorted(set(classes)), [classes.count(name) for name in sorted(set(classes))]); axes[3].set_title("Target class balance")
    fig.savefig(FIGURES / "dataset_distributions.png", dpi=200)
    plt.close(fig)


def dataset_overlays(rows: list[dict[str, str]], bscans: Path, metadata: Path, cfg: dict) -> None:
    """Plot up to eight diagnostic, label-only curves without fitting data."""
    targets = [row for row in rows if row["target_present"].lower() == "true"][:8]
    if not targets:
        status("Dataset overlays", "INSUFFICIENT DATA", "No target-present completed runs")
        return
    arrays = [np.load(bscans / f"run_{int(row['run_id']):04d}.npy") for row in targets]
    # Overlay panels use one scale so target examples can be compared visually.
    scale = max(float(np.percentile(abs(array), 99)) for array in arrays)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    pulse = float(cfg.get("pulse_allowance_s", 3e-9))
    for ax, row, array in zip(axes.flat, targets, arrays):
        run_id = int(row["run_id"])
        info = json.loads((metadata / f"run_{run_id:04d}.json").read_text(encoding="utf-8")); dt = float(info["dt_s"])
        x = cfg["scan_start_m"] + np.arange(array.shape[1]) * cfg["scan_step_m"]
        predicted = curve(float(cfg["ricker_time_zero_s"]), x, float(row["target_x_m"]), float(row["centre_depth_m"]), float(row["radius_m"]), float(row["eps_r"]))
        apex = int(np.argmin(abs(x - float(row["target_x_m"])))); peak_time, _ = background_removed_peak(array, float(predicted[apex]), dt, apex)
        image = ax.imshow(array, aspect="auto", cmap="seismic", vmin=-scale, vmax=scale, extent=[x[0], x[-1], array.shape[0] * dt * 1e9, 0])
        ax.plot(x, predicted * 1e9, color="lime", lw=1.2, label="label-derived geometric reference")
        inside = predicted[apex] + pulse <= array.shape[0] * dt
        ax.set(xlabel="Scan position (m)", ylabel="Recorded time (ns)", title=f"{run_id}: local peak gap={(peak_time-predicted[apex])*1e9:+.2f} ns; apex pulse in record={inside}")
    for ax in axes.flat[len(targets):]: ax.set_visible(False)
    fig.colorbar(image, ax=axes.ravel()[:len(targets)], label="Ez (V/m)")
    FIGURES.mkdir(parents=True, exist_ok=True); fig.savefig(FIGURES / "dataset_overlays.png", dpi=200); plt.close(fig)
    status("Dataset overlays", "PASS", "Saved diagnostic curves; local background-removed maxima are not asserted target reflections")


def direct_event_consistency(rows: list[dict[str, str]], bscans: Path, metadata: Path) -> None:
    no_targets = [row for row in rows if row["target_present"].lower() == "false"]
    if len(no_targets) < 3:
        status("Direct-event consistency", "INSUFFICIENT DATA", "Fewer than three completed no-target runs")
        return
    times = []
    for row in no_targets:
        run_id = int(row["run_id"]); array = np.load(bscans / f"run_{run_id:04d}.npy")
        dt = float(json.loads((metadata / f"run_{run_id:04d}.json").read_text(encoding="utf-8"))["dt_s"])
        stop = min(array.shape[0], round(3e-9 / dt)); times.append(np.argmax(abs(array[:stop, array.shape[1] // 2])) * dt)
    spread, tolerance = max(times) - min(times), 0.1e-9
    status("Direct-event consistency", "PASS" if spread <= tolerance else "FAIL", f"times_s={times}; spread={spread:.12e}; tolerance={tolerance:.12e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify existing B-scan files without running simulations.")
    parser.add_argument("--data-dir", default="data", help="Dataset directory relative to repository root.")
    parser.add_argument("--config", default=None, help="Configuration used to create the selected dataset. Defaults are inferred for data/ and data_lean/.")
    parser.add_argument("--step1-only", action="store_true", help="Only check and plot the real Step 1 B-scan.")
    args = parser.parse_args()
    step1_overlay(save=False)
    if not args.step1_only:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        if args.config is None:
            inferred = "configs/dataset_lean.yaml" if data_dir.name == "data_lean" else "configs/dataset.yaml"
        else:
            inferred = args.config
        cfg = load_config(inferred)
        verify_dataset(data_dir, cfg)


if __name__ == "__main__":
    main()
