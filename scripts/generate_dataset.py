"""Generate the Step 2 labelled gprMax dataset, safely and resumably.

The file named dataset.yaml deliberately uses JSON syntax: JSON is valid YAML
and Python can read it without adding an untested dependency. Each run's
parameters are derived solely from ``master_seed + run_id``. That lets a
stopped job resume without changing earlier labels or random draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dataset.yaml"
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
BSCANS = DATA / "bscans"
METADATA = DATA / "metadata"
TEMP = DATA / "temporary_h5"
LABELS = DATA / "labels.csv"
C = 299_792_458.0
DEFAULT_PULSE_ALLOWANCE_S = 3e-9
DEFAULT_TIMING_MARGIN_S = 1.5e-9


@dataclass(frozen=True)
class Plan:
    run_id: int
    seed: int
    target_present: bool
    target_type: str
    top_depth_m: float | None
    centre_depth_m: float | None
    radius_m: float | None
    target_x_m: float | None
    eps_r: float
    sigma: float
    n_clutter: int
    clutter: tuple[tuple[float, float, float, float], ...]
    split: str


def configure_paths(cfg: dict[str, Any]) -> None:
    """Keep outputs separate when a configuration requests it."""
    global DATA, INPUTS, BSCANS, METADATA, TEMP, LABELS
    DATA = ROOT / cfg.get("output_dir", "data")
    INPUTS, BSCANS, METADATA, TEMP = DATA / "inputs", DATA / "bscans", DATA / "metadata", DATA / "temporary_h5"
    LABELS = DATA / "labels.csv"


def validate_config(cfg: dict[str, Any]) -> None:
    required = {"master_seed", "total_runs", "domain_m", "surface_y_m", "cell_size_m", "time_window_s", "frequency_hz", "scan_start_m", "scan_step_m", "n_traces", "target_top_depth_m", "target_radius_m", "target_x_m", "soil_eps_r", "soil_sigma_s_m", "boundary_clearance_m", "ricker_time_zero_s", "split_fractions"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Configuration is missing required key(s): {', '.join(missing)}")
    if len(cfg["domain_m"]) != 2 or any(float(v) <= 0 for v in cfg["domain_m"]):
        raise ValueError("domain_m must contain two positive dimensions")
    if int(cfg["n_traces"]) < 2 or float(cfg["scan_step_m"]) <= 0:
        raise ValueError("n_traces must be at least 2 and scan_step_m must be positive")
    scan_end = cfg["scan_start_m"] + (cfg["n_traces"] - 1) * cfg["scan_step_m"]
    if not (0 <= cfg["scan_start_m"] <= scan_end <= cfg["domain_m"][0]):
        raise ValueError("Scan positions must lie inside the x domain")
    for key in ("target_top_depth_m", "target_radius_m", "target_x_m", "soil_eps_r", "soil_sigma_s_m"):
        values = cfg[key]
        if len(values) != 2 or not (float(values[0]) > 0 and float(values[0]) <= float(values[1])):
            raise ValueError(f"{key} must be an ascending pair of positive values")
    deepest_bottom = max(cfg["target_top_depth_m"]) + 2 * max(cfg["target_radius_m"])
    if cfg["surface_y_m"] - deepest_bottom < cfg["boundary_clearance_m"]:
        raise ValueError("Deepest/largest target violates lower boundary clearance")
    tolerance = 1e-12
    if min(cfg["target_x_m"]) - max(cfg["target_radius_m"]) < cfg["boundary_clearance_m"] - tolerance or cfg["domain_m"][0] - (max(cfg["target_x_m"]) + max(cfg["target_radius_m"])) < cfg["boundary_clearance_m"] - tolerance:
        raise ValueError("Target x range violates side boundary clearance")
    required = required_apex_window_s(cfg)
    if float(cfg["time_window_s"]) < required:
        raise ValueError(f"time_window_s is too short for the deepest target apex: requires at least {required:.12e} s")


def required_apex_window_s(cfg: dict[str, Any]) -> float:
    """Minimum recorded window for the deepest apex, not full hyperbola arms.

    The source-time reference is a recorded-time offset and is added once.
    Pulse/margin defaults are explicit so old configurations remain readable.
    """
    velocity = C / np.sqrt(max(cfg["soil_eps_r"]))
    travel = 2 * max(cfg["target_top_depth_m"]) / velocity
    pulse = float(cfg.get("pulse_allowance_s", DEFAULT_PULSE_ALLOWANCE_S))
    margin = float(cfg.get("timing_margin_s", DEFAULT_TIMING_MARGIN_S))
    return float(cfg["ricker_time_zero_s"]) + travel + pulse + margin


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate configuration key: {key}")
            output[key] = value
        return output
    try:
        with config_path.open(encoding="utf-8") as handle:
            cfg = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Cannot load configuration {config_path}: {error}") from error
    validate_config(cfg)
    configure_paths(cfg)
    return cfg


def uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def sample_plan_without_split(run_id: int, cfg: dict[str, Any]) -> Plan:
    seed = int(cfg["master_seed"]) + run_id
    rng = np.random.default_rng(seed)
    present = bool(rng.random() < cfg["target_present_probability"])
    target_type = "none"
    top = centre = radius = x_target = None
    occupied: list[tuple[float, float, float]] = []
    if present:
        target_type = "metal_pipe" if rng.random() < cfg["metal_pipe_probability_given_target"] else "air_void"
        top = uniform(rng, cfg["target_top_depth_m"])
        radius = uniform(rng, cfg["target_radius_m"])
        centre = top + radius
        x_target = uniform(rng, cfg["target_x_m"])
        occupied.append((x_target, centre, radius))

    eps_r = uniform(rng, cfg["soil_eps_r"])
    sigma = uniform(rng, cfg["soil_sigma_s_m"])
    clutter: list[tuple[float, float, float, float]] = []
    if rng.random() < cfg["clutter_probability"]:
        requested = int(rng.integers(cfg["clutter_count"][0], cfg["clutter_count"][1] + 1))
        # Rejection sampling keeps inclusions separated, avoiding accidental
        # merged geometry that would make a label ambiguous.
        for _ in range(requested):
            placed = False
            for _attempt in range(200):
                r = uniform(rng, cfg["clutter_radius_m"])
                x = float(rng.uniform(0.10 + r, cfg["domain_m"][0] - 0.10 - r))
                depth = float(rng.uniform(0.06 + r, cfg["surface_y_m"] - 0.10 - r))
                if all(np.hypot(x - ox, depth - od) >= r + oradius + 0.02 for ox, od, oradius in occupied):
                    clutter.append((x, depth, r, uniform(rng, cfg["clutter_eps_fraction"]) * eps_r))
                    occupied.append((x, depth, r))
                    placed = True
                    break
            if not placed:
                raise RuntimeError(f"Could not place clutter safely for run {run_id}")

    return Plan(
        run_id, seed, present, target_type, top, centre, radius, x_target,
        eps_r, sigma, len(clutter), tuple(clutter), "",
    )


def assign_splits(plans: list[Plan], cfg: dict[str, Any]) -> list[Plan]:
    """Assign a deterministic 70/15/15 split independently within each class."""
    split_by_id: dict[int, str] = {}
    split_rng = np.random.default_rng(int(cfg["master_seed"]) + 10_000)
    for present in (False, True):
        group = [plan.run_id for plan in plans if plan.target_present == present]
        split_rng.shuffle(group)
        n_train = round(cfg["split_fractions"]["train"] * len(group))
        n_val = round(cfg["split_fractions"]["val"] * len(group))
        for run_id in group[:n_train]:
            split_by_id[run_id] = "train"
        for run_id in group[n_train:n_train + n_val]:
            split_by_id[run_id] = "val"
        for run_id in group[n_train + n_val:]:
            split_by_id[run_id] = "test"
    return [Plan(**(asdict(plan) | {"split": split_by_id[plan.run_id]})) for plan in plans]


def all_plans(cfg: dict[str, Any]) -> list[Plan]:
    return assign_splits([sample_plan_without_split(i, cfg) for i in range(int(cfg["total_runs"]))], cfg)


def input_text(plan: Plan, cfg: dict[str, Any]) -> str:
    dx = cfg["cell_size_m"]
    domain_x, domain_y = cfg["domain_m"]
    lines = [
        f"#title: Step 2 dataset run {plan.run_id:04d}",
        "#domain_mode: TM",
        f"#domain: {domain_x:.3f} {domain_y:.3f} inf",
        f"#dx_dy_dz: {dx:.3f} {dx:.3f} {dx:.3f}",
        f"#time_window: {cfg['time_window_s']:.9g}",
        f"#material: {plan.eps_r:.9g} {plan.sigma:.9g} 1 0 soil",
        f"#waveform: ricker 1 {cfg['frequency_hz']:.9g} ricker_900MHz",
        f"#hertzian_dipole: z {cfg['scan_start_m']:.3f} {cfg['surface_y_m']:.3f} inf ricker_900MHz",
        f"#rx: {cfg['scan_start_m']:.3f} {cfg['surface_y_m']:.3f} inf",
        f"#src_steps: {cfg['scan_step_m']:.3f} 0 0",
        f"#rx_steps: {cfg['scan_step_m']:.3f} 0 0",
        f"#box: 0 0 0 {domain_x:.3f} {cfg['surface_y_m']:.3f} inf soil",
    ]
    if plan.target_present:
        material = "pec" if plan.target_type == "metal_pipe" else "air_void"
        if plan.target_type == "air_void":
            lines.append("#material: 1 0 1 0 air_void")
        y = cfg["surface_y_m"] - float(plan.centre_depth_m)
        lines.append(
            f"#cylinder: {plan.target_x_m:.9g} {y:.9g} 0 {plan.target_x_m:.9g} {y:.9g} inf {plan.radius_m:.9g} {material}"
        )
    for index, (x, depth, radius, eps) in enumerate(plan.clutter):
        material = f"clutter_{index}"
        y = cfg["surface_y_m"] - depth
        lines.append(f"#material: {eps:.9g} {plan.sigma:.9g} 1 0 {material}")
        lines.append(f"#cylinder: {x:.9g} {y:.9g} 0 {x:.9g} {y:.9g} inf {radius:.9g} {material}")
    return "\n".join(lines) + "\n"


def atomic_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def completed(plan: Plan, cfg: dict[str, Any]) -> bool:
    npy = BSCANS / f"run_{plan.run_id:04d}.npy"
    meta = METADATA / f"run_{plan.run_id:04d}.json"
    if not (npy.exists() and meta.exists()):
        return False
    try:
        data = np.load(npy, mmap_mode="r")
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        saved_shape = tuple(metadata["shape"])
        # gprMax includes the sample at t=0, so deriving a shape from
        # time_window/dt can be off by one. The merged-file shape saved at
        # generation time is the authoritative resume check.
        return (
            data.dtype == np.float32
            and tuple(data.shape) == saved_shape
            and data.shape[1] == cfg["n_traces"]
            and float(metadata["time_window_s"]) == float(cfg["time_window_s"])
            and np.isfinite(data).all()
            and np.ptp(data) != 0
            and metadata_matches_plan(metadata, plan)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def metadata_matches_plan(metadata: dict[str, Any], plan: Plan) -> bool:
    """Require saved metadata to be the exact deterministic plan.

    JSON ``null`` is the deliberate representation of target geometry when a
    run has no target. A target-present run must retain every planned float.
    """
    fields = ("run_id", "seed", "target_present", "target_type", "eps_r", "sigma", "n_clutter", "split")
    if any(metadata.get(field) != getattr(plan, field) for field in fields):
        return False
    # ``asdict`` holds a tuple, but JSON correctly reloads it as a list.
    # Compare the JSON representation explicitly instead of losing this check.
    if metadata.get("clutter") != [list(item) for item in plan.clutter]:
        return False
    geometry = ("top_depth_m", "centre_depth_m", "radius_m", "target_x_m")
    if not plan.target_present:
        return all(metadata.get(field) is None for field in geometry)
    return all(
        isinstance(metadata.get(field), (int, float))
        and not isinstance(metadata[field], bool)
        and metadata[field] == getattr(plan, field)
        for field in geometry
    )


def run_one(plan_dict: dict[str, Any], cfg: dict[str, Any], worker_count: int) -> dict[str, Any]:
    configure_paths(cfg)
    plan = Plan(**plan_dict)
    if completed(plan, cfg):
        return {"run_id": plan.run_id, "status": "skipped"}
    for directory in (INPUTS, BSCANS, METADATA, TEMP):
        directory.mkdir(parents=True, exist_ok=True)
    in_file = INPUTS / f"run_{plan.run_id:04d}.in"
    atomic_text(in_file, input_text(plan, cfg))
    base = TEMP / f"run_{plan.run_id:04d}.h5"
    expected_files = [TEMP / f"run_{plan.run_id:04d}{i}.h5" for i in range(1, cfg["n_traces"] + 1)]
    for path in expected_files:
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    env = os.environ.copy()
    # Multiple gprMax jobs already supply parallelism; share cores instead of
    # allowing every worker to start the machine's full OpenMP thread count.
    if worker_count == 1:
        # Let gprMax select its tested OpenMP default for a serial pilot. The
        # previous explicit 12-thread override did not reproduce Step 1's
        # observed 10-thread configuration on this Windows installation.
        env.pop("OMP_NUM_THREADS", None)
    else:
        env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // worker_count))
    command = [sys.executable, "-m", "gprMax", str(in_file), "-n", str(cfg["n_traces"]), "-outputfile", str(base), "--hide-progress-bars"]
    process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"gprMax failed for run {plan.run_id}:\n{process.stdout}\n{process.stderr}")
    if any(not path.exists() for path in expected_files):
        raise RuntimeError(f"gprMax did not create every trace file for run {plan.run_id}")
    from gprMax.toolboxes.Utilities.outputfiles_merge import merge_files
    merged = TEMP / f"run_{plan.run_id:04d}_merged.h5"
    merged.unlink(missing_ok=True)
    merge_files([str(path) for path in expected_files], merged_outputfile=merged)
    with h5py.File(merged, "r") as h5:
        bscan = h5["/rxs/rx1/Ez"][...]
        dt_s = float(h5.attrs["dt"])
    if bscan.shape[1] != cfg["n_traces"] or not np.isfinite(bscan).all() or np.ptp(bscan) == 0:
        raise RuntimeError(f"Merged output validation failed for run {plan.run_id}: {bscan.shape}")
    npy = BSCANS / f"run_{plan.run_id:04d}.npy"
    temporary_npy = npy.with_suffix(".npy.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, bscan.astype(np.float32, copy=False))
    temporary_npy.replace(npy)
    metadata = asdict(plan) | {"time_window_s": cfg["time_window_s"], "dt_s": dt_s, "shape": list(bscan.shape)}
    atomic_text(METADATA / f"run_{plan.run_id:04d}.json", json.dumps(metadata, indent=2) + "\n")
    for path in expected_files + [merged]:
        path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    return {"run_id": plan.run_id, "status": "generated", "seconds": elapsed, "shape": list(bscan.shape), "bytes": npy.stat().st_size}


def write_labels(plans: list[Plan]) -> None:
    rows = []
    fields = ["run_id", "seed", "target_present", "target_type", "top_depth_m", "centre_depth_m", "radius_m", "target_x_m", "eps_r", "sigma", "n_clutter", "split", "time_window_s", "dt_s"]
    for plan in plans:
        meta_file = METADATA / f"run_{plan.run_id:04d}.json"
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            rows.append({name: metadata.get(name, "") for name in fields})
    temporary = LABELS.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["run_id"])))
    temporary.replace(LABELS)


def pilot_ids(plans: list[Plan]) -> list[int]:
    selected = []
    for target_type, needed in (("metal_pipe", 2), ("air_void", 2), ("none", 2)):
        selected.extend(plan.run_id for plan in plans if plan.target_type == target_type and len([i for i in selected if plans[i].target_type == target_type]) < needed)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic gprMax Step 2 B-scans.")
    parser.add_argument("--count", type=int, default=None, help="Generate plans with run_id below this count.")
    parser.add_argument("--run-ids", type=int, nargs="+", help="Generate selected run IDs only.")
    parser.add_argument("--pilot", action="store_true", help="Generate six deterministic, class-varied pilot runs.")
    parser.add_argument("--workers", type=int, default=min(max((os.cpu_count() or 1) - 1, 1), 4))
    parser.add_argument("--config", default="configs/dataset.yaml", help="Configuration path relative to repository root.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    plans = all_plans(cfg)
    if args.pilot:
        ids = pilot_ids(plans)
    elif args.run_ids is not None:
        ids = args.run_ids
    else:
        ids = list(range(args.count if args.count is not None else cfg["total_runs"]))
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= len(plans):
        raise ValueError("Requested run IDs are outside the configured dataset")
    chosen = [plans[run_id] for run_id in ids]
    print(f"Generating {len(chosen)} run(s) with {args.workers} worker(s); master seed {cfg['master_seed']}.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, asdict(plan), cfg, args.workers) for plan in chosen]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result), flush=True)
    write_labels(plans)
    print(f"Wrote {LABELS}", flush=True)


if __name__ == "__main__":
    main()
