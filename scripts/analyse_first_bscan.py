"""Read the real gprMax Step 1 output and make honest, reproducible plots.

The apex picker is intentionally simple. It first picks the repeatable early
direct/source event as a measured time-zero reference. Then it subtracts the
trace-average (which removes laterally uniform source energy) and picks the
largest residual in a broad gate around the predicted pipe event. This is a
sanity check, not a general detector or a preprocessing pipeline.
"""

from pathlib import Path
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np


C = 299_792_458.0
EPS_R = 6.0
SOIL_VELOCITY = C / np.sqrt(EPS_R)
SURFACE_Y_M = 0.260
PIPE_CENTRE_Y_M = 0.140
PIPE_RADIUS_M = 0.032
TRUE_TOP_DEPTH_M = SURFACE_Y_M - (PIPE_CENTRE_Y_M + PIPE_RADIUS_M)
SCAN_START_M = 0.052
SCAN_STEP_M = 0.008
PIPE_X_M = 0.300

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "first_pipe_bscan_merged.h5"
FIGURES = ROOT / "results" / "figures"
SUMMARY_FILE = ROOT / "results" / "step1_summary.json"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    with h5py.File(INPUT_FILE, "r") as h5:
        # gprMax's merger stores a B-scan as (time samples, trace positions).
        ez = h5["/rxs/rx1/Ez"][...]
        dt_s = float(h5.attrs["dt"])

    n_time, n_traces = ez.shape
    times_s = np.arange(n_time) * dt_s
    positions_m = SCAN_START_M + np.arange(n_traces) * SCAN_STEP_M
    times_ns = times_s * 1e9

    print(f"Ez array shape: {ez.shape}")
    print("Axis 0 is time samples; axis 1 is stepped source/receiver positions.")
    print(f"Time increment from file: {dt_s:.12e} s")

    # A gprMax Ricker pulse starts after simulation time zero. Rather than
    # pretending recorded time is propagation time, use the strong, laterally
    # repeatable direct/source event at the centre trace as its time-zero.
    apex_trace = int(np.argmin(np.abs(positions_m - PIPE_X_M)))
    direct_gate = times_s < 2.3e-9
    direct_time_index = int(np.flatnonzero(direct_gate)[np.argmax(np.abs(ez[direct_gate, apex_trace]))])
    time_zero_s = times_s[direct_time_index]

    # The top reflection should be about 2d/v after that reference. The
    # spatial residual prevents the direct event (nearly identical in every
    # trace) from being selected as a pipe echo.
    expected_travel_time_s = 2 * TRUE_TOP_DEPTH_M / SOIL_VELOCITY
    residual = ez - ez.mean(axis=1, keepdims=True)
    gate = (times_s >= time_zero_s + expected_travel_time_s - 0.45e-9) & (
        times_s <= time_zero_s + expected_travel_time_s + 0.45e-9
    )
    gated = np.abs(residual[gate, apex_trace])
    apex_time_index = int(np.flatnonzero(gate)[np.argmax(gated)])
    recorded_echo_time_s = times_s[apex_time_index]
    propagation_time_s = recorded_echo_time_s - time_zero_s
    estimated_depth_m = SOIL_VELOCITY * propagation_time_s / 2
    difference_m = estimated_depth_m - TRUE_TOP_DEPTH_M

    # Plot trace positions horizontally and two-way time vertically.
    vmax = float(np.percentile(np.abs(ez), 99))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = ax.imshow(
        ez,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=[positions_m[0], positions_m[-1], times_ns[-1], times_ns[0]],
    )
    ax.plot(positions_m[apex_trace], times_ns[apex_time_index], "ko", ms=4, label="picked apex")
    ax.set(xlabel="Scan position (m)", ylabel="Recorded time (ns)", title="Step 1: simulated pipe B-scan (Ez)")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Ez (V/m)")
    fig.savefig(FIGURES / "first_bscan.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(times_ns, ez[:, apex_trace], lw=1)
    ax.axvline(times_ns[apex_time_index], color="tab:red", ls="--", label="picked pipe echo")
    ax.set(xlabel="Time (ns)", ylabel="Ez (V/m)", title=f"A-scan at x = {positions_m[apex_trace]:.3f} m")
    ax.legend()
    fig.savefig(FIGURES / "first_ascan.png", dpi=200)
    plt.close(fig)

    summary = {
        "array_shape_time_by_trace": [int(n_time), int(n_traces)],
        "dt_s": dt_s,
        "pipe_top_depth_true_m": TRUE_TOP_DEPTH_M,
        "apex_trace_index": apex_trace,
        "apex_position_m": float(positions_m[apex_trace]),
        "time_zero_reference_s": float(time_zero_s),
        "picked_echo_recorded_time_s": float(recorded_echo_time_s),
        "pipe_propagation_time_s": float(propagation_time_s),
        "estimated_top_depth_m": float(estimated_depth_m),
        "estimated_minus_true_m": float(difference_m),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
