# Step 1 decisions

## Environment

I used a Python virtual environment because the current official gprMax
installation guide recommends PyPI plus `venv` for ordinary modelling on
Windows. It supports Python 3.11--3.13 and recommends Python 3.12. The CPU
solver is sufficient here, so no MPI or GPU extra was installed.

## First model

- The model is 2D TMz. A long cylinder in the out-of-plane direction is the
  usual simple 2D representation of a pipe. It is an ideal PEC cylinder, not
  a realistic plastic or lossy metal pipe.
- Soil has relative permittivity 6 and conductivity 0.01 S/m. Air is left as
  the default material above the soil surface at y=0.260 m.
- The Ricker waveform has nominal frequency 900 MHz. In soil,
  `v = c/sqrt(6)` and `lambda = v/f = 0.136 m` (rounded). I selected a 4 mm
  cell, so the nominal wavelength spans about 34 cells. The waveform is
  broadband, hence the deliberately finer-than-10-cell nominal choice.
- The model uses a colocated Hertzian dipole and receiver at the soil surface
  and steps both by 8 mm. This is an idealised zero-offset B-scan; it makes
  the simple depth estimate `v*t/2` directly comparable to depth below the
  surface. A real antenna has size, coupling and usually a non-zero offset.
- The pipe centre is at (0.300 m, 0.140 m), radius 0.032 m. Its true top depth
  is `0.260 - (0.140 + 0.032) = 0.088 m`.
- The 4 ns time window is longer than the approximate two-way travel time to
  the pipe top (`2 * 0.088 / (c/sqrt(6))`, about 1.4 ns). It leaves time after
  the pipe echo, rather than cutting it off.

For the depth check, recorded gprMax time is not treated as pure propagation
time. The Ricker source and the colocated receiver create a repeatable early
direct/source event. The analysis picks that event from the real data as a
time-zero reference, then subtracts its recorded time from the later pipe
apex time before applying `depth = v*t/2`. It uses a trace-average residual
only to identify the spatially varying pipe event; the saved B-scan remains
the raw gprMax Ez data.

gprMax input files treat every line beginning with `#` as an executable input
command, so prose comments cannot be placed inside the `.in` file. The
reasons for each choice are recorded here and in the commented analysis code.

## Step 2 dataset geometry and numerical cost

Step 2 keeps the requested target top-depth range (0.10--0.40 m), radius
range (0.02--0.08 m), soil permittivity range (4--16), conductivity range,
clutter, and target classes. Only numerical cost was changed after the first
2 mm pilot proved impractically slow.

- The fixed 2D TM domain is 0.600 m by 0.800 m. The ground surface is at
  y=0.680 m. The deepest/largest possible pipe has its bottom at y=0.120 m,
  and inclusions are constrained to remain at least 0.100 m from the bottom
  boundary. The 0.800 m height leaves 0.120 m of air above the surface.
- The chosen cell size is 0.004 m. At eps_r=16, code calculated a 900 MHz
  wavelength of 0.083275682778 m, or 20.818920694444 cells. At 1.8 GHz,
  it calculated 0.041637841389 m, or 10.409460347222 cells. Thus it passes
  the requested minimum of 10 cells per wavelength at twice the centre
  frequency. A coarser cell was not used.
- The common time window is 20 ns. The widest permitted arm has lateral
  offset 0.368 m (scan x=0.548 m and target x=0.180 m). For eps_r=16,
  centre depth 0.480 m, and radius 0.080 m, code calculated the reflection
  path as 1.049667722972 m and travel time as 1.400525857087e-08 s. Adding
  the measured Step 1 time-zero (1.434061836743808e-09 s), a 3 ns Ricker
  pulse allowance, and 1.5 ns margin gives 1.993932040762e-08 s. 20 ns is
  the first rounded-up nanosecond window that contains this calculation.
- Target centres use x=0.180--0.420 m rather than the nominal middle 60% of
  the scan. This is the widest range that allows the largest 0.080 m target
  to stay 0.100 m from the 0 and 0.600 m absorbing boundaries.
- The resume check reads each completed run's saved metadata shape instead
  of recalculating `time_window/dt`. gprMax includes the t=0 sample, so the
  arithmetic calculation can be one sample short (as Step 1 demonstrated).

## Lean dataset experiment

`configs/dataset_lean.yaml` is a separate experiment; it does not replace the
original configuration and directs any future output to `data_lean/`.

- It uses 4 mm cells, target-top depths 0.10--0.30 m, radii 0.02--0.06 m,
  eps_r 4--12, and the same 900 MHz Ricker centre frequency. It scans 50
  positions from x=0.052 m in 0.010 m steps, ending at x=0.542 m.
- At eps_r=12, code calculated lambda=0.096158475737 m at 900 MHz, or
  24.039618934350 cells per wavelength. At 1.8 GHz it calculated
  lambda=0.048079237869 m, or 12.019809467175 cells. This is a numerical
  resolution check, not proof that the complete electromagnetic model is
  accurate.
- The lean domain is 0.600 m by 0.720 m with surface y=0.600 m. The maximum
  target bottom depth is 0.420 m, leaving 0.180 m to the lower boundary. The
  target x range was set to 0.161--0.439 m so the maximum-radius target has
  0.101 m clearance at either side.
- Using the verified Step 1 recorded-time reference of
  1.434061836743808e-09 s, eps_r=12, and a 0.300 m top depth, code calculated
  the apex at 8.367061362311e-09 s. The 13 ns window adds a deliberately
  separate 3 ns pulse allowance and 1.5 ns safety margin, giving a required
  1.286706136231e-08 s. The time-zero is added only once. The velocity is
  c/sqrt(eps_r), appropriate for this simple homogeneous, non-magnetic soil
  assumption. The 13 ns choice is designed to capture the deepest apex, not
  every late, wide hyperbola arm; clipping those arms is an intentional
  computational trade-off and has not been simulation-validated.

The generator now enforces this deepest-apex requirement for both experiment
files. Its explicit, testable defaults are a 3 ns pulse allowance and a
1.5 ns timing margin when a configuration does not provide those fields.
This validation is deliberately for the apex only, not a guarantee that all
wide hyperbola arms fit in the saved record.

## Lean run completion audit

Run 0 was initially reported as zero completed when the verifier was invoked
with `--data-dir data_lean` but without `--config`. It then loaded the original
configuration and checked the original `data/` paths. The verifier now infers
the lean configuration for `data_lean/` unless an explicit configuration is
given. A completed run requires its float32 array, metadata, planned run ID,
seed, class, split, configured time window, saved shape, finite values, and
non-constant values to agree. One completed simulation is not a complete or
fully validated dataset.

For target-absent runs, the agreed geometry representation is JSON `null` in
the per-run metadata and an empty CSV cell in `labels.csv` for each of
`top_depth_m`, `centre_depth_m`, `radius_m`, and `target_x_m`. This is not a
zero-sized target. The verifier requires precisely this pair of
representations; target-present runs must instead contain numeric geometry
that exactly matches their deterministic plan.
