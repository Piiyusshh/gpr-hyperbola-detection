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
