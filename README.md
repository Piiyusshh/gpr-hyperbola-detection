# gpr-hyperbola-detection

This is Step 1 of a small, reproducible preliminary study for automated GPR
analysis. It uses real gprMax FDTD simulation to produce and inspect one 2D
zero-offset B-scan of an ideal-metal pipe in a homogeneous soil half-space;
it does not yet create a dataset, preprocessing pipeline, or ML model.

## Reproduce Step 1 from a clean start (Windows PowerShell)

Use Python 3.12, then run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force results, results\figures | Out-Null

# Optional installation check: an official gprMax example.
.\.venv\Scripts\python.exe -m gprMax .venv\Lib\site-packages\gprMax\_examples\gpr\basic\cylinder_Ascan_2D.in -outputfile results\official_example.h5 --hide-progress-bars

# Run the 63 stepped A-scans and merge them into one B-scan.
.\.venv\Scripts\python.exe -m gprMax configs\first_pipe_bscan_2d.in -n 63 -outputfile results\first_pipe_bscan.h5 --hide-progress-bars
.\.venv\Scripts\python.exe -m gprMax.toolboxes.Utilities.outputfiles_merge results\first_pipe_bscan --output-file results\first_pipe_bscan_merged.h5

# Read the HDF5 file, print its shape, perform the sanity check, and save plots.
.\.venv\Scripts\python.exe scripts\analyse_first_bscan.py
```

The final command writes `results/figures/first_bscan.png`,
`results/figures/first_ascan.png`, and `results/step1_summary.json`.

See [the decisions](docs/DECISIONS.md), [setup notes](docs/SETUP_NOTES.md),
and [learning notes](docs/LEARNING.md) before moving to Step 2.
