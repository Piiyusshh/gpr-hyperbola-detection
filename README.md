# gpr-hyperbola-detection

This repository contains a reproducible preliminary research pipeline for
synthetic Ground Penetrating Radar (GPR) analysis. It uses gprMax FDTD
simulation to produce idealized 2D zero-offset B-scans, generates deterministic
labels and metadata for a synthetic dataset, verifies the saved data,
preprocesses B-scans, extracts numerical features, and evaluates a
nearest-centroid baseline classifier for binary target presence.

The current classifier has been evaluated only on synthetic gprMax data. It
has not been validated on measured or real-world GPR measurements and must not
be treated as a field-ready underground-object detector. The Step 1 apex timing
check is an internal geometric sanity check, not full hyperbola validation.
Current results do not establish reliable real-world generalization.

## Current status

The completed lean dataset contains 200 synthetic B-scans:

- 97 `metal_pipe` runs.
- 33 `air_void` runs.
- 70 `none` runs.
- 140 training, 30 validation, and 30 test runs.
- Each array has shape `(1379, 50)` and dtype `float32`.

The binary target-presence baseline achieved the following on the fixed
synthetic test split:

- Accuracy: 0.900.
- Macro-F1: 0.884.
- Majority-class accuracy: 0.633.

These are preliminary synthetic results from one simulation family, not field
performance claims.

## Repository structure

```text
configs/    gprMax inputs and deterministic dataset configurations
scripts/    simulation analysis, generation, verification, preprocessing, and baseline code
tests/      automated unit tests
docs/       modelling decisions, learning notes, reports, and audit guide
results/    selected lightweight summaries and diagnostic figures
data*/      generated datasets; excluded from ordinary Git publication
```

The original and lean dataset configurations are separate experiments. Do not
mix their seeds, output directories, labels, or split assignments.

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
and [learning notes](docs/LEARNING.md) for background and limitations.

## Step 2 configuration and verification

The original dataset experiment remains the default. Select the lean experiment
explicitly; this creates outputs beneath `data_lean/` and does not overwrite
the original experiment's data directory.

```powershell
# Show generator options without starting a simulation.
.\.venv\Scripts\python.exe scripts\generate_dataset.py --help

# Original configuration (only run when intentionally generating data).
.\.venv\Scripts\python.exe scripts\generate_dataset.py --config configs\dataset.yaml --pilot

# Lean configuration (only run when intentionally generating data).
.\.venv\Scripts\python.exe scripts\generate_dataset.py --config configs\dataset_lean.yaml --pilot

# Verification never launches gprMax.
.\.venv\Scripts\python.exe scripts\verify_dataset.py --step1-only
.\.venv\Scripts\python.exe scripts\verify_dataset.py --data-dir data_lean
```

The generator is resumable and skips existing valid runs. Dataset generation
can be expensive; do not run it merely to reproduce the reports when the
generated arrays are already available. Verification of a complete dataset
also refreshes diagnostic figures under `results/figures/`.

## Preliminary preprocessing and baseline

The completed lean dataset can be preprocessed without changing the raw
arrays. The reproducible preprocessing removes the per-time-sample mean across
traces in each B-scan, then scales that B-scan by its 99.5th-percentile
absolute amplitude. It is a signal-comparison baseline, not a claim that the
result is physically superior.

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_dataset.py --data-dir data_lean
.\.venv\Scripts\python.exe scripts\baseline_classifier.py --data-dir data_lean
```

The baseline predicts binary target presence (`target_type != none`) with a
dependency-free nearest-centroid classifier. It uses the configured
train/validation/test split and leaves the test split out of fitting. Actual
statistics, metrics, limitations, and generated artifacts are recorded in
[the research report](docs/RESEARCH_REPORT.md).

## Testing

Run the focused test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover configuration validation, deterministic plans, output
integrity, metadata/label consistency, preprocessing, feature extraction, and
baseline metric calculations.

## Data storage policy

Large generated arrays, processed arrays, temporary HDF5 files, logs, and
virtual environments are excluded by `.gitignore`. The source code,
configurations, tests, documentation, selected JSON summaries, and selected
diagnostic figures are suitable for Git publication. A complete dataset should
be distributed through a versioned research-data archive rather than committed
to ordinary Git history.

## Limitations and future research

The simulations use simplified 2D geometry, homogeneous or parametrically
sampled soil, idealized targets, synthetic clutter, and simplified source and
receiver representations. The current apex check does not validate full
hyperbola arms, and the labels indicate inserted simulation geometry rather
than independently verified observed reflections.

Future work should resolve the recorded Step 1 analysis-parameter discrepancy,
test grid, time-window, material, source/receiver, and boundary sensitivity,
validate target-reflection labels, evaluate independent simulation families,
and eventually compare with measured GPR data. See the
[complete audit and learning guide](docs/COMPLETE_PROJECT_AUDIT_AND_LEARNING_GUIDE.md)
for the detailed roadmap.
