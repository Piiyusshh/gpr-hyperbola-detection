# Preliminary dataset audit and baseline report

This report records the completed lean synthetic experiment as inspected on
2026-10-24. It is a software and signal-processing report, not a claim of
physical validation or real-world detector performance.

## Dataset audit

The lean dataset contains 200 unique run IDs, 0 through 199. Every run has a
`(1379, 50)` `float32` B-scan. All values are finite and non-constant. A
SHA-256 check of the array bytes found no exact duplicate arrays. The saved
labels and metadata agree with the deterministic plan and the verifier reports:

```text
Dataset status: COMPLETE - 200 of 200 completed
PASS: Dataset integrity - Verified 200 labelled B-scans
```

Class and split counts are:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `metal_pipe` | 67 | 17 | 13 | 97 |
| `air_void` | 24 | 3 | 6 | 33 |
| `none` | 49 | 10 | 11 | 70 |
| **Total** | **140** | **30** | **30** | **200** |

The split is deterministic and stratified only by target-present versus
target-absent during generation. The exact duplicate check did not find
cross-split copies. It does not rule out physically similar simulations or
other forms of dependence caused by the shared simulation design.

## Signal observations

Across all arrays, the mean standard deviation was approximately 771.2 in the
stored `Ez` units. The mean absolute-amplitude statistics were similar between
groups:

| Group | Mean standard deviation | Mean 99.5th-percentile absolute amplitude | Mean late absolute amplitude |
|---|---:|---:|---:|
| `none` (70) | 770.6 | 4280.1 | 214.9 |
| target present (130) | 771.8 | 4283.5 | 225.0 |
| `metal_pipe` (97) | 772.8 | 4288.2 | 227.0 |
| `air_void` (33) | 768.7 | 4269.5 | 219.3 |

These aggregate differences are small relative to the signal variation.
They show that amplitude alone should not be treated as a validated target
detector. The verifier's direct-event diagnostic found the same displayed
early-event time for all 200 runs, with zero measured spread at the reported
precision and a tolerance of 0.1 ns. That is a consistency result for this
simulation setup, not evidence that later target reflections have been
correctly identified.

Existing and newly generated figures:

- [dataset samples](../results/figures/dataset_samples.png)
- [dataset overlays](../results/figures/dataset_overlays.png)
- [preprocessing comparison](../results/figures/preprocessing_comparison.png)

The overlay curves are label-derived geometric references. A local maximum
near one of those curves is not automatically a detected target reflection.

## Preprocessing

The raw files under `data_lean/bscans/` were preserved. The new method writes
processed copies under
`data_lean/processed/background_removed_normalized/`:

1. Subtract the mean over scan traces at each time sample in each B-scan.
   This is a common simple background-removal operation and does not fit a
   parameter across dataset splits.
2. Divide each B-scan by its 99.5th-percentile absolute amplitude. This keeps
   the feature scale comparable between arrays without using validation or
   test data to estimate a global scale.

The method is intentionally simple. It may remove spatially broad target
energy and may amplify weak artefacts. Cleaner-looking images are not proof
of better physical fidelity.

## Baseline task and results

The first task is binary target presence:

```text
target present = target_type != none
```

This task is better supported than three-way classification because the
`air_void` class has only 33 examples, including 3 validation and 6 test
examples. The baseline extracts pooled time and trace amplitude features and
uses a nearest-centroid classifier. It has no external machine-learning
dependency.

The model is fitted on the 140 training runs. Validation is used only for
inspection; the 30-run test set is evaluated once after fitting.

| Evaluation | Accuracy | Macro-F1 | Confusion matrix (actual rows, predicted columns) |
|---|---:|---:|---|
| Validation | 0.933 | 0.925 | `[[9, 1], [1, 19]]` |
| Test | 0.900 | 0.884 | `[[8, 3], [0, 19]]` |
| Majority test baseline | 0.633 | 0.388 | `[[0, 11], [0, 19]]` |

Test support is 11 no-target and 19 target-present runs. The result is
reproducible with the current fixed split and saved in
[target_presence_metrics.json](../results/baseline/target_presence_metrics.json).
It is not evidence of performance on measured GPR data, and the small
synthetic test set gives uncertain estimates of generalisation.

## Hyperbola and physical limitations

The Step 1 apex comparison remains within one stored time sample:

```text
predicted apex minus saved picked apex = -5.467633733227e-12 s
criterion = 9.434617346999e-12 s
```

The calculation uses a homogeneous, non-magnetic velocity
`v = c / sqrt(eps_r)`, a recorded time-zero reference, and the target
geometry. It checks the apex only. Full hyperbola-arm agreement, antenna
coupling, boundary effects, numerical convergence, and material-model
validity remain unassessed.

The lean experiment is idealized 2D TM gprMax data with simplified
homogeneous soil, synthetic clutter, an idealized target representation, and
fixed numerical settings. Its 13 ns window is designed to include the
deepest expected apex under the configured approximation; it is not proof
that every late or wide hyperbola arm is captured.

### Step 1 analysis reproducibility note

The current [analyse_first_bscan.py](../scripts/analyse_first_bscan.py)
defaults to an apex-gate half-width of `0.45 ns`, while the saved
[step1_summary.json](../results/step1_summary.json) records
`"apex_gate_half_width_ns": 1.2`. The repository does not contain verified
evidence of the historical command or argument that produced the saved `1.2 ns`
summary. Neither value is changed by this report, and the saved result is not
regenerated here. This discrepancy must be resolved or explicitly labelled
before using Step 1 as a clean reproduction benchmark.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_dataset.py --data-dir data_lean --config configs\dataset_lean.yaml
.\.venv\Scripts\python.exe scripts\preprocess_dataset.py --data-dir data_lean
.\.venv\Scripts\python.exe scripts\baseline_classifier.py --data-dir data_lean
```

No simulations are required for these audit, preprocessing, or baseline
commands, and the raw simulation arrays should not be regenerated.

## Next research steps

Before drawing scientific conclusions, evaluate grid, time-window, material,
source/receiver, and boundary sensitivity; compare against independent
analytical or experimental references; investigate target-reflection picking
with explicit validation; and evaluate on held-out simulations with broader
parameter distributions and, ultimately, measured GPR data. A future model
should also report uncertainty and guard against simulation-family leakage.
