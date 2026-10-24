"""Focused tests for configuration mathematics and resumable output checks."""
from __future__ import annotations

import json
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_dataset as generator
from verify_dataset import C, curve, geometry_matches_metadata_and_label
from verify_dataset import verify_dataset


class DatasetPipelineTests(unittest.TestCase):
    def test_both_project_configs_have_sufficient_apex_window(self) -> None:
        for name in ("dataset.yaml", "dataset_lean.yaml"):
            cfg = generator.load_config(ROOT / "configs" / name)
            self.assertGreaterEqual(cfg["time_window_s"], generator.required_apex_window_s(cfg))

    def test_invalid_geometry_and_time_are_rejected(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml").copy()
        cfg["time_window_s"] = 1e-9
        with self.assertRaisesRegex(ValueError, "time_window_s"):
            generator.validate_config(cfg)
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml").copy()
        cfg["target_x_m"] = [0.01, 0.02]
        with self.assertRaisesRegex(ValueError, "side boundary"):
            generator.validate_config(cfg)

    def test_curve_apex_matches_the_analytical_expression(self) -> None:
        t0, x0, zc, radius, eps = 1.0e-9, 0.3, 0.12, 0.032, 6.0
        observed = curve(t0, np.array([x0]), x0, zc, radius, eps)[0]
        expected = t0 + 2 * (zc - radius) / (C / np.sqrt(eps))
        self.assertAlmostEqual(observed, expected)

    def test_completion_requires_matching_valid_saved_output(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml").copy()
        with tempfile.TemporaryDirectory() as temporary:
            cfg["output_dir"] = temporary
            generator.configure_paths(cfg)
            plan = generator.all_plans(cfg)[0]
            self.assertFalse(generator.completed(plan, cfg))
            generator.BSCANS.mkdir(parents=True)
            generator.METADATA.mkdir(parents=True)
            np.save(generator.BSCANS / "run_0000.npy", np.arange(150, dtype=np.float32).reshape(3, 50))
            metadata = {**generator.asdict(plan), "shape": [3, 50], "time_window_s": cfg["time_window_s"]}
            (generator.METADATA / "run_0000.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(generator.completed(plan, cfg))
            np.save(generator.BSCANS / "run_0000.npy", np.arange(150, dtype=np.float64).reshape(3, 50))
            self.assertFalse(generator.completed(plan, cfg))

    def test_existing_lean_run_zero_is_recognised_without_writing_files(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml")
        self.assertTrue(generator.completed(generator.all_plans(cfg)[0], cfg))

    def test_missing_metadata_rejects_completion(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml").copy()
        with tempfile.TemporaryDirectory() as temporary:
            cfg["output_dir"] = temporary
            generator.configure_paths(cfg)
            generator.BSCANS.mkdir(parents=True)
            np.save(generator.BSCANS / "run_0000.npy", np.arange(50, dtype=np.float32).reshape(1, 50))
            self.assertFalse(generator.completed(generator.all_plans(cfg)[0], cfg))

    def test_existing_label_reconciles_to_existing_completed_run(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml")
        with (ROOT / "data_lean" / "labels.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed_ids = {plan.run_id for plan in generator.all_plans(cfg) if generator.completed(plan, cfg)}
        self.assertEqual({int(row["run_id"]) for row in rows}, completed_ids)

    def test_no_target_null_metadata_and_empty_csv_geometry_reconcile(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml")
        plans = generator.all_plans(cfg)
        metadata = json.loads((ROOT / "data_lean" / "metadata" / "run_0001.json").read_text(encoding="utf-8"))
        with (ROOT / "data_lean" / "labels.csv").open(newline="", encoding="utf-8") as handle:
            row = next(row for row in csv.DictReader(handle) if row["run_id"] == "1")
        geometry = ("top_depth_m", "centre_depth_m", "radius_m", "target_x_m")
        self.assertFalse(plans[1].target_present)
        self.assertTrue(generator.metadata_matches_plan(metadata, plans[1]))
        self.assertTrue(all(metadata[field] is None and row[field] == "" for field in geometry))
        self.assertTrue(geometry_matches_metadata_and_label(plans[1], metadata, row))
        row["radius_m"] = "0.01"
        self.assertFalse(geometry_matches_metadata_and_label(plans[1], metadata, row))

    def test_target_present_geometry_must_match_metadata_csv_and_plan(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml")
        plan = generator.all_plans(cfg)[0]
        metadata = json.loads((ROOT / "data_lean" / "metadata" / "run_0000.json").read_text(encoding="utf-8"))
        self.assertTrue(plan.target_present)
        self.assertTrue(generator.metadata_matches_plan(metadata, plan))
        with (ROOT / "data_lean" / "labels.csv").open(newline="", encoding="utf-8") as handle:
            row = next(row for row in csv.DictReader(handle) if row["run_id"] == "0")
        self.assertTrue(geometry_matches_metadata_and_label(plan, metadata, row))
        row["radius_m"] = ""
        self.assertFalse(geometry_matches_metadata_and_label(plan, metadata, row))
        metadata["radius_m"] = None
        self.assertFalse(generator.metadata_matches_plan(metadata, plan))

    def test_complete_dataset_reports_pass(self) -> None:
        cfg = generator.load_config(ROOT / "configs" / "dataset_lean.yaml")
        captured = io.StringIO()
        # Verifier output is tested without replacing the user's diagnostic figures.
        with patch("verify_dataset.make_dataset_figures"), patch("verify_dataset.dataset_overlays"), patch("verify_dataset.direct_event_consistency"), redirect_stdout(captured):
            result = verify_dataset(ROOT / "data_lean", cfg)
        completed_count = sum(1 for plan in generator.all_plans(cfg) if generator.completed(plan, cfg))
        self.assertEqual(completed_count, 200)
        self.assertEqual(result[0]["status"], "PASS")
        self.assertIn("COMPLETE — 200 of 200", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
