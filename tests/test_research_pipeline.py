"""Focused tests for raw-preserving preprocessing and the baseline helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_pipeline import (  # noqa: E402
    background_remove,
    classification_metrics,
    extract_features,
    fit_centroid_classifier,
    predict_centroid,
    preprocess_array,
)


class ResearchPipelineTests(unittest.TestCase):
    def test_background_removal_removes_trace_mean_per_sample(self) -> None:
        array = np.array([[1, 3, 5], [2, 4, 8]], dtype=np.float32)
        result = background_remove(array)
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(array, np.array([[1, 3, 5], [2, 4, 8]], dtype=np.float32))

    def test_preprocessing_is_finite_and_feature_shape_is_stable(self) -> None:
        array = np.arange(120, dtype=np.float32).reshape(12, 10)
        processed = preprocess_array(array)
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.isfinite(processed).all())
        self.assertEqual(extract_features(processed).shape, (53,))

    def test_centroid_classifier_predicts_separable_training_examples(self) -> None:
        features = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
        labels = np.array([0, 0, 1, 1])
        model = fit_centroid_classifier(features, labels)
        np.testing.assert_array_equal(predict_centroid(model, features), labels)

    def test_metrics_include_support_and_confusion_matrix(self) -> None:
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["per_class"]["0"]["support"], 2)


if __name__ == "__main__":
    unittest.main()
