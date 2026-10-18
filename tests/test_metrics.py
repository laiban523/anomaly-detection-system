"""
tests/test_metrics.py

Unit tests for detection/metrics.py — the point-adjusted F1 evaluation module.

These tests are critical because a bug in point_adjust silently corrupts all
benchmark numbers.  Every edge case in the protocol is covered here.
"""

import numpy as np
import pytest

from detection.metrics import (
    compute_pa_metrics,
    compute_raw_metrics,
    full_report,
    point_adjust,
)

# point_adjust
class TestPointAdjust:

    def test_perfect_predictions_unchanged(self):
        """If pred already matches gt, adjust should be identical."""
        gt   = np.array([0, 0, 1, 1, 1, 0, 0])
        pred = np.array([0, 0, 1, 1, 1, 0, 0])
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, pred)

    def test_single_hit_credits_full_segment(self):
        """One detection inside a segment should credit the whole segment."""
        gt   = np.array([0, 0, 1, 1, 1, 1, 0])
        pred = np.array([0, 0, 0, 1, 0, 0, 0])   # only middle point flagged
        adj  = point_adjust(pred, gt)
        expected = np.array([0, 0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(adj, expected)

    def test_no_hit_in_segment_not_credited(self):
        """A segment with zero detections must not be credited."""
        gt   = np.array([0, 1, 1, 1, 0])
        pred = np.array([0, 0, 0, 0, 0])
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, pred)

    def test_false_positive_outside_segment_preserved(self):
        """FP outside any anomaly segment must stay as-is."""
        gt   = np.array([0, 0, 1, 1, 0, 0])
        pred = np.array([1, 0, 1, 0, 0, 1])   # FPs at index 0 and 5
        adj  = point_adjust(pred, gt)
        assert adj[0] == 1    # FP preserved
        assert adj[5] == 1    # FP preserved
        assert adj[2] == 1    # credited via hit
        assert adj[3] == 1    # credited via hit

    def test_multiple_segments(self):
        """Each segment is independently credited if hit."""
        gt   = np.array([1, 1, 0, 0, 1, 1])
        pred = np.array([1, 0, 0, 0, 0, 1])   # one hit per segment
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, np.array([1, 1, 0, 0, 1, 1]))

    def test_all_normal_gt(self):
        """No anomaly segments — adjust should be identical to input."""
        gt   = np.zeros(10, dtype=int)
        pred = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, pred)

    def test_all_anomaly_gt(self):
        """Entire series is one big anomaly segment — one hit credits all."""
        gt   = np.ones(8, dtype=int)
        pred = np.zeros(8, dtype=int)
        pred[3] = 1
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, np.ones(8, dtype=np.int8))

    def test_segment_at_start(self):
        """Anomaly segment starting at index 0."""
        gt   = np.array([1, 1, 1, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, np.array([1, 1, 1, 0, 0]))

    def test_segment_at_end(self):
        """Anomaly segment ending at the last index."""
        gt   = np.array([0, 0, 1, 1, 1])
        pred = np.array([0, 0, 0, 0, 1])
        adj  = point_adjust(pred, gt)
        np.testing.assert_array_equal(adj, np.array([0, 0, 1, 1, 1]))

    def test_output_dtype_is_int8(self):
        gt   = np.array([0, 1, 1, 0])
        pred = np.array([0, 1, 0, 0])
        adj  = point_adjust(pred, gt)
        assert adj.dtype == np.int8


# compute_raw_metrics
class TestRawMetrics:

    def test_perfect_predictions(self):
        gt   = np.array([1, 1, 0, 0, 1])
        pred = np.array([1, 1, 0, 0, 1])
        r = compute_raw_metrics(pred, gt)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"]    == pytest.approx(1.0)
        assert r["f1"]        == pytest.approx(1.0)

    def test_all_false_negatives(self):
        gt   = np.array([1, 1, 1])
        pred = np.zeros(3, dtype=int)
        r = compute_raw_metrics(pred, gt)
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"]    == pytest.approx(0.0)
        assert r["f1"]        == pytest.approx(0.0)

    def test_all_false_positives(self):
        gt   = np.zeros(3, dtype=int)
        pred = np.ones(3, dtype=int)
        r = compute_raw_metrics(pred, gt)
        assert r["precision"] == pytest.approx(0.0)
        assert r["recall"]    == pytest.approx(0.0)
        assert r["f1"]        == pytest.approx(0.0)

    def test_f1_harmonic_mean(self):
        # precision = 0.5, recall = 1.0 → F1 = 2/3
        gt   = np.array([1, 1])
        pred = np.array([1, 1])
        # inject a FP to drop precision
        gt   = np.array([0, 1, 1])
        pred = np.array([1, 1, 1])
        r = compute_raw_metrics(pred, gt)
        assert r["precision"] == pytest.approx(2/3, rel=1e-3)
        assert r["recall"]    == pytest.approx(1.0)
        assert r["f1"]        == pytest.approx(0.8, rel=1e-3)

    def test_all_zeros(self):
        gt   = np.zeros(5, dtype=int)
        pred = np.zeros(5, dtype=int)
        r = compute_raw_metrics(pred, gt)
        assert r["f1"] == pytest.approx(0.0)


# compute_pa_metrics
class TestPAMetrics:

    def test_single_hit_gives_perfect_recall(self):
        """One hit per segment → PA-Recall = 1.0."""
        gt   = np.array([0, 1, 1, 1, 0, 1, 1, 0])
        pred = np.array([0, 1, 0, 0, 0, 0, 1, 0])
        r = compute_pa_metrics(pred, gt)
        assert r["pa_recall"] == pytest.approx(1.0)

    def test_pa_f1_higher_than_raw(self):
        """PA-F1 should be >= Raw F1 when segments are partially detected."""
        gt   = np.array([0, 1, 1, 1, 1, 1, 0])
        pred = np.array([0, 1, 0, 0, 0, 0, 0])   # only first point of segment
        pa   = compute_pa_metrics(pred, gt)
        raw  = compute_raw_metrics(pred, gt)
        assert pa["pa_f1"] >= raw["f1"]

    def test_no_detection_zero_pa_f1(self):
        gt   = np.array([0, 1, 1, 0])
        pred = np.zeros(4, dtype=int)
        r = compute_pa_metrics(pred, gt)
        assert r["pa_f1"] == pytest.approx(0.0)


# full_report
class TestFullReport:

    def test_contains_all_keys(self):
        gt   = np.array([0, 1, 1, 0, 1])
        pred = np.array([0, 1, 0, 0, 1])
        r = full_report(pred, gt, name="test_detector")
        for key in ("precision", "recall", "f1", "pa_precision", "pa_recall", "pa_f1", "detector"):
            assert key in r

    def test_name_optional(self):
        gt   = np.array([0, 1])
        pred = np.array([0, 1])
        r = full_report(pred, gt)
        assert "detector" not in r

    def test_values_in_range(self):
        rng  = np.random.default_rng(0)
        gt   = rng.integers(0, 2, size=200)
        pred = rng.integers(0, 2, size=200)
        r    = full_report(pred, gt)
        for key in ("precision", "recall", "f1", "pa_precision", "pa_recall", "pa_f1"):
            assert 0.0 <= r[key] <= 1.0, f"{key} out of range: {r[key]}"
