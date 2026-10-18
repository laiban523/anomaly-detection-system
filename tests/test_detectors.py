"""
tests/test_detectors.py

Smoke tests for the three anomaly detectors.

These tests verify:
  - fit/predict contract (shapes, dtypes, value ranges)
  - correct RuntimeError when predict called before fit
  - severity labels are always valid strings
  - LSTM save/load round-trip preserves threshold
"""

import numpy as np
import pytest
import tempfile
import os

from detection.isolation_forest import IsolationForestDetector, DetectionResult
from detection.zscore import ZScoreDetector
from detection.lstm_detector import LSTMDetector


# Shared fixtures
N_TRAIN  = 300
N_TEST   = 100
N_FEATS  = 6
RNG      = np.random.default_rng(42)

TRAIN = RNG.standard_normal((N_TRAIN, N_FEATS)).astype(np.float32)
TEST  = RNG.standard_normal((N_TEST,  N_FEATS)).astype(np.float32)

VALID_SEVERITIES = {"normal", "low", "medium", "high"}


def _check_result(result: DetectionResult, n: int) -> None:
    """Shared assertions for any DetectionResult."""
    assert result.scores.shape      == (n,), "scores shape mismatch"
    assert result.predictions.shape == (n,), "predictions shape mismatch"
    assert result.severity.shape    == (n,), "severity shape mismatch"
    assert result.predictions.dtype == np.int8
    assert set(np.unique(result.predictions)).issubset({0, 1}), "predictions not binary"
    assert np.all(result.scores >= 0), "scores contain negatives"
    assert set(result.severity).issubset(VALID_SEVERITIES), "invalid severity label"



# Isolation Forest
class TestIsolationForestDetector:

    def test_fit_predict_shapes(self):
        det = IsolationForestDetector(contamination=0.05)
        det.fit(TRAIN)
        result = det.predict(TEST)
        _check_result(result, N_TEST)

    def test_scores_normalised(self):
        det = IsolationForestDetector(contamination=0.05)
        det.fit(TRAIN)
        result = det.predict(TEST)
        assert result.scores.max() <= 1.0 + 1e-6
        assert result.scores.min() >= 0.0 - 1e-6

    def test_predict_before_fit_raises(self):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(TEST)

    def test_contamination_affects_anomaly_rate(self):
        """Higher contamination → more flagged points."""
        det_low  = IsolationForestDetector(contamination=0.01)
        det_high = IsolationForestDetector(contamination=0.20)
        det_low.fit(TRAIN)
        det_high.fit(TRAIN)
        rate_low  = det_low.predict(TEST).predictions.mean()
        rate_high = det_high.predict(TEST).predictions.mean()
        assert rate_high > rate_low

# Z-Score
class TestZScoreDetector:

    def test_fit_predict_shapes(self):
        det = ZScoreDetector(window=20, threshold=3.0)
        det.fit(TRAIN)
        result = det.predict(TEST)
        _check_result(result, N_TEST)

    def test_predict_before_fit_raises(self):
        det = ZScoreDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(TEST)

    def test_obvious_spike_flagged(self):
        """Insert a large spike — should be flagged regardless of threshold."""
        train = np.zeros((200, 1), dtype=np.float32)
        test  = np.zeros((50,  1), dtype=np.float32)
        test[25, 0] = 100.0   # enormous spike

        det = ZScoreDetector(window=20, threshold=3.0)
        det.fit(train)
        result = det.predict(test)
        assert result.predictions[25] == 1, "spike at index 25 not flagged"

    def test_constant_series_no_flags(self):
        """Perfectly constant series should produce no anomaly flags."""
        train = np.ones((200, 3), dtype=np.float32)
        test  = np.ones((50,  3), dtype=np.float32)
        det = ZScoreDetector(window=20, threshold=3.0)
        det.fit(train)
        result = det.predict(test)
        assert result.predictions.sum() == 0, "constant series should not be flagged"

# LSTM Autoencoder
class TestLSTMDetector:

    def test_fit_predict_shapes(self):
        det = LSTMDetector(
            window_size=20, hidden_dim=8, n_layers=1,
            epochs=2, batch_size=16, train_step=5,
        )
        det.fit(TRAIN)
        result = det.predict(TEST)
        _check_result(result, N_TEST)

    def test_predict_before_fit_raises(self):
        det = LSTMDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(TEST)

    def test_save_load_preserves_threshold(self):
        det = LSTMDetector(
            window_size=20, hidden_dim=8, n_layers=1,
            epochs=2, batch_size=16, train_step=5,
        )
        det.fit(TRAIN)
        original_threshold = det._threshold

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lstm_test.pt")
            det.save(path)
            loaded = LSTMDetector.load(path)

        assert loaded._threshold == pytest.approx(original_threshold, rel=1e-5)
        assert loaded._fitted is True

    def test_save_load_consistent_predictions(self):
        """Loaded model should produce identical predictions to original."""
        det = LSTMDetector(
            window_size=20, hidden_dim=8, n_layers=1,
            epochs=2, batch_size=16, train_step=5,
        )
        det.fit(TRAIN)
        preds_original = det.predict(TEST).predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lstm_test.pt")
            det.save(path)
            loaded = LSTMDetector.load(path)

        preds_loaded = loaded.predict(TEST).predictions
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_save_before_fit_raises(self):
        det = LSTMDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            det.save("should_not_exist.pt")
