"""
detection/isolation_forest.py

Multivariate anomaly detector using scikit-learn's IsolationForest.

Isolation Forest works by randomly partitioning the feature space and
measuring how quickly each point is isolated.  Anomalies are isolated faster
(shorter average path length) because they occupy sparse, low-density regions.
This makes it well-suited for multivariate sensor data where anomalies manifest
as unusual *combinations* of feature values, not just individual spikes.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DETECTOR_NAME = "isolation_forest"


@dataclass(frozen=True)
class DetectionResult:
    """
    Immutable result container shared by both detectors so the storage layer
    can treat them uniformly.
    """

    scores: np.ndarray        # Normalised anomaly score in [0, 1] (1 = most anomalous)
    predictions: np.ndarray   # int8 array: 1 = anomaly, 0 = normal
    severity: np.ndarray      # object array: 'normal' | 'low' | 'medium' | 'high'


class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest with StandardScaler preprocessing.

    Usage:
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(train_data)        # train_data assumed to be mostly normal
        result = detector.predict(test_data)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, train_data: np.ndarray) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest on training data."""
        logger.info(
            "Fitting IsolationForest on %d samples, %d features, contamination=%.3f",
            *train_data.shape,
            self.contamination,
        )
        X = self._scaler.fit_transform(train_data)
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, data: np.ndarray) -> DetectionResult:
        """
        Score and classify each row in data.

        sklearn's decision_function returns negative scores where more negative
        means more anomalous.  We flip and normalise to [0, 1] so that
        higher = more anomalous, consistent with the Z-score detector.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        X = self._scaler.transform(data)

        # Raw scores: more negative → more anomalous
        raw = self._model.decision_function(X)
        # Flip so higher = more anomalous, normalise to [0, 1]
        flipped = -raw
        lo, hi = flipped.min(), flipped.max()
        scores = (flipped - lo) / (hi - lo + 1e-10)

        # sklearn: -1 anomaly, +1 normal
        sk_pred = self._model.predict(X)
        predictions = np.where(sk_pred == -1, 1, 0).astype(np.int8)

        severity = _scores_to_severity(scores, predictions)

        return DetectionResult(scores=scores, predictions=predictions, severity=severity)


# Shared severity helper (used by both detectors)
def _scores_to_severity(scores: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    Map anomaly scores to severity labels.

    Severity is only assigned to flagged points (predictions == 1).
    The thresholds are percentile-based within the anomalous subset so they
    are data-adaptive rather than hard-coded.
    """
    severity = np.full(len(scores), "normal", dtype=object)
    mask = predictions == 1
    if mask.any():
        anom_scores = scores[mask]
        p33 = np.percentile(anom_scores, 33)
        p66 = np.percentile(anom_scores, 66)
        severity[mask & (scores <= p33)] = "low"
        severity[mask & (scores > p33) & (scores <= p66)] = "medium"
        severity[mask & (scores > p66)] = "high"
    return severity
