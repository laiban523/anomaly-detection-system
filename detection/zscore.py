"""
detection/zscore.py

Univariate rolling Z-score anomaly detector.

For each timestep the detector computes a rolling mean and standard deviation
over the preceding `window` timesteps for every feature dimension.  A point is
flagged when the maximum |z-score| across all features exceeds `threshold`.

Contrast with Isolation Forest:
  - Z-score catches *univariate spikes* — a single feature deviating sharply.
  - Isolation Forest catches *multivariate anomalies* — unusual combinations
    that might not look extreme in any single dimension.

Running both in parallel on the same data and comparing their flags is a
meaningful analytical exercise, not just feature-padding.
"""

import logging

import numpy as np
import pandas as pd

from detection.isolation_forest import DetectionResult, _scores_to_severity

logger = logging.getLogger(__name__)

DETECTOR_NAME = "zscore"


class ZScoreDetector:
    """
    Rolling Z-score detector with train-set baseline statistics.

    The rolling window uses min_periods=1 so every timestep, even the first,
    receives a score (using whatever data is available in the window).
    When std is zero or NaN the training-set std is substituted to avoid
    division-by-zero producing NaN scores.
    """

    def __init__(self, window: int = 100, threshold: float = 3.0) -> None:
        self.window = window
        self.threshold = threshold
        self._train_std: np.ndarray | None = None

    def fit(self, train_data: np.ndarray) -> "ZScoreDetector":
        """Store per-feature standard deviation from training data as fallback."""
        self._train_std = train_data.std(axis=0).astype(np.float32) + 1e-10
        logger.info(
            "ZScoreDetector fitted | window=%d | threshold=%.1f | features=%d",
            self.window,
            self.threshold,
            train_data.shape[1],
        )
        return self

    def predict(self, data: np.ndarray) -> DetectionResult:
        """
        Compute per-timestep max |z-score| across all feature dimensions.

        Returns a DetectionResult where:
          - scores   = max |z-score| at each timestep (not normalised to [0,1]
                       intentionally, so raw z-values remain interpretable)
          - predictions = 1 where scores > threshold
          - severity  = percentile-bucketed within flagged points
        """
        if self._train_std is None:
            raise RuntimeError("Call fit() before predict().")

        df = pd.DataFrame(data.astype(np.float64))
        max_z = np.zeros(len(data), dtype=np.float64)

        for col_idx in df.columns:
            series = df[col_idx]
            roll_mean = series.rolling(window=self.window, min_periods=1).mean()
            roll_std = (
                series.rolling(window=self.window, min_periods=1)
                .std()
                .fillna(float(self._train_std[col_idx]))
            )
            # Replace zero std with training-set fallback to stay numerically stable
            roll_std = roll_std.where(roll_std > 1e-10, float(self._train_std[col_idx]))

            z = ((series - roll_mean) / roll_std).abs()
            max_z = np.maximum(max_z, z.values)

        predictions = (max_z > self.threshold).astype(np.int8)
        severity = _scores_to_severity(max_z, predictions)

        return DetectionResult(
            scores=max_z,
            predictions=predictions,
            severity=severity,
        )
