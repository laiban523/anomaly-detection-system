"""
storage/writer.py

Writes detection results to the anomaly_records table in batches.

Design decisions:
  - bulk_insert_mappings is used instead of add_all() for performance;
    it bypasses ORM overhead and is significantly faster for large inserts.
  - Every timestep is written (not just anomalies) so the full score time
    series is available to the dashboard without re-running the pipeline.
  - ground_truth is written per-timestep when the labels array is provided,
    enabling per-row precision/recall evaluation in the dashboard.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from detection.isolation_forest import DetectionResult
from storage.db import AnomalyRecord, get_session

logger = logging.getLogger(__name__)

# Synthetic base timestamp — all timesteps get a deterministic wall-clock time
# so the dashboard has a proper time axis without real timestamps.
_BASE_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)


def write_results(
    results: DetectionResult,
    data: np.ndarray,
    channel: str,
    detector_name: str,
    ground_truth: np.ndarray | None = None,
    start_idx: int = 0,
    session=None,
) -> int:
    """
    Persist a DetectionResult batch to the database.

    Args:
        results:       Output from IsolationForestDetector.predict() or
                       ZScoreDetector.predict().
        data:          Raw feature array for this batch, shape (n, n_features).
        channel:       Channel ID string, e.g. "P-1".
        detector_name: "isolation_forest" or "zscore".
        ground_truth:  Full test-set label array (int8).  Indexed by
                       start_idx + i so slicing is handled here.
        start_idx:     Offset of this batch within the full test array.
        session:       Existing SQLAlchemy session.  A new one is created if
                       None and is closed automatically after the commit.

    Returns:
        Number of rows written.
    """
    own_session = session is None
    if own_session:
        session = get_session()

    n = len(results.predictions)

    rows = []
    for i in range(n):
        global_idx = start_idx + i
        gt_val: bool | None = None
        if ground_truth is not None and global_idx < len(ground_truth):
            gt_val = bool(ground_truth[global_idx])

        rows.append(
            {
                "timestamp": _BASE_TIME + timedelta(minutes=global_idx),
                "channel": channel,
                "detector": detector_name,
                "anomaly_score": float(results.scores[i]),
                "is_anomaly": bool(results.predictions[i]),
                "severity": str(results.severity[i]),
                "raw_values": data[i].tolist() if data is not None else None,
                "ground_truth": gt_val,
                "created_at": datetime.now(timezone.utc),
            }
        )

    try:
        session.bulk_insert_mappings(AnomalyRecord, rows)
        session.commit()
        logger.debug(
            "Wrote %d rows | channel=%s detector=%s start=%d",
            n,
            channel,
            detector_name,
            start_idx,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

    return n
