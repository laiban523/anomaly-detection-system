"""
detection/metrics.py

Point-adjusted evaluation metrics for time-series anomaly detection.

Standard raw precision/recall is misleading on datasets like SMAP/MSL
because anomalies come in contiguous multi-timestep windows.  A detector
that fires once inside a 50-timestep anomaly window correctly detected the
anomaly, but raw metrics penalise the other 49 un-flagged points.

The point-adjust protocol (Xu et al. 2018, used in almost every SMAP/MSL
benchmark paper) fixes this:

    If ANY predicted point falls inside a contiguous ground-truth anomaly
    segment, ALL points in that segment are counted as correctly detected.

This file provides:
  - point_adjust()         — adjust raw predictions using the protocol above
  - compute_pa_metrics()   — precision, recall, F1 on point-adjusted preds
  - compute_raw_metrics()  — standard (unadjusted) metrics for comparison
"""

from __future__ import annotations

import numpy as np


# Core protocol
def point_adjust(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Apply the point-adjust protocol to binary predictions.

    Parameters
    ----------
    y_pred : 1-D int/bool array — raw detector predictions (1=anomaly)
    y_true : 1-D int/bool array — ground-truth labels (1=anomaly)

    Returns
    -------
    y_adj : 1-D int8 array — adjusted predictions
    """
    pred = np.asarray(y_pred, dtype=bool)
    gt   = np.asarray(y_true, dtype=bool)
    adj  = pred.copy()

    # Find contiguous anomaly segments in ground truth via diff trick
    padded = np.concatenate(([False], gt, [False]))
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]   # segment start indices (inclusive)
    ends   = np.where(diff == -1)[0]   # segment end indices (exclusive)

    for s, e in zip(starts, ends):
        if pred[s:e].any():            # any detection inside the segment?
            adj[s:e] = True            # credit the whole segment

    return adj.astype(np.int8)


# Metric helpers
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def compute_pa_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """
    Point-adjusted precision, recall, and F1.

    Returns
    -------
    dict with keys: pa_precision, pa_recall, pa_f1
    """
    adj  = point_adjust(y_pred, y_true)
    gt   = np.asarray(y_true, dtype=bool)
    pred = adj.astype(bool)

    tp = int(( pred &  gt).sum())
    fp = int(( pred & ~gt).sum())
    fn = int((~pred &  gt).sum())

    p, r, f = _prf(tp, fp, fn)
    return {"pa_precision": p, "pa_recall": r, "pa_f1": f}


def compute_raw_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """
    Standard (unadjusted) precision, recall, and F1.

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    pred = np.asarray(y_pred, dtype=bool)
    gt   = np.asarray(y_true, dtype=bool)

    tp = int(( pred &  gt).sum())
    fp = int(( pred & ~gt).sum())
    fn = int((~pred &  gt).sum())

    p, r, f = _prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f}


def full_report(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    name: str = "",
) -> dict[str, float]:
    """Return both raw and point-adjusted metrics in one dict."""
    raw = compute_raw_metrics(y_pred, y_true)
    pa  = compute_pa_metrics(y_pred, y_true)
    report = {**raw, **pa}
    if name:
        report["detector"] = name
    return report
