"""
api/routes.py

FastAPI application exposing anomaly detection results via REST.

Endpoints:
  GET /health              — liveness check
  GET /channels            — list channels present in DB
  GET /anomalies           — paginated anomaly records with filtering
  GET /anomalies/{id}      — single record by primary key
  GET /stats               — aggregate counts and anomaly rate
  GET /comparison          — side-by-side detector stats for one channel
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func

from storage.db import AnomalyRecord, get_session

app = FastAPI(
    title="IoT Anomaly Detection API",
    version="1.0.0",
    description="Query anomaly records produced by the SMAP/MSL detection pipeline.",
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class AnomalyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    channel: str
    detector: str
    anomaly_score: float
    is_anomaly: bool
    severity: str
    ground_truth: Optional[bool]
    created_at: datetime


class StatsResponse(BaseModel):
    channel: Optional[str]
    detector: Optional[str]
    total_records: int
    anomaly_count: int
    anomaly_rate: float
    severity_counts: dict[str, int]
    precision: Optional[float]   # TP / (TP + FP) — None if no ground truth
    recall: Optional[float]      # TP / (TP + FN) — None if no ground truth


class ComparisonResponse(BaseModel):
    channel: str
    isolation_forest: StatsResponse
    zscore: StatsResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stats(session, channel: str | None, detector: str | None) -> StatsResponse:
    q = session.query(AnomalyRecord)
    if channel:
        q = q.filter(AnomalyRecord.channel == channel)
    if detector:
        q = q.filter(AnomalyRecord.detector == detector)

    total = q.count()
    anom_q = q.filter(AnomalyRecord.is_anomaly.is_(True))
    anomaly_count = anom_q.count()

    sev_rows = (
        session.query(AnomalyRecord.severity, func.count(AnomalyRecord.id))
        .filter(AnomalyRecord.is_anomaly.is_(True))
    )
    if channel:
        sev_rows = sev_rows.filter(AnomalyRecord.channel == channel)
    if detector:
        sev_rows = sev_rows.filter(AnomalyRecord.detector == detector)
    severity_counts = {k: v for k, v in sev_rows.group_by(AnomalyRecord.severity).all()}

    # Precision / recall against ground truth where available
    gt_q = q.filter(AnomalyRecord.ground_truth.isnot(None))
    precision = recall = None
    if gt_q.count() > 0:
        tp = q.filter(AnomalyRecord.is_anomaly.is_(True),  AnomalyRecord.ground_truth.is_(True)).count()
        fp = q.filter(AnomalyRecord.is_anomaly.is_(True),  AnomalyRecord.ground_truth.is_(False)).count()
        fn = q.filter(AnomalyRecord.is_anomaly.is_(False), AnomalyRecord.ground_truth.is_(True)).count()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return StatsResponse(
        channel=channel,
        detector=detector,
        total_records=total,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_count / total if total > 0 else 0.0,
        severity_counts=severity_counts,
        precision=precision,
        recall=recall,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/channels", response_model=list[str])
def get_channels():
    """Return distinct channel IDs present in the database."""
    session = get_session()
    try:
        rows = session.query(AnomalyRecord.channel).distinct().order_by(AnomalyRecord.channel).all()
        return [r[0] for r in rows]
    finally:
        session.close()


@app.get("/anomalies", response_model=list[AnomalyResponse])
def get_anomalies(
    channel: Optional[str] = Query(None, description="Filter by channel ID"),
    detector: Optional[str] = Query(None, description="'isolation_forest' or 'zscore'"),
    severity: Optional[str] = Query(None, description="'low', 'medium', or 'high'"),
    anomalies_only: bool = Query(True, description="Return only flagged rows"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    session = get_session()
    try:
        q = session.query(AnomalyRecord)
        if channel:
            q = q.filter(AnomalyRecord.channel == channel)
        if detector:
            q = q.filter(AnomalyRecord.detector == detector)
        if severity:
            q = q.filter(AnomalyRecord.severity == severity)
        if anomalies_only:
            q = q.filter(AnomalyRecord.is_anomaly.is_(True))
        q = q.order_by(AnomalyRecord.timestamp).offset(offset).limit(limit)
        return q.all()
    finally:
        session.close()


@app.get("/anomalies/{record_id}", response_model=AnomalyResponse)
def get_anomaly(record_id: int):
    session = get_session()
    try:
        record = session.get(AnomalyRecord, record_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
        return record
    finally:
        session.close()


@app.get("/stats", response_model=StatsResponse)
def get_stats(
    channel: Optional[str] = Query(None),
    detector: Optional[str] = Query(None),
):
    session = get_session()
    try:
        return _build_stats(session, channel, detector)
    finally:
        session.close()


@app.get("/comparison/{channel}", response_model=ComparisonResponse)
def get_comparison(channel: str):
    """
    Side-by-side stats for Isolation Forest vs Z-Score on the same channel.
    The meaningful difference: IF catches multivariate anomalies, Z-score
    catches univariate spikes.  This endpoint makes that explicit.
    """
    session = get_session()
    try:
        return ComparisonResponse(
            channel=channel,
            isolation_forest=_build_stats(session, channel, "isolation_forest"),
            zscore=_build_stats(session, channel, "zscore"),
        )
    finally:
        session.close()
