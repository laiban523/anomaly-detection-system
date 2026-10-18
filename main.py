"""
main.py -- IoT Sensor Anomaly Detection Pipeline

Pipeline stages:
  1. Ingestion  -- load SMAP/MSL channel (train + test splits)
  2. Detection  -- train three detectors on the training split,
                   run all three on the full test set:
                     Isolation Forest   (multivariate, tree-based)
                     Z-Score            (univariate, statistical baseline)
                     LSTM Autoencoder   (multivariate, sequence-aware)
  3. Storage    -- write all results to PostgreSQL in batches
  4. Evaluation -- log raw + point-adjusted F1 for each detector

Environment variables (see .env.example):
  DATABASE_URL, CHANNEL, BATCH_SIZE, IF_CONTAMINATION,
  ZSCORE_WINDOW, ZSCORE_THRESHOLD, LSTM_MODEL_DIR
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

CHANNEL          = os.environ.get("CHANNEL", "P-1")
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "500"))
IF_CONTAMINATION = float(os.environ.get("IF_CONTAMINATION", "0.01"))
ZSCORE_WINDOW    = int(os.environ.get("ZSCORE_WINDOW", "100"))
ZSCORE_THRESHOLD = float(os.environ.get("ZSCORE_THRESHOLD", "3.0"))
LSTM_MODEL_DIR   = os.environ.get("LSTM_MODEL_DIR", "data")


# Evaluation helper
def _evaluate(name: str, predictions: np.ndarray, ground_truth: np.ndarray) -> None:
    """Log raw and point-adjusted precision, recall, and F1."""
    from detection.metrics import full_report
    report = full_report(predictions, ground_truth, name=name)

    logger.info(
        "[%s] Raw    -> Precision=%.3f  Recall=%.3f  F1=%.3f",
        name, report["precision"], report["recall"], report["f1"],
    )
    logger.info(
        "[%s] PA-adj -> Precision=%.3f  Recall=%.3f  F1=%.3f  <- benchmark metric",
        name, report["pa_precision"], report["pa_recall"], report["pa_f1"],
    )

# Pipeline
def run() -> None:
    from ingestion.loader import load_channel
    from ingestion.simulator import stream_data
    from detection.isolation_forest import IsolationForestDetector, DETECTOR_NAME as IF_NAME
    from detection.zscore import ZScoreDetector, DETECTOR_NAME as ZS_NAME
    from detection.lstm_detector import LSTMDetector, DETECTOR_NAME as LSTM_NAME
    from storage.db import init_db, get_engine, get_session
    from storage.writer import write_results

    # Database 
    logger.info("Initialising database schema...")
    engine = get_engine()
    init_db(engine)

    # Ingestion 
    logger.info("Loading channel '%s' from SMAP/MSL dataset...", CHANNEL)
    train_data, test_data, ground_truth = load_channel(CHANNEL)
    logger.info(
        "Shapes -- train: %s | test: %s | anomaly rate in test: %.2f%%",
        train_data.shape, test_data.shape, ground_truth.mean() * 100,
    )

    # Train detectors 
    logger.info("Training Isolation Forest (contamination=%.3f)...", IF_CONTAMINATION)
    if_detector = IsolationForestDetector(contamination=IF_CONTAMINATION)
    if_detector.fit(train_data)

    logger.info("Fitting Z-Score detector (window=%d, threshold=%.1f)...",
                ZSCORE_WINDOW, ZSCORE_THRESHOLD)
    zs_detector = ZScoreDetector(window=ZSCORE_WINDOW, threshold=ZSCORE_THRESHOLD)
    zs_detector.fit(train_data)

    # LSTM: load cached model if it exists, otherwise train and save
    lstm_model_path = str(Path(LSTM_MODEL_DIR) / f"lstm_{CHANNEL}.pt")
    if Path(lstm_model_path).exists():
        logger.info("Loading cached LSTM model from %s...", lstm_model_path)
        lstm_detector = LSTMDetector.load(lstm_model_path)
    else:
        logger.info("Training LSTM Autoencoder (this takes ~2-5 min on CPU)...")
        lstm_detector = LSTMDetector(
            window_size=100,
            hidden_dim=64,
            n_layers=2,
            epochs=30,
            batch_size=64,
            lr=1e-3,
            threshold_percentile=99.5,
            train_step=10,
        )
        lstm_detector.fit(train_data)
        lstm_detector.save(lstm_model_path)

    # Detect on full test set 
    logger.info("Running detection on %d test timesteps...", len(test_data))
    if_results   = if_detector.predict(test_data)
    zs_results   = zs_detector.predict(test_data)
    lstm_results = lstm_detector.predict(test_data)

    # Evaluation (raw + point-adjusted) 
    logger.info("EVALUATION  (PA = point-adjusted, benchmark standard)")
    _evaluate("IsolationForest",  if_results.predictions,   ground_truth)
    _evaluate("ZScore",           zs_results.predictions,   ground_truth)
    _evaluate("LSTMAutoencoder",  lstm_results.predictions, ground_truth)

    # -- 6. Write to PostgreSQL in batches -----------------------------------
    logger.info("Writing results to PostgreSQL (batch_size=%d)...", BATCH_SIZE)
    session = get_session(engine)

    ResultType = type(if_results)

    def _slice(result, s, e):
        return ResultType(
            scores=result.scores[s:e],
            predictions=result.predictions[s:e],
            severity=result.severity[s:e],
        )

    try:
        total_if = total_zs = total_lstm = 0

        for start_idx, batch in stream_data(test_data, batch_size=BATCH_SIZE):
            end_idx = start_idx + len(batch)

            if_batch   = _slice(if_results,   start_idx, end_idx)
            zs_batch   = _slice(zs_results,   start_idx, end_idx)
            lstm_batch = _slice(lstm_results, start_idx, end_idx)

            total_if += write_results(
                results=if_batch, data=batch, channel=CHANNEL,
                detector_name=IF_NAME, ground_truth=ground_truth,
                start_idx=start_idx, session=session,
            )
            total_zs += write_results(
                results=zs_batch, data=batch, channel=CHANNEL,
                detector_name=ZS_NAME, ground_truth=ground_truth,
                start_idx=start_idx, session=session,
            )
            total_lstm += write_results(
                results=lstm_batch, data=batch, channel=CHANNEL,
                detector_name=LSTM_NAME, ground_truth=ground_truth,
                start_idx=start_idx, session=session,
            )

            logger.info(
                "  [%d-%d] IF: %d  ZS: %d  LSTM: %d anomalies",
                start_idx, end_idx - 1,
                int(if_batch.predictions.sum()),
                int(zs_batch.predictions.sum()),
                int(lstm_batch.predictions.sum()),
            )

    except Exception:
        logger.exception("Pipeline failed during DB write.")
        raise
    finally:
        session.close()

    logger.info(
        "Pipeline complete. Wrote IF=%d  ZS=%d  LSTM=%d rows for channel '%s'.",
        total_if, total_zs, total_lstm, CHANNEL,
    )
    logger.info("Dashboard: http://localhost:8501  |  API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    run()
