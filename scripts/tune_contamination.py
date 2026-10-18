"""
scripts/tune_contamination.py

Sensitivity analysis for Isolation Forest's IF_CONTAMINATION hyperparameter.

Sweeps contamination values over the training/test split of a given channel
and reports PA-F1, PA-Precision, PA-Recall, and Raw F1 for each value.

Usage:
    python scripts/tune_contamination.py              # uses default channel P-1
    python scripts/tune_contamination.py --channel S-1
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ingestion.loader import load_channel
from detection.isolation_forest import IsolationForestDetector
from detection.metrics import full_report

CONTAMINATION_VALUES = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


def run(channel: str) -> None:
    print(f"\nLoading channel '{channel}'...")
    train, test, gt = load_channel(channel)
    anomaly_rate = gt.mean() * 100
    print(f"Anomaly rate in test: {anomaly_rate:.2f}%")
    print(f"Train shape: {train.shape} | Test shape: {test.shape}\n")

    header = f"{'Contamination':>15}  {'PA-F1':>7}  {'PA-Prec':>8}  {'PA-Rec':>7}  {'Raw-F1':>7}"
    print(header)
    print("-" * len(header))

    best_f1, best_c = 0.0, None
    for c in CONTAMINATION_VALUES:
        det = IsolationForestDetector(contamination=c)
        det.fit(train)
        res = det.predict(test)
        r = full_report(res.predictions, gt)

        marker = " <-- best" if r["pa_f1"] > best_f1 else ""
        if r["pa_f1"] > best_f1:
            best_f1, best_c = r["pa_f1"], c
            marker = " <-- best"

        print(
            f"{c:>15.2f}  {r['pa_f1']:>7.3f}  {r['pa_precision']:>8.3f}"
            f"  {r['pa_recall']:>7.3f}  {r['f1']:>7.3f}{marker}"
        )

    print(f"\nBest contamination for channel '{channel}': {best_c} (PA-F1={best_f1:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep IF_CONTAMINATION values.")
    parser.add_argument("--channel", default="P-1", help="SMAP/MSL channel ID")
    args = parser.parse_args()
    run(args.channel)
