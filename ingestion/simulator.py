"""
ingestion/simulator.py

Simulates a streaming data source by yielding fixed-size batches from a
pre-loaded numpy array.  In a real system this would be replaced by a Kafka
consumer or an MQTT subscriber; here it lets the pipeline code stay
stream-shaped (batch-oriented generator) without requiring live hardware.
"""

import time
from collections.abc import Generator

import numpy as np


def stream_data(
    data: np.ndarray,
    batch_size: int = 500,
    delay: float = 0.0,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (start_index, batch) tuples, optionally sleeping between batches.

    Args:
        data:       Array of shape (n_timesteps, n_features).
        batch_size: Number of rows per batch.
        delay:      Seconds to sleep between batches.  0 = no artificial delay.

    Yields:
        (start_idx, batch) where batch has shape (<=batch_size, n_features).
    """
    n = len(data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, data[start:end]
        if delay > 0:
            time.sleep(delay)
