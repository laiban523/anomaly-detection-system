import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _has_raw_splits(data_dir):
    return (data_dir / "train").is_dir() and (data_dir / "test").is_dir()


def _load_raw_channel(channel, data_dir):
    train_path = data_dir / "train" / f"{channel}.npy"
    test_path  = data_dir / "test"  / f"{channel}.npy"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Channel '{channel}' not found in train/. "
            "Run list_channels() to see available channels."
        )

    train_data = np.load(train_path).astype(np.float32)
    test_data  = np.load(test_path).astype(np.float32)

    labels_df    = pd.read_csv(data_dir / "labeled_anomalies.csv")
    channel_rows = labels_df[labels_df["chan_id"] == channel]

    test_labels = np.zeros(len(test_data), dtype=np.int8)
    for _, row in channel_rows.iterrows():
        for start, end in ast.literal_eval(row["anomaly_sequences"]):
            test_labels[start:end + 1] = 1

    logger.info(
        "Loaded channel %s | train %s | test %s | anomaly rate %.2f%%",
        channel, train_data.shape, test_data.shape, test_labels.mean() * 100,
    )
    return train_data, test_data, test_labels


def _errors_dir(data_dir):
    p = data_dir / "2018-05-19_15.00.10" / "smoothed_errors"
    return p if p.is_dir() else None


def _load_errors_channel(channel, data_dir):
    errors_path = _errors_dir(data_dir) / f"{channel}.npy"
    if not errors_path.exists():
        raise FileNotFoundError(f"Channel '{channel}' not found in smoothed_errors/.")

    raw = np.load(errors_path).astype(np.float32)
    test_data = raw.reshape(-1, 1)
    n = len(test_data)

    labels_df    = pd.read_csv(data_dir / "labeled_anomalies.csv")
    channel_rows = labels_df[labels_df["chan_id"] == channel]

    test_labels = np.zeros(n, dtype=np.int8)
    for _, row in channel_rows.iterrows():
        offset = int(row["num_values"]) - n
        for start, end in ast.literal_eval(row["anomaly_sequences"]):
            adj_s = max(0, start - offset)
            adj_e = min(n - 1, end - offset)
            if adj_s <= adj_e:
                test_labels[adj_s:adj_e + 1] = 1

    normal = test_data[test_labels == 0]
    if len(normal) < 50:
        normal = test_data[:int(0.6 * n)]
    rng = np.random.default_rng(42)
    train_data = normal[rng.permutation(len(normal))]

    logger.info(
        "Loaded (smoothed_errors) channel %s | train %s | test %s | anomaly rate %.2f%%",
        channel, train_data.shape, test_data.shape, test_labels.mean() * 100,
    )
    return train_data, test_data, test_labels


def _generate_synthetic(channel, n_train=12000, n_test=6000, n_features=25,
                        anomaly_rate=0.05, seed=42):
    logger.warning(
        "No real dataset found -- generating synthetic data for channel '%s'.", channel
    )
    rng = np.random.default_rng(seed)

    def _signal(n, t_offset):
        t   = np.linspace(t_offset, t_offset + 5 * np.pi, n)
        out = np.zeros((n, n_features), dtype=np.float32)
        for i in range(n_features):
            freq      = 0.5 + rng.uniform(0, 0.5)
            phase     = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(0.5, 2.0)
            noise     = rng.normal(0, 0.15, n).astype(np.float32)
            out[:, i] = (amplitude * np.sin(freq * t + phase)).astype(np.float32) + noise
        return out

    train_data  = _signal(n_train, 0.0)
    test_data   = _signal(n_test, 5 * np.pi)
    test_labels = np.zeros(n_test, dtype=np.int8)

    for _ in range(max(1, int(n_test * anomaly_rate / 8))):
        s   = int(rng.integers(0, n_test - 25))
        e   = min(s + int(rng.integers(6, 25)), n_test)
        col = int(rng.integers(0, n_features))
        test_data[s:e, col] += float(rng.choice([-1, 1])) * float(rng.uniform(4, 8))
        test_labels[s:e] = 1

    logger.info(
        "Synthetic channel '%s' | train %s | test %s | anomaly rate %.2f%%",
        channel, train_data.shape, test_data.shape, test_labels.mean() * 100,
    )
    return train_data, test_data, test_labels


def load_channel(channel="P-1", data_dir=_DEFAULT_DATA_DIR):
    """
    Load train data, test data, and ground-truth anomaly labels for a channel.

    Priority: raw train/test .npy files -> smoothed_errors -> synthetic fallback.

    Returns:
        train_data   ndarray (n_train, n_features) float32
        test_data    ndarray (n_test,  n_features) float32
        test_labels  ndarray (n_test,)             int8  (1=anomaly, 0=normal)
    """
    if _has_raw_splits(data_dir):
        try:
            return _load_raw_channel(channel, data_dir)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)

    if _errors_dir(data_dir) is not None:
        try:
            return _load_errors_channel(channel, data_dir)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)

    return _generate_synthetic(channel)


def list_channels(spacecraft=None, data_dir=_DEFAULT_DATA_DIR):
    """Return sorted list of available channel IDs."""
    labels_path = data_dir / "labeled_anomalies.csv"

    if _has_raw_splits(data_dir) and labels_path.exists():
        train_channels = {f.stem for f in (data_dir / "train").glob("*.npy")}
        df = pd.read_csv(labels_path)
        if spacecraft:
            df = df[df["spacecraft"] == spacecraft.upper()]
        return sorted(train_channels & set(df["chan_id"].tolist()))

    if _errors_dir(data_dir) is not None and labels_path.exists():
        err_channels = {f.stem for f in _errors_dir(data_dir).glob("*.npy")}
        df = pd.read_csv(labels_path)
        if spacecraft:
            df = df[df["spacecraft"] == spacecraft.upper()]
        return sorted(err_channels & set(df["chan_id"].tolist()))

    return ["P-1", "S-1", "E-1", "M-1"]
