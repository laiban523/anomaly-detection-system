"""
detection/lstm_detector.py

LSTM Autoencoder anomaly detector for multivariate time-series.

Architecture
------------
Encoder  : 2-layer LSTM  (n_features → hidden_dim)
Decoder  : 2-layer LSTM  (hidden_dim → hidden_dim)
Projection: Linear       (hidden_dim → n_features)

The model is trained on normal (training-split) data to minimise
reconstruction MSE.  At inference time, timesteps whose reconstruction
error exceeds a percentile-based threshold are flagged as anomalies.

Why this matters for SMAP/MSL
------------------------------
Isolation Forest treats each timestep as an independent point — it has
no concept of time.  The LSTM autoencoder learns *temporal structure*:
expected transitions, periodicities, and inter-feature dependencies.
This makes it sensitive to anomalies that are only visible as a deviation
from the expected *sequence*, not as an outlier in the current snapshot.

Usage
-----
    detector = LSTMDetector()
    detector.fit(train_data)                  # train_data: (n, 25) float32
    result   = detector.predict(test_data)    # returns DetectionResult
    detector.save("data/lstm_P-1.pt")

    # Later:
    detector = LSTMDetector.load("data/lstm_P-1.pt")
    result   = detector.predict(test_data)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from detection.isolation_forest import DetectionResult, _scores_to_severity

logger = logging.getLogger(__name__)

DETECTOR_NAME = "lstm_autoencoder"


# Model definition
class _LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.

    Encoder reads the full input window and compresses it to a hidden state.
    The final hidden state is replicated across all decoder timesteps, so
    the decoder must reconstruct the entire input from that bottleneck.
    """

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.encoder  = nn.LSTM(n_features,  hidden_dim, n_layers, batch_first=True)
        self.decoder  = nn.LSTM(hidden_dim,  hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)           # h_n: (n_layers, B, H)
        # Bottleneck: last layer's hidden state repeated for every timestep
        context = h_n[-1].unsqueeze(1).expand(-1, seq_len, -1)   # (B, T, H)
        out, _  = self.decoder(context)
        return self.fc_out(out)                  # (B, T, F)


# Detector
class LSTMDetector:
    """
    LSTM Autoencoder anomaly detector.

    Parameters
    ----------
    window_size          : number of timesteps per sliding window (default 100)
    hidden_dim           : LSTM hidden units (default 64)
    n_layers             : LSTM layers in encoder & decoder (default 2)
    epochs               : training epochs (default 30)
    batch_size           : mini-batch size for training & inference (default 64)
    lr                   : Adam learning rate (default 1e-3)
    threshold_percentile : percentile of training errors used as anomaly
                           threshold (default 99.5)
    train_step           : window stride during training — larger values
                           reduce overlap and speed up training (default 10)
    """

    def __init__(
        self,
        window_size: int           = 100,
        hidden_dim: int            = 64,
        n_layers: int              = 2,
        epochs: int                = 30,
        batch_size: int            = 64,
        lr: float                  = 1e-3,
        threshold_percentile: float = 99.5,
        train_step: int            = 10,
    ) -> None:
        self.window_size           = window_size
        self.hidden_dim            = hidden_dim
        self.n_layers              = n_layers
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.lr                    = lr
        self.threshold_percentile  = threshold_percentile
        self.train_step            = train_step

        self._model:     _LSTMAutoencoder | None = None
        self._mean:      np.ndarray | None       = None
        self._std:       np.ndarray | None       = None
        self._threshold: float                   = 0.0
        self._fitted:    bool                    = False
        self._device = torch.device("cpu")

    # Internal helpers
    def _normalise(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / (self._std + 1e-8)

    def _make_windows(self, data: np.ndarray, step: int = 1) -> np.ndarray:
        """Slide a window of size `window_size` over `data` with given step."""
        n = len(data)
        idx = range(0, n - self.window_size + 1, step)
        return np.stack([data[i : i + self.window_size] for i in idx])

    def _reconstruction_errors(self, data_norm: np.ndarray) -> np.ndarray:
        """
        Compute per-timestep reconstruction MSE.

        Each window's MSE is assigned to its *last* timestep (online setting:
        the model has seen the whole window before flagging the final point).
        Timesteps not covered by any window (the first window_size-1 points)
        are filled with the mean error of covered timesteps.
        """
        self._model.eval()
        n       = len(data_norm)
        errors  = np.zeros(n, dtype=np.float32)
        covered = np.zeros(n, dtype=bool)

        windows = self._make_windows(data_norm, step=1)
        tensor  = torch.tensor(windows, dtype=torch.float32)

        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                batch = tensor[i : i + self.batch_size].to(self._device)
                recon = self._model(batch)
                # MSE averaged over (T, F) for each sample in the batch
                mse = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                for k, err in enumerate(mse):
                    ts = i + k + self.window_size - 1   # last timestep of window
                    if ts < n:
                        errors[ts]  = err
                        covered[ts] = True

        # Fill leading uncovered timesteps with the mean of covered ones
        if covered.any():
            fill = float(errors[covered].mean())
            errors[~covered] = fill

        return errors


    # Public API
    def fit(self, train_data: np.ndarray) -> "LSTMDetector":
        """
        Fit normalisation statistics and train the autoencoder.

        Parameters
        ----------
        train_data : (n_timesteps, n_features) array — should be the normal
                     training split, free of (or with minimal) anomalies.
        """
        n_features = train_data.shape[1]

        # Fit normalisation on training data
        self._mean = train_data.mean(axis=0)
        self._std  = train_data.std(axis=0)
        X_norm     = self._normalise(train_data)

        # Build training windows with stride to limit overlap
        windows = self._make_windows(X_norm, step=self.train_step)
        logger.info(
            "LSTM training: %d windows  (window=%d, stride=%d, features=%d)",
            len(windows), self.window_size, self.train_step, n_features,
        )

        # Initialise model
        self._model = _LSTMAutoencoder(n_features, self.hidden_dim, self.n_layers)
        self._model.to(self._device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        dataset   = torch.tensor(windows, dtype=torch.float32)

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            # Shuffle each epoch
            perm    = torch.randperm(len(dataset))
            dataset = dataset[perm]
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i : i + self.batch_size].to(self._device)
                optimiser.zero_grad()
                loss = criterion(self._model(batch), batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

            if epoch % 10 == 0 or epoch == self.epochs:
                logger.info(
                    "LSTM epoch [%02d/%d]  avg_loss=%.6f",
                    epoch, self.epochs, epoch_loss / max(n_batches, 1),
                )

        # Set anomaly threshold from training reconstruction errors
        train_errors     = self._reconstruction_errors(X_norm)
        self._threshold  = float(np.percentile(train_errors, self.threshold_percentile))
        logger.info(
            "LSTM anomaly threshold (p%.1f of train errors): %.6f",
            self.threshold_percentile, self._threshold,
        )

        self._fitted = True
        return self

    def predict(self, data: np.ndarray) -> DetectionResult:
        """
        Score each timestep and return a DetectionResult.

        Parameters
        ----------
        data : (n_timesteps, n_features) array — test / production data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        data_norm = self._normalise(data)
        errors    = self._reconstruction_errors(data_norm)

        # Normalise reconstruction errors to [0, 1] as anomaly score
        lo, hi = errors.min(), errors.max()
        scores = (errors - lo) / (hi - lo + 1e-10)

        predictions = (errors > self._threshold).astype(np.int8)
        severity    = _scores_to_severity(scores, predictions)

        return DetectionResult(scores=scores, predictions=predictions, severity=severity)

    # Persistence
    def save(self, path: str) -> None:
        """Serialise model weights + metadata to a .pt checkpoint."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted — nothing to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "mean":        self._mean,
                "std":         self._std,
                "threshold":   self._threshold,
                "n_features":  int(self._mean.shape[0]),
                "config": {
                    "window_size":          self.window_size,
                    "hidden_dim":           self.hidden_dim,
                    "n_layers":             self.n_layers,
                    "threshold_percentile": self.threshold_percentile,
                },
            },
            path,
        )
        logger.info("LSTM checkpoint saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMDetector":
        """Restore a LSTMDetector from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]

        detector = cls(
            window_size          = cfg["window_size"],
            hidden_dim           = cfg["hidden_dim"],
            n_layers             = cfg["n_layers"],
            threshold_percentile = cfg["threshold_percentile"],
        )
        detector._model = _LSTMAutoencoder(
            ckpt["n_features"], cfg["hidden_dim"], cfg["n_layers"]
        )
        detector._model.load_state_dict(ckpt["model_state"])
        detector._model.eval()
        detector._mean      = ckpt["mean"]
        detector._std       = ckpt["std"]
        detector._threshold = ckpt["threshold"]
        detector._fitted    = True
        logger.info("LSTM checkpoint loaded ← %s", path)
        return detector
