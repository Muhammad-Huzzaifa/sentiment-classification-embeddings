"""Model definitions."""

from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_STATE


class SklearnClassifier:
    """Simple logistic regression wrapper."""

    def __init__(self) -> None:
        """Initializes classifier."""
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits model.

        Args:
            x_train: Train features.
            y_train: Train labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Runs inference.

        Args:
            x_test: Test features.

        Returns:
            Predicted labels.
        """
        return self.model.predict(x_test)

    def save(self, filepath: str | Path) -> None:
        """Saves model.

        Args:
            filepath: Output path.
        """
        joblib.dump(self.model, Path(filepath))


class RNNClassifier(nn.Module):
    """LSTM classifier for sequence inputs."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 1,
    ) -> None:
        """Initializes network.

        Args:
            embedding_dim: Input embedding size.
            hidden_dim: LSTM hidden size.
            output_dim: Number of classes.
            num_layers: Number of LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass.

        Args:
            x: Input tensor shaped (batch, seq_len, emb_dim).

        Returns:
            Logits shaped (batch, output_dim).
        """
        lengths = (x.abs().sum(dim=2) > 0).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])
