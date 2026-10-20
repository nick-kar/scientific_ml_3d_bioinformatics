"""Training and evaluation utilities for the PointNet classifier."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pointnet import PointNetClassifier


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Train a classifier using cross-entropy loss."""
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for points, labels in train_loader:
            points = points.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * points.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1:02d}/{epochs} - loss: {average_loss:.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    points: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a trained model on a held-out set."""
    model.eval()

    x = torch.as_tensor(points, dtype=torch.float32)
    y = np.asarray(labels, dtype=np.int64)

    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
    }


def make_loader(
    points: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from point-cloud arrays."""
    x = torch.as_tensor(points, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("points must have shape (samples, points, 3)")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("labels must have one value per point cloud")
    if len(x) == 0:
        raise ValueError("The dataset must not be empty")

    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
    )
