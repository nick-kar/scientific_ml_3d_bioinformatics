"""Lightweight PointNet-style network for 3D point-cloud classification."""

import torch
from torch import nn


class PointNetClassifier(nn.Module):
    """Classify point clouds represented by (x, y, z) coordinates."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Tensor shaped (batch_size, num_points, 3).

        Returns:
            Class logits shaped (batch_size, num_classes).
        """
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "Expected points shaped (batch_size, num_points, 3)."
            )

        point_features = self.point_encoder(points)
        global_features = torch.max(point_features, dim=1).values

        return self.classifier(global_features)
