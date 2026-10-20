import numpy as np
import pytest
import torch

from src.pointnet import PointNetClassifier
from src.train_pointnet import evaluate_model, make_loader


def test_model_output_shape():
    model = PointNetClassifier(num_classes=4)
    points = torch.rand(2, 128, 3)

    output = model(points)

    assert output.shape == (2, 4)


def test_model_is_invariant_to_point_order():
    torch.manual_seed(42)
    model = PointNetClassifier(num_classes=3)
    model.eval()

    points = torch.rand(2, 64, 3)
    permutation = torch.randperm(64)

    with torch.no_grad():
        original = model(points)
        shuffled = model(points[:, permutation, :])

    assert torch.allclose(original, shuffled, atol=1e-6)


def test_model_rejects_invalid_input_shape():
    model = PointNetClassifier()

    with pytest.raises(ValueError):
        model(torch.rand(2, 64, 2))


def test_make_loader():
    points = np.random.rand(8, 32, 3).astype(np.float32)
    labels = np.array([0, 1] * 4)

    loader = make_loader(points, labels, batch_size=4, shuffle=False)
    batch_points, batch_labels = next(iter(loader))

    assert batch_points.shape == (4, 32, 3)
    assert batch_labels.shape == (4,)


def test_make_loader_rejects_mismatched_labels():
    points = np.random.rand(4, 32, 3).astype(np.float32)
    labels = np.array([0, 1])

    with pytest.raises(ValueError):
        make_loader(points, labels)


def test_evaluation_returns_valid_metrics():
    torch.manual_seed(42)
    model = PointNetClassifier(num_classes=2)
    points = np.random.rand(6, 32, 3).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1])

    metrics = evaluate_model(model, points, labels)

    assert set(metrics) == {"accuracy", "macro_f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
