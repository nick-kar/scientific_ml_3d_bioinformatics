"""Train and evaluate PointNet on the official ModelNet10 split."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pointnet import PointNetClassifier


def load_split(root, split, classes, points_per_mesh, limit, seed):
    point_clouds, labels = [], []
    rng = random.Random(seed)

    for label, category in enumerate(classes):
        files = sorted((root / category / split).glob("*.off"))
        if limit is not None and len(files) > limit:
            files = sorted(rng.sample(files, limit))

        for path in files:
            mesh = trimesh.load(path, force="mesh", process=False)
            if len(mesh.faces) == 0:
                raise ValueError(f"Mesh has no faces: {path}")

            points, _ = trimesh.sample.sample_surface(
                mesh,
                points_per_mesh,
                seed=rng.randrange(0, 2**32),
            )

            # Center each point cloud and scale it to a unit sphere.
            points = points.astype(np.float32)
            points -= points.mean(axis=0, keepdims=True)
            scale = np.linalg.norm(points, axis=1).max()
            if scale > 0:
                points /= scale

            point_clouds.append(points)
            labels.append(label)

    if not point_clouds:
        raise ValueError(f"No meshes found for split '{split}'.")

    return np.stack(point_clouds), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ModelNet10/ModelNet10")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/modelnet10_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    root = Path(args.data)
    classes = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()
    )
    if len(classes) != 10:
        raise ValueError(
            f"Expected 10 ModelNet classes, found {len(classes)} in {root}"
        )

    print("Classes:", classes, flush=True)
    print("Loading training meshes...", flush=True)
    x_train, y_train = load_split(
        root, "train", classes, args.points,
        args.max_train_per_class, args.seed,
    )
    print("Loading test meshes...", flush=True)
    x_test, y_test = load_split(
        root, "test", classes, args.points,
        args.max_test_per_class, args.seed + 1,
    )

    print(f"Train: {x_train.shape}; test: {x_test.shape}", flush=True)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = PointNetClassifier(num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for points, labels in train_loader:
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- loss: {total_loss / len(y_train):.4f}",
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        predictions = logits.argmax(dim=1).numpy()

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "dataset": "ModelNet10",
        "split": "official train/test",
        "seed": args.seed,
        "points_per_mesh": args.points,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "classes": classes,
        "epochs": args.epochs,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "limitations": (
            "PointNet-style baseline; single seed; no hyperparameter tuning. "
            "Use full official splits for final reported results."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print("\nPer-class results:")
    for category in classes:
        row = report[category]
        print(
            f"{category:12s} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1-score']:.3f} "
            f"n={row['support']}"
        )
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
