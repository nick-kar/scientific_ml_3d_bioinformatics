import numpy as np

from src.train_modelnet10 import load_split


TETRAHEDRON = """OFF
4 4 0
0 0 0
1 0 0
0 1 0
0 0 1
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
"""


def make_dataset(root):
    for category in ("class_a", "class_b"):
        folder = root / category / "train"
        folder.mkdir(parents=True)
        (folder / f"{category}_001.off").write_text(
            TETRAHEDRON, encoding="utf-8"
        )


def test_load_split_returns_expected_shapes_and_labels(tmp_path):
    make_dataset(tmp_path)

    points, labels = load_split(
        tmp_path, "train", ["class_a", "class_b"],
        points_per_mesh=32, limit=None, seed=42,
    )

    assert points.shape == (2, 32, 3)
    assert labels.shape == (2,)
    assert labels.tolist() == [0, 1]
    assert np.isfinite(points).all()


def test_point_clouds_are_centered_and_normalized(tmp_path):
    make_dataset(tmp_path)

    points, _ = load_split(
        tmp_path, "train", ["class_a", "class_b"],
        points_per_mesh=32, limit=None, seed=42,
    )

    np.testing.assert_allclose(points.mean(axis=1), 0.0, atol=1e-5)
    norms = np.linalg.norm(points, axis=2)
    assert np.all(norms.max(axis=1) <= 1.00001)
    assert np.allclose(norms.max(axis=1), 1.0, atol=1e-5)


def test_sampling_is_reproducible_for_fixed_seed(tmp_path):
    make_dataset(tmp_path)

    first_points, first_labels = load_split(
        tmp_path, "train", ["class_a", "class_b"],
        points_per_mesh=32, limit=None, seed=123,
    )
    second_points, second_labels = load_split(
        tmp_path, "train", ["class_a", "class_b"],
        points_per_mesh=32, limit=None, seed=123,
    )

    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_labels, second_labels)
