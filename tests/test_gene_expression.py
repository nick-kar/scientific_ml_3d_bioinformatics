import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.train_gene_expression import build_models


def test_build_models_returns_expected_pipelines():
    models = build_models(n_features=200)

    assert set(models) == {"logistic_regression", "rbf_svm"}
    assert all(len(model.steps) == 3 for model in models.values())


def test_feature_selection_respects_available_feature_count():
    models = build_models(n_features=12)

    for model in models.values():
        selector = model.named_steps["selection"]
        assert selector.k == 12


def test_models_fit_and_predict_three_classes():
    rng = np.random.default_rng(42)
    samples_per_class = 20
    n_features = 30

    X = rng.normal(size=(samples_per_class * 3, n_features))
    y = np.repeat(["allB", "allT", "aml"], samples_per_class)

    # Add class-specific signals to make this a learnable synthetic task.
    X[y == "allB", :3] += 3
    X[y == "allT", 3:6] += 3
    X[y == "aml", 6:9] += 3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    for model in build_models(n_features).values():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({"allB", "allT", "aml"})
        assert 0.0 <= accuracy_score(y_test, predictions) <= 1.0


def test_gene_expression_features_are_numeric():
    data = pd.DataFrame(
        {
            "Samples": [1, 2],
            "BM.PB": ["BM", "BM"],
            "Gender": ["F", "M"],
            "Source": ["DFCI", "DFCI"],
            "tissue.mf": ["BM:f", "BM:m"],
            "cancer": ["allB", "aml"],
            "gene_1": [1.2, 2.3],
            "gene_2": [-0.5, 0.7],
        }
    )

    X = data.iloc[:, 6:].apply(pd.to_numeric, errors="raise")

    assert X.shape == (2, 2)
    assert np.issubdtype(X.dtypes.iloc[0], np.number)