"""Train leakage-safe classifiers on the Golub leukemia gene-expression data."""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "golub.csv"
OUTPUT_DIR = ROOT / "outputs"


def build_models(n_features: int) -> dict:
    """Return pipelines with preprocessing fitted only on training data."""
    k = min(100, n_features)

    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selection", SelectKBest(score_func=f_classif, k=k)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selection", SelectKBest(score_func=f_classif, k=k)),
                ("classifier", SVC(kernel="rbf", class_weight="balanced")),
            ]
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place golub.csv inside the data directory."
        )

    data = pd.read_csv(DATA_PATH)
    target_column = "cancer"

    if target_column not in data.columns:
        raise ValueError("The dataset must contain a 'cancer' column.")

    # The first six columns contain sample metadata, not gene measurements.
    X = data.iloc[:, 6:].apply(pd.to_numeric, errors="raise")
    y = data[target_column].astype(str)

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values must be handled before training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": "Golub leukemia gene-expression dataset",
        "samples": int(len(data)),
        "gene_features": int(X.shape[1]),
        "class_counts": {
            str(label): int(count)
            for label, count in y.value_counts().items()
        },
        "split": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "random_state": 42,
        },
        "models": {},
    }

    print(f"Dataset: {len(data)} samples, {X.shape[1]} gene features")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("\nTraining and evaluating models...")

    for name, model in build_models(X.shape[1]).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        results["models"][name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=sorted(y.unique()),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=sorted(y.unique())
            ).tolist(),
        }

        print(f"\n{name}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Macro-F1: {macro_f1:.3f}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=sorted(y.unique()),
                zero_division=0,
            )
        )

    results["class_labels"] = sorted(y.unique())

    results_path = OUTPUT_DIR / "gene_expression_results.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()