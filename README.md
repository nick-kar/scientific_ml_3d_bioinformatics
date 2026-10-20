# Scientific ML: 3D Point Clouds & Biomedical Bioinformatics

A research-oriented machine learning project combining **3D geometric deep learning** and **high-dimensional biomedical classification** using PyTorch and scikit-learn.

## Project Overview

This repository explores two distinct scientific ML tasks:

### 1. 3D Point-Cloud Classification

A lightweight PointNet-style neural network for classifying 3D point clouds represented by XYZ coordinates.

**Implemented**
- PyTorch point-cloud feature extraction
- Symmetric max pooling for permutation-invariant global features
- Classification head with dropout
- Training and evaluation utilities
- Input validation and automated tests

**Current status:** The model and utilities are implemented and tested. A real-dataset benchmark and corresponding performance results are not yet available.

### 2. Biomedical Gene-Expression Classification

A three-class leukemia classification experiment using the Golub gene-expression dataset.

**Dataset characteristics**
- 72 samples
- 7,129 gene-expression features per sample
- Three classes: `allB`, `allT`, and `aml`
- No missing gene-expression values in the downloaded CSV

**Models**
- Logistic Regression with class balancing
- RBF Support Vector Machine with class balancing

**Methodology**
- Stratified train/test split with a fixed random seed
- Standardization and univariate feature selection inside scikit-learn pipelines
- Feature selection restricted to the training data during fitting
- Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices
- JSON output for recorded evaluation results

## Initial Results

Results from one stratified 75%/25% train/test split using `random_state=42`:

| Metric | Logistic Regression | RBF SVM |
|---|---:|---:|
| Accuracy | 0.944 | 0.889 |
| Macro-F1 | 0.957 | 0.871 |
| Correct test predictions | 17/18 | 16/18 |

Logistic Regression performed better on this particular split.

**Limitations:** The test set contains only 18 samples, including two `allT` samples. Performance estimates are therefore uncertain and may vary substantially across different splits. These preliminary results are not evidence of clinical readiness or guaranteed generalization to new populations.

## Technology Stack

- Python
- PyTorch
- NumPy and pandas
- scikit-learn
- pytest

## Repository Structure

```text
scientific_ml_3d_bioinformatics/
├── assets/
├── data/
│   └── .gitkeep
├── outputs/
├── src/
│   ├── pointnet.py
│   └── train_pointnet.py
│   └── train_gene_expression.py
├── tests/
│   ├── test_pointnet.py
│   └── test_gene_expression.py
├── .gitignore
├── README.md
└── requirements.txt
```

The downloaded dataset and generated outputs are excluded from version control.

## Setup

Install the dependencies in an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

Run the automated tests:

```bash
python -m pytest -v
```

## Run the Gene-Expression Experiment

Download the Golub CSV dataset from:

https://openintro.org/data/csv/golub.csv

Save it as `data/golub.csv`, then run:

```bash
python -m src.train_gene_expression
```

The evaluation results are written to:

```text
outputs/gene_expression_results.json
```

## Testing

The current test suite contains 10 passing tests covering:
- PointNet output shapes and permutation invariance
- Invalid point-cloud input handling
- Data loading and evaluation utilities
- Classifier pipeline construction and feature selection
- Three-class model fitting and prediction on synthetic data
- Numeric gene-expression feature extraction

Passing automated tests verifies the tested software behavior; it does not establish scientific validity or real-world model performance.

## Future Work

- Benchmark point-cloud classification on a documented public dataset
- Add cross-validation and uncertainty-aware evaluation for gene-expression classification
- Compare feature-selection strategies and model performance
- Add reproducible experiment configurations and visualizations
- Integrate continuous testing with GitHub Actions

## Intended Use

This repository is an educational and research portfolio project demonstrating scientific machine learning, model evaluation, and software testing. It is not a medical diagnostic system.