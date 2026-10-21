# Scientific ML: 3D Point Clouds & Biomedical Bioinformatics

Two applied machine-learning pipelines exploring 3D geometric deep learning and biomedical gene-expression classification.

## Projects

### 1. 3D Point-Cloud Classification — ModelNet10

Implemented a lightweight PointNet-style classifier in PyTorch for classifying 3D objects from sampled mesh-surface points.

**Pipeline**
- Load ModelNet10 `.off` meshes using `trimesh`.
- Sample 512 points from each mesh surface.
- Center and normalize each point cloud.
- Encode point features with a shared multilayer perceptron and aggregate them using symmetric max pooling.
- Train with cross-entropy loss and Adam; evaluate using accuracy, macro-F1, and per-class precision/recall/F1.
- Preserve the dataset's official training and test split.

**Full-split results**

| Metric | Result |
|---|---:|
| Training objects | 3,991 |
| Test objects | 908 |
| Classes | 10 |
| Points per object | 512 |
| Training epochs | 15 |
| Test accuracy | 84.36% |
| Test macro-F1 | 0.8350 |

The experiment uses a single seed and a lightweight PointNet-style architecture. These results are from this implementation and configuration; they should not be interpreted as reproducing the original PointNet paper's benchmark.

### 2. Biomedical Gene-Expression Classification — Golub Dataset

Compared Logistic Regression and RBF-kernel SVM pipelines for classifying leukemia-related gene-expression profiles.

**Pipeline**
- Extract numeric gene-expression features from the dataset.
- Split data using a stratified train/test split.
- Standardize features and select up to 100 features using `SelectKBest` with an ANOVA F-test.
- Train and compare Logistic Regression and RBF SVM classifiers with balanced class weights.
- Evaluate with accuracy, macro-F1, and per-class metrics.

**Held-out results**

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 94.4% | 0.957 |
| RBF SVM | 88.9% | 0.871 |

The test split contains only 18 samples, including just two `allT` samples. Results are therefore preliminary and sensitive to the split; they do not establish clinical validity or generalization to new patient populations.

## Tech Stack

- Python
- PyTorch
- scikit-learn
- NumPy and pandas
- trimesh
- pytest
- GitHub Actions

## Repository Structure

```text
.
├── .github/workflows/tests.yml
├── data/                         # Local datasets; not tracked by Git
├── outputs/                      # Local experiment results; not tracked by Git
├── src/
│   ├── pointnet.py
│   ├── train_pointnet.py
│   ├── train_modelnet10.py
│   └── train_gene_expression.py
├── tests/
│   ├── test_pointnet.py
│   ├── test_modelnet10.py
│   └── test_gene_expression.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.9 or later.

```bash
python -m pip install -r requirements.txt
```

### ModelNet10 data

Download ModelNet10 from the [official Princeton dataset page](https://modelnet.cs.princeton.edu/download.html) and extract it so the directory structure is:

```text
data/ModelNet10/ModelNet10/
├── bathtub/
│   ├── train/
│   └── test/
├── bed/
├── chair/
└── ... other categories
```

The extracted dataset is intentionally excluded from Git. Download and use the dataset in accordance with its original terms.

Run the full 3D experiment:

```bash
python -m src.train_modelnet10 --points 512 --epochs 15 --batch-size 32 --output outputs/modelnet10_results.json
```

Run a smaller pilot:

```bash
python -m src.train_modelnet10 --max-train-per-class 100 --max-test-per-class 20 --points 512 --epochs 10 --batch-size 16 --output outputs/modelnet10_pilot_results.json
```

### Gene-expression experiment

Place the Golub CSV file at `data/golub.csv`, then run:

```bash
python -m src.train_gene_expression
```

The exact supported command-line options are defined in each script.

## Tests

Run all automated tests:

```bash
python -m pytest -v
```

The test suite covers model output and permutation invariance, data validation, gene-expression pipelines, mesh loading, point-cloud normalization, and reproducible sampling.

## Limitations

- The 3D model is a compact PointNet-style baseline, not a reproduction of the full original architecture.
- ModelNet10 results use one seed and one training configuration; no hyperparameter search or multi-seed confidence intervals were performed.
- The gene-expression experiment uses a single small held-out test split. Cross-validation and external validation would be needed for stronger conclusions.
- Neither pipeline is intended for clinical decision-making.
