# MIT Data Science & ML — Cheatsheet

A clean, topic-by-topic summary of the **MIT Professional Education — Applied Data Science Program** (Great Learning). The original course work lives in Google Drive (`MIT GT - Data Science`); this repo is the **cleaned-up version**.

Each topic folder has:
- **`<topic>.md`**: theory: what each model is, when to use it, key hyperparameters, how to evaluate it, common mistakes.
- **`<topic>.ipynb`**: implementation: every model of the topic on **one dataset**, ending in a comparison.

## Contents

| # | Topic | Cheatsheet | Notebook | Status |
|---|---|---|---|---|
| 00 | **Python, pandas, NumPy, plots** | [python_pandas_numpy.md](00_foundations/python_pandas_numpy.md) | [python_pandas_numpy.ipynb](00_foundations/python_pandas_numpy.ipynb) | ✅ |
| 00 | **Statistics & hypothesis testing** | [statistics.md](00_foundations/statistics.md) | [statistics.ipynb](00_foundations/statistics.ipynb) | ✅ |
| 01 | **Preprocessing** | [preprocessing.md](01_toolkit/preprocessing.md) | [preprocessing.ipynb](01_toolkit/preprocessing.ipynb) | ✅ |
| 01 | **Dimensionality reduction** (PCA, t-SNE) | [dimensionality_reduction.md](01_toolkit/dimensionality_reduction.md) | [dimensionality_reduction.ipynb](01_toolkit/dimensionality_reduction.ipynb) | ✅ |
| 01 | **Evaluation metrics (all topics)** | [evaluation_metrics.md](01_toolkit/evaluation_metrics.md) | — | ✅ |
| 02 | **Clustering** | [clustering.md](02_clustering/clustering.md) | [clustering.ipynb](02_clustering/clustering.ipynb) | ✅ |
| 03 | **Regression** | [regression.md](03_regression/regression.md) | [regression.ipynb](03_regression/regression.ipynb) | ✅ |
| 04 | **Classification** | [classification.md](04_classification/classification.md) | [classification.ipynb](04_classification/classification.ipynb) | ✅ |
| 05 | **Deep Learning** (ANN, CNN) | [deep_learning.md](05_deep_learning/deep_learning.md) | [deep_learning.ipynb](05_deep_learning/deep_learning.ipynb) | ✅ |
| 06 | Recommendation Systems | — | — | 🔜 |
| — | [Projects](projects/) | Cleaned portfolio projects (one per topic + Hackathon) | | 🔜 |

## Which model when? (quick guide)

| I want to… | Topic | Start with |
|---|---|---|
| Group similar rows, no target | Clustering | K-Means → check GMM / DBSCAN |
| Predict a number | Regression | Linear Regression → Random Forest / Boosting |
| Predict a category | Classification | Logistic Regression → Random Forest / Boosting |
| Reduce # of columns / visualize | Toolkit | PCA (linear), t-SNE (visual only) |
| Images, or very complex patterns | Deep Learning | CNN (images), ANN (tabular) |
| Recommend items to users | Recommendation | Popularity → Collaborative filtering → Matrix factorization (SVD) |

## Setup

```bash
uv sync
uv run jupyter lab
```

Datasets are in [`data/`](data/).
