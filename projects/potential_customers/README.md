# ExtraaLearn: Potential Customers Prediction

> **Type:** Supervised learning, binary classification · **Course:** Module 4 Project Assessment
> **Notebooks:** [04_classification/classification.ipynb](../../04_classification/classification.ipynb) (8 ML models) · [05_deep_learning/deep_learning.ipynb](../../05_deep_learning/deep_learning.ipynb) (ANN on the same data)
> **Data:** `extraalearn.csv` ([sources](../../data/README.md))

## Problem
An online education startup gets thousands of leads. Which ones will **convert to paid customers**? The sales team wants to focus on them. 4,612 leads, **~30% convert** (imbalanced).

**Metric choice:** missing a future customer (false negative) means lost revenue, while calling a lead who doesn't convert costs only a little time, so the model optimizes **recall**, with F1 and ROC-AUC as checks.

## Approach
1. Dropped the ID and 14 duplicates; Yes/No → 1/0; one-hot encoding; **stratified** 70/30 split.
2. Imbalance: `class_weight="balanced"` + **threshold tuning** on the precision-recall curve (chosen on train, evaluated on test).
3. Models: Logistic Regression, KNN, SVM, Decision Tree (default vs pruned), tuned Random Forest, AdaBoost, Gradient Boosting, plus an ANN (Dropout + L2 + EarlyStopping).

## Results (test set)

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression, threshold 0.32 | **0.92** | 0.65 | 0.88 |
| Decision Tree, pruned (depth 3) | 0.90 | 0.72 | 0.89 |
| **Random Forest, tuned** ✅ | 0.88 | 0.76 | **0.93** |
| ANN, regularized | 0.80 | 0.75 | 0.91 |
| Gradient Boosting | 0.76 | **0.77** | **0.93** |
| Decision Tree, default | 0.67 (train 1.00, overfits) | 0.67 | 0.76 |

## Insights & recommendations
- Top drivers: **first interaction through the website** and **time spent on the website** (~32% of importance each), then **profile completion**.
- Push leads toward the website and toward completing their profile; prioritize leads with high time on site.
- The pruned 3-level tree gives the sales team **rules they can read**, at almost the same recall.
- On this small tabular dataset, the neural network doesn't beat Random Forest.

## Skills shown
Imbalanced classification · metric selection from business cost · threshold tuning · pruning · ensembles · GridSearchCV · ANN with regularization
