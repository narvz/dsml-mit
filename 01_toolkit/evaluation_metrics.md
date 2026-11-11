# Evaluation Metrics Cheatsheet

One place for every metric used in the course. Each topic cheatsheet links back here.

---

## 1. Regression (predicting a number)

| Metric | Formula (idea) | Read it as | Notes |
|---|---|---|---|
| **MAE** | mean(\|y − ŷ\|) | "On average I'm off by X units" | Same units as target. Robust to outliers. |
| **MSE** | mean((y − ŷ)²) | Penalizes big errors more | Units are squared → hard to explain. Used as the loss. |
| **RMSE** | √MSE | "Typical error" in target units | Most common to report. Sensitive to outliers. |
| **MAPE** | mean(\|y − ŷ\| / \|y\|) | "Off by X % on average" | Breaks when y ≈ 0. Easy for business people. |
| **R²** | 1 − SS_res / SS_tot | % of the variance in y explained by the model | 1 = perfect, 0 = as good as predicting the mean, < 0 = worse than the mean. |
| **Adjusted R²** | R² penalized by # of features | Same as R² but fair when comparing models with different # of features | Always ≤ R². Use it to compare linear models. |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)   # n = rows, p = features
```

**Overfitting check:** compare train vs test. Train R² ≫ test R² (or train RMSE ≪ test RMSE) → overfitting.

---

## 2. Classification (predicting a class)

### Confusion matrix (binary)

|                | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP (Type I error) |
| **Actual 1** | FN (Type II error) | TP |

### Metrics

| Metric | Formula | Question it answers | Use when… |
|---|---|---|---|
| **Accuracy** | (TP + TN) / all | How many did I get right? | Classes are balanced. **Misleading on imbalanced data.** |
| **Precision** | TP / (TP + FP) | Of the ones I flagged as 1, how many really were 1? | **False positives are costly** (spam filter, sending an expensive offer). |
| **Recall** (Sensitivity, TPR) | TP / (TP + FN) | Of all the real 1s, how many did I catch? | **False negatives are costly** (disease, fraud, employee attrition). |
| **F1-score** | 2·P·R / (P + R) | Balance of precision and recall | Imbalanced data, or both errors matter. |
| **Specificity** (TNR) | TN / (TN + FP) | Of all the real 0s, how many did I keep as 0? | Medical tests, together with recall. |
| **ROC-AUC** | Area under TPR vs FPR curve | How well the model ranks 1s above 0s, for any threshold | Comparing models independent of the threshold. 0.5 = random. |
| **PR-AUC** | Area under Precision vs Recall curve | Same idea, focused on the positive class | **Very imbalanced** data (better than ROC-AUC there). |

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

**Threshold tuning:** `predict()` uses 0.5. For a better recall/precision trade-off, use `predict_proba()` and pick a threshold from the precision-recall curve.

**Multiclass:** `average='macro'` (each class weighs the same) vs `average='weighted'` (by class size).

> **How to choose the metric — think about the business cost of each error first.**
> Missing a churner (FN) costs more than a wasted coupon (FP)? → Recall.
> Not sure / both matter / new company with no cost info → F1.

---

## 3. Clustering (no labels)

| Metric | Range | Better | What it measures | Used with |
|---|---|---|---|---|
| **WCSS / Inertia** (Elbow plot) | 0 → ∞ | Lower, but look for the "elbow" | Compactness: sum of squared distances to the cluster center | K-Means, K-Medoids |
| **Silhouette score** | −1 → 1 | Higher | Compactness + separation. 1 = perfect, 0 = on the border, −1 = wrong cluster | All models |
| **Cophenetic correlation** | 0 → 1 | Higher (> 0.7 good) | How well the dendrogram keeps the real distances | Hierarchical (to pick distance + linkage) |
| **BIC / AIC** | — | **Lower** | Fit quality penalized by complexity (# clusters) | GMM |
| **Cluster sizes** | — | Balanced-ish | One giant cluster + tiny ones = useless segmentation | All models (sanity check) |

```python
from sklearn.metrics import silhouette_score
silhouette_score(X, labels)
```

> A high silhouette is not everything: DBSCAN may score 0.5 by putting everyone in *one* cluster + noise. Always check cluster sizes and whether the profiles make **business** sense.

---

## 4. Dimensionality reduction

| Metric | Read it as |
|---|---|
| **Explained variance ratio** (PCA) | % of the original information each component keeps. Pick the # of components that reaches ~80–90% cumulative. |

```python
pca.explained_variance_ratio_.cumsum()
```

---

## 5. Recommendation systems

| Metric | Type | What it measures |
|---|---|---|
| **RMSE** | Rating prediction | How far predicted ratings are from real ratings |
| **Precision@k** | Ranking | Of the top-k items recommended, how many are relevant |
| **Recall@k** | Ranking | Of all relevant items, how many made it to the top-k |
| **F1@k** | Ranking | Balance of Precision@k and Recall@k |

"Relevant" = the user's real rating ≥ a threshold (e.g. 3.5).

---

## 6. Deep learning (training curves)

- Plot **training vs validation loss** (and accuracy) per epoch.
- Validation loss goes up while training loss goes down → **overfitting** → early stopping, dropout, more data, augmentation.
- Then evaluate with the classification / regression metrics above.

---

## 7. Validation strategy (applies to all supervised models)

| Technique | When |
|---|---|
| **Train / test split** (`stratify=y` for classification) | Always. Never evaluate on training data. |
| **Train / validation / test** | When you tune hyperparameters (tune on validation, report on test). |
| **K-fold cross-validation** | Small datasets, or to get a stable score. `StratifiedKFold` for classification. |
| **GridSearchCV / RandomizedSearchCV** | Hyperparameter tuning with CV built in. Set `scoring=` to the metric you chose above. |
