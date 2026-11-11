# Classification Cheatsheet

> **Course module:** 4. Classification and Hypothesis Testing (hypothesis testing will go in `00_foundations`)
> **Code:** [classification.ipynb](classification.ipynb): 8 models on the same dataset (ExtraaLearn lead conversion)
> **Metrics:** [evaluation_metrics.md § Classification](../01_toolkit/evaluation_metrics.md#2-classification-predicting-a-class)

Classification = **supervised** learning where the target `y` is a **category** (converted / not, churn / stay, disease / healthy).

---

## The workflow

1. **Define the business cost of each error**, which decides the metric (recall, precision or F1). **Do this first.**
2. **EDA**: class balance, conversion rate by each feature.
3. **Clean**: drop IDs, duplicates, encode Yes/No as 1/0.
4. **One-hot** categoricals (`drop_first=True`).
5. **Stratified** train/test split (`stratify=y`).
6. **Handle imbalance**: `class_weight="balanced"`, threshold tuning, or resampling (SMOTE).
7. **Baseline**: Logistic Regression.
8. **Try other models** and **tune** with `GridSearchCV(scoring=<your metric>)`.
9. **Compare on test**: chosen metric + F1 + ROC-AUC, plus train vs test for overfitting.
10. **Explain**: odds ratios, tree rules, feature importance, then turn them into business actions.

---

## Models at a glance

| Model | Idea | Scaling needed? | `class_weight`? | Overfitting risk | Explainability | Key hyperparameters |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear score passed through a sigmoid → probability | **Yes** (for regularization) | ✅ | Low | ⭐⭐⭐ Odds ratios | `C` (inverse of regularization), `penalty` |
| **KNN** | Class of the k nearest neighbors (vote) | **Yes** | ❌ | k small means high | ⭐⭐ | `n_neighbors`, `weights`, `metric` |
| **SVM** | Boundary with the **widest margin**; kernels for non-linear | **Yes** | ✅ | Medium | ⭐ | `C`, `kernel`, `gamma` |
| **Naive Bayes** | Bayes' theorem, assumes features are independent | No | ❌ | Low | ⭐⭐ | — (very fast, good for text) |
| **Decision Tree** | If/else splits that make the nodes "purer" | No | ✅ | **High** | ⭐⭐⭐ Visual rules | `max_depth`, `min_samples_leaf`, `criterion`, `ccp_alpha` |
| **Random Forest** | Many trees on bootstrap samples + random features, **vote** (bagging) | No | ✅ | Medium | ⭐ Feature importance | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| **AdaBoost** | Sequential weak learners; **reweights misclassified rows** | No | ❌ | Medium | ⭐ | `n_estimators`, `learning_rate` |
| **Gradient Boosting / XGBoost** | Sequential trees fitting the **errors** of the previous ones | No | ❌ (`scale_pos_weight` in XGB) | Medium | ⭐ | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

---

## Logistic Regression

$$P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}$$

- Despite its name, it's a **classifier**. It outputs a probability; the **threshold** (default 0.5) turns it into a class.
- **Odds ratio** = `exp(β)`: if > 1, the feature **increases** the odds of class 1; if < 1, it decreases them.
  - Example: odds ratio 2.0 means each unit (or std, if scaled) up **doubles** the odds.
- `statsmodels.Logit` gives p-values (like OLS). The same multicollinearity / VIF concerns apply.

## KNN

- No training: it stores the data and, for a new row, looks at the **k closest** rows.
- k small → noisy, overfits; k large → too smooth. Use an **odd k** for binary problems (avoids ties). Tune with CV.
- Slow on large data; distances become meaningless with many features (**curse of dimensionality**).

## SVM

- Finds the boundary that is **as far as possible** from both classes (max margin). The closest points are the **support vectors**.
- `C` high → small margin, fits training data hard (overfit); `C` low → wider margin, more errors allowed.
- **Kernel trick** (`rbf`, `poly`) for non-linear boundaries without creating the features yourself.
- `probability=True` is needed for `predict_proba` (slower).

## Decision Tree

- At each node it picks the split that reduces **impurity** the most:
  - **Gini** = 1 − Σpᵢ² · **Entropy** = −Σpᵢ log₂ pᵢ (similar results; gini is a bit faster).
- Grows until the leaves are pure, so it **memorizes** (train recall = 1.0). Prune it:
  - **Pre-pruning**: `max_depth`, `min_samples_leaf`, `min_samples_split`, tuned with GridSearchCV.
  - **Post-pruning**: cost-complexity `ccp_alpha` (`cost_complexity_pruning_path`).
- A shallow tree (depth 3–4) gives **rules the business can read**.

## Ensembles

| | **Bagging** (Random Forest) | **Boosting** (AdaBoost, Gradient Boosting, XGBoost) |
|---|---|---|
| Trees built | In **parallel**, independent | **Sequentially**, each fixes the previous |
| Each tree | Deep (low bias, high variance) | Shallow / "weak" (high bias) |
| Reduces | **Variance** | **Bias** |
| Combine | Majority **vote** / average | **Weighted** sum |
| Overfitting | Hard to overfit with more trees | Can overfit: lower `learning_rate` + more trees |

- **AdaBoost** increases the **weight** of misclassified rows so the next learner focuses on them.
- **Gradient Boosting** fits the next tree to the **residual errors** (gradient of the loss). **XGBoost** = faster, regularized version.

---

## Imbalanced data

| Technique | How | Notes |
|---|---|---|
| **Stratified split** | `train_test_split(..., stratify=y)` | Always |
| **Class weights** | `class_weight="balanced"` | Easiest. Mistakes on the minority class count more |
| **Threshold tuning** | Pick from the precision-recall curve on **train/validation** | Works with every model that has `predict_proba` |
| **Oversampling (SMOTE)** | `imblearn.over_sampling.SMOTE`, **only on train** | Creates synthetic minority rows |
| **Undersampling** | Drop majority rows | Loses information |
| **Right metric** | Recall / F1 / PR-AUC, **not accuracy** | Accuracy on 70/30 data: 70% by always predicting 0 |

---

## Choosing the metric (business first)

| Situation | Costly error | Optimize |
|---|---|---|
| Lead conversion, churn, fraud, disease | **Missing a positive** (FN) | **Recall** |
| Spam filter, expensive offer, legal action | **False alarm** (FP) | **Precision** |
| Both matter / costs unknown | — | **F1** |
| Compare models before choosing a threshold | — | **ROC-AUC** (PR-AUC if very imbalanced) |

---

## Common mistakes

- Reporting **accuracy** on imbalanced data.
- Choosing the **threshold on the test set** (leakage). Choose it on train / validation.
- Not using `stratify=y`.
- Forgetting to **scale** for Logistic Regression, KNN and SVM (use a `Pipeline`).
- Applying **SMOTE before the split**, so synthetic copies leak into the test set.
- Trusting a default Decision Tree (it always overfits).
- Optimizing the metric in GridSearchCV that you **don't** care about (`scoring` defaults to accuracy).

---

## Source notebooks (Drive: `MIT GT - Data Science/4. Classification and Hypothesis Testing`)

- `Project Assessment/Learner Notebook - Full Code Version - Potential Customers Prediction.ipynb`: main source
- `Mentor Learning Session/Employee Attrition - Ro.ipynb`
- `Guided Project/Hospital Length of Stay (LOS) Prediction - Ro.ipynb`
- `Practice Project/Tourism Package Prediction` (learner + solution)
