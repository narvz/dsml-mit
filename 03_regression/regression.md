# Regression Cheatsheet

> **Course module:** 3. Regression and Prediction
> **Code:** [regression.ipynb](regression.ipynb): 6 models on the same dataset (SuperKart sales)
> **Metrics:** [evaluation_metrics.md § Regression](../01_toolkit/evaluation_metrics.md#1-regression-predicting-a-number)

Regression = **supervised** learning where the target `y` is a **number** (sales, price, length of stay…).

---

## The workflow

1. **EDA**: target distribution, relation of each feature with the target, correlations.
2. **Clean & engineer**: fix labels, missing values, outliers; turn raw columns into useful ones (ID prefix → category, year → age).
3. **Encode** categoricals: one-hot with `drop_first=True` (avoids the dummy variable trap).
4. **Train/test split**, **before** fitting anything.
5. **Multicollinearity check (VIF)**: drop redundant features (needed for linear models).
6. **Baseline: Linear Regression** (statsmodels OLS for p-values + assumptions).
7. **Try other models**: regularized linear, trees, ensembles.
8. **Compare** train vs test (overfitting) and use **cross-validation** for stable scores.
9. **Tune** the best model (GridSearchCV), then **explain** it (coefficients / feature importance).

---

## Models at a glance

| Model | Idea | Scaling needed? | Overfitting risk | Explainability | Key hyperparameters |
|---|---|---|---|---|---|
| **Linear Regression (OLS)** | Straight-line fit, minimizes squared errors | No (yes to compare coefficients) | Low | ⭐⭐⭐ Coefficients + p-values | — |
| **Ridge (L2)** | Linear + penalty on large coefficients (shrinks them) | **Yes** | Low | ⭐⭐⭐ | `alpha` |
| **Lasso (L1)** | Linear + penalty that can set coefficients to **0**, so it also selects features | **Yes** | Low | ⭐⭐⭐ | `alpha` |
| **Decision Tree** | Splits data with if/else rules | No | **High** | ⭐⭐ (visual rules) | `max_depth`, `min_samples_leaf` |
| **Random Forest** | Many trees on random samples/features, averaged (**bagging**) | No | Medium | ⭐ Feature importance | `n_estimators`, `max_depth`, `max_features` |
| **Gradient Boosting** | Trees built one after another, each fixes the previous errors (**boosting**) | No | Medium | ⭐ Feature importance | `n_estimators`, `learning_rate`, `max_depth` |

> Other boosting libraries (XGBoost, LightGBM, CatBoost) work the same way, just faster and with more options.

---

## Linear Regression

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

- **Coefficient βᵢ**: change in `y` when `xᵢ` goes up by 1, **holding the others fixed**.
- **Intercept β₀**: `y` when all features are 0.
- **statsmodels vs sklearn**:
  - `sm.OLS(y, sm.add_constant(X)).fit().summary()` gives R², adj. R², **p-values**, confidence intervals. Use it to **explain**.
  - `LinearRegression()` gives the same coefficients and fits in pipelines / CV. Use it to **predict**.
- **p-value of a coefficient** < 0.05 means the feature is significant. Drop non-significant ones **one at a time** (highest p first), because p-values change each time you drop one.

### The assumptions (check them on the residuals)

| Assumption | How to check | If it fails |
|---|---|---|
| **Linearity** | Residuals vs fitted plot: no pattern | Transform features (log, square), add interactions, or use trees |
| **Mean of residuals ≈ 0** | `resid.mean()` | Usually OK if there is an intercept |
| **Homoscedasticity** (constant error variance) | Residuals vs fitted: no **funnel**. **Goldfeld-Quandt** test: p > 0.05 means OK | Log-transform the target, weighted least squares |
| **Normality of residuals** | Histogram + **Q-Q plot** (points on the line). Shapiro test | Transform the target; less critical with large n |
| **No multicollinearity** | **VIF < 5** | Drop / combine redundant features, use Ridge |
| **Independent errors** | Durbin-Watson ≈ 2 (in `summary()`) | Matters for time series |

### Multicollinearity & VIF

- **VIF** = how well a feature is predicted by the **other** features. 1 = independent, > 5 high, **∞ = perfectly redundant**.
- It does **not** hurt predictions much, but makes **coefficients and p-values unreliable** (they swap sign, get huge).
- Typical cause: several columns that describe the **same thing**. In SuperKart, `Store_Id`, `Store_Type`, `Store_Size`, city tier and `Store_Age` all identify the same 4 stores, so keep one.
- Remove the highest-VIF feature, recompute, repeat.

---

## Regularization (Ridge & Lasso)

- Adds a **penalty** to the loss so coefficients stay small, which reduces overfitting when there are many / correlated features.
  - **Ridge (L2):** penalty = α·Σβ². Shrinks all coefficients, never to exactly 0. Good with multicollinearity.
  - **Lasso (L1):** penalty = α·Σ|β|. Can set coefficients to **0**, giving automatic **feature selection**.
  - **ElasticNet**: mix of both.
- **Always scale** first (the penalty is unfair to features with big units).
- α = 0 gives plain OLS; α too high underfits. Tune α with CV (`RidgeCV`, `LassoCV`).

---

## Tree-based models

- **Decision Tree**: splits on the feature/threshold that reduces the error (MSE) the most. Without limits it memorizes the training data (**train R² = 1**). Control it with `max_depth`, `min_samples_leaf` (**pruning**).
- **Random Forest (bagging)**: many deep trees, each on a bootstrap sample and a random subset of features, **averaged**. This lowers variance and is a strong default.
- **Gradient Boosting (boosting)**: shallow trees added **sequentially**, each fitting the **residuals** of the previous ones. Usually the most accurate on tabular data, but needs tuning (`learning_rate` ↓ means more `n_estimators`).
- No scaling needed, they handle non-linearity and interactions automatically, and give **feature importance**, but no p-values or coefficients.

---

## Bias-variance & validation

| | Train error | Test error | Fix |
|---|---|---|---|
| **Underfitting** (high bias) | High | High | More complex model, more features |
| **Good fit** | Low | Low (close to train) | — |
| **Overfitting** (high variance) | Very low | Much higher | Simpler model, regularization, pruning, more data |

- **K-fold cross-validation** (`cross_val_score`, `KFold(shuffle=True)`): report **mean ± std**.
- **GridSearchCV**: tries every hyperparameter combination with CV. `scoring="neg_root_mean_squared_error"` (sklearn maximizes, hence the negative).

---

## Common mistakes

- Reading coefficients **with multicollinearity** (VIF = ∞), which gives meaningless signs and sizes.
- Fitting the scaler / encoder on the **full** data before splitting (**data leakage**). Use a `Pipeline`.
- Leaving **ID columns** in (unique per row, so the model memorizes).
- Judging a model only on **train** metrics.
- Applying Ridge / Lasso **without scaling**.
- Using MAPE when the target can be **≈ 0**.
- Assuming the most complex model is best: if linear is close, it's easier to explain and deploy.

---

## Source notebooks (Drive: `MIT GT - Data Science/3. Regression and Prediction`)

- `Guided Project/SuperKart Sales Prediction - Ro.ipynb`: main source (OLS, VIF, assumptions, CV)
- `Capstone Project/SuperKart Sales Project.ipynb`
- `Module/`: `Case_Study_BigMart_Sales_Prediction`, `Practical_Application_Effects_of_Advertising_on_Sales`, `Auto Case Study - Ro`
- `Hands-on Quiz/LinearRegression_HandsOnQuiz`, `Practice Quizz/Basic Linear Regression`
