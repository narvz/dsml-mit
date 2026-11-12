# SuperKart Sales Prediction

> **Type:** Supervised learning, regression · **Course:** Module 3 Guided Project + Capstone
> **Notebook:** [03_regression/regression.ipynb](../../03_regression/regression.ipynb) · **Data:** [superkart.csv](../../data/superkart.csv)

## Problem
A supermarket chain wants to **forecast the sales revenue of each product in each store** for the next quarter, to plan inventory and sales strategy across city tiers. 8,763 product × store records.

## Approach
1. **Feature engineering:** product ID prefix → category (FD / DR / NC), establishment year → store age, 16 product types → perishable flag, fixed `"reg"` → `"Regular"`.
2. **Multicollinearity:** VIF showed **perfect redundancy**, since store ID, type, size, city tier and age all identify the same 4 stores, and "non-consumable" = "no sugar". Kept one of each, so every VIF is < 4.
3. **Explainable baseline:** statsmodels OLS, p-value feature selection, and checks of **every linear regression assumption** (linearity, homoscedasticity with Goldfeld-Quandt, normal residuals with a Q-Q plot).
4. **Models compared:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, plus 5-fold CV and GridSearch tuning.

## Results (test set)

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| **Random Forest** ✅ | **0.935** | 269 | **3.7%** |
| Gradient Boosting (tuned) | 0.931 | 277 | 4.2% |
| Decision Tree | 0.876 (train 1.00, overfits) | 372 | 4.7% |
| Linear / Ridge / Lasso | 0.83 | 433 | 9.6% |

## Insights
- **Product price (MRP)** drives ~47% of the predictions, **store type** ~40%, product weight ~11%. Sugar content, shelf area and perishability barely matter.
- Linear regression meets all assumptions and is fully explainable, so it's the model to show stakeholders. Random Forest is the one for accurate forecasts.

## Skills shown
Feature engineering · VIF / multicollinearity · OLS inference & assumption testing · regularization · tree ensembles · cross-validation · hyperparameter tuning
