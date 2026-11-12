# Shinkansen Travel Experience: Passenger Satisfaction ⭐

> **Type:** Binary classification, end-to-end · **Course:** MIT ADSP Hackathon
> **Notebook:** [shinkansen.ipynb](shinkansen.ipynb) · **Data:** `data/shinkansen/` ([sources](../../data/README.md)) · **Output:** [submission.csv](submission.csv)

## Problem
A high-speed rail operator collected **travel data** (who travelled, class, distance, delays) and a **15-question service survey** for 94,379 trips. Predict whether each passenger's **overall experience was positive**, and find **which service aspects drive satisfaction**, so investment goes where it matters. Leaderboard metric: **accuracy** (~55/45 classes).

## Approach
1. **Data:** joined the two sources 1:1 on `ID` (validated), for train and test.
2. **Data quality:** up to ~10% missing in 5 columns; survey answers are ordered text scales → **ordinal encoding 0–5** (with an assertion that no unexpected label slips through).
3. **EDA:** satisfaction rate by every rating level (heatmap), by segment, and by delay.
4. **Pipeline:** `ColumnTransformer` with median imputation + missing indicators (numeric / ordinal) and "Missing" category + one-hot (nominal), inside every model's `Pipeline`, so there's **no leakage** and the same transform is applied at submission time.
5. **Models:** Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, MLP on an 80/20 stratified hold-out.
6. **Tuning:** `RandomizedSearchCV` (12 combinations × 3-fold CV) on HistGradientBoosting.
7. **Explainability:** permutation importance on the original columns.
8. **Delivery:** refit on all data → `submission.csv` + the full pipeline saved with `joblib` (raw rows in, prediction out: ready for an API).

## Results (20% hold-out, 18,876 trips)

| Model | Train acc | Val acc | Val AUC |
|---|---|---|---|
| **HistGradientBoosting, tuned** ✅ | 0.978 | **0.954** | **0.993** |
| HistGradientBoosting | 0.958 | 0.950 | 0.992 |
| Random Forest | 0.991 | 0.946 | 0.990 |
| Neural net (MLP) | 0.954 | 0.943 | 0.990 |
| Decision Tree (depth 12) | 0.953 | 0.935 | 0.971 |
| Logistic Regression | 0.833 | 0.835 | 0.907 |

The original hackathon notebooks reached ~0.944 validation accuracy with an ANN. The tuned boosting pipeline gets **0.954**, trains in seconds, and is easier to explain.

## Key findings
| Finding | Evidence |
|---|---|
| **Onboard entertainment** is the steepest driver | 21% satisfied at "Poor" → 95% at "Excellent" |
| **Seat comfort** is the top model feature | Permutation importance 0.16 (next: 0.045) |
| **Digital journey** (online booking, support, boarding) | each ~20–30% → ~75% across its scale |
| **At-risk segments** | Disloyal customers 24% satisfied · Eco class 40% · personal travel 47% |
| **Gender gap** | Women 65% vs men 44% |
| **Low impact** | Arrival-time convenience & platform location (flat curves), seat class (55% vs 55%) |
| **Delays** | 59% satisfied on time → 44% with > 60 min delay |
| ⚠️ **Data-quality flag** | 3,463 passengers rated seat comfort "Extremely Poor" and **100%** were satisfied, which points to a likely mis-coded label |

## Recommendations
1. Invest in **seat comfort and onboard entertainment** first, but validate the seat-comfort labels with the data owner.
2. Improve the **online journey as one package** (booking + support + boarding).
3. Run **retention offers** for disloyal, Eco-class and personal-travel passengers.
4. **Investigate the gender gap** with a follow-up survey.
5. Deprioritize platform location and arrival-time changes; handle long delays with communication and compensation.

## Fixes vs the original hackathon notebooks
- The original re-fitted the scaler on the test set (`scaler.fit_transform(submission_X)`), so test rows were scaled differently from training rows. Now a single fitted `Pipeline` transforms both.
- Manual imputation steps repeated per dataset → one `ColumnTransformer`.
- Three ANN notebook variants → one comparison table with 5 model families plus tuning.

## Skills shown
Data integration · ordinal encoding · `ColumnTransformer` / `Pipeline` · model comparison · RandomizedSearchCV · permutation importance · data-quality skepticism · business recommendations · model serialization (MLOps-ready)
