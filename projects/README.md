# Projects

Portfolio versions of the course projects: **problem → approach → results → business recommendations**.

| Project | Type | Best model | Result | Notebook |
|---|---|---|---|---|
| ⭐ [**Shinkansen Travel Experience**](shinkansen/) | Classification, end-to-end (Hackathon) | Tuned HistGradientBoosting | **95.4% accuracy**, AUC 0.993 | [shinkansen.ipynb](shinkansen/shinkansen.ipynb) |
| [Customer Personality Segmentation](customer_segmentation/) | Clustering | K-Means (k=3) on PCA | 3 actionable segments | [clustering.ipynb](../02_clustering/clustering.ipynb) |
| [SuperKart Sales Prediction](superkart_sales/) | Regression | Random Forest | R² 0.935, MAPE 3.7% | [regression.ipynb](../03_regression/regression.ipynb) |
| [ExtraaLearn Potential Customers](potential_customers/) | Imbalanced classification | Tuned Random Forest | Recall 0.88, AUC 0.93 | [classification.ipynb](../04_classification/classification.ipynb) |

The Shinkansen project is self-contained in its folder. The other three are case-study write-ups whose code lives in the topic notebooks, so nothing is duplicated.

## Not cleaned up yet

| Project | Source (Drive) | Note |
|---|---|---|
| Hospital Length of Stay | `4. .../Guided Project/Hospital LOS Prediction - Ro` + `Module/.../Hospital_LOS_Prediction_Part 1–3, Serialization` | Good MLOps candidate (it already has a serialization notebook) |
| Book Recommendation | `6. .../Guided Project/Book Recommendation - Ro` | Data is 22–73 MB; the recommendation topic uses MovieLens instead |
