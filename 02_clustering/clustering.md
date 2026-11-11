# Clustering Cheatsheet

> **Course module:** 2. Making Sense of Unstructured Data
> **Code:** [clustering.ipynb](clustering.ipynb) — all 5 models on the same dataset (Customer Personality)
> **Metrics:** [evaluation_metrics.md § Clustering](../01_toolkit/evaluation_metrics.md#3-clustering-no-labels)

Clustering = **unsupervised** learning. There is no target `y`; the model groups rows that look alike. The goal is usually **segmentation** (customers, products, players) so the business can act differently on each group.

---

## The workflow (same for every model)

1. **EDA**: distributions, outliers, correlations.
2. **Clean**: drop IDs / constant columns, missing values, duplicates, impossible outliers.
3. **Encode** categoricals (one-hot).
4. **Scale**: **mandatory**, because every model here is based on distances. `StandardScaler`.
5. **(Optional) PCA**: fewer columns, less noise, faster. Keep ~80% of the variance.
6. **Choose k** (elbow, silhouette, BIC…), then **fit** the final model.
7. **Profile the clusters**: mean/median of each original feature per cluster, then give each one a business name.
8. **Compare models**: silhouette + cluster sizes + "does it make business sense?"

---

## Models at a glance

| Model | Center / idea | Need to pick k? | Cluster shape | Handles outliers | Choose k / params with |
|---|---|---|---|---|---|
| **K-Means** | Mean of the points (centroid) | Yes | Round, similar size | ❌ Sensitive | Elbow (WCSS) + Silhouette |
| **K-Medoids** | A real data point (medoid) | Yes | Round | ✅ More robust | Elbow + Silhouette |
| **Hierarchical** | Tree of merges (dendrogram) | Cut the tree | Any (depends on linkage) | Depends on linkage | Cophenetic corr. (distance + linkage), Silhouette (k) |
| **GMM** | Mixture of Gaussians (probabilistic) | Yes | Elliptical, can overlap | ❌ | **BIC / AIC (lower = better)** + Silhouette |
| **DBSCAN** | Density (points with enough neighbors) | **No** | **Any shape** | ✅ Labels them as noise (`-1`) | `eps`, `min_samples` grid search → Silhouette |

---

## K-Means

- Picks k centroids → assigns each point to the nearest → moves the centroid to the mean → repeats until stable.
- **Hyperparameters:** `n_clusters`, `random_state` (results depend on the starting points), `n_init`.
- **Choose k:**
  - **Elbow plot:** WCSS (`inertia_`) vs k → pick the "elbow" where the curve flattens.
  - **Silhouette:** pick the k with the highest score, or a slightly bigger k if you need more profiles for the business.
- ✅ Fast, simple, easy to explain. ❌ Assumes round, similar-size clusters; outliers pull the means.

## K-Medoids

- Same as K-Means, but the center has to be a **real data point**, so it is more robust to outliers.
- `kmedoids` package (`scikit-learn-extra` is no longer maintained and breaks with NumPy 2).
- Slower on large data (it compares pairwise distances).

## Hierarchical (Agglomerative)

- **Agglomerative (bottom-up):** every point starts as its own cluster, then the 2 closest clusters merge, repeated until one is left. **Divisive** = top-down (rarely used).
- **Two decisions:**
  - **Distance:** euclidean, manhattan (`cityblock`), chebyshev, mahalanobis…
  - **Linkage:** how to measure the distance between two *groups*:
    - `single`: closest points → long "chain" clusters
    - `complete`: farthest points
    - `average`: average of all pairs
    - `ward`: merge that increases variance the least → **balanced, compact clusters** (euclidean only)
- **Cophenetic correlation:** how well the dendrogram keeps the real distances (> 0.7 good). Use it to compare distance + linkage pairs.
- **Cut the tree** (choose k) with silhouette, or visually on the dendrogram.
- ⚠️ The highest cophenetic pair (e.g. `average`) can still give **useless clusters** (1 giant cluster + tiny ones). Always check sizes; `ward` is usually more useful for segmentation.

## Gaussian Mixture Model (GMM)

- Assumes the data is a mix of k Gaussian distributions. Each point gets a **probability** of belonging to each cluster (soft clustering, via `predict_proba`).
- Clusters can be **elliptical and overlap**, so it is more flexible than K-Means.
- **Choose k with BIC / AIC**: they reward fit but penalize more clusters. **Lower = better**; pick the minimum.
- `GaussianMixture(n_components=k, random_state=0)`

## DBSCAN

- For each point: if at least `min_samples` points are within radius `eps`, it is a **core** point. Points near a core point are **border** points; everything else is **noise** (`-1`).
- ✅ No need to choose k, finds any shape, detects outliers.
- ❌ Very sensitive to `eps` / `min_samples`; struggles when clusters have different densities.
- **Tuning:** grid search over `eps` × `min_samples`, keep the pair with the best silhouette (only when it finds more than 1 label).
- ⚠️ A high silhouette may just mean "one dense blob + noise". That is useful for **outlier detection**, not for segmentation.

---

## Cluster profiling (the part the business cares about)

```python
df['Cluster'] = model.labels_
df.groupby('Cluster')[num_cols].mean()          # or .median()
df['Cluster'].value_counts()                     # sizes
# Scale the means to compare features on one chart:
StandardScaler().fit_transform(means).T.plot(kind='bar')
```

Give each cluster a **name + description**, for example: *"Affluent, high spenders, no kids, buy in store, respond to campaigns"*.

---

## Common mistakes

- Forgetting to **scale**, so `Income` (thousands) dominates `Kidhome` (0–2).
- Leaving **IDs / constant columns** in. They add noise.
- Choosing k **only** by the metric. Business usefulness matters too (2 clusters is often too few to act on).
- Not checking **cluster sizes** (a 2024 / 26 / 5 split is not a segmentation).
- Profiling on **PCA components**. Always profile with the **original** columns.
- **Text vs code mismatch** (writing "choose 5", then fitting 7). Keep the conclusion and the code in sync.

---

## Source notebooks (Drive: `MIT GT - Data Science/2. Making Sense of Unstructured Data`)

- `Capstone/Capstone: Customer Personality Segmentation.ipynb`: main source for this notebook
- `Guided Project/Fantasy Sport Clustering Analysis - Ro.ipynb`
- `Module/.../PCA & K-Means Ro.ipynb`, `MLS_Case_Study_Unsupervised_Learning.ipynb`
