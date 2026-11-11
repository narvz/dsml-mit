# Dimensionality Reduction Cheatsheet (PCA · t-SNE)

> **Course module:** 2. Making Sense of Unstructured Data
> **Code:** [dimensionality_reduction.ipynb](dimensionality_reduction.ipynb) (Customer Personality + sklearn digits)
> **Used in:** [clustering.ipynb](../02_clustering/clustering.ipynb) (PCA before clustering)

**Why reduce dimensions?** Fewer columns means less noise, faster models, less multicollinearity, and the ability to **plot** high-dimensional data in 2D. It also helps with the **curse of dimensionality** (in many dimensions, distances become meaningless).

---

## PCA vs t-SNE

| | **PCA** | **t-SNE** |
|---|---|---|
| Type | Linear | Non-linear |
| Preserves | **Global** variance | **Local** neighborhoods |
| Use for | Feature reduction **before modeling**, visualization | **Visualization only** |
| Deterministic | Yes | No (depends on `random_state`) |
| New data | `pca.transform(X_new)` ✅ | ❌ No transform: must re-run |
| Interpretable | Loadings show what each PC means | Axes mean nothing |
| Speed | Fast | Slow on large data |
| Key parameter | `n_components` | `perplexity` (5–50) |

Other options: **UMAP** (like t-SNE, faster, keeps more global structure), **LDA** (supervised, uses the labels), **autoencoders** (neural nets).

---

## PCA (Principal Component Analysis)

- Finds new axes (**principal components**) that are **linear combinations** of the original columns:
  - **PC1** = direction of **maximum variance**, PC2 = next most variance and **orthogonal** (uncorrelated) to PC1, and so on.
- Math: **eigenvectors** of the covariance matrix = directions (**loadings**); **eigenvalues** = variance along each one.
- PCs are **uncorrelated**, which removes multicollinearity.

### Steps

```python
X_s = StandardScaler().fit_transform(X)              # 1. ALWAYS scale
pca = PCA().fit(X_s)                                 # 2. fit with all components
pca.explained_variance_ratio_.cumsum()               # 3. choose how many
X_pca = PCA(n_components=0.80).fit_transform(X_s)    # 4. keep 80% of the variance
```

### How many components?

| Method | Rule |
|---|---|
| **Cumulative explained variance** | Fewest PCs reaching **~80–90%** (`n_components=0.80`) |
| **Scree plot** | Elbow where the bars flatten |
| **Kaiser rule** | Keep PCs with eigenvalue > 1 (on scaled data) |
| Visualization | 2 or 3 |

### Interpreting: loadings

- `pca.components_[i]` = weights of the original columns in PC *i*. Large |weight| = that column drives the PC.
- Example (Customer Personality): **PC1 ≈ "spending power"**: income, wine/meat spend, catalog & store purchases (+) vs kids at home & web visits (−).

### ⚠️ Why scaling matters

Without scaling, PCA is dominated by the column with the **largest units**. On Customer Personality, unscaled PC1 "explains" 99.98% of the variance, and it's just `Income`. Scaled, PC1 explains ~28% and mixes many features.

---

## t-SNE (t-distributed Stochastic Neighbor Embedding)

- Converts distances into **probabilities of being neighbors** in high-D, then finds a 2D layout where those neighbor probabilities match. It uses a **t-distribution** in 2D so clusters don't crowd together.
- **`perplexity`** ≈ how many neighbors each point considers:
  - low (5): many small, fragmented clusters · high (100): more global, clusters merge · usual: **5–50**.
- Also: `learning_rate`, `max_iter` (more iterations = more stable), `random_state`.

### ⚠️ How NOT to read a t-SNE plot

- Distances **between** clusters mean nothing.
- Cluster **sizes** mean nothing (dense and sparse groups look similar).
- Different `random_state` / perplexity give different pictures, so try several.
- Don't feed t-SNE output into a model. **Use it to look, not to learn.**

---

## Common mistakes

- Running PCA **without scaling**.
- Profiling clusters on **PC values** instead of the original columns.
- Fitting PCA on train + test together (**leakage**). Fit on train, `transform` test (or put it in a `Pipeline`).
- Expecting PCA to help tree models. Trees handle many features fine, and PCA makes them less interpretable.
- Treating t-SNE clusters as proof that clusters exist. Confirm with a clustering algorithm + metrics.

---

## Source notebooks (Drive: `MIT GT - Data Science/2. Making Sense of Unstructured Data/Module`)

- `PCA & K-Means Ro`, `PCA and K-Means Decipher Genome`, `MLS Case_Study_PCA`, `Hand-On Quiz PCA`
- `1.17 t-distributed Stochastic Neighbour Embedding (t-SNE) - MNIST Digit Visualization`
