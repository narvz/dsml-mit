# Customer Personality Segmentation

> **Type:** Unsupervised learning (clustering) · **Course:** Module 2 Capstone
> **Notebook:** [02_clustering/clustering.ipynb](../../02_clustering/clustering.ipynb) · **Data:** [customer_personality.csv](../../data/customer_personality.csv)

## Problem
A retailer wants to **segment its 2,240 customers** by demographics, spending and channel behavior so it can personalize campaigns, retain high-value customers and plan inventory.

## Approach
1. **Cleaning:** dropped the ID and constant columns, imputed `Income` with the median, removed 182 duplicates and 3 impossible birth years (< 1900), grouped rare marital statuses → 2,055 customers.
2. **Preparation:** one-hot encoding → `StandardScaler` → **PCA** (15 of 35 components keep 80% of the variance).
3. **Models compared:** K-Means, K-Medoids, Hierarchical (Ward), Gaussian Mixture, DBSCAN. k chosen with the elbow, silhouette and BIC.
4. **Selection:** silhouette **plus** cluster sizes **plus** business sense.

## Results

| Model | Silhouette | Cluster sizes |
|---|---|---|
| **K-Means (k=3)** ✅ | 0.21 | 954 / 603 / 498 |
| GMM (k=3) | 0.22 | 1,062 / 754 / 239 |
| K-Medoids (k=3) | 0.18 | 847 / 605 / 603 |
| Hierarchical, Ward (k=3) | 0.17 | 830 / 637 / 588 |
| DBSCAN | 0.50 | 1,998 / 57 (one blob + noise) |

DBSCAN's high score is misleading because it doesn't segment. K-Means was chosen for its balanced, explainable groups.

## Segments & actions

| Segment | Profile | Action |
|---|---|---|
| **Affluent spenders** | ~78k income, no kids, top spend on wine & meat, few deals, best campaign response | Premium / loyalty program |
| **Established families** | ~58k income, teens at home, good spend, most deal & web purchases | Bundles, discounts, web promos |
| **Budget young families** | ~35k income, small kids, low spend, many web visits but few purchases | Low-price offers, convert web traffic |

## Skills shown
EDA · outlier & missing value treatment · scaling · PCA · 5 clustering algorithms · cluster validation · business profiling
