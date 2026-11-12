# Recommendation Systems Cheatsheet

> **Course module:** 6. Recommendation Systems
> **Code:** [recommendation_systems.ipynb](recommendation_systems.ipynb): 7 models on MovieLens 100k with the `surprise` library
> **Metrics:** [evaluation_metrics.md § Recommendation systems](../01_toolkit/evaluation_metrics.md#5-recommendation-systems)

A recommender predicts **which items a user will like** from past interactions (ratings, clicks, purchases), item attributes, or both. The data is a **user × item matrix** that is mostly empty (MovieLens: 94% sparse; Amazon: >99%). The job is to fill in the blanks.

---

## Types of recommenders

| Type | Idea | Needs | ✅ | ❌ |
|---|---|---|---|---|
| **Rank-based (popularity)** | Top items by average rating with a minimum number of ratings | Ratings | Simple, works for **new users** | Same for everyone, no personalization |
| **User-user CF** | Users similar to you liked X | Ratings | Intuitive, finds new tastes | Users have few ratings and change; slow with many users |
| **Item-item CF** | People who liked this also liked X | Ratings | More stable than user-user, explainable ("because you liked…") | Popular items dominate |
| **Matrix factorization (SVD)** | Learn hidden user & item **latent factors** | Ratings | Best accuracy, handles sparsity, scales | Black box, needs tuning |
| **Clustering-based** (CoClustering) | Group users & items, predict from cluster averages | Ratings | Fast | Coarse |
| **Content-based** | Items similar in **attributes** (genre, text, brand) to what you liked | Item features | Works for **new items**, no other users needed | Over-specializes ("more of the same") |
| **Hybrid** | Combine content + collaborative | Both | Best of both, handles cold start | More complex |

**CF = collaborative filtering**: uses only the interaction matrix, not item features.

---

## Rank-based

```python
stats = df.groupby("item")["rating"].agg(avg_rating="mean", count_rating="count")
stats[stats.count_rating > min_interactions].sort_values("avg_rating", ascending=False).head(n)
```

**Always set a minimum number of interactions.** Otherwise an item with a single 5★ rating ranks first.

---

## Similarity-based CF (KNN)

Predicted rating of user *u* for item *i* = **similarity-weighted average** of the ratings from the *k* most similar neighbors (users who rated *i*, or items *u* rated).

| Similarity | Notes |
|---|---|
| **Cosine** | Angle between rating vectors |
| **MSD** (mean squared difference) | Surprise default; often best on small data |
| **Pearson** | Cosine on mean-centered ratings: corrects users who always rate high / low |
| **Pearson baseline** | Pearson using baseline (bias) estimates: robust with few common ratings |

Key hyperparameters (`KNNBasic`):
- `k`: max number of neighbors · `min_k`: minimum neighbors needed, otherwise fall back to the global mean
- `sim_options={"name": "cosine" | "msd" | "pearson" | "pearson_baseline", "user_based": True | False}`
- Variants: `KNNWithMeans` (centers each user's ratings), `KNNWithZScore`, `KNNBaseline`.

---

## Matrix factorization (SVD)

$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^\top p_u$$

- **μ** = global mean, **b_u** = user bias (generous / harsh rater), **b_i** = item bias (good / bad item), **p_u, q_i** = latent factor vectors.
- Learned by **SGD**, minimizing squared error + **regularization** (prevents overfitting to users with few ratings).
- Hyperparameters: `n_factors` (latent dimensions), `n_epochs`, `lr_all` (learning rate), `reg_all` (regularization).
- The "SVD" in Surprise is **Funk SVD** (made famous by the Netflix Prize). It's not the linear-algebra SVD, which can't handle missing values.
- **SVD++** also uses *which* items were rated (implicit feedback): more accurate, slower.

---

## The `surprise` workflow

```python
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=0.2, random_state=1)

model = SVD(random_state=1).fit(trainset)
preds = model.test(testset);  accuracy.rmse(preds)
model.predict(uid=42, iid=50).est                      # one prediction

gs = GridSearchCV(SVD, {"n_factors": [50, 100], "reg_all": [0.02, 0.1]}, measures=["rmse"], cv=3).fit(data)
gs.best_params["rmse"]
final = SVD(**gs.best_params["rmse"]).fit(data.build_full_trainset())   # refit on ALL data to recommend
```

To **recommend**: predict every item the user has **not** rated, sort by predicted rating, take the top n.

---

## Evaluation

| Metric | Measures | Better |
|---|---|---|
| **RMSE** | Error of the predicted rating | Lower |
| **Precision@k** | Of the top-k recommended, fraction actually relevant | Higher |
| **Recall@k** | Of all relevant items, fraction that made the top-k | Higher |
| **F1@k** | Balance of Precision@k and Recall@k | Higher |

- **Relevant** = true rating ≥ threshold (e.g. **3.5**). **Recommended** = predicted rating ≥ threshold **and** in the top k.
- **Business view:** FN = relevant item never shown (lost sale); FP = irrelevant item shown (annoyed user). Precision@k matters when the list is short (only k slots).
- Beyond accuracy: **coverage** (how much of the catalog gets recommended), **diversity**, **novelty**, and **A/B tests** in production.

---

## Cold start

| Problem | Fix |
|---|---|
| **New user** (no history) | Popularity, ask for preferences on signup, demographics |
| **New item** (no ratings) | Content-based (attributes), promote to explore |
| Unknown ids in `surprise` | **KNN**: `predict()` returns the **global mean** with `was_impossible=True` (check `details`). **SVD**: unknown user → **μ + item bias** (a popularity-like guess) |

---

## Common mistakes

- Popularity ranking **without a minimum** number of ratings.
- Recommending items the user **already rated / bought**.
- Evaluating only RMSE. A good RMSE doesn't guarantee a good **top-k list** (in the notebook, tuned SVD has the best RMSE but baseline user-user KNN has a slightly higher F1@10).
- Forgetting to **refit on the full data** before making final recommendations.
- Not filtering very sparse users / items before modeling (the course kept users with ≥ 50 ratings), which leaves too little signal and too much memory use.
- Using **user-user** CF when there are far more users than items. Item-item scales better.

---

## Source notebooks (Drive: `MIT GT - Data Science/6. Recommendation Systems`)

- `Project Assessment/Recommendation_Systems_Learner_Notebook_Full_Code_updated.ipynb`: Amazon Electronics (7.8M ratings → users ≥ 50 ratings). Main source of the workflow, `precision_recall_at_k` and the 3.5 threshold
- `Guided Project/Book Recommendation - Ro.ipynb` (Book-Crossing: Books / Users / Ratings)
- `Mentor Learning Session/MLS_Yelp_Reviews_Notebook_updated.ipynb`, `Module/Million_Songs_updated.ipynb`
- The original datasets are 22–320 MB, so this cheatsheet uses MovieLens 100k (same 1–5 rating format)
