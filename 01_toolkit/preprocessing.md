# Preprocessing Cheatsheet

> **Used in:** every topic. **Code:** [preprocessing.ipynb](preprocessing.ipynb) (Customer Personality + ExtraaLearn)

Everything between **loading the data** and **fitting a model**. Order matters:

1. **Quality checks**: dtypes, missing, duplicates, constant / ID columns, messy labels
2. **Fix errors**: typos, impossible values, duplicate labels (`"reg"` → `"Regular"`)
3. **Split train / test** ← before anything that *learns* from the data
4. **Impute** missing values
5. **Handle outliers** (cap / transform)
6. **Encode** categoricals
7. **Scale** numerics
8. **Feature engineering** (can also happen before the split if it uses no statistics of the data, e.g. year → age)

Steps 4–7 go inside a **`Pipeline`** so they're fitted on train only.

---

## Data quality checks

| Check | Code | Action |
|---|---|---|
| Missing | `df.isna().sum()` | Impute or drop |
| Duplicates | `df.duplicated().sum()`. Check **again after dropping the ID** | `drop_duplicates()` |
| ID / unique per row | `df.nunique() == len(df)` | Drop (or extract info, e.g. prefix `FD` / `DR` / `NC`) |
| Constant columns | `df.nunique() == 1` | Drop |
| Messy categories | `value_counts()` | Merge typos / rare levels (`"Alone"` → `"Single"`, `"YOLO"` → `"Other"`) |
| Wrong dtypes | `df.info()` | `astype`, `pd.to_datetime` |

---

## Missing values

| Strategy | When | Code |
|---|---|---|
| **Drop rows** | Very few missing, random | `df.dropna(subset=[col])` |
| **Drop column** | Mostly missing (> 50–60%) | `df.drop(columns=col)` |
| **Median** | Numeric, **skewed / outliers** | `SimpleImputer(strategy="median")` |
| **Mean** | Numeric, ~symmetric, no outliers | `SimpleImputer(strategy="mean")` |
| **Mode** | Categorical | `SimpleImputer(strategy="most_frequent")` |
| **Constant / "Unknown"** | Categorical, when missing means something | `SimpleImputer(strategy="constant", fill_value="Unknown")` |
| **Group median** | A related column explains it | `df.groupby("g")[col].transform("median")` |
| **KNN imputer** | Other columns predict it well | `KNNImputer(n_neighbors=5)` (scale first) |
| **Missing flag** | Missingness itself is informative | `df[col + "_missing"] = df[col].isna().astype(int)` |

> ⚠️ **Mean vs median:** the **median** is robust to outliers, the mean is not. The original Cheat Sheet doc had this reversed.

---

## Outliers

**First ask: error or real?**
- **Error** (born in 1893, income 666,666, negative age): fix or remove.
- **Real but extreme** (a big spender): keep it. It's information. Cap or transform if it hurts the model.

| Detect | Rule |
|---|---|
| **IQR** | < Q1 − 1.5·IQR or > Q3 + 1.5·IQR (boxplot whiskers) |
| **z-score** | \|z\| > 3 (assumes ~normal) |
| Visual | Boxplot, histogram, scatter |

| Treat | Code | Notes |
|---|---|---|
| Remove | `df = df[df.col >= lo]` | Only for errors |
| **Cap (winsorize)** | `s.clip(lower=lo, upper=hi)` | Keeps the row |
| **Log transform** | `np.log1p(s)` | Right-skewed money / count data |
| Robust methods | `RobustScaler`, tree models, median | Less sensitive by design |

Tree-based models barely care about outliers. Linear models, K-Means, PCA and neural nets do.

---

## Encoding categoricals

| Method | When | Code |
|---|---|---|
| **One-hot / dummies** | **Nominal** (no order): color, city, store type | `pd.get_dummies(df, drop_first=True)` / `OneHotEncoder(handle_unknown="ignore")` |
| **Ordinal** | Categories **with order**: education, size S/M/L | `s.map({"Low": 0, "Med": 1, "High": 2})` / `OrdinalEncoder(categories=[order])` |
| **Binary map** | Yes/No | `(s == "Yes").astype(int)` |
| **Label encoding** | **Target** `y` only (or tree models) | `LabelEncoder()` |
| Frequency / target encoding | High cardinality (100s of levels) | `s.map(s.value_counts())`; target encoding needs CV to avoid leakage |

- `drop_first=True` avoids the **dummy variable trap** (perfect multicollinearity) in linear models. Trees don't need it.
- Assigning 1, 2, 3… to **nominal** categories makes the model think 3 > 1. Use one-hot.
- In production, `OneHotEncoder` remembers train categories; `get_dummies` on new data can produce **different columns**.

---

## Scaling

| Scaler | Result | Use when |
|---|---|---|
| **StandardScaler** | mean 0, std 1 (z-score) | **Default** |
| **MinMaxScaler** | range 0–1 | Bounded inputs (pixels, some neural nets). **Sensitive to outliers** |
| **RobustScaler** | median 0, scaled by IQR | **Outliers present** |
| Log / Box-Cox / Yeo-Johnson | Less skew | Skewed features (`PowerTransformer`) |

| Needs scaling | Doesn't |
|---|---|
| KNN, K-Means, hierarchical, DBSCAN, SVM, PCA, Ridge / Lasso / Logistic (regularized), neural networks | Decision trees, Random Forest, Gradient Boosting / XGBoost |

---

## Feature engineering

| Idea | Example |
|---|---|
| Date → age / duration | `2023 - Store_Establishment_Year`, `Dt_Customer` → year / tenure |
| Date parts | `.dt.year`, `.dt.month`, `.dt.dayofweek`, is_weekend |
| Combine | `Kidhome + Teenhome` = Children; sum of all `Mnt*` = Total_Spend |
| Ratios | spend / purchases = average ticket (guard against / 0) |
| Group categories | 16 product types → perishable / non-perishable |
| Extract from text / IDs | `Product_Id.str[:2]` → FD / DR / NC |
| Flags | accepted any campaign, has kids, missing indicator |
| Binning | `pd.cut(age, [0, 30, 50, 100])` |
| Interactions | `price * area` (helps linear models) |

---

## Data leakage

**Leakage** = information the model wouldn't have at prediction time gets into training, so test scores look great and production fails.

| Leak | Fix |
|---|---|
| Fit scaler / imputer / encoder on **all** data | **Split first**, `fit` on train, `transform` test. Use a `Pipeline` |
| SMOTE / oversampling before the split | Resample **train only** |
| Feature computed from the target (or from the future) | Remove it. Ask "would I know this at prediction time?" |
| Choosing thresholds / hyperparameters on the test set | Use validation / CV |
| Duplicates spread across train and test | Drop duplicates before splitting |

---

## Pipeline + ColumnTransformer (the production way)

```python
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression())])
cross_val_score(model, X_train, y_train, cv=5)      # preprocessing is re-fitted inside every fold
model.fit(X_train, y_train); model.predict(X_new)   # raw data in, prediction out
```

- No leakage, one object to save (`joblib.dump(model, "model.pkl")`) and deploy. This is the bridge to **MLOps**.
- `GridSearchCV` can tune preprocessing too: `param_grid={"prep__num__impute__strategy": ["mean", "median"]}`.

---

## Source material

- Drive `Cheat Sheet` doc: EDA / Data Transformation / Data Preparation sections
- Every project notebook's cleaning steps (Customer Personality, SuperKart, ExtraaLearn)
