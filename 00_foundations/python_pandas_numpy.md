# Python for Data Science Cheatsheet (NumPy · pandas · seaborn)

> **Course module:** 1. Foundations, Week 1: Python for Data Science
> **Code:** [python_pandas_numpy.ipynb](python_pandas_numpy.ipynb): every snippet below runs on the SuperKart dataset

---

## NumPy

| Task | Code |
|---|---|
| Create | `np.array([1,2,3])`, `np.arange(0, 10, 2)`, `np.linspace(0, 1, 5)`, `np.zeros((2,3))`, `np.ones(3)`, `np.eye(3)` |
| Shape | `a.shape`, `a.ndim`, `a.dtype`, `a.reshape(3, 4)`, `a.ravel()`, `a.T` |
| Index / slice | `m[0, 1]`, `m[:, 0]` (column), `m[1:, 2:]`, `a[a > 2]` (mask), `a[[0, 2]]` (fancy) |
| Vectorized math | `a * 10`, `a ** 2`, `np.sqrt(a)`, `np.log1p(a)`, `np.exp(a)`. **No loops** |
| Aggregate | `a.sum()`, `a.mean()`, `np.median(a)`, `a.std()`, `a.min()`, `a.argmax()`, `np.percentile(a, 90)` |
| `axis` | `axis=0` = down the rows (result per **column**), `axis=1` = across (result per **row**) |
| If / else | `np.where(cond, x, y)` |
| Random | `rng = np.random.default_rng(42)`, then `rng.normal(0, 1, 5)`, `rng.integers(0, 10, 5)`, `rng.choice(arr, 3)` |
| Linear algebra | `A @ B` (matrix product), `np.linalg.inv(A)`, `np.linalg.solve(A, b)`, `np.linalg.eig(A)` |
| Combine | `np.concatenate([a, b])`, `np.vstack`, `np.hstack` |

**Broadcasting:** arrays with compatible shapes combine automatically, e.g. `(3,4) - (4,)` subtracts the vector from every row.

---

## pandas

### Load & inspect

| Task | Code |
|---|---|
| Read | `pd.read_csv(path, sep="\t")`, `pd.read_excel`, `pd.read_parquet`, `pd.read_sql(query, conn)` |
| First look | `df.head()`, `df.sample(5)`, `df.shape`, `df.info()`, `df.dtypes` |
| Summaries | `df.describe().T`, `df.describe(include=["object", "string"]).T` |
| Counts | `s.value_counts()`, `s.value_counts(normalize=True)`, `s.nunique()`, `s.unique()` |
| Quality | `df.isna().sum()`, `df.duplicated().sum()` |

### Select & filter

| Task | Code |
|---|---|
| Columns | `df["col"]` (Series), `df[["a", "b"]]` (DataFrame) |
| By label | `df.loc[rows, cols]`. End **included** |
| By position | `df.iloc[0:5, 0:3]`. End **excluded** |
| Filter | `df[(df.a > 1) & (df.b == "x")]`: use `&`, `\|`, `~` with parentheses |
| In a list | `df[df.col.isin(["a", "b"])]` |
| SQL-like | `df.query("price > 200 and size == 'High'")` |

### Create & transform

| Task | Code |
|---|---|
| New column | `df["age"] = 2023 - df["year"]` |
| Strings | `s.str[:2]`, `s.str.lower()`, `s.str.contains("x")`, `s.str.split("-")`, `s.str.replace("a", "b")` |
| Replace | `s.replace({"reg": "Regular"})` |
| Bins | `pd.cut(s, bins=[0, 100, 200], labels=["low", "high"])`, `pd.qcut(s, 4)` (quartiles) |
| Function | `s.map(dict_or_func)`, `s.apply(func)`, `df.apply(func, axis=1)` (slow, last resort) |
| Dates | `pd.to_datetime(s, format="%d-%m-%Y")`, then `.dt.year`, `.dt.month`, `.dt.day_name()` |
| Types | `s.astype(int)`, `s.astype("category")` |
| Rename / drop | `df.rename(columns={"a": "b"})`, `df.drop(columns=[...])` |
| Sort | `df.sort_values("col", ascending=False)`, `df.nlargest(5, "col")` |

### Group, pivot, merge (SQL / Excel equivalents)

| Task | Code | Equivalent |
|---|---|---|
| Group | `df.groupby("g").agg(total=("x", "sum"), avg=("y", "mean"))` | `GROUP BY` |
| Group → same shape | `df.groupby("g")["x"].transform("median")` | window function |
| Pivot | `pd.pivot_table(df, index="a", columns="b", values="x", aggfunc="mean")` | Excel pivot |
| Cross-tab | `pd.crosstab(df.a, df.b, normalize="index")` | count pivot |
| Join | `left.merge(right, on="key", how="left")` | `LEFT JOIN` |
| Stack | `pd.concat([df1, df2])` | `UNION ALL` |
| Reshape | `df.melt(...)` (wide→long), `df.pivot(...)` (long→wide) | unpivot / pivot |

> ⚠️ **pandas 3**: Copy-on-Write is the default, so `df["col"].fillna(x, inplace=True)` no longer modifies `df`. Write `df["col"] = df["col"].fillna(x)` instead.

---

## Visualization: Plot Guide

| Plot | Function | Shows | Variables |
|---|---|---|---|
| Histogram | `sns.histplot(x=, kde=True, bins=)` | Distribution: shape, skew, modes | 1 numeric |
| Boxplot | `sns.boxplot(x=, y=)` | Median, IQR (box), whiskers 1.5×IQR, outliers | numeric (+ category) |
| Countplot | `sns.countplot(x=)` | **Count** per category | 1 categorical |
| Barplot | `sns.barplot(x=, y=, errorbar=None)` | **Mean** per category (+95% CI by default) | category + numeric |
| Line | `sns.lineplot(x=, y=)` | Trend over time / an ordered variable | 2 (+hue) |
| Scatter | `sns.scatterplot(x=, y=, hue=)` | Relation between 2 numerics, clusters, outliers | 2 numeric |
| lmplot | `sns.lmplot(x=, y=)` | Scatter + regression line + CI | 2 numeric |
| Jointplot | `sns.jointplot(x=, y=, kind="kde")` | Scatter + marginal distributions | 2 numeric |
| Violin | `sns.violinplot(x=, y=)` | Boxplot + density shape | numeric by category |
| Strip / Swarm | `sns.stripplot`, `sns.swarmplot` | Every point per category (swarm: no overlap) | numeric by category |
| Catplot | `sns.catplot(kind="box", col=)` | Any categorical plot, faceted | many |
| Pairplot | `sns.pairplot(df, hue=)` | All pairwise scatters + distributions | many numeric |
| Heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True)` | Matrices (correlation, confusion) | many |

- `barplot` = **mean**, `countplot` = **count**. Easy to mix up.
- seaborn ≥ 0.12: `ci=None` became **`errorbar=None`**.
- Univariate = 1 variable · bivariate = 2 · multivariate = 3+ (`hue`, `col`, `row`, pairplot).
- Reusable helpers in the notebook: `histogram_boxplot()` (box + hist with mean & median) and `labeled_barplot()` (counts / % on bars).

---

## Source notebooks (Drive: `MIT GT - Data Science/1. Foundations: Python & Statistics/Week 1`)

- `Numpy notes`, `Pandas Notes`, `Data Visualization notes`, `Plot Guide` (doc)
- `Python_for_Data_Science_NumPy`, `Pandas_for_Data_Science_Pandas`, `Data_Visualization_with_Python`
- Case studies: `MLS 1/Uber Case Study`, `CardioGoodFitness`, `Quiz - Python for Data Science`
