# Statistics & Hypothesis Testing Cheatsheet

> **Course modules:** 1. Foundations, Week 2: Statistics for Data Science + the hypothesis testing part of 4. Classification and Hypothesis Testing
> **Code:** [statistics.ipynb](statistics.ipynb): every test on a real question from ExtraaLearn / SuperKart

---

## Descriptive statistics

| Measure | What it tells | Notes |
|---|---|---|
| **Mean** | Average | Pulled by outliers |
| **Median** | Middle value | Robust to outliers. Use it for skewed data |
| **Mode** | Most frequent | For categoricals |
| **Std / variance** | Spread around the mean | Variance = std² (squared units) |
| **IQR** = Q3 − Q1 | Spread of the middle 50% | Robust. Outlier rule: < Q1 − 1.5·IQR or > Q3 + 1.5·IQR |
| **Skewness** | Asymmetry | > 0 right tail (mean > median) · < 0 left tail |
| **Kurtosis** | Tail heaviness | High = more extreme values |
| **Correlation** (Pearson) | Linear relation, −1 to 1 | ≠ causation. Spearman for monotonic / ranks |

**Degrees of freedom** = number of values free to vary. With n observations and a known mean, only n − 1 are free, so the sample std divides by **n − 1**.

---

## Probability distributions

### scipy.stats functions (same API for every distribution)

| Function | Gives | Example |
|---|---|---|
| `pmf(k)` / `pdf(x)` | Probability (density) **at** a value | P(X = 6) |
| `cdf(x)` | **P(X ≤ x)** | P(X ≤ 3) |
| `1 - cdf(x)` / `sf(x)` | P(X > x) | P(X ≥ 10) = `1 - binom.cdf(9, n, p)` |
| `ppf(q)` | Inverse of cdf: the value at percentile q | 90th percentile, critical values |
| `rvs(size)` | Random samples | simulations |

### The main ones

| Distribution | Type | Models | Parameters | Mean |
|---|---|---|---|---|
| **Bernoulli** | discrete | one yes/no trial | p | p |
| **Binomial** | discrete | # successes in n independent trials | n, p | n·p |
| **Poisson** | discrete | # events in a fixed interval (calls per hour) | λ | λ |
| **Uniform** | continuous | all values equally likely | a, b | (a+b)/2 |
| **Normal** | continuous | sums / averages of many effects, measurement errors | μ, σ | μ |
| **t (Student)** | continuous | like normal, fatter tails; small samples, σ unknown | df | 0 |
| **Exponential** | continuous | time between events | λ | 1/λ |

**Normal: empirical rule (68–95–99.7):** 68% of values within μ ± 1σ, 95% within ±2σ, 99.7% within ±3σ.
Example: delivery time μ = 30, σ = 5 gives 68% in 25–35 min, 95% in 20–40, 99.7% in 15–45.

**z-score** = (x − μ) / σ: how many std a value is from the mean. Standard normal = μ 0, σ 1.

---

## Sampling & the Central Limit Theorem

- **CLT:** for n ≥ ~30, the distribution of the **sample mean** is ≈ normal, **whatever the population's shape**.
- **Standard error** SE = σ / √n: the std of the sample mean. Bigger n gives a more precise estimate.
- This is why t-tests work on skewed data when n is large.

## Confidence intervals

- **CI for the mean:** x̄ ± t(1−α/2, n−1) · s/√n, with `stats.t.interval(0.95, df=n-1, loc=x̄, scale=stats.sem(x))`.
- **Meaning:** if we repeated the sampling many times, **95% of the intervals** would contain the true mean. (Not "95% probability the mean is in *this* interval".)
- Wider with higher confidence or smaller n.
- **CI for a proportion:** p̂ ± z · √(p̂(1−p̂)/n).

---

## Hypothesis testing

### The recipe
1. **H0** (null): no effect / no difference / independent. **Ha** (alternative): what you want to show.
2. Pick **α** (significance level, usually **0.05**) and the test (table below). One-tailed (`alternative="greater"`/`"less"`) or two-tailed.
3. Check the test's **assumptions**.
4. Compute the **p-value** = probability of data at least this extreme **if H0 were true**.
5. **p < α → reject H0**. **p ≥ α → fail to reject H0** (not the same as proving H0).

### Errors

| | H0 true | H0 false |
|---|---|---|
| **Reject H0** | **Type I error** (false positive), probability = α | ✅ Correct (**power** = 1 − β) |
| **Fail to reject** | ✅ Correct | **Type II error** (false negative), probability = β |

Same idea as the confusion matrix: Type I = FP, Type II = FN. Bigger samples give more power.

### Which test?

| Question | Data | Test | scipy / statsmodels |
|---|---|---|---|
| Is the mean = a value? | 1 numeric | **One-sample t** | `stats.ttest_1samp(x, popmean)` |
| Do 2 independent groups have the same mean? | numeric × 2 groups | **Two-sample t** (Welch if variances differ) | `stats.ttest_ind(a, b, equal_var=False)` |
| Same subjects before / after? | paired numeric | **Paired t** | `stats.ttest_rel(before, after)` |
| Do 3+ groups have the same mean? | numeric × 3+ groups | **One-way ANOVA** (+ **Tukey HSD** post-hoc) | `stats.f_oneway(*groups)`, `pairwise_tukeyhsd` |
| Are 2 proportions equal? | binary × 2 groups | **Two-proportion z-test** | `proportions_ztest(count, nobs)` |
| Are 2 categorical variables related? | contingency table | **Chi-square independence** | `stats.chi2_contingency(pd.crosstab(a, b))` |
| Same, but not normal / small n / ordinal | — | **Mann-Whitney U** (2 groups), **Wilcoxon** (paired), **Kruskal-Wallis** (3+) | `stats.mannwhitneyu`, `stats.wilcoxon`, `stats.kruskal` |

### Assumption checks (read them the other way around: H0 = "assumption holds")

| Assumption | Test | p < 0.05 means |
|---|---|---|
| Normality | **Shapiro-Wilk** `stats.shapiro(x)` (n < 5000) | Not normal: rely on CLT (large n) or go non-parametric |
| Equal variances | **Levene** `stats.levene(a, b, ...)` | Unequal: Welch t-test / non-parametric test |
| Chi-square | Expected counts ≥ 5 in every cell | Merge categories or use Fisher's exact test |
| Independence | Random sampling, no repeated subjects | Use paired tests |

---

## Common mistakes

- "p > 0.05 so H0 is true." No: you only failed to find evidence against it.
- **Statistically significant ≠ important.** With huge n, tiny differences are significant. Look at the **effect size** (the actual difference).
- Running many tests and reporting the one with p < 0.05 (**p-hacking**). Use ANOVA + Tukey, or correct with Bonferroni (α / number of tests).
- Using a t-test on 3+ groups pairwise instead of ANOVA.
- Reading an assumption test's "reject" as good news.
- Confusing correlation with causation.

---

## Source notebooks (Drive: `MIT GT - Data Science/1. Foundations: Python & Statistics/Week 2`)

- `Notes.ipynb` (binomial pmf/cdf, empirical rule, degrees of freedom, pdf/cdf/ppf)
- `Notebook_Inferential_Statistics`, `Notebook_Hypothesis_Testing`, lecture slides (Inferential Statistics, Hypothesis Testing)
- `MLS 2/Medicon Dose Testing Ro`, `Mobile Internet Usage Analysis`, `Practice Exercise`
