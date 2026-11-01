# Module 3 · Experimentation & A/B Testing
## Days 24–27: Analysis, CUPED, Multiple Testing and Sequential Tests

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 21–23
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 27:** Students analyse an experiment with regression, robust standard errors and CUPED; quantify how multiple comparisons, peeking and "best segment" reporting inflate false discoveries; and design a sequential test that can stop early without breaking its error rate.

---

### Where we are

The email_v2 test is planned (Day 21), sized (Day 22) and assigned with a stratified, audited service (Day 23). Now it runs, and we analyse it. Every day this week we compare what a method reports with the lab's **truth**: email_v2 raises revenue by **$3.48 per customer (+5%)** over four weeks, and orders by **0.057 (+5%)**.

We run email_v2 **once** for the readout, with the Day 23 assignment, through a small helper used on Days 24–25:

```python
"""The email_v2 experiment, run once, joined to pre-period covariates (Days 24-25)."""
import polars as pl

from causalkit.assignment import add_strata, assign_stratified
from causalkit.lab import population, run


def email_v2_results(seed: int = 24) -> pl.DataFrame:
    pop = population()
    z = assign_stratified(pop, "email_v2")
    out = run(pop, z, "email_v2", seed=seed)
    covariates = add_strata(pop).select("customer_id", "loyalty", "device", "region",
                                        "pre_orders", "pre_revenue", "stratum")
    return out.join(covariates, on="customer_id")
```

---

# DAY 24 — Analysing Experiments with Regression

| Time | Block |
|---|---|
| 0:00 – 0:40 | 24.1 A difference in means is a regression |
| 0:40 – 1:30 | 24.2 Robust standard errors and relative lifts |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 24.3 Covariates: same target, less noise |
| 2:45 – 3:20 | 24.4 Binary outcomes |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 24.5 Day 24 mini-project: a standard readout |
| 4:45 – 5:00 | 24.6 Save your work with Git |

---

## 24.1 A difference in means is a regression (40 min)

**Concept.** Regressing the outcome on a 0/1 treatment dummy, `revenue ~ treated`, gives:

- intercept = mean of the control group,
- coefficient on `treated` = treated mean − control mean.

So the regression *is* the difference in means. Its value comes from what it adds: standard errors, covariates, interactions, clustering, all in one familiar framework.

Create `day24/ols_equals_diff.py`:

```python
# day24/ols_equals_diff.py   (python -m day24.ols_equals_diff)
"""The workhorse: a difference in means IS a regression on a treatment dummy."""
import numpy as np
import statsmodels.formula.api as smf

from causalkit.experiment_data import email_v2_results
from causalkit.lab import truth

df = email_v2_results().to_pandas()

diff = df.loc[df.treated == 1, "revenue"].mean() - df.loc[df.treated == 0, "revenue"].mean()
classical = smf.ols("revenue ~ treated", df).fit()
robust = smf.ols("revenue ~ treated", df).fit(cov_type="HC2")

print(f"Difference in means          {diff:+.3f}")
print(f"OLS coefficient on treated   {classical.params['treated']:+.3f}")
print(f"Classical SE {classical.bse['treated']:.3f}   Robust (HC2) SE {robust.bse['treated']:.3f}")
lo, hi = robust.conf_int().loc["treated"]
print(f"95% CI [{lo:+.3f}, {hi:+.3f}]   p = {robust.pvalues['treated']:.4f}")

# Relative lift and its CI (delta method for a ratio of two means)
t = df.loc[df.treated == 1, "revenue"]; c = df.loc[df.treated == 0, "revenue"]
ratio = t.mean() / c.mean()
se_log = np.sqrt(t.var() / (len(t) * t.mean() ** 2) + c.var() / (len(c) * c.mean() ** 2))
print(f"Relative lift {ratio - 1:+.2%}   95% CI [{ratio * np.exp(-1.96 * se_log) - 1:+.2%}, "
      f"{ratio * np.exp(1.96 * se_log) - 1:+.2%}]")
print(f"Truth: {truth('email_v2')['revenue']:+.3f} per customer, {truth('email_v2')['revenue_relative']:+.1%}")
```

```bash
python -m day24.ols_equals_diff
```

```
Difference in means          +2.446
OLS coefficient on treated   +2.446
Classical SE 1.011   Robust (HC2) SE 1.011
95% CI [+0.464, +4.428]   p = 0.0156
Relative lift +3.49%   95% CI [+0.65%, +6.41%]
Truth: +3.480 per customer, +5.0%
```

This one experiment estimates +$2.45, below the true +$3.48 but well inside its confidence interval [0.46, 4.43]. That's what a single experiment looks like: right on average, off by about one standard error in any given run. Report the interval, not just the point.

---

## 24.2 Robust standard errors and relative lifts (50 min)

**Robust (HC2) standard errors** don't assume both groups have the same variance. With equal group sizes they're almost identical to classical ones (as above). With unequal groups and a treatment that changes the variance, classical standard errors can be badly wrong. Use robust ones by default; it costs nothing.

**Relative lift** ("+3.5%") is how results are usually communicated. Its confidence interval needs care, because it's a ratio of two noisy means. The script uses the **delta method** on the log of the ratio, which gives an asymmetric interval [+0.65%, +6.41%].

**Try it.** Treat only 10% of customers instead of half. Compare classical and robust standard errors.

<details>
<summary>Solution</summary>

```python
# day24/unequal_split.py   (python -m day24.unequal_split)
import numpy as np
import statsmodels.formula.api as smf

from causalkit.lab import population, run

pop = population()
rng = np.random.default_rng(241)
z = (rng.random(pop.height) < 0.10).astype(np.int8)          # only 10% treated
df = run(pop, z, "email_v2", seed=241).to_pandas()
for cov in ["nonrobust", "HC2"]:
    fit = smf.ols("revenue ~ treated", df).fit(cov_type=cov)
    print(f"{cov:<10} effect {fit.params['treated']:+.3f}  SE {fit.bse['treated']:.3f}")
```

```
nonrobust  effect +6.806  SE 1.635
HC2        effect +6.806  SE 1.847
```

With a 10/90 split, the classical SE understates the uncertainty by 11%. Treated customers buy more (the email works), so their revenue is more variable, and the small group's variance dominates the standard error. (The estimate itself is noisy, +6.8 against a truth of +3.48: only 4,000 customers were treated.)

</details>

---

## 24.3 Covariates: same target, less noise (60 min)

**Concept.** Adding pre-treatment covariates to the regression doesn't change what is estimated in a randomised experiment (randomisation already removed bias). It removes the part of the outcome's variation that the covariates explain, so the estimate is less noisy. Two common specifications:

- **ANCOVA**: `revenue ~ treated + pre_revenue + loyalty`.
- **Lin (2013)**: interact treatment with **centred** covariates, `revenue ~ treated * (pre_c + loy_c)`. Lin proved this version never makes precision worse, even if the effect differs across customers.

Create `day24/covariate_adjustment.py`. Each replay draws a fresh *complete* randomisation (not stratified), so covariates have chance imbalances to correct:

```python
# day24/covariate_adjustment.py   (python -m day24.covariate_adjustment)
"""
Adding pre-treatment covariates: same target, less noise. Three specifications, 300 replays,
each with a fresh complete randomisation (no stratification, so covariates have imbalance to fix):
  plain     revenue ~ treated
  ANCOVA    revenue ~ treated + pre_revenue + loyalty
  Lin       revenue ~ treated * (centred covariates)   (Lin 2013: never hurts precision)
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from causalkit.lab import population, run, truth

pop = population()
rng = np.random.default_rng(24)
cov = pop.select("customer_id", "loyalty", "pre_revenue").with_columns(
    pre_c=pl.col("pre_revenue") - pl.col("pre_revenue").mean(),
    loy_c=pl.col("loyalty") - pl.col("loyalty").mean())

SPECS = {"plain": "revenue ~ treated",
         "ANCOVA": "revenue ~ treated + pre_revenue + loyalty",
         "Lin": "revenue ~ treated * (pre_c + loy_c)"}
est = {k: [] for k in SPECS}
covered = {k: 0 for k in SPECS}
target = truth("email_v2")["revenue"]
for r in range(300):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    df = run(pop, z, "email_v2", seed=1000 + r).join(cov, on="customer_id").to_pandas()
    for name, f in SPECS.items():
        fit = smf.ols(f, df).fit(cov_type="HC2")
        est[name].append(fit.params["treated"])
        lo, hi = fit.conf_int().loc["treated"]
        covered[name] += lo <= target <= hi

rows = [{"specification": k, "mean": round(np.mean(v), 3), "SD across replays": round(np.std(v), 3),
         "95% CI coverage": f"{covered[k] / 300:.0%}"} for k, v in est.items()]
print(pl.DataFrame(rows))
print(f"True effect {target:+.3f}")
```

```bash
python -m day24.covariate_adjustment
```

```
│ specification ┆ mean  ┆ SD across replays ┆ 95% CI coverage │
│ plain         ┆ 3.481 ┆ 1.086             ┆ 94%             │
│ ANCOVA        ┆ 3.487 ┆ 0.942             ┆ 92%             │
│ Lin           ┆ 3.487 ┆ 0.942             ┆ 92%             │
True effect +3.480
```

All three are unbiased. Covariates cut the standard deviation by 13%. ANCOVA and Lin agree here because the effect is a constant 5% for everyone. Coverage of 92–94% over 300 replays is within simulation error of 95% (±2.5 points).

> Only **pre-treatment** covariates. A covariate measured after treatment is a bad control (Day 17), even in a perfect experiment.

---

## 24.4 Binary outcomes (35 min)

For a 0/1 outcome like conversion, the same regression is called a **linear probability model** (LPM). Its coefficient is the effect in percentage points, directly. A logit model gives the same answer once converted to an **average marginal effect**, but its raw coefficient (log-odds) is hard to explain to a manager.

Create `day24/binary_outcome.py`:

```python
# day24/binary_outcome.py   (python -m day24.binary_outcome)
"""Conversion (0/1): the linear probability model gives the effect in percentage points directly."""
import numpy as np
import statsmodels.formula.api as smf

from causalkit.experiment_data import email_v2_results
from causalkit.lab import truth

df = email_v2_results().to_pandas()

lpm = smf.ols("converted ~ treated", df).fit(cov_type="HC2")
logit = smf.logit("converted ~ treated", df).fit(disp=False)
ame = logit.get_margeff().summary_frame().loc["treated"]          # average marginal effect

print(f"Linear probability model: {lpm.params['treated'] * 100:+.2f} points  (SE {lpm.bse['treated'] * 100:.2f})")
print(f"Logit, average marginal effect: {ame['dy/dx'] * 100:+.2f} points  (SE {ame['Std. Err.'] * 100:.2f})")
print(f"Logit coefficient (log-odds, harder to explain): {logit.params['treated']:+.4f}")
print(f"Truth: {truth('email_v2')['converted'] * 100:+.2f} points")
```

```
Linear probability model: +2.42 points  (SE 0.49)
Logit, average marginal effect: +2.42 points  (SE 0.49)
Logit coefficient (log-odds, harder to explain): +0.1019
Truth: +1.51 points
```

In a randomised experiment, the LPM with robust standard errors is the standard choice. (This run overshoots the truth by 1.9 standard errors: unusual, but it happens about 6% of the time. Don't adjust your method to the truth after seeing it.)

---

## 24.5 Day 24 mini-project: a standard readout (70 min)

**Goal.** Package the analysis into a function every experiment can use: one row per metric with control mean, effect, robust SE, CI, relative lift and p-value.

Create `causalkit/analysis.py`:

```python
"""
A standard experiment readout (Day 24 mini-project; CUPED added on Day 25).

    from causalkit.analysis import readout
    readout(df, metrics=["revenue", "orders", "converted"], covariates=["pre_revenue", "loyalty"])
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf


def readout(df: pl.DataFrame, metrics: list[str], covariates: list[str] | None = None,
            treatment: str = "treated") -> pl.DataFrame:
    """One row per metric: control mean, absolute effect, robust SE, 95% CI, relative lift, p-value."""
    pdf = df.to_pandas()
    rows = []
    for m in metrics:
        rhs = " + ".join([treatment] + (covariates or []))
        fit = smf.ols(f"{m} ~ {rhs}", pdf).fit(cov_type="HC2")
        control_mean = pdf.loc[pdf[treatment] == 0, m].mean()
        eff, se = fit.params[treatment], fit.bse[treatment]
        rows.append({
            "metric": m,
            "control_mean": round(control_mean, 3),
            "effect": round(eff, 3),
            "se": round(se, 3),
            "ci_low": round(eff - 1.96 * se, 3),
            "ci_high": round(eff + 1.96 * se, 3),
            "lift": f"{eff / control_mean:+.2%}",
            "p_value": f"{fit.pvalues[treatment]:.4f}" if fit.pvalues[treatment] >= 0.0001 else "<0.0001",
        })
    return pl.DataFrame(rows)
```

Then `day24/readout.py`:

```python
# day24/readout.py   (python -m day24.readout)
"""Mini-project: the email_v2 readout, without and with covariate adjustment, against the truth."""
import polars as pl

from causalkit.analysis import readout
from causalkit.experiment_data import email_v2_results
from causalkit.lab import truth

df = email_v2_results()
METRICS = ["revenue", "orders", "converted", "margin"]

with pl.Config(tbl_width_chars=120):
    print("Unadjusted")
    print(readout(df, METRICS))
    print("Adjusted for pre_revenue, pre_orders and strata (strata already include loyalty)")
    print(readout(df, METRICS, covariates=["pre_revenue", "pre_orders", "C(stratum)"]))
t = truth("email_v2")
print("Truth:", {m: round(t[m], 3) for m in METRICS})
```

```bash
python -m day24.readout
```

```
Unadjusted
│ metric    ┆ control_mean ┆ effect ┆ se    ┆ ci_low ┆ ci_high ┆ lift   ┆ p_value │
│ revenue   ┆ 70.095       ┆ 2.446  ┆ 1.011 ┆ 0.464  ┆ 4.428   ┆ +3.49% ┆ 0.0156  │
│ orders    ┆ 1.143        ┆ 0.057  ┆ 0.014 ┆ 0.029  ┆ 0.084   ┆ +4.95% ┆ <0.0001 │
│ converted ┆ 0.598        ┆ 0.024  ┆ 0.005 ┆ 0.015  ┆ 0.034   ┆ +4.05% ┆ <0.0001 │
│ margin    ┆ 21.029       ┆ 0.734  ┆ 0.303 ┆ 0.139  ┆ 1.328   ┆ +3.49% ┆ 0.0156  │
Adjusted for pre_revenue, pre_orders and strata (strata already include loyalty)
│ revenue   ┆ 70.095       ┆ 2.634  ┆ 0.873 ┆ 0.923  ┆ 4.345   ┆ +3.76% ┆ 0.0026  │
│ orders    ┆ 1.143        ┆ 0.053  ┆ 0.012 ┆ 0.029  ┆ 0.077   ┆ +4.61% ┆ <0.0001 │
│ converted ┆ 0.598        ┆ 0.024  ┆ 0.005 ┆ 0.014  ┆ 0.033   ┆ +3.94% ┆ <0.0001 │
│ margin    ┆ 21.029       ┆ 0.79   ┆ 0.262 ┆ 0.277  ┆ 1.304   ┆ +3.76% ┆ 0.0026  │
Truth: {'revenue': 3.48, 'orders': 0.057, 'converted': 0.015, 'margin': 1.044}
```

Adjustment shrinks every revenue and margin SE by about 14%. Don't add `loyalty` next to `C(stratum)`: the strata already contain loyalty, so the two are perfectly collinear and the regression can't separate them.

Write a **Results** section in `docs/EXPERIMENT_PLAN_email_v2.md` using the adjusted table and the plan's decision rule.

---

## 24.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 24: regression analysis, robust SEs, covariates, readout function"
git push
```

### Day 24 checklist

- [ ] I can show that a difference in means equals a regression on a treatment dummy
- [ ] I use robust standard errors by default and can report a relative lift with its CI
- [ ] I can explain why pre-treatment covariates reduce noise without changing the target
- [ ] I can analyse a binary outcome with a linear probability model
- [ ] I have a reusable readout function for every experiment

---

# DAY 25 — CUPED and Variance Reduction

| Time | Block |
|---|---|
| 0:00 – 0:40 | 25.1 Why the pre-period is a gift |
| 0:40 – 1:30 | 25.2 CUPED by hand |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 25.3 1,000 replays: CUPED and capping |
| 2:40 – 3:20 | 25.4 Choosing the covariate |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 25.5 Day 25 mini-project: can CUPED rescue the guardrail? |
| 4:45 – 5:00 | 25.6 Save your work with Git |

---

## 25.1 Why the pre-period is a gift (40 min)

**Concept.** Most of the variation in a customer's revenue during a test has nothing to do with the test: heavy shoppers spend a lot whichever group they're in. That variation is **noise** for the comparison. But it's predictable: a customer's revenue last month predicts their revenue this month.

**CUPED** (Controlled-experiment Using Pre-Experiment Data, Deng, Xu, Kohavi and Walker at Microsoft, 2013) subtracts the predictable part:

```text
Y_cuped = Y − θ × (X − mean(X))        θ = cov(X, Y) / var(X)
```

where X is the same metric measured **before** the test. Because X is pre-treatment, both groups have the same expected X, so the expected difference between groups is unchanged. The variance falls by a factor of (1 − ρ²), where ρ is the correlation between X and Y. CUPED is the regression of Day 24.3 written as a recipe, and it's now standard at large experimentation platforms.

---

## 25.2 CUPED by hand (50 min)

Create `day25/cuped_by_hand.py`:

```python
# day25/cuped_by_hand.py   (python -m day25.cuped_by_hand)
"""
CUPED (Deng et al., Microsoft, 2013): subtract the part of the outcome that the pre-period predicts.
    Y_cuped = Y - theta * (X - mean(X)),   theta = cov(X, Y) / var(X)
Same expected difference between groups (X is pre-treatment), smaller variance: by a factor (1 - rho^2).
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from causalkit.experiment_data import email_v2_results

df = email_v2_results()
y, x, t = (df[c].to_numpy().astype(float) for c in ["revenue", "pre_revenue", "treated"])

theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
rho = np.corrcoef(x, y)[0, 1]
y_cuped = y - theta * (x - x.mean())


def diff_and_se(v: np.ndarray) -> tuple[float, float]:
    a, b = v[t == 1], v[t == 0]
    return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


d0, se0 = diff_and_se(y)
d1, se1 = diff_and_se(y_cuped)
print(f"theta = {theta:.3f}   corr(pre_revenue, revenue) = {rho:.3f}   1 - rho^2 = {1 - rho**2:.3f}")
print(f"Plain:  effect {d0:+.3f}  SE {se0:.3f}")
print(f"CUPED:  effect {d1:+.3f}  SE {se1:.3f}   (variance x {(se1 / se0) ** 2:.3f})")

reg = smf.ols("revenue ~ treated + pre_revenue", df.to_pandas()).fit(cov_type="HC2")
print(f"Regression on treated + pre_revenue: {reg.params['treated']:+.3f}  SE {reg.bse['treated']:.3f}"
      "   (CUPED is this regression, done in two steps)")
print(f"Equivalent sample size: CUPED is worth {1 / (1 - rho**2):.2f}x as many customers")
```

```bash
python -m day25.cuped_by_hand
```

```
theta = 0.299   corr(pre_revenue, revenue) = 0.493   1 - rho^2 = 0.757
Plain:  effect +2.446  SE 1.011
CUPED:  effect +2.553  SE 0.880   (variance x 0.757)
Regression on treated + pre_revenue: +2.553  SE 0.880   (CUPED is this regression, done in two steps)
Equivalent sample size: CUPED is worth 1.32x as many customers
```

With a correlation of 0.49, CUPED keeps 76% of the variance: the same precision as running the plain test on 32% more customers. Free, from data the company already has.

---

## 25.3 1,000 replays: CUPED and capping (55 min)

**Concept.** Revenue has a long right tail: a few very large orders add enormous noise. **Capping** (winsorising) replaces values above a high percentile with that percentile. It reduces variance further, but it changes the estimand: you now measure the effect on *capped* revenue. Whether that's acceptable is a business choice, made in the plan, before the data arrive.

Create `day25/cuped_replays.py`:

```python
# day25/cuped_replays.py   (python -m day25.cuped_replays)
"""
1,000 replays with fresh randomisation: plain, CUPED, and CUPED on revenue capped at the 99th percentile.
Capping (winsorising) cuts variance further, but changes WHAT is estimated: be explicit about it.
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run, truth

pop = population()
x = pop["pre_revenue"].to_numpy()
rng = np.random.default_rng(25)


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


est = {"plain": [], "CUPED": [], "CUPED + cap at p99": []}
for r in range(1000):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    y = run(pop, z, "email_v2", seed=2000 + r)["revenue"].to_numpy()
    cap = np.percentile(y, 99)                       # pooled cap: same rule for both groups
    for name, v in [("plain", y), ("CUPED", cuped(y, x)), ("CUPED + cap at p99", cuped(np.minimum(y, cap), x))]:
        est[name].append(v[z == 1].mean() - v[z == 0].mean())

target = truth("email_v2")["revenue"]
rows = [{"estimator": k, "mean": round(np.mean(v), 3), "SD": round(np.std(v), 3),
         "SD vs plain": f"{np.std(v) / np.std(est['plain']):.0%}"} for k, v in est.items()]
print(pl.DataFrame(rows))
print(f"True effect on (uncapped) revenue: {target:+.3f}")
```

```bash
python -m day25.cuped_replays
```

```
│ estimator          ┆ mean  ┆ SD    ┆ SD vs plain │
│ plain              ┆ 3.494 ┆ 1.002 ┆ 100%        │
│ CUPED              ┆ 3.505 ┆ 0.887 ┆ 88%         │
│ CUPED + cap at p99 ┆ 3.287 ┆ 0.826 ┆ 82%         │
True effect on (uncapped) revenue: +3.480
```

- **CUPED** is unbiased and 12% less noisy.
- **Capping** cuts noise another 6 points, but its average (3.29) is 6% below the true effect on revenue: it estimates the effect on capped revenue, which is smaller because the email also raises some big orders.

Use capping when the decision tolerates it ("we care about typical customers, not whales"), state it in the plan, and apply the same cap to both groups.

---

## 25.4 Choosing the covariate (40 min)

The best CUPED covariate is the one most correlated with the outcome, usually **the same metric in the pre-period**. More covariates can help a little; use regression (Day 24) to combine them.

**Try it.** For orders, revenue and conversion, which pre-period covariate gives the largest variance reduction: `pre_orders` or `pre_revenue`?

<details>
<summary>Solution</summary>

```python
# day25/which_covariate.py   (python -m day25.which_covariate)
import numpy as np

from causalkit.experiment_data import email_v2_results

df = email_v2_results()
for metric in ["orders", "revenue", "converted"]:
    y = df[metric].to_numpy().astype(float)
    for cov in ["pre_orders", "pre_revenue"]:
        rho = np.corrcoef(df[cov].to_numpy(), y)[0, 1]
        print(f"{metric:<10} with {cov:<12} rho = {rho:.3f}  variance kept = {1 - rho**2:.2f}")
```

```
orders     with pre_orders   rho = 0.447  variance kept = 0.80
orders     with pre_revenue  rho = 0.359  variance kept = 0.87
revenue    with pre_orders   rho = 0.369  variance kept = 0.86
revenue    with pre_revenue  rho = 0.493  variance kept = 0.76
converted  with pre_orders   rho = 0.259  variance kept = 0.93
converted  with pre_revenue  rho = 0.206  variance kept = 0.96
```

Each metric's own pre-period version wins. Conversion benefits least: a 0/1 outcome over four weeks is mostly luck. A pre-period *conversion* indicator (did they buy in the previous four weeks?) would help it more than either.

</details>

---

## 25.5 Day 25 mini-project: can CUPED rescue the guardrail? (70 min)

**Goal.** Day 22 found that detecting a $0.50 per-customer margin loss would take 110,000 customers. How far does CUPED close the gap?

Create `day25/guardrail_plan.py`:

```python
# day25/guardrail_plan.py   (python -m day25.guardrail_plan)
"""
Mini-project: Day 22 showed the free-shipping margin guardrail needs ~110,000 customers.
How much does CUPED (with one or two pre-period covariates) bring that down?
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import norm

from causalkit.lab import population, run

pop = population()
last_month = run(pop, np.zeros(pop.height), "aa_test", seed=99).join(
    pop.select("customer_id", "pre_revenue", "pre_orders", "loyalty"), on="customer_id").to_pandas()

Z = norm.ppf(0.975) + norm.ppf(0.80)
LOSS = 0.50                                         # the margin loss per customer we must be able to see
rows = []
for label, covs in [("no CUPED", []), ("CUPED: pre_revenue", ["pre_revenue"]),
                    ("CUPED: pre_revenue + pre_orders + loyalty", ["pre_revenue", "pre_orders", "loyalty"])]:
    if covs:
        r2 = smf.ols("margin ~ " + " + ".join(covs), last_month).fit().rsquared
    else:
        r2 = 0.0
    sd = last_month["margin"].std() * np.sqrt(1 - r2)
    n = 2 * Z**2 * sd**2 / LOSS**2
    rows.append({"design": label, "R2 from pre-period": round(r2, 3),
                 "customers per group": int(round(n, -2)), "total": int(round(2 * n, -2))})
with pl.Config(fmt_str_lengths=50, tbl_width_chars=110):
    print(pl.DataFrame(rows))
print("Available: 40,000 customers. Day 30 will need another idea as well: see 'non-inferiority'.")
```

```bash
python -m day25.guardrail_plan
```

```
│ design                                    ┆ R2 from pre-period ┆ customers per group ┆ total  │
│ no CUPED                                  ┆ 0.0                ┆ 55400               ┆ 110800 │
│ CUPED: pre_revenue                        ┆ 0.255              ┆ 41300               ┆ 82600  │
│ CUPED: pre_revenue + pre_orders + loyalty ┆ 0.267              ┆ 40600               ┆ 81300  │
Available: 40,000 customers. Day 30 will need another idea as well: see 'non-inferiority'.
```

CUPED saves about 28,000 customers, but 81,000 is still twice what NorthMart has. The remaining gap is a question of **framing**, not statistics: a guardrail doesn't need to *detect* a small loss; it needs to *rule out* a loss too large to accept. That's a **non-inferiority** test, which Day 30 uses.

---

## 25.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 25: CUPED, capping, covariate choice, guardrail planning"
git push
```

### Day 25 checklist

- [ ] I can compute CUPED by hand and explain why it doesn't bias the estimate
- [ ] I can predict CUPED's variance reduction from the pre-period correlation
- [ ] I can explain what capping buys and what it changes about the estimand
- [ ] I can choose the best pre-period covariate for a metric

---

# DAY 26 — Multiple Testing and Peeking

| Time | Block |
|---|---|
| 0:00 – 0:30 | 26.1 The garden of forking paths |
| 0:30 – 1:30 | 26.2 Twenty comparisons in an A/A test |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 26.3 Corrections: Bonferroni, Holm, Benjamini–Hochberg |
| 2:30 – 3:15 | 26.4 Peeking: checking every day |
| 3:15 – 3:30 | Break |
| 3:30 – 4:45 | 26.5 Day 26 mini-project: the winner's curse |
| 4:45 – 5:00 | 26.6 Save your work with Git |

---

## 26.1 The garden of forking paths (30 min)

**Concept.** A 5% false positive rate holds for **one** pre-specified test. Real readouts look at many metrics, many segments and many days. Each extra look is another chance for noise to cross the line. Three habits produce most false discoveries in business experiments:

1. **Many comparisons:** 4 metrics × 5 regions × 3 devices = 60 chances to find something.
2. **Peeking:** checking results daily and stopping as soon as they look good.
3. **Reporting the best:** presenting the segment or metric with the biggest effect, without saying it was chosen *because* it was biggest.

None of these is dishonest in intent. All of them are fixed by the same tool: **decide in the plan** which comparisons are confirmatory, and correct for the rest.

---

## 26.2 Twenty comparisons in an A/A test (60 min)

Create `day26/many_comparisons.py`. It runs 300 A/A tests and reads each one 20 ways (4 metrics × 5 regions):

```python
# day26/many_comparisons.py   (python -m day26.many_comparisons)
"""
An A/A test read 20 ways (4 metrics x 5 regions). Nothing works, yet something usually 'wins'.
Then three corrections: Bonferroni, Holm (family-wise error) and Benjamini-Hochberg (false discovery rate).
"""
import numpy as np
import polars as pl
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from causalkit.lab import population, run

pop = population()
region = pop["region"].to_numpy()
METRICS = ["revenue", "orders", "converted", "margin"]
rng = np.random.default_rng(26)

any_hit = {"none": 0, "bonferroni": 0, "holm": 0, "fdr_bh": 0}
REPS = 300
for r in range(REPS):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    out = run(pop, z, "aa_test", seed=3000 + r)
    pvals = []
    for m in METRICS:
        y = out[m].to_numpy()
        for g in np.unique(region):
            sel = region == g
            pvals.append(ttest_ind(y[sel & (z == 1)], y[sel & (z == 0)], equal_var=False).pvalue)
    pvals = np.array(pvals)
    any_hit["none"] += (pvals < 0.05).any()
    for method in ["bonferroni", "holm", "fdr_bh"]:
        any_hit[method] += multipletests(pvals, alpha=0.05, method=method)[0].any()

print(f"{len(METRICS) * 5} comparisons per A/A test, {REPS} A/A tests")
with pl.Config(fmt_str_lengths=70, tbl_width_chars=110):
    print(pl.DataFrame({"correction": list(any_hit),
                        "share of A/A tests with at least one significant result":
                            [round(v / REPS, 3) for v in any_hit.values()]}))
print("Without correction, 1 - 0.95^20 = 0.64 would be expected if the 20 tests were independent.")
```

```bash
python -m day26.many_comparisons
```

```
20 comparisons per A/A test, 300 A/A tests
│ correction ┆ share of A/A tests with at least one significant result │
│ none       ┆ 0.467                                                   │
│ bonferroni ┆ 0.033                                                   │
│ holm       ┆ 0.033                                                   │
│ fdr_bh     ┆ 0.063                                                   │
Without correction, 1 - 0.95^20 = 0.64 would be expected if the 20 tests were independent.
```

Nearly half of all A/A tests (where nothing happened) produce at least one "significant" regional result. It's below the 64% of 20 independent tests because the four metrics are correlated (a region's revenue and margin move together), but it's still a coin toss.

---

## 26.3 Corrections: Bonferroni, Holm, Benjamini–Hochberg (45 min)

| Method | Controls | Rule (m tests) | Use it for |
|---|---|---|---|
| **Bonferroni** | family-wise error rate (FWER): P(any false positive) ≤ α | reject if p < α / m | few tests, every claim must hold |
| **Holm** | FWER, uniformly more powerful than Bonferroni | step-down: compare the k-th smallest p with α / (m − k + 1) | the default when you need FWER |
| **Benjamini–Hochberg** | false discovery rate (FDR): expected share of false claims among claims ≤ α | step-up: largest k with p₍ₖ₎ ≤ k·α / m | many exploratory tests, a few false leads acceptable |

In an A/A test every discovery is false, so FDR equals FWER and all three keep the rate near 5%. The difference shows when there **are** real effects.

**Try it.** Run the same 20 comparisons on email_v2, which really works in every region. On average, how many of the 20 real effects does each method find?

<details>
<summary>Solution</summary>

```python
# day26/real_effect_corrections.py   (python -m day26.real_effect_corrections)
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from causalkit.lab import population, run

pop = population()
region = pop["region"].to_numpy()
rng = np.random.default_rng(262)
found = {"none": 0, "holm": 0, "fdr_bh": 0}
for r in range(100):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    out = run(pop, z, "email_v2", seed=8000 + r)
    p = []
    for m in ["revenue", "orders", "converted", "margin"]:
        y = out[m].to_numpy()
        for g in np.unique(region):
            s = region == g
            p.append(ttest_ind(y[s & (z == 1)], y[s & (z == 0)], equal_var=False).pvalue)
    p = np.array(p)
    found["none"] += (p < 0.05).sum()
    for method in ["holm", "fdr_bh"]:
        found[method] += multipletests(p, method=method)[0].sum()
print({k: round(float(v) / 100, 2) for k, v in found.items()}, "true effects found per experiment (of 20)")
```

```
{'none': 7.34, 'holm': 2.36, 'fdr_bh': 4.76} true effects found per experiment (of 20)
```

Holm finds about 2 of the 20 real regional effects, Benjamini–Hochberg twice as many. Neither finds most of them: a region is a fifth of the sample, so each regional test is underpowered (Day 22). Splitting a test into many segments doesn't create information; it divides it.

</details>

---

## 26.4 Peeking: checking every day (45 min)

**Concept.** A p-value is valid at the sample size you planned. If you compute it every day and stop the first time it dips below 0.05, you give noise 28 chances instead of one.

Create `day26/peeking.py`. It simulates 2,000 A/A tests where 700 new customers per group arrive each day, with revenue drawn from last month's real distribution:

```python
# day26/peeking.py   (python -m day26.peeking)
"""
Peeking: an A/A test checked every day for 28 days, stopped the first time p < 0.05.
Customers' 4-week revenue is resampled from last month's real distribution.
"""
import numpy as np
from scipy.stats import norm

from causalkit.lab import population, run

pop = population()
revenue = run(pop, np.zeros(pop.height), "aa_test", seed=99)["revenue"].to_numpy()
rng = np.random.default_rng(260)

SIMS, DAYS, PER_DAY = 2000, 28, 700            # 700 new customers per group per day (~40k in total)
stopped_by = np.zeros(SIMS, dtype=bool)
final_sig = np.zeros(SIMS, dtype=bool)
first_day = np.full(SIMS, -1)
for s in range(SIMS):
    a = rng.choice(revenue, size=(DAYS, PER_DAY))           # group A, by day
    b = rng.choice(revenue, size=(DAYS, PER_DAY))           # group B: same distribution (A/A)
    n = PER_DAY * np.arange(1, DAYS + 1)
    sa, sb = a.sum(1).cumsum(), b.sum(1).cumsum()
    qa, qb = (a**2).sum(1).cumsum(), (b**2).sum(1).cumsum()
    va, vb = qa / n - (sa / n) ** 2, qb / n - (sb / n) ** 2
    zstat = (sa / n - sb / n) / np.sqrt(va / n + vb / n)
    p = 2 * norm.sf(np.abs(zstat))
    hits = np.flatnonzero(p < 0.05)
    if hits.size:
        stopped_by[s] = True
        first_day[s] = hits[0] + 1
    final_sig[s] = p[-1] < 0.05

print(f"Analyse once, on day 28:                false positive rate {final_sig.mean():.3f}")
print(f"Peek daily, stop at the first p < 0.05: false positive rate {stopped_by.mean():.3f}")
print(f"Of the peeking 'wins', share declared in the first week: {(first_day[stopped_by] <= 7).mean():.0%}")
```

```bash
python -m day26.peeking
```

```
Analyse once, on day 28:                false positive rate 0.057
Peek daily, stop at the first p < 0.05: false positive rate 0.292
Of the peeking 'wins', share declared in the first week: 58%
```

Daily peeking turns a 5% error rate into **29%**. Most false wins appear in the first week, when samples are small and noisy, which is exactly when an eager team is most tempted to call the result. Day 27 shows how to look early *legitimately*.

---

## 26.5 Day 26 mini-project: the winner's curse (75 min)

**Goal.** email_v2 lifts orders by exactly 5% in **every** region. A dashboard shows the regional results and the team reports the best one: "+X% in our best region!" How big does that winner look on average, and which region tends to win?

Create `day26/winners_curse.py`:

```python
# day26/winners_curse.py   (python -m day26.winners_curse)
"""
Mini-project: the 'best region' in an experiment. email_v2 lifts orders by 5% in EVERY region.
Report the region with the biggest estimated lift, 300 times: how big does the winner look?
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run

pop = population()
region = pop["region"].to_numpy()
regions = np.unique(region)
rng = np.random.default_rng(261)

winners, winner_names, all_lifts = [], [], []
for r in range(300):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    y = run(pop, z, "email_v2", seed=4000 + r)["orders"].to_numpy()
    lifts = {g: y[(region == g) & (z == 1)].mean() / y[(region == g) & (z == 0)].mean() - 1 for g in regions}
    best = max(lifts, key=lifts.get)
    winners.append(lifts[best]); winner_names.append(best); all_lifts.extend(lifts.values())

print(f"True lift in every region:            +5.0%")
print(f"Average estimated lift, all regions:  {np.mean(all_lifts):+.1%}")
print(f"Average lift of the 'winning' region: {np.mean(winners):+.1%}")
print(pl.Series("winning region", winner_names).value_counts().sort("count", descending=True))
```

```bash
python -m day26.winners_curse
```

```
True lift in every region:            +5.0%
Average estimated lift, all regions:  +5.2%
Average lift of the 'winning' region: +8.7%
│ winning region ┆ count │
│ ATL            ┆ 81    │
│ AB             ┆ 61    │
│ BC             ┆ 61    │
│ QC             ┆ 60    │
│ ON             ┆ 37    │
```

- The average region is estimated correctly (+5.2%); the *selected* region overstates the effect by 74% (+8.7%).
- The **smallest** regions win most often (ATL and AB are 12% of customers each; ON is 38%). Small samples are noisier, so their extremes are more extreme. "Our best results come from small markets" is often just this.

Add a **Reporting policy** section to `docs/EXPERIMENT_PLAN_email_v2.md`: one primary metric (confirmatory, no correction), secondary metrics with Holm, segments as exploratory with Benjamini–Hochberg and a sentence warning that the largest segment effect is biased upward.

---

## 26.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 26: multiple comparisons, corrections, peeking, winner's curse"
git push
```

### Day 26 checklist

- [ ] I can explain why many comparisons inflate false positives, and estimate by how much
- [ ] I can choose between Bonferroni, Holm and Benjamini–Hochberg
- [ ] I can show by simulation what daily peeking does to the error rate
- [ ] I can explain the winner's curse and why small segments win most often
- [ ] I have a written reporting policy for metrics and segments

---

# DAY 27 — Sequential Testing

| Time | Block |
|---|---|
| 0:00 – 0:40 | 27.1 Why stop early, and how to do it honestly |
| 0:40 – 1:40 | 27.2 Calibrating boundaries by simulation |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 27.3 Pocock versus O'Brien–Fleming |
| 2:40 – 3:15 | 27.4 Weekly outcomes from the lab |
| 3:15 – 3:30 | Break |
| 3:30 – 4:45 | 27.5 Day 27 mini-project: four stopping rules, 400 experiments |
| 4:45 – 5:00 | 27.6 Save your work with Git |

---

## 27.1 Why stop early, and how to do it honestly (40 min)

**Concept.** There are good reasons to look before the planned end: stop a harmful change quickly, or ship a clear winner sooner. Day 26 showed that looking with the usual 1.96 threshold every time breaks the error rate. The fix, developed for clinical trials, is a **group-sequential design**:

1. Fix the number of looks in advance (here K = 4, once a week).
2. At each look, compare the z-statistic with a **boundary** higher than 1.96.
3. Choose the boundaries so the probability of *ever* crossing one, if there's no effect, is exactly 5%.

The boundaries "spend" the 5% error budget across the looks. Where you spend it is a design choice.

---

## 27.2 Calibrating boundaries by simulation (60 min)

If there's no effect, the z-statistic at successive looks behaves like a random walk divided by √k: each look adds a new week of data to the old. We simulate 400,000 such walks and search for the boundary scale that gives a 5% crossing rate.

Create `day27/boundaries.py`:

```python
# day27/boundaries.py   (python -m day27.boundaries)
"""
Group-sequential boundaries for K = 4 weekly looks, found by simulation.
Under 'no effect', the z-statistic at look k behaves like a random walk scaled by sqrt(k).
  Pocock:          same critical value at every look
  O'Brien-Fleming: very strict early, close to 1.96 at the end:  c * sqrt(K / k)
Each is calibrated so the chance of EVER crossing is 5%.
"""
import numpy as np

K, SIMS, ALPHA = 4, 400_000, 0.05
rng = np.random.default_rng(27)
paths = rng.standard_normal((SIMS, K)).cumsum(axis=1) / np.sqrt(np.arange(1, K + 1))   # z at each look
looks = np.arange(1, K + 1)


def crossing_rate(bounds: np.ndarray) -> float:
    return (np.abs(paths) > bounds).any(axis=1).mean()


def calibrate(shape: np.ndarray) -> np.ndarray:
    lo, hi = 1.0, 5.0
    for _ in range(40):                                  # bisection on the scale c
        c = (lo + hi) / 2
        lo, hi = (c, hi) if crossing_rate(c * shape) > ALPHA else (lo, c)
    return hi * shape


naive = np.full(K, 1.96)
pocock = calibrate(np.ones(K))
obf = calibrate(np.sqrt(K / looks))

if __name__ == "__main__":
    for name, b in [("naive 1.96 every week", naive), ("Pocock", pocock), ("O'Brien-Fleming", obf)]:
        print(f"{name:<22} boundaries {np.round(b, 3)}   overall false positive rate {crossing_rate(b):.3f}")
```

```bash
python -m day27.boundaries
```

```
naive 1.96 every week  boundaries [1.96 1.96 1.96 1.96]   overall false positive rate 0.126
Pocock                 boundaries [2.361 2.361 2.361 2.361]   overall false positive rate 0.050
O'Brien-Fleming        boundaries [4.049 2.863 2.338 2.024]   overall false positive rate 0.050
```

These match the published tables (Pocock 1977: 2.361; O'Brien and Fleming 1979: 2.024 at the final of four looks). Simulation lets you calibrate any design, including uneven looks or a different number of them, without tables.

Even four looks at 1.96 give a 12.6% false positive rate: less than daily peeking's 29%, but still 2.5 times the promise.

---

## 27.3 Pocock versus O'Brien–Fleming (45 min)

| | Pocock | O'Brien–Fleming |
|---|---|---|
| Boundaries | equal at every look | very high early, near 1.96 at the end |
| Early stopping | easier | only for overwhelming effects |
| Final look | stricter (2.36): loses power if you run to the end | almost a fixed test (2.02) |
| Typical use | when stopping early is the main goal | when the planned test is the main goal and early looks are a safety net |

O'Brien–Fleming is the more common choice in practice: it costs almost nothing if you end up running the full test, and it protects against stopping on an early fluke.

> **Going further:** "always-valid" methods (mixture sequential probability ratio tests, confidence sequences) allow looking at *any* time, with no fixed schedule. Several commercial platforms use them. They're built on the same idea: pay for flexibility with wider early boundaries.

---

## 27.4 Weekly outcomes from the lab (35 min)

Sequential tests need results as they accumulate. The lab's `run_weekly()` (at the end of `causalkit/lab.py`) returns one row per customer per week, so we can compute cumulative revenue after weeks 1, 2, 3 and 4 for the same customers. Read its docstring, then look at how the mini-project uses it.

---

## 27.5 Day 27 mini-project: four stopping rules, 400 experiments (75 min)

**Goal.** Run email_v2 (real effect) and an A/A test 200 times each with weekly looks. For four stopping rules, record how often each declares significance and how many weeks the test runs on average.

Create `day27/sequential_in_lab.py`:

```python
# day27/sequential_in_lab.py   (python -m day27.sequential_in_lab)
"""
Mini-project: run email_v2 (real effect) and an A/A test 200 times each, looking at cumulative
revenue every week. Compare three stopping rules on false positives, power and duration.
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run_weekly
from day27.boundaries import naive, obf, pocock

pop = population()
rng = np.random.default_rng(270)
RULES = {"fixed: look once at week 4": np.array([np.inf, np.inf, np.inf, 1.96]),
         "naive: 1.96 every week": naive, "Pocock": pocock, "O'Brien-Fleming": obf}


def weekly_z(experiment: str, seed: int) -> np.ndarray:
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    w = run_weekly(pop, z, experiment, seed=seed).sort("customer_id", "week")
    rev = w["revenue"].to_numpy().reshape(pop.height, 4).cumsum(axis=1)   # cumulative by customer
    t = z.astype(bool)
    diff = rev[t].mean(0) - rev[~t].mean(0)
    se = np.sqrt(rev[t].var(0, ddof=1) / t.sum() + rev[~t].var(0, ddof=1) / (~t).sum())
    return diff / se


results = []
for experiment in ["aa_test", "email_v2"]:
    zs = np.array([weekly_z(experiment, 5000 + r) for r in range(200)])
    for name, bounds in RULES.items():
        crossed = np.abs(zs) > bounds
        stop_week = np.where(crossed.any(1), crossed.argmax(1) + 1, 4)
        results.append({"experiment": experiment, "rule": name,
                        "share significant": round(crossed.any(1).mean(), 3),
                        "average weeks run": round(stop_week.mean(), 2)})

with pl.Config(fmt_str_lengths=40, tbl_width_chars=110, tbl_rows=-1):
    print(pl.DataFrame(results))
```

```bash
python -m day27.sequential_in_lab
```

```
│ experiment ┆ rule                       ┆ share significant ┆ average weeks run │
│ aa_test    ┆ fixed: look once at week 4 ┆ 0.04              ┆ 4.0               │
│ aa_test    ┆ naive: 1.96 every week     ┆ 0.13              ┆ 3.72              │
│ aa_test    ┆ Pocock                     ┆ 0.045             ┆ 3.9               │
│ aa_test    ┆ O'Brien-Fleming            ┆ 0.035             ┆ 3.99              │
│ email_v2   ┆ fixed: look once at week 4 ┆ 0.945             ┆ 4.0               │
│ email_v2   ┆ naive: 1.96 every week     ┆ 0.96              ┆ 1.69              │
│ email_v2   ┆ Pocock                     ┆ 0.915             ┆ 2.06              │
│ email_v2   ┆ O'Brien-Fleming            ┆ 0.94              ┆ 2.64              │
```

Read the two halves together:

- **Naive peeking** finds email_v2 fastest (1.7 weeks), but would have "found" an effect in 13% of A/A tests: the speed is bought with false discoveries.
- **O'Brien–Fleming** keeps the A/A error rate at the planned level, keeps 94% power (the fixed test has 94.5%), and still stops email_v2 after 2.6 weeks on average: **a third of the test time saved, for free**.
- **Pocock** stops a little earlier but loses a few points of power by being stricter at the end.

(Rates from 200 replays have a simulation error of about ±1.5 points for the A/A rows.)

Add a **Monitoring** section to `docs/EXPERIMENT_PLAN_email_v2.md`: four weekly looks, O'Brien–Fleming boundaries, and a rule that only the primary metric can trigger an early stop.

---

## 27.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 27: group-sequential boundaries, early stopping in the lab"
git push
```

**Coming next (Days 28–30).** Everything so far assumed one customer's treatment doesn't affect another, and that we can randomise customers. Days 28–29 drop both assumptions. Day 30 is the capstone: the free-shipping experiment, from plan to decision.

### Day 27 checklist

- [ ] I can explain why repeated looks need higher thresholds, and how boundaries spend α
- [ ] I can calibrate group-sequential boundaries by simulation
- [ ] I can choose between Pocock and O'Brien–Fleming for a business test
- [ ] I can show that a sequential design saves time without inflating false positives
