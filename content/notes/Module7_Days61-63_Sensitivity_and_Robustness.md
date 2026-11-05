# Module 7 · Sensitivity, Robustness & Decisions
## Days 61–63: Sensitivity to Hidden Confounding, Robustness Values and Specification Curves

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–6
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 63:** Students can quantify how strong a hidden confounder would have to be to change a conclusion (DoubleML sensitivity bounds, robustness values, E-values), make that strength meaningful by benchmarking against observed covariates, and show how much an answer depends on analysis choices with refutation tests and a specification curve.

---

### Where we are

Every module so far ended with an assumption the data couldn't check: no hidden confounders (Modules 4 and 6), parallel trends (Module 5), exclusion (Module 4). Module 7 stops treating those assumptions as yes/no and asks the two questions a decision-maker actually has:

1. **How wrong would the assumption have to be to change the decision?** (Days 61–63)
2. **Given everything we know and don't know, what should we do?** (Days 64–70)

| Days | Question | Tools |
|---|---|---|
| 61–63 | How fragile is the estimate? | DoubleML sensitivity, robustness values, E-values, specification curves |
| 64–66 | Whom should we treat, and is the programme worth it? | uplift curves, EconML policy trees, cost-benefit analysis |
| 67–70 | How do we decide under uncertainty, and when should we learn more first? | decision criteria, value of information, the decision memo |

The running example is Module 6's coupon programme, now with real money attached: **a coupon costs $5, NorthMart keeps 30% of extra spend, so a coupon pays for itself if it raises spending by more than $5 / 0.30 = $16.67.** The true average effect is $16.51, just below that line. Every method in this module is tested on how well it handles a close call.

### Small changes to earlier files

Two simulators gain a hidden column so we can measure the real confounder later (an analyst wouldn't have it): `_browsing` in `causalkit/coupons.py` and `_affinity` in `causalkit/loyalty.py`. Add `"_browsing": browsing` to the dictionary at the end of `coupons()`, and `"_affinity": affinity` next to `"_ps": ps` in `world()`. Nothing else changes; every earlier result is identical.

```bash
mkdir -p day61 day62 day63 day64 day65 day66 day67 day68 day69 day70
```

No new packages this module. (The usual uplift package, scikit-uplift, no longer runs with current numpy and scikit-learn; on Day 64 we write the 40 lines we need instead.)

---

# DAY 61 — Sensitivity to Hidden Confounding

| Time | Block |
|---|---|
| 0:00 – 0:40 | 61.1 What a hidden confounder does, in one formula |
| 0:40 – 1:30 | 61.2 Sensitivity analysis with DoubleML |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 61.3 Reading the contour plot |
| 2:30 – 3:10 | 61.4 Robustness values say nothing about whether confounding exists |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 61.5 Day 61 mini-project: benchmarking |
| 4:40 – 5:00 | 61.6 Save your work with Git |

---

## 61.1 What a hidden confounder does, in one formula (40 min)

The coupon world of Day 54: NorthMart's targeting model also used a real-time **browsing signal** that was never stored. `coupons(hidden=0.5)` builds it in. Because it's a simulator, we can run the regression the analyst can't.

Create `day61/ovb_by_hand.py`:

```python
# day61/ovb_by_hand.py   (python -m day61.ovb_by_hand)
"""
The omitted-variable bias formula, with a confounder we can see because we built the simulator.
NorthMart's targeting model also used a real-time browsing signal that was never stored (hidden=0.5).
  short regression: spend on coupon + features            (what the analyst can run)
  long regression:  spend on coupon + features + browsing (what the analyst wishes they could run)
OVB = (effect of browsing on spend) x (difference in browsing between couponed and other customers, given X).
"""
import statsmodels.formula.api as smf

from causalkit.coupons import X_COLS, coupons

df = coupons(hidden=0.5).to_pandas()
controls = " + ".join(X_COLS)
short = smf.ols(f"spend ~ coupon + {controls}", df).fit()
long = smf.ols(f"spend ~ coupon + _browsing + {controls}", df).fit()
imbalance = smf.ols(f"_browsing ~ coupon + {controls}", df).fit()

gamma, delta = long.params["_browsing"], imbalance.params["coupon"]
print(f"short regression (browsing omitted):  {short.params['coupon']:6.2f}")
print(f"long regression (browsing included):  {long.params['coupon']:6.2f}")
print(f"difference:                           {short.params['coupon'] - long.params['coupon']:6.2f}")
print(f"OVB formula: gamma x delta = {gamma:.2f} x {delta:.3f} = {gamma * delta:.2f}")
print("   gamma: extra $ of spend per 1 SD of browsing;  delta: how much more browsing couponed customers did")
```

```bash
python -m day61.ovb_by_hand
```

```
short regression (browsing omitted):   21.61
long regression (browsing included):   13.34
difference:                             8.28
OVB formula: gamma x delta = 14.85 x 0.557 = 8.28
   gamma: extra $ of spend per 1 SD of browsing;  delta: how much more browsing couponed customers did
```

The **omitted-variable bias** formula is exact: bias = (how much the confounder moves the outcome) × (how unequal it is between treated and untreated, given the controls). A confounder that does only one of the two does no harm. Every sensitivity method in this module is a way of putting plausible numbers on those two factors **without** seeing the confounder.

(The long regression's $13.34 is below the true $16.51 because linear controls can't capture the confounding by the observed features; that's Day 51's lesson, separate from today's.)

---

## 61.2 Sensitivity analysis with DoubleML (50 min)

**Concept.** Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022, "Long story short") extend the OVB formula to DML. A hidden confounder is described by two numbers, both shares between 0 and 1:

- **cf_y**: how much of the outcome's remaining variation it would explain;
- **cf_d**: roughly, how much of the treatment's remaining variation it would explain.

(A third parameter, ρ, is the correlation between the two; ρ = 1 is the worst case.) For any (cf_y, cf_d), DoubleML returns the range the true effect could lie in. The **robustness value (RV)** is the smallest equal strength (cf_y = cf_d) that could move the estimate to a chosen threshold.

Create `day61/dml_sensitivity.py`. Its `fit_irm` function is reused tomorrow:

```python
# day61/dml_sensitivity.py   (python -m day61.dml_sensitivity)
"""
In real data you can't run the long regression. Sensitivity analysis asks instead: how strong would a hidden
confounder have to be to change the conclusion? DoubleML implements Chernozhukov, Cinelli, Newey, Sharma &
Syrgkanis (2022): a confounder's strength is described by
    cf_y: share of the outcome's remaining variation it explains,
    cf_d: how much it changes the treatment's (Riesz representer's) variation, roughly the same idea for the coupon,
and the bounds say how far the estimate could move.
The business threshold: a coupon pays for itself if it raises spend by more than $16.67 (Day 65).
"""
import doubleml as dml
import matplotlib.pyplot as plt
import numpy as np

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

np.random.seed(61)  # DoubleML draws its cross-fitting folds from numpy's global RNG
BREAK_EVEN = 16.67


def fit_irm(df):
    data = dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS)
    return dml.DoubleMLIRM(data, ml_g=reg(), ml_m=clf(), n_folds=5, trimming_threshold=0.02).fit()


if __name__ == "__main__":
    full = coupons(hidden=0.5)
    irm = fit_irm(public(full).to_pandas())
    print(f"IRM ATE: {irm.coef[0]:.2f} (SE {irm.se[0]:.2f})   truth {truth(full)['ATE']:.2f}")
    for cf in [0.01, 0.03, 0.05, 0.10]:
        irm.sensitivity_analysis(cf_y=cf, cf_d=cf, rho=1.0)
        p = irm.sensitivity_params
        print(f"confounder with cf_y = cf_d = {cf:.2f}: estimate between {p['theta']['lower'][0]:6.2f} and "
              f"{p['theta']['upper'][0]:6.2f}  (CI {p['ci']['lower'][0]:6.2f} to {p['ci']['upper'][0]:6.2f})")
    irm.sensitivity_analysis(null_hypothesis=0.0)
    rv0 = irm.sensitivity_params["rv"][0]
    irm.sensitivity_analysis(null_hypothesis=BREAK_EVEN)
    rv_be = irm.sensitivity_params["rv"][0]
    print(f"Robustness value, effect = 0:            {rv0:.1%}")
    print(f"Robustness value, effect = break-even:   {rv_be:.1%}")

    grid = np.linspace(0.0, 0.2, 21)
    lower = np.zeros((len(grid), len(grid)))
    for i, cy in enumerate(grid):
        for j, cd in enumerate(grid):
            irm.sensitivity_analysis(cf_y=cy, cf_d=min(cd, 0.99), rho=1.0)
            lower[i, j] = irm.sensitivity_params["theta"]["lower"][0]
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(grid, grid, lower, levels=[0, 8, BREAK_EVEN, 20, 22, 24], colors="#64748b")
    ax.clabel(cs, fmt="%.1f")
    ax.contour(grid, grid, lower, levels=[BREAK_EVEN], colors="#dc2626", linewidths=2)
    ax.set_xlabel("cf_d (strength with the coupon)")
    ax.set_ylabel("cf_y (strength with spend)")
    ax.set_title("Lower bound of the coupon effect ($)")
    fig.tight_layout()
    fig.savefig("figures/day61_sensitivity_contour.png", dpi=120)
    print("Saved figures/day61_sensitivity_contour.png")
```

```bash
python -m day61.dml_sensitivity
```

```
IRM ATE: 24.61 (SE 0.64)   truth 16.51
confounder with cf_y = cf_d = 0.01: estimate between  23.74 and  25.47  (CI  22.69 to  26.53)
confounder with cf_y = cf_d = 0.03: estimate between  21.97 and  27.24  (CI  20.92 to  28.29)
confounder with cf_y = cf_d = 0.05: estimate between  20.17 and  29.04  (CI  19.12 to  30.10)
confounder with cf_y = cf_d = 0.10: estimate between  15.49 and  33.72  (CI  14.44 to  34.78)
Robustness value, effect = 0:            24.7%
Robustness value, effect = break-even:   8.8%
Saved figures/day61_sensitivity_contour.png
```

Two very different statements from the same estimate:

- "Coupons raise spending": robust. A confounder would need 25% strength on both sides to explain it away entirely.
- "Coupons pay for themselves": fragile. A confounder of 8.8% strength could push the effect below $16.67.

**The robustness value depends on the question.** Always compute it against the threshold the decision depends on, not against zero.

---

## 61.3 Reading the contour plot (45 min)

Open `figures/day61_sensitivity_contour.png`. Each point is a hypothetical confounder (strength with the coupon on the x-axis, with spend on the y-axis); the lines show the lowest plausible effect. The red line is the break-even: any confounder above and to the right of it could make the coupon unprofitable.

The plot turns an argument nobody can win ("there might be confounders") into one people can have ("is there plausibly anything as strong as this, that we didn't measure?"). That's a question for the people who built the targeting model, not for the data.

---

## 61.4 Robustness values say nothing about whether confounding exists (40 min)

**Try it.** Rerun the sensitivity analysis on `coupons(hidden=0.0)`, a world with **no** hidden confounder. What's the robustness value for break-even? Create `day61/no_hidden.py`.

<details>
<summary>Solution</summary>

```python
# day61/no_hidden.py   (python -m day61.no_hidden)
import numpy as np

from causalkit.coupons import coupons, public, truth
from day61.dml_sensitivity import BREAK_EVEN, fit_irm

np.random.seed(61)
full = coupons(hidden=0.0)
irm = fit_irm(public(full).to_pandas())
irm.sensitivity_analysis(null_hypothesis=BREAK_EVEN)
print(f"No hidden confounder: ATE {irm.coef[0]:.2f} (truth {truth(full)['ATE']:.2f}); "
      f"robustness value for break-even {irm.sensitivity_params['rv'][0]:.1%}")
```

```
No hidden confounder: ATE 16.25 (truth 16.51); robustness value for break-even 0.5%
```

0.5%: extremely fragile, in a world with no confounding at all. The RV measures how far the estimate sits from the threshold relative to its precision. Here the estimate ($16.25) is almost exactly at break-even, so the tiniest confounder could tip it either way. A low RV says "the conclusion depends on there being almost no hidden confounding", not "there is hidden confounding".

</details>

---

## 61.5 Day 61 mini-project: benchmarking (75 min)

"cf = 0.088" means nothing to a marketing director. **Benchmarking** gives it meaning: pretend an observed feature had been left out, and measure its strength on the same scale. Then say: "a hidden confounder as strong as app use would move the estimate by $X". Create `day61/benchmarking.py`:

```python
# day61/benchmarking.py   (python -m day61.benchmarking)
"""
Mini-project: 'cf = 0.05' means nothing to a marketing director. Benchmarking does: pretend an OBSERVED
feature had been left out, and measure how strong it is on the same scale. 'A hidden confounder as strong
as app use would move the estimate by $X.' Then, because this is a simulator, measure the real hidden
browsing signal the same way.
"""
import doubleml as dml
import numpy as np
import polars as pl

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg
from day61.dml_sensitivity import fit_irm

np.random.seed(61)
full = coupons(hidden=0.5)
irm = fit_irm(public(full).to_pandas())
rows = []
for feature in ["deal_share", "app_user", "email_open", "pre_spend"]:
    b = irm.sensitivity_benchmark(benchmarking_set=[feature]).iloc[0]
    rows.append({"left out": feature, "cf_y": round(b.cf_y, 3), "cf_d": round(b.cf_d, 3), "rho": round(b.rho, 2),
                 "estimate moves by": round(b.delta_theta, 2)})

df = full.to_pandas()
cols = X_COLS + ["_browsing"]
np.random.seed(61)
oracle = dml.DoubleMLIRM(dml.DoubleMLData(df[cols + ["coupon", "spend"]], "spend", "coupon", cols),
                         ml_g=reg(), ml_m=clf(), n_folds=5, trimming_threshold=0.02).fit()
b = oracle.sensitivity_benchmark(benchmarking_set=["_browsing"]).iloc[0]
rows.append({"left out": "browsing (the real hidden one)", "cf_y": round(b.cf_y, 3), "cf_d": round(b.cf_d, 3),
             "rho": round(b.rho, 2), "estimate moves by": round(b.delta_theta, 2)})
with pl.Config(tbl_width_chars=110, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
print(f"Estimate with browsing included: {oracle.coef[0]:.2f}")
```

```bash
python -m day61.benchmarking
```

```
│ left out                       ┆ cf_y  ┆ cf_d  ┆ rho  ┆ estimate moves by │
│ deal_share                     ┆ 0.034 ┆ 0.138 ┆ 0.45 ┆ 2.48              │
│ app_user                       ┆ 0.165 ┆ 0.049 ┆ 0.63 ┆ 4.77              │
│ email_open                     ┆ 0.074 ┆ 0.002 ┆ 0.21 ┆ 0.2               │
│ pre_spend                      ┆ 1.0   ┆ 0.104 ┆ -0.71┆ -20.68            │
│ browsing (the real hidden one) ┆ 0.154 ┆ 0.114 ┆ 0.78 ┆ 8.36              │
Estimate with browsing included: 16.24
```

- A confounder as strong as **app use** (cf_y 0.165, cf_d 0.049) moves the estimate by $4.77; one as strong as **deal-seeking** by $2.48.
- **Past spending** is in a different league (it explains almost all of spend's variation). Nobody believes a confounder that strong is hiding, so it isn't a useful benchmark.
- The **real** hidden browsing signal: cf_y 0.154, cf_d 0.114, and it moves the estimate by $8.36, from $24.61 to $16.24. It's about as strong as app use on spend and stronger on the coupon. A careful analyst who said "something about as strong as app use could be missing" would have concluded that the coupon's profitability was in doubt, which was right.

Write in `day61/NOTES.md`: which observed feature is the most plausible benchmark for "what we didn't measure", and why?

---

## 61.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 61: OVB formula, DoubleML sensitivity bounds, robustness values, benchmarking"
git push
```

### Day 61 checklist

- [ ] I can state the omitted-variable bias formula and explain why both factors are needed
- [ ] I can run DoubleML's sensitivity analysis and read bounds and robustness values
- [ ] I compute robustness values against the decision threshold, not just against zero
- [ ] I can benchmark a hypothetical confounder against observed features
- [ ] I know that a low robustness value doesn't prove confounding exists

---

# DAY 62 — E-values and Robustness Values

| Time | Block |
|---|---|
| 0:00 – 0:40 | 62.1 Partial R² and the robustness value |
| 0:40 – 1:30 | 62.2 Back to the loyalty programme |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 62.3 Benchmarks a CFO can follow |
| 2:30 – 3:20 | 62.4 E-values for yes/no outcomes |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 62.5 Day 62 mini-project: the sensitivity contour plot |
| 4:40 – 5:00 | 62.6 Save your work with Git |

---

## 62.1 Partial R² and the robustness value (40 min)

For regression estimates, Cinelli & Hazlett (2020) describe a hidden confounder Z by two **partial R²** values:

- R²(D~Z | X): the share of the treatment's variation, after the controls, that Z explains;
- R²(Y~Z | D, X): the share of the outcome's variation, after treatment and controls, that Z explains.

Everything follows from the regression output you already have. The **robustness value** RV_q is the smallest strength (equal on both sides) a confounder needs to reduce the estimate by a fraction q:

```
f = q × |t| / √dof          RV_q = ½ (√(f⁴ + 4f²) − f²)
```

A confounder weaker than RV on both sides can't do it; one stronger on both sides can. To ask "could a confounder push the estimate below a threshold c?", use t = (estimate − c) / SE with q = 1.

Create `causalkit/sensitivity.py`:

```python
"""Sensitivity to an omitted confounder in OLS (Cinelli & Hazlett, 2020) and E-values (VanderWeele & Ding, 2017).

The language: a hidden confounder Z is described by two partial R2 values,
    r2dz: share of the treatment's residual variance (after the controls) that Z explains,
    r2yz: share of the outcome's residual variance (after treatment and controls) that Z explains.
"""
import numpy as np


def partial_r2(t: float, dof: float) -> float:
    """Partial R2 of a regressor with t-statistic t: t^2 / (t^2 + dof)."""
    return t**2 / (t**2 + dof)


def robustness_value(t: float, dof: float, q: float = 1.0) -> float:
    """Minimum strength (same partial R2 with treatment and outcome) a confounder needs to reduce the
    estimate by a fraction q (q = 1: to zero). Pass t = (estimate - target) / se for other targets with q = 1."""
    f = q * abs(t) / np.sqrt(dof)
    return 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)


def bias(se: float, dof: float, r2yz: float, r2dz: float) -> float:
    """Largest possible bias of the OLS estimate from a confounder with strengths (r2yz, r2dz)."""
    return se * np.sqrt(dof * r2yz * r2dz / (1 - r2dz))


def benchmark(r2yxj: float, r2dxj: float, k: float = 1.0) -> tuple[float, float]:
    """Strengths (r2yz, r2dz) of a confounder k times as strong as observed covariate Xj.
    r2dxj: partial R2 of Xj with the treatment given the other controls;
    r2yxj: partial R2 of Xj with the outcome given the treatment and the other controls."""
    r2dz = k * r2dxj / (1 - r2dxj)
    r2zxj = k * r2dxj**2 / ((1 - k * r2dxj) * (1 - r2dxj))
    r2yz = ((np.sqrt(k) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj))
    return min(r2yz, 1.0), min(r2dz, 1.0)


def e_value(rr: float) -> float:
    """E-value: the risk ratio a confounder needs with BOTH treatment and outcome to explain away rr."""
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))
```

---

## 62.2 Back to the loyalty programme (50 min)

Module 4's question: does joining the loyalty programme raise next-year spending by more than **$150**, the break-even (Day 40)? World A has no hidden confounder; world B has the unmeasured brand affinity. Create `day62/robustness_value.py`:

```python
# day62/robustness_value.py   (python -m day62.robustness_value)
"""
Back to Module 4's loyalty programme. In world B, an unmeasured brand affinity drives both joining and
spending. The robustness value (Cinelli & Hazlett, 2020) asks: how strongly would a hidden confounder have
to be associated with BOTH joining and spending (as a share of their remaining variation) to change the
conclusion? The business line from Day 40: joining pays off if it raises spending by more than $150.
"""
import numpy as np  # noqa: F401  (used inside the formula: np.sqrt)
import statsmodels.formula.api as smf

from causalkit.loyalty import truth, world
from causalkit.sensitivity import partial_r2, robustness_value

FORMULA = ("spend ~ joined + pre_spend + I(pre_spend**2) + pre_orders + engaged + tenure_years"
           " + np.sqrt(tenure_years) + age + mobile")
BREAK_EVEN = 150.0

if __name__ == "__main__":
    for title, hidden in [("world A (no hidden confounder)", 0.0), ("world B (hidden brand affinity)", 1.0)]:
        full = world(hidden=hidden)
        fit = smf.ols(FORMULA, full.to_pandas()).fit()
        est, se, t, dof = fit.params["joined"], fit.bse["joined"], fit.tvalues["joined"], fit.df_resid
        print(f"{title}: OLS {est:.1f} (SE {se:.1f}); true ATE {truth(full)['ATE']:.1f}")
        print(f"   partial R2 of joining with spend: {partial_r2(t, dof):.1%}")
        print(f"   robustness value to explain the effect away entirely:   {robustness_value(t, dof):.1%}")
        print(f"   robustness value to push it below the $150 break-even:  {robustness_value((est - BREAK_EVEN) / se, dof):.1%}")
```

```bash
python -m day62.robustness_value
```

```
world A (no hidden confounder): OLS 151.7 (SE 1.9); true ATE 158.4
   partial R2 of joining with spend: 23.4%
   robustness value to explain the effect away entirely:   42.1%
   robustness value to push it below the $150 break-even:  0.6%
world B (hidden brand affinity): OLS 226.1 (SE 2.3); true ATE 158.4
   partial R2 of joining with spend: 32.4%
   robustness value to explain the effect away entirely:   49.3%
   robustness value to push it below the $150 break-even:  20.7%
```

Read world B like a skeptical reader would, without the truth column: the estimate is $226, and a confounder would need 20.7% partial R² with both joining and spending to push it below $150. That sounds robust. Is a confounder that strong plausible? The robustness value can't answer that. Benchmarks can.

---

## 62.3 Benchmarks a CFO can follow (45 min)

Engagement (opening marketing emails) is a confounder we did measure. How strong is it, and what would a hidden confounder 1, 2 or 3 times as strong do? Create `day62/benchmarks.py`:

```python
# day62/benchmarks.py   (python -m day62.benchmarks)
"""
'A confounder with partial R2 of 21%' still means nothing to a CFO. Benchmark it: how strong is ENGAGEMENT
(opening marketing emails), a confounder we did measure? Then ask what a hidden confounder 1, 2 or 3 times
as strong as engagement would do. Finally, because it's a simulator, measure the real hidden affinity.
"""
import numpy as np  # noqa: F401  (used inside the formula: np.sqrt)
import polars as pl
import statsmodels.formula.api as smf

from causalkit.loyalty import world
from causalkit.sensitivity import benchmark, bias, partial_r2
from day62.robustness_value import BREAK_EVEN, FORMULA

df = world(hidden=1.0).to_pandas()
y_fit = smf.ols(FORMULA, df).fit()
d_fit = smf.ols(FORMULA.replace("spend ~ joined +", "joined ~"), df).fit()
est, se, dof = y_fit.params["joined"], y_fit.bse["joined"], y_fit.df_resid
r2yx = partial_r2(y_fit.tvalues["engaged"], dof)
r2dx = partial_r2(d_fit.tvalues["engaged"], d_fit.df_resid)
print(f"Engagement: partial R2 {r2yx:.1%} with spend, {r2dx:.1%} with joining")

rows = []
for k in [1, 2, 3]:
    r2yz, r2dz = benchmark(r2yx, r2dx, k)
    adj = est - bias(se, dof, r2yz, r2dz)
    rows.append({"hidden confounder": f"{k}x as strong as engagement", "r2 with spend": round(r2yz, 3),
                 "r2 with joining": round(r2dz, 3), "adjusted estimate": round(adj, 1),
                 "still above $150?": "yes" if adj > BREAK_EVEN else "no"})
long_y = smf.ols(FORMULA + " + _affinity", df).fit()
long_d = smf.ols(FORMULA.replace("spend ~ joined +", "joined ~") + " + _affinity", df).fit()
rows.append({"hidden confounder": "the real brand affinity", "r2 with spend": round(partial_r2(long_y.tvalues['_affinity'], long_y.df_resid), 3),
             "r2 with joining": round(partial_r2(long_d.tvalues['_affinity'], long_d.df_resid), 3),
             "adjusted estimate": round(long_y.params["joined"], 1),
             "still above $150?": "yes" if long_y.params["joined"] > BREAK_EVEN else "no"})
with pl.Config(tbl_width_chars=120, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
print(f"Unadjusted OLS: {est:.1f}")
```

```bash
python -m day62.benchmarks
```

```
Engagement: partial R2 11.6% with spend, 2.8% with joining
│ hidden confounder          ┆ r2 with spend ┆ r2 with joining ┆ adjusted estimate ┆ still above $150? │
│ 1x as strong as engagement ┆ 0.139         ┆ 0.029           ┆ 205.1             ┆ yes               │
│ 2x as strong as engagement ┆ 0.278         ┆ 0.058           ┆ 183.4             ┆ yes               │
│ 3x as strong as engagement ┆ 0.418         ┆ 0.086           ┆ 161.1             ┆ yes               │
│ the real brand affinity    ┆ 0.319         ┆ 0.132           ┆ 153.9             ┆ yes               │
Unadjusted OLS: 226.1
```

The statement for the CFO: "Even a hidden factor three times as strong as email engagement would leave the effect above break-even ($161)." And that was true: the real brand affinity is a bit over twice engagement's strength with spend and about four and a half times with joining. Adjusting for it gives $153.9, above $150 by a whisker.

Notice what the benchmarks don't say: the estimate moved from $226 to $154. The **conclusion** (above $150) survived; the **number** did not. If the decision depended on the size of the effect (how much to invest, say), the sensitivity analysis would be a warning, not a reassurance.

---

## 62.4 E-values for yes/no outcomes (50 min)

**Concept.** The **E-value** (VanderWeele & Ding, 2017) is the simplest sensitivity statistic of all, for risk ratios: the minimum strength of association, on the risk-ratio scale, that a hidden confounder would need with **both** treatment and outcome to explain the estimate away.

```
E-value = RR + √(RR × (RR − 1))
```

No modelling of the confounder at all, which makes it popular in medicine. Here the yes/no outcome is "high spender next year" (over $1,000). Create `day62/e_value.py` (about 30 seconds; the bootstrap refits the model 400 times):

```python
# day62/e_value.py   (python -m day62.e_value)
"""
E-values (VanderWeele & Ding, 2017) for a yes/no outcome: does joining make a customer more likely to be a
'high spender' (over $1,000 next year)? The E-value is the risk ratio a hidden confounder would need with
BOTH joining and high spending to explain the observed risk ratio away. No model of the confounder needed.
The risk ratio is standardised (g-computation, Day 31) from a logistic model with squares and interactions;
its 95% CI comes from 200 bootstrap samples.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from causalkit.loyalty import world
from causalkit.sensitivity import e_value

COLS = ["joined", "pre_spend", "pre_orders", "engaged", "tenure_years", "age", "mobile"]


def standardised_rr(df):
    """Logistic model with all pairwise interactions and squares; predict everyone joined vs not joined."""
    model = make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=1e4, max_iter=2000))
    model.fit(df[COLS], df.high)
    p1 = model.predict_proba(df[COLS].assign(joined=1))[:, 1].mean()
    p0 = model.predict_proba(df[COLS].assign(joined=0))[:, 1].mean()
    return p1 / p0


for title, hidden in [("world A", 0.0), ("world B", 1.0)]:
    df = world(hidden=hidden).to_pandas().assign(high=lambda d: (d.spend > 1000).astype(int))
    rr = standardised_rr(df)
    rng = np.random.default_rng(0)
    boots = [standardised_rr(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(200)]
    lo = np.percentile(boots, 2.5)
    true_rr = (df._y1 > 1000).mean() / (df._y0 > 1000).mean()
    print(f"{title}: risk ratio {rr:.2f} (95% CI from {lo:.2f}); true risk ratio {true_rr:.2f}")
    print(f"   E-value: {e_value(rr):.2f} for the estimate, {e_value(lo):.2f} for the CI's lower end")
```

```bash
python -m day62.e_value
```

```
world A: risk ratio 1.36 (95% CI from 1.33); true risk ratio 1.38
   E-value: 2.05 for the estimate, 2.00 for the CI's lower end
world B: risk ratio 1.59 (95% CI from 1.55); true risk ratio 1.36
   E-value: 2.56 for the estimate, 2.47 for the CI's lower end
```

In world B: "a hidden confounder would need to make customers about 2.5 times as likely to join, and 2.5 times as likely to be high spenders, to explain away the effect". Report the E-value for the CI's lower end too: it tells you what it takes to make the effect statistically indistinguishable from zero.

E-values are easy to compute and to misread. A large E-value doesn't mean confounding is absent; and because E-values only address "explain away entirely", they can't answer threshold questions like break-even. Use them for yes/no outcomes in a world that thinks in risk ratios; use robustness values and benchmarks for decisions.

**Try it.** In world B, what strength would a confounder need to halve the effect, or cut it by a quarter? Use `robustness_value(t, dof, q)`. Create `day62/halving.py`.

<details>
<summary>Solution</summary>

```python
# day62/halving.py   (python -m day62.halving)
import numpy as np  # noqa: F401  (used inside the formula: np.sqrt)
import statsmodels.formula.api as smf

from causalkit.loyalty import world
from causalkit.sensitivity import robustness_value
from day62.robustness_value import FORMULA

fit = smf.ols(FORMULA, world(hidden=1.0).to_pandas()).fit()
t, dof = fit.tvalues["joined"], fit.df_resid
for q in [1.0, 0.5, 0.25]:
    print(f"RV to reduce the estimate by {q:.0%}: {robustness_value(t, dof, q):.1%}")
```

```
RV to reduce the estimate by 100%: 49.3%
RV to reduce the estimate by 50%: 29.1%
RV to reduce the estimate by 25%: 15.9%
```

The real affinity reduced the estimate by 32% (from $226 to $154), and its strength (32% with spend, 13% with joining) sits between the 25% and 50% robustness values. Smaller changes need much weaker confounders: a confounder at 16% on both sides can wipe out a quarter of an effect that looks overwhelmingly significant.

</details>

---

## 62.5 Day 62 mini-project: the sensitivity contour plot (65 min)

Cinelli & Hazlett's signature figure: adjusted estimates for every hypothetical confounder, with benchmarks marked. Create `day62/contour.py`:

```python
# day62/contour.py   (python -m day62.contour)
"""
Mini-project: the sensitivity contour plot (Cinelli & Hazlett's signature figure). Every point is a
hypothetical confounder (its strength with joining, its strength with spending); the contour lines show the
adjusted estimate. Mark the break-even line, the engagement benchmarks, and the real affinity.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from causalkit.loyalty import world
from causalkit.sensitivity import benchmark, bias, partial_r2
from day62.robustness_value import BREAK_EVEN, FORMULA

df = world(hidden=1.0).to_pandas()
y_fit = smf.ols(FORMULA, df).fit()
d_fit = smf.ols(FORMULA.replace("spend ~ joined +", "joined ~"), df).fit()
est, se, dof = y_fit.params["joined"], y_fit.bse["joined"], y_fit.df_resid

r2d = np.linspace(0.001, 0.3, 120)
r2y = np.linspace(0.001, 0.6, 120)
D, Y = np.meshgrid(r2d, r2y)
adjusted = est - bias(se, dof, Y, D)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
cs = ax.contour(D, Y, adjusted, levels=[0, 50, 100, 200], colors="#94a3b8")
ax.clabel(cs, fmt="%.0f")
be = ax.contour(D, Y, adjusted, levels=[BREAK_EVEN], colors="#dc2626", linewidths=2)
ax.clabel(be, fmt="break-even %.0f")
r2yx = partial_r2(y_fit.tvalues["engaged"], dof)
r2dx = partial_r2(d_fit.tvalues["engaged"], d_fit.df_resid)
for k in [1, 2, 3]:
    ry, rd = benchmark(r2yx, r2dx, k)
    ax.plot(rd, ry, "s", color="#2563eb")
    ax.annotate(f"{k}x engagement", (rd, ry), textcoords="offset points", xytext=(6, -10), fontsize=8)
long_y = smf.ols(FORMULA + " + _affinity", df).fit()
long_d = smf.ols(FORMULA.replace("spend ~ joined +", "joined ~") + " + _affinity", df).fit()
ta = (partial_r2(long_d.tvalues["_affinity"], long_d.df_resid), partial_r2(long_y.tvalues["_affinity"], long_y.df_resid))
ax.plot(*ta, "*", ms=14, color="#d97706")
ax.annotate("real brand affinity", ta, textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("partial R2 of the confounder with joining")
ax.set_ylabel("partial R2 of the confounder with spending")
ax.set_title(f"Adjusted effect of joining (unadjusted {est:.0f})")
fig.tight_layout()
fig.savefig("figures/day62_contour.png", dpi=120)
print("Saved figures/day62_contour.png")
print(f"Adjusted estimate at the real affinity's strength (worst case): {est - bias(se, dof, ta[1], ta[0]):.1f}")
```

```bash
python -m day62.contour
```

```
Saved figures/day62_contour.png
Adjusted estimate at the real affinity's strength (worst case): 153.9
```

Open the figure. The red line is break-even; the blue squares are 1×, 2× and 3× engagement; the star is the real affinity, just on the profitable side of the red line. For an OLS estimate the bias formula is exact, so the worst-case adjustment at the affinity's strength reproduces the long regression ($153.9) to the decimal.

Put this figure in any report that rests on "we controlled for everything important". It shows the reader exactly how much "everything" has to cover.

---

## 62.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 62: partial R2, robustness values, benchmarks, E-values, contour plot"
git push
```

### Day 62 checklist

- [ ] I can compute a robustness value from a t-statistic, for zero and for a decision threshold
- [ ] I can benchmark a hidden confounder as k times an observed covariate
- [ ] I can compute and interpret an E-value for a risk ratio, and its limits
- [ ] I distinguish "the conclusion survives" from "the number survives"
- [ ] I can draw a sensitivity contour plot with benchmarks

---

# DAY 63 — Refutation Tests and Specification Curves

| Time | Block |
|---|---|
| 0:00 – 0:30 | 63.1 Two other ways an estimate can be fragile |
| 0:30 – 1:30 | 63.2 Refutation tests |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 63.3 What refutation tests can and can't do |
| 2:30 – 3:10 | 63.4 The garden of forking paths |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 63.5 Day 63 mini-project: a specification curve |
| 4:40 – 5:00 | 63.6 Save your work with Git |

---

## 63.1 Two other ways an estimate can be fragile (30 min)

Days 61–62 asked about one threat: a confounder outside the data. Two more are inside the analysis:

1. **A broken pipeline.** A bug, a leak of the outcome into the features, an estimator that finds effects anywhere. Refutation tests check for these.
2. **Analyst choices.** Which features, which learner, how to treat outliers, which estimand. Each choice is defensible; together they can move the answer a lot. A specification curve shows how much.

---

## 63.2 Refutation tests (60 min)

**Concept.** Change the data in ways that **should** change the answer in a known way, and check the estimator reacts correctly. You met DoWhy's versions in Module 2; here they are by hand with DoubleML, so you can apply them to any estimator.

| Test | What you change | A correct estimator gives |
|---|---|---|
| placebo treatment | shuffle who got a coupon | ≈ 0 |
| random common cause | add a column of noise as a confounder | no change |
| data subsets | re-estimate on random 80% subsets | stable estimates |
| placebo outcome | an outcome the coupon can't affect (last year's basket size) | ≈ 0 |

Create `day63/refutations.py` (about a minute):

```python
# day63/refutations.py   (python -m day63.refutations)
"""
Refutation tests (the idea behind DoWhy's refuters, Module 2, done by hand with DoubleML): change the data in
ways that SHOULD (or should NOT) change the answer, and check that the estimator reacts correctly.
  placebo treatment:   shuffle who got a coupon. The effect should vanish.
  random common cause: add a column of noise as a 'confounder'. The effect shouldn't move.
  data subsets:        re-estimate on random 80% subsets. The effect should be stable.
  placebo outcome:     an outcome the coupon can't affect (last year's basket size). Effect should be 0.
A refutation test can only show that something is wrong; passing doesn't prove the estimate right.
"""
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg

np.random.seed(63)  # DoubleML draws its cross-fitting folds from numpy's global RNG


def plr(df, x_cols=X_COLS, y="spend", d="coupon"):
    data = dml.DoubleMLData(df[x_cols + [d, y]], y, d, x_cols)
    fit = dml.DoubleMLPLR(data, ml_l=reg(), ml_m=clf(), n_folds=5).fit()
    return fit.coef[0], fit.se[0]


if __name__ == "__main__":
    df = public(coupons()).to_pandas()
    rng = np.random.default_rng(0)
    est, se = plr(df)
    print(f"original estimate:                  {est:6.2f} (SE {se:.2f})")

    placebo = [plr(df.assign(coupon=rng.permutation(df.coupon.to_numpy())))[0] for _ in range(5)]
    print(f"placebo treatment (5 shuffles):     {np.mean(placebo):6.2f}  (each: {', '.join(f'{p:.2f}' for p in placebo)})")

    noisy = df.assign(random_cause=rng.normal(size=len(df)))
    print(f"with a random common cause:         {plr(noisy, X_COLS + ['random_cause'])[0]:6.2f}")

    subsets = [plr(df.sample(frac=0.8, random_state=s))[0] for s in range(5)]
    print(f"80% subsets (5):                    {np.mean(subsets):6.2f}  (SD {np.std(subsets):.2f})")

    x_wo = [c for c in X_COLS if c != "basket"]
    p_est, p_se = plr(df, x_wo, y="basket")
    print(f"placebo outcome (basket size):      {p_est:6.2f}  (SE {p_se:.2f})")
```

```bash
python -m day63.refutations
```

```
original estimate:                   16.19 (SE 0.56)
placebo treatment (5 shuffles):       0.06  (each: -0.68, 0.16, 0.35, -0.15, 0.60)
with a random common cause:          16.13
80% subsets (5):                     16.18  (SD 0.28)
placebo outcome (basket size):        0.12  (SE 0.22)
```

All four pass. Shuffled coupons give effects scattered around zero, roughly within one standard error; noise doesn't move the estimate; subsets agree; and the coupon "changes" last year's basket size by $0.12, indistinguishable from zero.

---

## 63.3 What refutation tests can and can't do (40 min)

Day 61's world with the hidden browsing signal would pass every one of these tests too: shuffling, noise and subsets can't detect a confounder that isn't in the data. Refutation tests catch **mistakes in the analysis**, not **mistakes in the assumptions**. The one exception is the placebo outcome: if the confounder affects the placebo outcome too, the test can catch it. That's why good placebo outcomes (things the treatment can't touch but the confounders would) are worth searching for: last year's spending, behaviour before the campaign was announced, a product category the coupon didn't cover.

**Try it.** Replace spend with pure noise and re-estimate (DoWhy calls this the "dummy outcome" refuter). Create `day63/dummy_outcome.py`.

<details>
<summary>Solution</summary>

```python
# day63/dummy_outcome.py   (python -m day63.dummy_outcome)
import numpy as np

from causalkit.coupons import X_COLS, coupons, public
from day63.refutations import plr

df = public(coupons()).to_pandas()
df["noise_outcome"] = np.random.default_rng(1).normal(150, 60, len(df))
est, se = plr(df, X_COLS, y="noise_outcome")
print(f"Effect of the coupon on pure noise: {est:.2f} (SE {se:.2f})")
```

```
Effect of the coupon on pure noise: -0.04 (SE 0.94)
```

Zero, as it should be. If an estimator found an effect on noise, it would be fitting something about the treatment assignment itself: a leak, or overfitting without cross-fitting (Day 52).

</details>

---

## 63.4 The garden of forking paths (40 min)

**Concept.** Gelman & Loken's phrase: an analysis isn't one path but a garden of them. Each fork is a reasonable choice (include these features? cap outliers? which learner?), and an analyst who tries a few and reports the one that "works" can find almost anything, without any intent to mislead.

The **specification curve** (Simonsohn, Simmons & Nelson, 2020) is the antidote: list every defensible combination of choices, run them all, and show every estimate, sorted, with a panel underneath marking which choices produced which estimate. Readers see at once whether the conclusion depends on the choices, and which choice matters.

---

## 63.5 Day 63 mini-project: a specification curve (75 min)

Four choices, two options each, 16 specifications. Create `day63/spec_curve.py` (under a minute):

```python
# day63/spec_curve.py   (python -m day63.spec_curve)
"""
Mini-project: a specification curve (Simonsohn, Simmons & Nelson, 2020). An analyst makes many defensible
choices; report ALL combinations, sorted, so readers see how much the answer depends on them.
Choices here (2 x 2 x 2 x 2 = 16 specifications):
  features:  all 24 | without the 10 noise columns
  learners:  lasso (linear) | LightGBM
  model:     PLR (variance-weighted effect) | IRM (ATE)
  outcome:   spend | spend capped at its 99th percentile (common practice with heavy tails)
"""
import itertools

import doubleml as dml
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causalkit.coupons import BEHAVIOUR, X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

np.random.seed(63)
full = coupons()
df = public(full).to_pandas()
df["spend_capped"] = df.spend.clip(upper=df.spend.quantile(0.99))
choices = {"features": ["all 24", "no noise"], "learners": ["lasso", "LightGBM"], "model": ["PLR", "IRM"],
           "outcome": ["spend", "capped"]}
rows = []
for feats, learn, model, outcome in itertools.product(*choices.values()):
    x = X_COLS if feats == "all 24" else BEHAVIOUR + ["lapsing"]
    y = "spend" if outcome == "spend" else "spend_capped"
    data = dml.DoubleMLData(df[x + ["coupon", y]], y, "coupon", x)
    if learn == "lasso":
        g, m = make_pipeline(StandardScaler(), LassoCV(cv=3)), make_pipeline(StandardScaler(), LogisticRegressionCV(cv=3))
    else:
        g, m = reg(), clf()
    fit = (dml.DoubleMLPLR(data, g, m, n_folds=5) if model == "PLR"
           else dml.DoubleMLIRM(data, g, m, n_folds=5, trimming_threshold=0.02)).fit()
    rows.append({"features": feats, "learners": learn, "model": model, "outcome": outcome,
                 "estimate": fit.coef[0], "lo": fit.confint().iloc[0, 0], "hi": fit.confint().iloc[0, 1]})
spec = pl.DataFrame(rows).sort("estimate")
with pl.Config(tbl_rows=16, tbl_width_chars=110):
    print(spec.with_columns(pl.col("estimate", "lo", "hi").round(2)))
print(f"Range of estimates: {spec['estimate'].min():.2f} to {spec['estimate'].max():.2f}; truth ATE {truth(full)['ATE']:.2f}")
print("Median estimate by choice:")
for c in choices:
    print("   " + c + ": " + ", ".join(f"{k} {v:.2f}" for k, v in
                                     spec.group_by(c).agg(pl.col("estimate").median()).sort(c).iter_rows()))

fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
xs = np.arange(spec.height)
top.errorbar(xs, spec["estimate"], yerr=[spec["estimate"] - spec["lo"], spec["hi"] - spec["estimate"]],
             fmt="o", color="#2563eb", ms=4)
top.axhline(truth(full)["ATE"], color="#dc2626", ls="--", lw=1, label="true ATE")
top.set_ylabel("coupon effect ($)")
top.legend()
labels = [(c, v) for c, vals in choices.items() for v in vals]
for i, (c, v) in enumerate(labels):
    on = (spec[c] == v).to_numpy()
    bottom.scatter(xs[on], np.full(on.sum(), i), s=14, color="#334155")
bottom.set_yticks(range(len(labels)), [f"{c}: {v}" for c, v in labels], fontsize=8)
bottom.set_xlabel("specification (sorted by estimate)")
fig.tight_layout()
fig.savefig("figures/day63_spec_curve.png", dpi=120)
print("Saved figures/day63_spec_curve.png")
```

```bash
python -m day63.spec_curve
```

```
│ features ┆ learners ┆ model ┆ outcome ┆ estimate ┆ lo    ┆ hi    │
│ no noise ┆ lasso    ┆ IRM   ┆ capped  ┆ 11.32    ┆ 9.18  ┆ 13.46 │
│ all 24   ┆ lasso    ┆ IRM   ┆ capped  ┆ 11.51    ┆ 9.44  ┆ 13.57 │
...
│ all 24   ┆ LightGBM ┆ IRM   ┆ capped  ┆ 16.03    ┆ 14.86 ┆ 17.2  │
│ no noise ┆ LightGBM ┆ PLR   ┆ spend   ┆ 16.06    ┆ 14.95 ┆ 17.16 │
Range of estimates: 11.32 to 16.06; truth ATE 16.51
Median estimate by choice:
   features: all 24 14.95, no noise 14.94
   learners: LightGBM 15.94, lasso 13.00
   model: IRM 14.15, PLR 14.97
   outcome: capped 14.75, spend 14.94
```

Open `figures/day63_spec_curve.png`. The bottom panel makes it obvious: **every low estimate uses lasso**, and every LightGBM estimate is between $15.8 and $16.1. The feature set and outcome capping hardly matter. One choice does all the work, and it's one that Day 53's nuisance-RMSE comparison would settle (LightGBM predicts better).

What would a fishing analyst report? With a business case that needs coupons to look weak, $11.32 (lasso, IRM, capped); with one that needs them to look strong, $16.06. Both come with confidence intervals that look perfectly respectable. Only the full curve tells the truth about the analysis.

Write in `day63/NOTES.md`: which choices would you fix in advance in a pre-analysis plan (Day 21), and why?

---

## 63.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 63: refutation tests by hand, specification curve"
git push
```

**Coming next (Days 64–66).** The question changes from "how sure are we about the effect?" to "what do we do about it?" Uplift modelling, policy learning, and the cost-benefit analysis of a whole programme.

### Day 63 checklist

- [ ] I can run placebo-treatment, random-cause, subset, placebo-outcome and dummy-outcome refutations
- [ ] I know that refutation tests catch analysis mistakes, not missing confounders (except good placebo outcomes)
- [ ] I can build and read a specification curve with a choice panel
- [ ] I can identify which analyst choice drives an answer, and how to settle it

