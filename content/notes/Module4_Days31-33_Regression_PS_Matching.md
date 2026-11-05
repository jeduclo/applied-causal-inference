# Module 4 · Selection on Observables & Instrumental Variables
## Days 31–33: Regression Adjustment, Propensity Scores and Matching

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–3
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 33:** Students can estimate ATE, ATT and ATC by g-computation with bootstrap standard errors, fit and check a propensity score, use it to stratify and match, and judge a matched sample with balance diagnostics, while knowing that good balance is necessary, not sufficient.

---

### Where we are

Module 3 showed what a well-run experiment buys. But most business decisions were never randomised: customers **chose** to join the loyalty programme, regions **chose** to run a campaign. Module 4 returns to observational data with two strategies:

| Days | Strategy | Key assumption |
|---|---|---|
| 31–36 | **Selection on observables**: compare joiners with similar non-joiners | every confounder is measured (Day 15's backdoor criterion) |
| 37–38 | **Instrumental variables**: use something random that nudges people to join | the nudge affects spending *only* through joining |
| 39–40 | Compare them all, then the capstone | |

The question for all ten days: **how much does joining NorthMart's loyalty programme raise a customer's spending over the next year?** Joiners spend $593 more than non-joiners. Most of that gap is selection: big spenders are the ones who join.

### The loyalty simulator

Create `causalkit/loyalty.py`. As in Module 3, **read only the docstrings**: the code contains the capstone answers. The simulator knows every customer's spending both with and without the programme, so every estimator this module meets is graded against the truth.

<details>
<summary>causalkit/loyalty.py (provided; click to expand)</summary>

```python
"""
NorthMart's loyalty programme (simulated). Provided for Module 4: read the docstrings, not the code,
until the Day 40 capstone is done (the code contains the answers).

    from causalkit.loyalty import world, truth

    df = world()                         # 20,000 customers; who joined the programme, and next-year spend
    df = world(hidden=1.0)               # same, but an unobserved trait also drives joining and spending
    truth(df)                            # true ATE, ATT, ATC and LATE (for checking only)

Columns an analyst has:
    customer_id, tenure_years, age, mobile, engaged (opens marketing emails),
    pre_spend (last 12 months, $), pre_orders (last 12 months),
    invited (1 = randomly invited to join with a $20 sign-up bonus),
    joined (1 = joined the loyalty programme: the TREATMENT),
    spend (next 12 months, $: the OUTCOME)
Columns starting with '_' are the simulator's hidden truth.
"""
import numpy as np
import polars as pl


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def world(n: int = 20_000, seed: int = 31, hidden: float = 0.0, overlap: float = 1.0,
          invite_effect: float = 0.9, bonus_spend_effect: float = 0.0) -> pl.DataFrame:
    """Simulate one loyalty-programme dataset.
    hidden:  strength of an UNOBSERVED trait (brand affinity) that drives both joining and spending.
             0 = selection on observables holds (Days 31-36); 1 = it fails (Days 37-40).
    overlap: >1 makes joining more predictable from the covariates (worse overlap, Day 35).
    invite_effect: how strongly the random invitation raises joining (weak instruments, Day 38).
    bonus_spend_effect: $ the invitation adds to spend DIRECTLY (breaks the exclusion restriction, Day 38)."""
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(0, 8, n).round(1)
    age = np.clip(rng.normal(45, 14, n), 18, 80).round()
    mobile = (rng.random(n) < 0.6).astype(np.int8)
    engaged = (rng.random(n) < _sigmoid(-0.5 + 0.2 * tenure)).astype(np.int8)
    pre_spend = rng.lognormal(np.log(400) + 0.08 * tenure, 0.7, n).round(2)
    pre_orders = rng.poisson(pre_spend / 80)
    affinity = rng.normal(size=n)                                  # hidden brand affinity
    invited = (rng.random(n) < 0.5).astype(np.int8)                # randomised encouragement

    ls = np.log(pre_spend / 400)
    index = overlap * (-2.0 + 1.1 * ls + 0.9 * engaged + 0.15 * tenure - 0.01 * (age - 45)) + hidden * 1.0 * affinity
    u = rng.random(n)
    joined_if_not_invited = (u < _sigmoid(index)).astype(np.int8)
    joined_if_invited = (u < _sigmoid(index + invite_effect)).astype(np.int8)
    joined = np.where(invited == 1, joined_if_invited, joined_if_not_invited)
    ps = _sigmoid(index + invite_effect * invited)                 # true propensity (given everything)

    y0 = (150 + 0.75 * pre_spend + 0.0001 * pre_spend**2 + 110 * engaged + 60 * np.sqrt(tenure)
          + 15 * mobile + hidden * 90 * affinity + bonus_spend_effect * invited + rng.normal(0, 120, n))
    tau = 50 + 0.12 * pre_spend + 40 * engaged                     # effect of joining, larger for big spenders
    y1 = y0 + tau
    spend = np.where(joined == 1, y1, y0).round(2)

    return pl.DataFrame({
        "customer_id": [f"M{i:05d}" for i in range(1, n + 1)],
        "tenure_years": tenure, "age": age.astype(np.int16), "mobile": mobile, "engaged": engaged,
        "pre_spend": pre_spend, "pre_orders": pre_orders, "invited": invited,
        "joined": joined.astype(np.int8), "spend": spend,
        "_y0": y0, "_y1": y1, "_tau": tau, "_ps": ps, "_affinity": affinity,
        "_complier": ((joined_if_invited == 1) & (joined_if_not_invited == 0)).astype(np.int8),
    })


def truth(df: pl.DataFrame) -> dict:
    """True average effects of joining in THIS dataset (they need the hidden columns)."""
    t = df["_tau"].to_numpy()
    d = df["joined"].to_numpy() == 1
    c = df["_complier"].to_numpy() == 1
    return {"ATE": float(t.mean()), "ATT": float(t[d].mean()), "ATC": float(t[~d].mean()),
            "LATE": float(t[c].mean()), "share_joined": float(d.mean()), "share_compliers": float(c.mean())}


def public(df: pl.DataFrame) -> pl.DataFrame:
    """Drop the hidden columns: what an analyst would actually receive."""
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

Two worlds matter:

| World | Call | What's true |
|---|---|---|
| **A** | `world()` | everything that drives joining is in the data: selection on observables holds |
| **B** | `world(hidden=1.0)` | an unmeasured trait, *brand affinity*, drives both joining and spending |

A third column matters from Day 37: `invited`. Half of customers were **randomly** sent a $20 sign-up offer last year. It changed who joined, and nothing else.

### New folders and packages

```
causal-decision-lab/
├── causalkit/
│   ├── loyalty.py        # provided: the loyalty simulator
│   └── estimators.py     # built Days 31–37: every estimator in this module
├── data/make_loyalty_scenario.py   # Day 40 capstone data
└── day31/ ... day40/
```

Add to `requirements.txt`, then `pip install -r requirements.txt`:

```text
# Module 4: instrumental variables (scikit-learn is already installed with DoWhy; pinned explicitly)
linearmodels==7.1
scikit-learn==1.9.1
```

```bash
mkdir -p day31 day32 day33 day34 day35 day36 day37 day38 day39 day40
```

### The estimator library

Every day adds functions to `causalkit/estimators.py`. Here is the complete file for reference; each day explains the part it uses. **Type each function in on the day it's introduced**, not all at once.

<details>
<summary>causalkit/estimators.py (complete, built Days 31–37; click to expand)</summary>

```python
"""
Module 4 estimators, built up day by day. Every function takes a pandas DataFrame with
columns `joined` (treatment), `spend` (outcome) and the covariates you name.

    from causalkit.estimators import COVARIATES, g_computation, propensity, ...
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

COVARIATES = ["pre_spend", "pre_orders", "engaged", "tenure_years", "age", "mobile"]
OUTCOME_FORMULA = ("spend ~ pre_spend + I(pre_spend**2) + pre_orders + engaged"
                   " + tenure_years + I(tenure_years**0.5) + age + mobile")


# ---------- Day 31: regression adjustment (g-computation) ----------
def g_computation(df: pd.DataFrame, formula: str = OUTCOME_FORMULA) -> dict:
    """Fit one outcome model per group, predict BOTH potential outcomes for everyone, average."""
    m1 = smf.ols(formula, df[df.joined == 1]).fit()
    m0 = smf.ols(formula, df[df.joined == 0]).fit()
    y1_hat, y0_hat = m1.predict(df), m0.predict(df)
    diff = y1_hat - y0_hat
    d = df.joined == 1
    return {"ATE": diff.mean(), "ATT": diff[d].mean(), "ATC": diff[~d].mean()}


def bootstrap(estimator, df: pd.DataFrame, reps: int = 200, seed: int = 0, **kwargs) -> dict:
    """Standard errors by resampling customers with replacement and re-estimating."""
    rng = np.random.default_rng(seed)
    draws = [estimator(df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True), **kwargs)
             for _ in range(reps)]
    return {k: float(np.std([d[k] for d in draws], ddof=1)) for k in draws[0]}


# ---------- Day 32: propensity scores ----------
PS_FORMULA = "joined ~ np.log(pre_spend) + pre_orders + engaged + tenure_years + age + mobile"


def propensity(df: pd.DataFrame, formula: str = PS_FORMULA) -> np.ndarray:
    """P(joined = 1 | covariates), by logistic regression."""
    return smf.logit(formula, df).fit(disp=False).predict(df).to_numpy()


def smd(a: pd.Series, b: pd.Series, wa=None, wb=None) -> float:
    """Standardised mean difference, optionally weighted (Days 33-34)."""
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    return (ma - mb) / np.sqrt((a.var() + b.var()) / 2)


# ---------- Day 33: matching ----------
MATCH_VARS = ["log_pre_spend", "engaged", "tenure_years", "age"]


def match_att(df: pd.DataFrame, on: str = "ps", caliper: float | None = None,
              bias_correct: bool = False) -> dict:
    """1-to-1 nearest-neighbour matching WITH replacement, for the ATT.
    on="ps": match on the logit of the propensity score; on="mahalanobis": on the covariates.
    caliper (in SDs of the matching distance): drop treated customers with no close match.
    bias_correct: adjust each pair for its remaining covariate gap (Abadie & Imbens 2011)."""
    from sklearn.neighbors import NearestNeighbors

    d = df.assign(log_pre_spend=np.log(df.pre_spend)).reset_index(drop=True)
    if on == "ps":
        p = propensity(d)
        X = np.log(p / (1 - p)).reshape(-1, 1)
    else:
        Z = d[MATCH_VARS].to_numpy(float)
        L = np.linalg.cholesky(np.linalg.inv(np.cov(Z, rowvar=False)))
        X = Z @ L                                            # Mahalanobis distance = Euclidean here
    t, c = np.flatnonzero(d.joined == 1), np.flatnonzero(d.joined == 0)
    dist, idx = NearestNeighbors(n_neighbors=1).fit(X[c]).kneighbors(X[t])
    dist, match = dist[:, 0], c[idx[:, 0]]
    keep = np.ones(len(t), bool) if caliper is None else dist <= caliper * X.std()
    y_t, y_c = d.spend.to_numpy()[t[keep]], d.spend.to_numpy()[match[keep]]
    if bias_correct:
        m0 = smf.ols(OUTCOME_FORMULA, d.iloc[c]).fit()
        y_c = y_c + (m0.predict(d.iloc[t[keep]]).to_numpy() - m0.predict(d.iloc[match[keep]]).to_numpy())
    return {"ATT": float((y_t - y_c).mean()), "matched_treated": int(keep.sum()),
            "distinct_controls_used": int(len(np.unique(match[keep]))),
            "_pairs": (t[keep], match[keep])}


# ---------- Day 34: inverse probability weighting ----------
def ipw_weights(d: np.ndarray, ps: np.ndarray, estimand: str = "ATE") -> np.ndarray:
    """ATE: everyone re-weighted to look like the whole population.
    ATT: non-joiners re-weighted to look like the joiners (joiners keep weight 1)."""
    if estimand == "ATE":
        return np.where(d == 1, 1 / ps, 1 / (1 - ps))
    return np.where(d == 1, 1.0, ps / (1 - ps))


def ipw(df: pd.DataFrame, ps: np.ndarray | None = None, normalised: bool = True) -> dict:
    """IPW estimates of the ATE and ATT. normalised=True is the Hajek version (weights sum to 1 per group)."""
    ps = propensity(df) if ps is None else ps
    d, y = df.joined.to_numpy(), df.spend.to_numpy()
    out = {}
    for est in ["ATE", "ATT"]:
        w = ipw_weights(d, ps, est)
        if normalised:
            out[est] = np.average(y[d == 1], weights=w[d == 1]) - np.average(y[d == 0], weights=w[d == 0])
        else:   # Horvitz-Thompson: divide by the number of units, not the sum of weights
            n = len(y) if est == "ATE" else (d == 1).sum()
            out[est] = (w * y * d).sum() / n - (w * y * (1 - d)).sum() / n
    return out


def effective_sample_size(w: np.ndarray) -> float:
    """How many equally weighted observations the weights are 'worth' (Kish)."""
    return w.sum() ** 2 / (w**2).sum()


# ---------- Day 36: augmented IPW (doubly robust), with cross-fitting ----------
def _fit_predict_outcome(train: pd.DataFrame, test: pd.DataFrame, method: str, formula: str):
    if method == "ols":
        return {g: smf.ols(formula, train[train.joined == g]).fit().predict(test).to_numpy() for g in [0, 1]}
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = {}
    for g in [0, 1]:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(train.loc[train.joined == g, COVARIATES], train.spend[train.joined == g])
        out[g] = m.predict(test[COVARIATES])
    return out


def _fit_predict_ps(train: pd.DataFrame, test: pd.DataFrame, method: str, formula: str) -> np.ndarray:
    if method == "logit":
        p = smf.logit(formula, train).fit(disp=False).predict(test).to_numpy()
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        p = m.fit(train[COVARIATES], train.joined).predict_proba(test[COVARIATES])[:, 1]
    return np.clip(p, 0.01, 0.99)


def aipw(df: pd.DataFrame, outcome: str = "ols", ps_model: str = "logit",
         outcome_formula: str = OUTCOME_FORMULA, ps_formula: str = PS_FORMULA,
         folds: int = 5, seed: int = 0) -> dict:
    """Doubly robust estimates of the ATE and ATT, with standard errors from the influence function.
    Each customer's predictions come from models fitted on the OTHER folds (cross-fitting)."""
    from sklearn.model_selection import KFold

    df = df.reset_index(drop=True)
    n = len(df)
    mu0, mu1, ps = np.empty(n), np.empty(n), np.empty(n)
    splits = KFold(folds, shuffle=True, random_state=seed).split(df) if folds > 1 else [(np.arange(n), np.arange(n))]
    for train_idx, test_idx in splits:
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        mu = _fit_predict_outcome(train, test, outcome, outcome_formula)
        mu0[test_idx], mu1[test_idx] = mu[0], mu[1]
        ps[test_idx] = _fit_predict_ps(train, test, ps_model, ps_formula)

    d, y = df.joined.to_numpy(), df.spend.to_numpy()
    psi_ate = mu1 - mu0 + d * (y - mu1) / ps - (1 - d) * (y - mu0) / (1 - ps)
    p1 = d.mean()
    psi_att = (d * (y - mu0) - (1 - d) * ps / (1 - ps) * (y - mu0)) / p1
    return {"ATE": psi_ate.mean(), "ATE_se": psi_ate.std(ddof=1) / np.sqrt(n),
            "ATT": psi_att.mean(), "ATT_se": psi_att.std(ddof=1) / np.sqrt(n)}


# ---------- Day 37: instrumental variables ----------
IV_CONTROLS = ["pre_spend", "pre_spend_sq", "pre_orders", "engaged", "tenure_years", "sqrt_tenure", "age", "mobile"]


def two_sls(df: pd.DataFrame, controls: list[str] | None = IV_CONTROLS):
    """2SLS of spend on joined, instrumenting joined with invited. Returns the linearmodels result."""
    from linearmodels.iv import IV2SLS

    d = df.assign(const=1.0, pre_spend_sq=df.pre_spend**2 / 1e4, sqrt_tenure=np.sqrt(df.tenure_years))
    exog = d[["const"] + (controls or [])]
    return IV2SLS(d.spend, exog, d.joined, d.invited).fit(cov_type="robust")
```

</details>

---

# DAY 31 — Regression Adjustment Done Right

| Time | Block |
|---|---|
| 0:00 – 0:40 | 31.1 The question and the naive answer |
| 0:40 – 1:20 | 31.2 Why one regression coefficient isn't enough |
| 1:20 – 1:35 | Break |
| 1:35 – 2:35 | 31.3 G-computation: predict both worlds for everyone |
| 2:35 – 3:15 | 31.4 Bootstrap standard errors |
| 3:15 – 3:30 | Break |
| 3:30 – 4:40 | 31.5 Day 31 mini-project: a specification table |
| 4:40 – 5:00 | 31.6 Save your work with Git |

---

## 31.1 The question and the naive answer (40 min)

Create `day31/naive_and_ols.py`:

```python
# day31/naive_and_ols.py   (python -m day31.naive_and_ols)
"""
Who joined NorthMart's loyalty programme, and what did it do to their spending?
In this world, everything that drove joining is in the data (selection on observables holds).
So regression CAN work. Does it, and what exactly does it estimate?
"""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.loyalty import public, truth, world

full = world()
df = public(full)
t = truth(full)
print(df.head(5))

naive = df.filter(pl.col("joined") == 1)["spend"].mean() - df.filter(pl.col("joined") == 0)["spend"].mean()
pdf = df.to_pandas()
specs = {
    "naive difference": "spend ~ joined",
    "linear controls": "spend ~ joined + pre_spend + pre_orders + engaged + tenure_years + age + mobile",
    "flexible controls": "spend ~ joined + pre_spend + I(pre_spend**2) + pre_orders + engaged"
                         " + tenure_years + I(tenure_years**0.5) + age + mobile",
}
print(f"\nJoiners spent ${naive:,.0f} more than non-joiners next year.")
for name, f in specs.items():
    fit = smf.ols(f, pdf).fit(cov_type="HC2")
    print(f"{name:<18} coefficient on joined {fit.params['joined']:+8.1f}  (SE {fit.bse['joined']:.1f})")
print(f"\nTruth: ATE {t['ATE']:.1f}   ATT {t['ATT']:.1f}   ATC {t['ATC']:.1f}")
```

```bash
python -m day31.naive_and_ols
```

```
Joiners spent $593 more than non-joiners next year.
naive difference   coefficient on joined   +593.0  (SE 9.9)
linear controls    coefficient on joined   +118.8  (SE 3.7)
flexible controls  coefficient on joined   +151.7  (SE 2.0)

Truth: ATE 158.4   ATT 186.0   ATC 132.7
```

Three lessons in four lines:

1. The naive gap ($593) is mostly **selection**: joiners were already big spenders.
2. Adding controls linearly removes most of it, but **under-shoots** ($119). Spending next year rises faster than linearly with spending last year, and a straight line can't capture that.
3. With `pre_spend²` and `sqrt(tenure)` the coefficient is close to the ATE. But which effect is it? The truth has three: ATE (everyone), ATT (joiners) and ATC (non-joiners). The effect is larger for big spenders, who are more likely to join, so **ATT > ATE > ATC**.

---

## 31.2 Why one regression coefficient isn't enough (40 min)

**Concept.** When the effect varies across customers, the coefficient on `joined` in `spend ~ joined + X` is a **weighted average** of customer-level effects, with weights chosen by the regression, not by you. It's usually closer to the effect for customers whose joining is hardest to predict, which is rarely the population a decision is about.

Recall the estimand card from Day 12. Before estimating, decide:

| Decision question | Estimand |
|---|---|
| Was the programme worth it **for the members we have**? | ATT |
| Should we **auto-enrol everyone**? | ATE |
| Should we **push the non-members** to join? | ATC |

The estimator should then target that number directly. That's what g-computation does.

---

## 31.3 G-computation: predict both worlds for everyone (60 min)

**Concept.** Fit an outcome model that includes the treatment and the confounders. Then, for **every** customer, predict spending twice: once with `joined = 1`, once with `joined = 0`. The difference is each customer's predicted effect. Average it over everyone for the ATE, over joiners for the ATT, over non-joiners for the ATC. This is also called **regression adjustment** or the **standardisation** estimator, and it's Day 11's potential-outcomes table filled in by a model.

To let the effect vary, the outcome model interacts `joined` with the covariates. In `causalkit/estimators.py`, type in `COVARIATES`, `OUTCOME_FORMULA`, `g_computation` and `bootstrap`.

Create `day31/g_computation.py`:

```python
# day31/g_computation.py   (python -m day31.g_computation)
"""
G-computation (regression adjustment done right): model the outcome separately for joiners and
non-joiners, predict what EVERY customer would spend in both cases, then average the difference
over whichever population the estimand names.
"""
from causalkit.estimators import bootstrap, g_computation
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)

est = g_computation(df)
se = bootstrap(g_computation, df, reps=200)
for k in ["ATE", "ATT", "ATC"]:
    print(f"{k}: estimate {est[k]:7.1f}  (bootstrap SE {se[k]:.1f})   truth {t[k]:7.1f}")
```

```bash
python -m day31.g_computation
```

```
ATE: estimate   156.8  (bootstrap SE 1.9)   truth   158.4
ATT: estimate   183.9  (bootstrap SE 2.2)   truth   186.0
ATC: estimate   131.6  (bootstrap SE 2.1)   truth   132.7
```

All three estimands, each within about one standard error of the truth, from one fitted model.

---

## 31.4 Bootstrap standard errors (40 min)

**Concept.** G-computation's estimate is an average of predictions, so the regression's standard errors don't apply to it directly. The **bootstrap** gets a standard error by brute force: resample customers with replacement, re-run the whole estimator, repeat 200 times, and take the standard deviation of the 200 estimates. It works for any estimator, as long as you resample the units that were independently sampled (here, customers) and repeat **every** step, including model fitting.

Read `bootstrap` in `estimators.py`: it takes any estimator function and its arguments, so we'll reuse it for other estimators.

**Try it.** `invited` was randomised, so it isn't a confounder. Add `+ invited` to the outcome formula. What happens to the estimates and their standard errors? Create `day31/add_invited.py`.

<details>
<summary>Solution</summary>

```python
# day31/add_invited.py   (python -m day31.add_invited)
from causalkit.estimators import OUTCOME_FORMULA, bootstrap, g_computation
from causalkit.loyalty import public, world

df = public(world()).to_pandas()
for name, f in [("without invited", OUTCOME_FORMULA), ("with invited", OUTCOME_FORMULA + " + invited")]:
    est = g_computation(df, f)
    se = bootstrap(g_computation, df, reps=100, formula=f)
    print(f"{name:<16} ATE {est['ATE']:.1f} (SE {se['ATE']:.2f})   ATT {est['ATT']:.1f} (SE {se['ATT']:.2f})")
```

```
without invited  ATE 156.8 (SE 2.01)   ATT 183.9 (SE 2.18)
with invited     ATE 156.8 (SE 2.10)   ATT 184.0 (SE 2.30)
```

The estimates don't move, and the standard errors get slightly **larger**. `invited` predicts *joining* but not *spending* (except through joining), so it adds noise without removing any bias. The rule from Day 16: control for causes of the outcome, not for things that only cause the treatment. Hold that thought: on Day 37, this same variable becomes the hero.

</details>

---

## 31.5 Day 31 mini-project: a specification table (70 min)

The estimate is only as good as the outcome model. Let's see how much the model matters. Create `day31/specification_table.py`:

```python
# day31/specification_table.py   (python -m day31.specification_table)
"""
Mini-project: the outcome model decides the answer. G-computation with three outcome models:
a linear one, the flexible one, and a machine-learning model (gradient boosting) that finds the
shape on its own.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from causalkit.estimators import COVARIATES, g_computation
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)

LINEAR = "spend ~ " + " + ".join(COVARIATES)


def g_comp_boosting(df):
    X = df[COVARIATES]
    preds = {}
    for g in [0, 1]:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(X[df.joined == g], df.spend[df.joined == g])
        preds[g] = m.predict(X)
    diff = preds[1] - preds[0]
    d = (df.joined == 1).to_numpy()
    return {"ATE": diff.mean(), "ATT": diff[d].mean(), "ATC": diff[~d].mean()}


if __name__ == "__main__":
    rows = []
    for name, est in [("linear outcome model", g_computation(df, LINEAR)),
                      ("flexible outcome model", g_computation(df)),
                      ("gradient boosting", g_comp_boosting(df))]:
        rows.append({"outcome model": name, **{k: round(v, 1) for k, v in est.items()}})
    rows.append({"outcome model": "TRUTH", **{k: round(t[k], 1) for k in ["ATE", "ATT", "ATC"]}})
    with pl.Config(fmt_str_lengths=30):
        print(pl.DataFrame(rows))
```

```bash
python -m day31.specification_table
```

```
│ outcome model          ┆ ATE   ┆ ATT   ┆ ATC   │
│ linear outcome model   ┆ 139.5 ┆ 195.0 ┆ 88.0  │
│ flexible outcome model ┆ 156.8 ┆ 183.9 ┆ 131.6 │
│ gradient boosting      ┆ 170.0 ┆ 209.7 ┆ 133.1 │
│ TRUTH                  ┆ 158.4 ┆ 186.0 ┆ 132.7 │
```

- The **linear** model gets the ATC badly wrong ($88 vs $133): it extrapolates a straight line into regions where it doesn't fit.
- The **flexible** model is right because we wrote it with the true functional form. In real data you never know that form.
- **Gradient boosting** learns the shape from the data, but overstates the ATT by $24. Machine-learning models shrink towards the average to avoid overfitting, and that shrinkage biases a difference of two predictions. Day 36 shows how to fix this.

The specification table is a habit: if your conclusion changes across reasonable outcome models, the model is doing the work, not the data. Write two sentences in `day31/NOTES.md` on which row you'd report and why.

---

## 31.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 31: loyalty simulator, g-computation, bootstrap, specification table"
git push
```

### Day 31 checklist

- [ ] I can explain why the naive difference and a single regression coefficient may not answer the decision question
- [ ] I can choose between ATE, ATT and ATC from the decision being made
- [ ] I can estimate all three by g-computation, with bootstrap standard errors
- [ ] I can explain why controlling for a cause of treatment only (like `invited`) doesn't help
- [ ] I check how much my answer depends on the outcome model

---

# DAY 32 — Propensity Scores

| Time | Block |
|---|---|
| 0:00 – 0:40 | 32.1 One number that summarises selection |
| 0:40 – 1:30 | 32.2 Estimating the propensity score |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 32.3 The balancing property |
| 2:40 – 3:20 | 32.4 Subclassification on the propensity score |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 32.5 Day 32 mini-project: what a missing confounder does |
| 4:40 – 5:00 | 32.6 Save your work with Git |

---

## 32.1 One number that summarises selection (40 min)

**Concept.** The **propensity score** is each customer's probability of joining given their covariates: e(X) = P(joined = 1 | X). Rosenbaum and Rubin (1983) proved two things:

1. **Balancing property:** among customers with the same propensity score, the covariates have the same distribution in joiners and non-joiners. Comparing them is like a small randomised experiment.
2. **Sufficiency:** if adjusting for X removes confounding, so does adjusting for e(X) alone.

So instead of matching on six covariates, we can match on one number. The catch: e(X) must be **estimated**, and the result is only as good as the model and the covariates in it.

Two conditions make it work, and you should be able to name them:

| Condition | Meaning | Checkable? |
|---|---|---|
| **Unconfoundedness** | no unmeasured cause of both joining and spending | no: a DAG argument (Day 15) |
| **Overlap (positivity)** | every customer had some chance of each choice: 0 < e(X) < 1 | yes: look at the scores (Day 35) |

---

## 32.2 Estimating the propensity score (50 min)

Type `PS_FORMULA`, `propensity` and `smd` into `estimators.py`. `smd` is the standardised mean difference from Day 23's balance table, now with optional weights.

Create `day32/estimate_ps.py`:

```python
# day32/estimate_ps.py   (python -m day32.estimate_ps)
"""The propensity score: each customer's probability of joining, given what we know about them."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from causalkit.estimators import PS_FORMULA, propensity
from causalkit.loyalty import public, world

full = world()
df = public(full).to_pandas()

model = smf.logit(PS_FORMULA, df).fit(disp=False)
print(model.summary().tables[1])

df["ps"] = propensity(df)
print("\nPropensity score by group (quantiles 1%, 25%, 50%, 75%, 99%):")
for g, label in [(1, "joined"), (0, "did not join")]:
    print(f"  {label:<13}", np.round(np.quantile(df.ps[df.joined == g], [0.01, 0.25, 0.5, 0.75, 0.99]), 3))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df.ps[df.joined == 0], bins=50, alpha=0.6, label="did not join", color="#d95926")
ax.hist(df.ps[df.joined == 1], bins=50, alpha=0.6, label="joined", color="#3987e5")
ax.set_xlabel("Estimated propensity to join"); ax.set_ylabel("Customers"); ax.legend()
ax.set_title("Overlap: both groups appear at every propensity level")
fig.tight_layout(); fig.savefig("figures/day32_propensity_overlap.png", dpi=150)
print("Saved figures/day32_propensity_overlap.png")
```

```bash
python -m day32.estimate_ps
```

```
                        coef    std err          z      P>|z|      [0.025      0.975]
Intercept            -7.4894      0.249    -30.090      0.000      -7.977      -7.002
np.log(pre_spend)     1.0603      0.043     24.910      0.000       0.977       1.144
pre_orders            0.0002      0.004      0.048      0.962      -0.008       0.008
engaged               0.8460      0.033     25.954      0.000       0.782       0.910
tenure_years          0.1525      0.007     21.252      0.000       0.138       0.167
age                  -0.0089      0.001     -7.650      0.000      -0.011      -0.007
mobile                0.0275      0.032      0.852      0.394      -0.036       0.091

Propensity score by group (quantiles 1%, 25%, 50%, 75%, 99%):
  joined        [0.136 0.438 0.603 0.741 0.924]
  did not join  [0.061 0.228 0.366 0.529 0.844]
Saved figures/day32_propensity_overlap.png
```

Open the figure. The two histograms are shifted, which is selection, but they **overlap** over almost the whole range: for nearly every joiner there are non-joiners with a similar score. That's what makes the comparisons on Days 32–36 possible.

**Don't read the logit coefficients causally**, and don't drop "insignificant" variables like `mobile`. The propensity model's only job is to balance confounders, not to explain joining.

---

## 32.3 The balancing property (55 min)

Create `day32/balancing_property.py`:

```python
# day32/balancing_property.py   (python -m day32.balancing_property)
"""
The balancing property: customers with the SAME propensity score look alike, whether they joined or not.
Check it within propensity-score quintiles.
"""
import pandas as pd
import polars as pl

from causalkit.estimators import COVARIATES, propensity, smd
from causalkit.loyalty import public, world

df = public(world()).to_pandas()
df["ps"] = propensity(df)
df["quintile"] = pd.qcut(df.ps, 5, labels=False)

rows = []
for col in COVARIATES:
    row = {"covariate": col,
           "overall": round(smd(df[col][df.joined == 1], df[col][df.joined == 0]), 3)}
    for q in range(5):
        s = df[df.quintile == q]
        row[f"Q{q + 1}"] = round(smd(s[col][s.joined == 1], s[col][s.joined == 0]), 3)
    rows.append(row)
print("Standardised mean differences, joiners vs non-joiners:")
print(pl.DataFrame(rows))
```

```bash
python -m day32.balancing_property
```

```
Standardised mean differences, joiners vs non-joiners:
│ covariate    ┆ overall ┆ Q1     ┆ Q2     ┆ Q3     ┆ Q4     ┆ Q5     │
│ pre_spend    ┆ 0.649   ┆ 0.189  ┆ 0.054  ┆ -0.006 ┆ 0.025  ┆ 0.32   │
│ pre_orders   ┆ 0.595   ┆ 0.1    ┆ 0.066  ┆ 0.017  ┆ -0.02  ┆ 0.271  │
│ engaged      ┆ 0.465   ┆ 0.071  ┆ -0.001 ┆ 0.015  ┆ 0.073  ┆ -0.009 │
│ tenure_years ┆ 0.553   ┆ 0.147  ┆ 0.006  ┆ 0.069  ┆ -0.033 ┆ 0.071  │
│ age          ┆ -0.097  ┆ -0.026 ┆ -0.0   ┆ 0.012  ┆ 0.014  ┆ -0.074 │
│ mobile       ┆ 0.017   ┆ 0.051  ┆ -0.025 ┆ -0.024 ┆ 0.009  ┆ 0.035  │
```

Overall, joiners differ by 0.5–0.65 standard deviations on four covariates. Within the middle quintiles of the propensity score, almost every difference falls under 0.1, the usual threshold. The edge quintiles (Q1, Q5) are wider bins, with a lot of variation inside them, so balance there is worse. Finer bins fix that, as the next section shows.

---

## 32.4 Subclassification on the propensity score (40 min)

**Concept.** Cut the score into strata, compute the joiner/non-joiner difference within each, and average the differences weighted by each stratum's share of everyone (ATE) or of joiners (ATT). It's Day 23's stratified estimator, with strata built from the score.

Create `day32/ps_stratification.py`:

```python
# day32/ps_stratification.py   (python -m day32.ps_stratification)
"""
Mini-project: subclassification on the propensity score. Within each stratum compare joiners
with non-joiners, then average the stratum effects (weighted by stratum size for the ATE,
by number of joiners for the ATT). How many strata are enough?
"""
import pandas as pd
import polars as pl

from causalkit.estimators import propensity
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)
df["ps"] = propensity(df)


def subclassify(df: pd.DataFrame, k: int) -> dict:
    s = df.assign(stratum=pd.qcut(df.ps, k, labels=False))
    g = s.groupby(["stratum", "joined"]).spend.mean().unstack()
    effect = g[1] - g[0]
    n_all = s.groupby("stratum").size()
    n_treated = s[s.joined == 1].groupby("stratum").size()
    return {"ATE": (effect * n_all).sum() / n_all.sum(), "ATT": (effect * n_treated).sum() / n_treated.sum()}


rows = [{"strata": k, **{e: round(v, 1) for e, v in subclassify(df, k).items()}} for k in [1, 5, 10, 20, 50]]
print(pl.DataFrame(rows))
print(f"Truth: ATE {t['ATE']:.1f}   ATT {t['ATT']:.1f}")
```

```bash
python -m day32.ps_stratification
```

```
│ strata ┆ ATE   ┆ ATT   │
│ 1      ┆ 593.0 ┆ 593.0 │
│ 5      ┆ 224.1 ┆ 281.4 │
│ 10     ┆ 194.5 ┆ 244.7 │
│ 20     ┆ 181.4 ┆ 227.1 │
│ 50     ┆ 173.5 ┆ 214.9 │
Truth: ATE 158.4   ATT 186.0
```

The textbook claim is that five strata remove about 90% of the bias. Here five strata remove about 85% (593 → 224 against a truth of 158), and even 50 strata still overstate the ATE by $15. Why? Within a stratum, joiners still have somewhat higher `pre_spend`, and spending is **convex** in `pre_spend`, so small leftover differences become large outcome differences. Subclassification is a good diagnostic and a rough first estimate, not a final one.

---

## 32.5 Day 32 mini-project: what a missing confounder does (65 min)

Suppose the CRM team forgot to export `engaged` (opens marketing emails). Re-estimate the propensity score without it and repeat the 20-strata ATE. Create `day32/omit_engaged.py`.

<details>
<summary>Solution</summary>

```python
# day32/omit_engaged.py   (python -m day32.omit_engaged)
import pandas as pd

from causalkit.estimators import propensity
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
for name, f in [("full model", None),
                ("engaged omitted", "joined ~ np.log(pre_spend) + pre_orders + tenure_years + age + mobile")]:
    ps = propensity(df) if f is None else propensity(df, f)
    s = df.assign(stratum=pd.qcut(ps, 20, labels=False))
    g = s.groupby(["stratum", "joined"]).spend.mean().unstack()
    n = s.groupby("stratum").size()
    print(f"{name:<16} ATE with 20 strata: {((g[1] - g[0]) * n).sum() / n.sum():.1f}")
print(f"Truth: ATE {truth(full)['ATE']:.1f}")
```

```
full model       ATE with 20 strata: 181.4
engaged omitted  ATE with 20 strata: 205.9
Truth: ATE 158.4
```

The bias grows by $25. Nothing in the output warns you: the score from the smaller model balances the covariates it *includes* just as well. **Balance diagnostics check the variables you measured, never the ones you didn't.** Unconfoundedness is an assumption you defend with a DAG and domain knowledge, not a property of the data.

</details>

Add a short "What we assume" section to `day32/NOTES.md`: list the confounders, and for each one say why it affects both joining and spending.

---

## 32.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 32: propensity score, balancing property, subclassification"
git push
```

### Day 32 checklist

- [ ] I can define the propensity score and state the balancing property
- [ ] I can name unconfoundedness and overlap, and say which one the data can check
- [ ] I can estimate a propensity score and plot its overlap by group
- [ ] I can estimate ATE and ATT by subclassification, and explain why leftover imbalance remains
- [ ] I know that balance on measured covariates says nothing about unmeasured ones

---

# DAY 33 — Matching

| Time | Block |
|---|---|
| 0:00 – 0:40 | 33.1 Matching as a constructed control group |
| 0:40 – 1:40 | 33.2 Four ways to match |
| 1:40 – 1:55 | Break |
| 1:55 – 2:50 | 33.3 Judging a match: balance and the Love plot |
| 2:50 – 3:20 | 33.4 Why good balance can still give a biased answer |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 33.5 Day 33 mini-project: check the whole distribution |
| 4:40 – 5:00 | 33.6 Save your work with Git |

---

## 33.1 Matching as a constructed control group (40 min)

**Concept.** For each joiner, find the non-joiner who looks most like them and use that person's spending as the joiner's "would-have-spent" value. Average the gaps and you have the **ATT**: the effect for the people who actually joined. Matching is popular with decision-makers because the comparison is concrete: "customers like Ana who didn't join spent $X".

The choices that matter:

| Choice | Options | What we use |
|---|---|---|
| distance | propensity score, Mahalanobis distance on covariates | both |
| ratio | 1:1, 1:k | 1:1 |
| replacement | with or without | **with**: each joiner gets their best match, even if it's reused |
| caliper | max allowed distance; joiners without a close match are dropped | optional |
| bias correction | adjust each pair for its remaining covariate gap (Abadie & Imbens) | optional |

Type `MATCH_VARS` and `match_att` into `estimators.py`. Mahalanobis distance rescales covariates so that each counts in units of its own spread, and accounts for their correlation; after the Cholesky transform it is plain Euclidean distance, so scikit-learn's `NearestNeighbors` does the search.

---

## 33.2 Four ways to match (60 min)

Create `day33/matching.py`:

```python
# day33/matching.py   (python -m day33.matching)
"""For every joiner, find the most similar non-joiner; compare their spending. Four versions."""
import polars as pl

from causalkit.estimators import match_att
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)

rows = []
for name, kw in [("propensity score", {"on": "ps"}),
                 ("propensity score, caliper 0.05 SD", {"on": "ps", "caliper": 0.05}),
                 ("Mahalanobis on covariates", {"on": "mahalanobis"}),
                 ("Mahalanobis + bias correction", {"on": "mahalanobis", "bias_correct": True})]:
    r = match_att(df, **kw)
    rows.append({"matching on": name, "ATT": round(r["ATT"], 1), "joiners matched": r["matched_treated"],
                 "distinct non-joiners used": r["distinct_controls_used"]})
with pl.Config(fmt_str_lengths=40, tbl_width_chars=110):
    print(pl.DataFrame(rows))
print(f"Truth: ATT {t['ATT']:.1f}")
```

```bash
python -m day33.matching
```

```
│ matching on                       ┆ ATT   ┆ joiners matched ┆ distinct non-joiners used │
│ propensity score                  ┆ 203.6 ┆ 9630            ┆ 4424                      │
│ propensity score, caliper 0.05 SD ┆ 189.2 ┆ 9547            ┆ 4424                      │
│ Mahalanobis on covariates         ┆ 229.1 ┆ 9630            ┆ 4726                      │
│ Mahalanobis + bias correction     ┆ 185.2 ┆ 9630            ┆ 4726                      │
Truth: ATT 186.0
```

- Plain matching overstates the ATT by $18–43, even with 20,000 customers.
- The **caliper** drops 83 joiners who had no close match. The estimate improves, but notice the cost: it is now the effect for 9,547 joiners, a slightly different population. Every caliper changes the estimand a little.
- **Only ~4,500 distinct non-joiners** serve as matches for 9,630 joiners: some are reused many times. Matching with replacement trades variance for less bias.
- **Bias correction** almost removes the error ($185.2 vs $186.0). It fits an outcome model on non-joiners and uses it to adjust each pair for the gap the match left. That's matching and g-computation combined, a preview of Day 36's idea of using two models at once.

---

## 33.3 Judging a match: balance and the Love plot (55 min)

**Concept.** You judge a match by **balance**, never by the estimate it gives. Standard practice: every covariate's SMD below 0.1 after matching, shown in a **Love plot** (one dot per covariate, before and after).

Create `day33/balance_after_matching.py`:

```python
# day33/balance_after_matching.py   (python -m day33.balance_after_matching)
"""
Mini-project: did matching balance the covariates? Standardised mean differences before and after,
for propensity-score and Mahalanobis matching, and a 'love plot' of the result.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from causalkit.estimators import COVARIATES, match_att, smd
from causalkit.loyalty import public, world

df = public(world()).to_pandas()

pairs = {name: match_att(df, on=on)["_pairs"] for name, on in [("PS matching", "ps"), ("Mahalanobis", "mahalanobis")]}
rows = []
for col in COVARIATES:
    row = {"covariate": col,
           "before": round(smd(df[col][df.joined == 1], df[col][df.joined == 0]), 3)}
    for name, (t, c) in pairs.items():
        row[name] = round(smd(df[col].iloc[t], df[col].iloc[c]), 3)
    rows.append(row)
table = pl.DataFrame(rows)
print(table)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
y = np.arange(len(COVARIATES))
for col, marker in [("before", "o"), ("PS matching", "s"), ("Mahalanobis", "^")]:
    ax.scatter(table[col].abs(), y, marker=marker, label=col, s=45)
ax.axvline(0.1, color="grey", linestyle="--", linewidth=1)
ax.set_yticks(y, COVARIATES); ax.set_xlabel("|standardised mean difference|"); ax.legend()
ax.set_title("Balance before and after matching")
fig.tight_layout(); fig.savefig("figures/day33_love_plot.png", dpi=150)
print("Saved figures/day33_love_plot.png")
```

```bash
python -m day33.balance_after_matching
```

```
│ covariate    ┆ before ┆ PS matching ┆ Mahalanobis │
│ pre_spend    ┆ 0.649  ┆ 0.014       ┆ 0.051       │
│ pre_orders   ┆ 0.595  ┆ 0.014       ┆ 0.025       │
│ engaged      ┆ 0.465  ┆ -0.002      ┆ 0.0         │
│ tenure_years ┆ 0.553  ┆ 0.009       ┆ 0.009       │
│ age          ┆ -0.097 ┆ 0.024       ┆ -0.001      │
│ mobile       ┆ 0.017  ┆ 0.006       ┆ -0.005      │
Saved figures/day33_love_plot.png
```

By the usual rule, both matches pass easily: every SMD is under 0.06.

---

## 33.4 Why good balance can still give a biased answer (30 min)

Yet PS matching overstates the ATT by $18. How?

The SMD compares **means**. The outcome depends on `pre_spend²`: next year's spending grows faster than linearly with last year's. Two groups can have the same mean `pre_spend` but different spreads, and then they have different means of `pre_spend²`, and different average spending even without the programme.

Two lessons:

1. Check balance on the **functions of covariates the outcome depends on** (squares, logs, interactions), and on spreads, not just means.
2. Matching's leftover gaps are where **bias correction** (or Day 36's AIPW) earns its keep.

---

## 33.5 Day 33 mini-project: check the whole distribution (65 min)

Compute, before and after PS matching, the SMD of `pre_spend`, the SMD of `pre_spend²`, and the **variance ratio** of `pre_spend` (joiners' variance divided by matched non-joiners'; good balance is between 0.8 and 1.25). Create `day33/variance_ratio.py`. Use `match_att(df, on="ps")["_pairs"]` to get the matched rows.

<details>
<summary>Solution</summary>

```python
# day33/variance_ratio.py   (python -m day33.variance_ratio)
import numpy as np

from causalkit.estimators import match_att, smd
from causalkit.loyalty import public, world

df = public(world()).to_pandas()
t, c = match_att(df, on="ps")["_pairs"]
x = df.pre_spend
for name, a, b in [("before matching", x[df.joined == 1], x[df.joined == 0]),
                   ("after PS matching", x.iloc[t], x.iloc[c])]:
    print(f"{name:<18} SMD of pre_spend {smd(a, b):+.3f}   SMD of pre_spend^2 {smd(a**2, b**2):+.3f}"
          f"   variance ratio {np.var(a) / np.var(b):.2f}")
```

```
before matching    SMD of pre_spend +0.649   SMD of pre_spend^2 +0.412   variance ratio 3.02
after PS matching  SMD of pre_spend +0.014   SMD of pre_spend^2 +0.038   variance ratio 1.18
```

After matching, the squared term is still 2.7 times less balanced than the mean, and the variance ratio sits near the top of the acceptable range. Each gap is "acceptable" on its own, but the outcome is steep in exactly that direction: big spenders spend disproportionately more, so a slightly wider spread among joiners adds dollars. Report variance ratios alongside SMDs in every balance table you publish.

</details>

---

## 33.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 33: PS and Mahalanobis matching, calipers, bias correction, Love plot"
git push
```

**Coming next (Days 34–36).** Matching throws away most non-joiners and reuses a few. Weighting keeps everyone. Days 34–36 build inverse probability weighting, confront its main weakness (extreme weights when overlap is poor), and then combine an outcome model with weights in the doubly robust estimator, AIPW.

### Day 33 checklist

- [ ] I can match on the propensity score or on Mahalanobis distance, with replacement and a caliper
- [ ] I can explain how a caliper changes the population the estimate describes
- [ ] I judge a match by covariate balance and a Love plot, not by its estimate
- [ ] I check balance on squares and spreads, not just means
- [ ] I can explain what bias-corrected matching adds

