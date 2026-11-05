# Module 6 · Causal Machine Learning & Heterogeneity
## Days 51–54: From Prediction to Double Machine Learning

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–5
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 54:** Students can explain why a good predictive model can give a bad effect estimate, estimate effects by partialling out with cross-fitted machine learning (DML) by hand and with DoubleML, say what a partially linear model's single coefficient means when effects vary, and estimate ATE, ATT and LATE with DoubleML's interactive models.

---

### Where we are

Modules 4 and 5 estimated **one** effect, with a handful of carefully chosen controls. Two things change in Module 6:

1. **Many features, unknown functional form.** A marketing database has dozens of customer features, and nobody knows how they combine. Machine learning is good at that. Days 51–54 show how to use it **without** breaking causal inference.
2. **Effects that differ across customers.** The real question is usually *for whom*: who should get the coupon, which segments can absorb a price rise. Days 55–57 estimate and validate **heterogeneous** effects; Days 58–60 apply all of it to prices.

| Days | Question | Tools |
|---|---|---|
| 51–54 | How do we use ML for one effect, with valid confidence intervals? | DoubleML |
| 55–57 | Who responds, and how do we know the model is right? | EconML: meta-learners, causal forests, DRTester |
| 58–60 | How price-sensitive is each segment, and where can prices rise? | DoubleML, EconML, pyfixest |

### Two simulators

| File | What it simulates | Used on |
|---|---|---|
| `causalkit/coupons.py` | 20,000 customers; a targeting model sends $10-off coupons; each customer's true effect is known | Days 51–57 |
| `causalkit/demand.py` | 80 stores × 5 categories × 104 weeks; a pricing team sets prices from costs and its demand forecast | Days 58–60 |

<details>
<summary>causalkit/coupons.py (provided; click to expand)</summary>

```python
"""
NorthMart's targeted coupons (simulated). Provided for Module 6: read the docstrings, not the code,
until the Day 60 capstone is done.

A targeting model sends a $10-off coupon to customers it thinks will respond, mostly deal-seekers and
lapsing customers. Did the coupon raise their spending next month, and for WHOM?

    from causalkit.coupons import coupons, truth, X_COLS

    df = coupons()                       # 20,000 customers; coupons targeted by the marketing model
    df = coupons(randomized=True)        # same customers, coupons assigned by a coin flip (an A/B test)
    df = coupons(holdout=True, hidden=1.0)
        # half of customers randomly made ineligible; the targeting model also used a real-time
        # browsing signal that was never stored (an unobserved confounder)
    truth(df)                            # true ATE, ATT, LATE (for checking only)

Columns an analyst has:
    customer_id, 24 features (X_COLS: behaviour, demographics, and 10 columns z1..z10 of pure noise),
    eligible (1 = could receive a coupon; random when holdout=True), coupon (the TREATMENT),
    spend (next 30 days, $: the OUTCOME)
Columns starting with '_' are the simulator's hidden truth: _tau (each customer's effect), _ps, _complier,
_browsing (the unstored browsing signal).
"""
import numpy as np
import polars as pl

BEHAVIOUR = ["tenure_years", "age", "pre_spend", "visits", "deal_share", "app_user", "email_open",
             "basket", "returns_rate", "region_east", "region_west", "region_north", "region_south"]
NOISE = [f"z{i}" for i in range(1, 11)]
X_COLS = BEHAVIOUR + ["lapsing"] + NOISE


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _features(n: int, rng: np.random.Generator) -> dict:
    region = rng.integers(0, 5, n)
    f = {
        "tenure_years": np.round(rng.gamma(2.0, 2.0, n), 1),
        "age": np.clip(rng.normal(44, 13, n), 18, 85).round(),
        "pre_spend": np.round(np.exp(rng.normal(5.6, 0.6, n)), 2),
        "visits": rng.poisson(5, n).astype(float),
        "deal_share": np.round(rng.beta(2, 5, n), 3),
        "app_user": rng.binomial(1, 0.45, n).astype(float),
        "email_open": np.round(rng.beta(2, 3, n), 3),
        "basket": np.round(np.exp(rng.normal(3.6, 0.35, n)), 2),
        "returns_rate": np.round(rng.beta(1, 12, n), 3),
    }
    for k, name in enumerate(["east", "west", "north", "south"], start=1):
        f[f"region_{name}"] = (region == k).astype(float)
    f["lapsing"] = (f["visits"] <= 2).astype(float)
    for z in NOISE:
        f[z] = np.round(rng.normal(0, 1, n), 3)
    return f


def cate(f: dict) -> np.ndarray:
    """True effect of the coupon on next-month spend ($) for customers with features f."""
    return (-8 + 70 * f["deal_share"] + 10 * f["app_user"] - 12 * (f["pre_spend"] > 500)
            + 6 * np.tanh(f["visits"] - 4) + 0.15 * (f["age"] - 44) * (f["deal_share"] > 0.4))


def coupons(n: int = 20_000, seed: int = 51, randomized: bool = False, holdout: bool = False,
            hidden: float = 0.0) -> pl.DataFrame:
    """randomized: coupons by coin flip (p = 0.5) instead of the targeting model.
    holdout:    half of customers are randomly made ineligible (they can't receive a coupon).
    hidden:     strength of an unstored browsing signal that drives both targeting and spending."""
    rng = np.random.default_rng(seed)
    f = _features(n, rng)
    browsing = rng.normal(0, 1, n)
    lp = np.log(f["pre_spend"] / 300)
    g = (90 + 75 * lp + 18 * lp**2 + 12 * np.sqrt(f["visits"]) * (1 + f["app_user"]) + 60 * f["email_open"] ** 2
         + 15 * np.sin(f["age"] / 8) + 0.6 * f["basket"] - 40 * f["returns_rate"] - 25 * f["lapsing"]
         + 8 * f["region_west"] + 30 * hidden * browsing)
    tau = cate(f)
    score = (-1.1 + 4.0 * (f["deal_share"] - 0.28) + 1.3 * f["lapsing"] + 0.7 * f["app_user"] - 0.9 * lp
             + 0.8 * np.sin(3 * f["email_open"]) - 0.3 * (f["tenure_years"] > 6) + 1.2 * hidden * browsing)
    ps = np.full(n, 0.5) if randomized else np.clip(_sigmoid(score), 0.03, 0.97)
    targeted = rng.random(n) < ps
    eligible = rng.random(n) < 0.5 if holdout else np.ones(n, bool)
    coupon = targeted & eligible
    spend = np.maximum(g + tau * coupon + rng.normal(0, 35, n), 0)
    cols = {"customer_id": [f"C{i:05d}" for i in range(1, n + 1)]}
    cols |= {k: f[k] for k in X_COLS}
    cols |= {"eligible": eligible.astype(np.int64), "coupon": coupon.astype(np.int64), "spend": np.round(spend, 2),
             "_tau": tau, "_ps": ps, "_complier": targeted, "_browsing": browsing}
    return pl.DataFrame(cols)


def truth(df: pl.DataFrame) -> dict:
    tau = df["_tau"].to_numpy()
    d = df["coupon"].to_numpy() == 1
    out = {"ATE": round(float(tau.mean()), 2), "ATT": round(float(tau[d].mean()), 2)}
    if df["eligible"].mean() < 1:
        out["LATE"] = round(float(tau[df["_complier"].to_numpy()].mean()), 2)
    return out


def public(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

Module 6 also has a small shared file, so every day uses the same machine-learning settings. Create `causalkit/learners.py`:

```python
"""Learners shared by Module 6 (Days 51-57): one place to change them."""
import lightgbm as lgb


def reg(**kw):
    """Gradient-boosted regression trees with settings that work well on the coupon data."""
    return lgb.LGBMRegressor(**({"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                                 "min_child_samples": 40, "verbose": -1, "random_state": 0} | kw))


def clf(**kw):
    """Gradient-boosted classification trees (for propensity scores)."""
    return lgb.LGBMClassifier(**({"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                                  "min_child_samples": 40, "verbose": -1, "random_state": 0} | kw))


def variance_weighted(tau, ps):
    """What a partially linear model targets when effects vary: CATEs weighted by e(X)(1 - e(X))."""
    w = ps * (1 - ps)
    return float((tau * w).sum() / w.sum())
```

### New packages

Add to `requirements.txt`, then `pip install -r requirements.txt`:

```text
# Module 6: causal machine learning
doubleml==0.11.4
econml==0.17.0
lightgbm==4.7.0
```

```bash
mkdir -p day51 day52 day53 day54 day55 day56 day57 day58 day59 day60
```

Add one line to `.gitignore` (Days 55–60 save predictions between scripts):

```text
data/*.npz
```

Some scripts in this module take several minutes; the notes say which. Start them, then read the next section while they run.

---

# DAY 51 — From Prediction to Causal ML

| Time | Block |
|---|---|
| 0:00 – 0:40 | 51.1 Two different questions |
| 0:40 – 1:30 | 51.2 A great predictor, a biased effect |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 51.3 Why: regularisation and confounding |
| 2:30 – 3:10 | 51.4 Is it just confounding? |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 51.5 Day 51 mini-project: is a better predictor a better estimator? |
| 4:40 – 5:00 | 51.6 Save your work with Git |

---

## 51.1 Two different questions (40 min)

| | Prediction | Causal inference |
|---|---|---|
| Question | What will this customer spend? | How much does the coupon **change** what they spend? |
| Judged by | out-of-sample accuracy (R², RMSE) | bias and coverage of the effect estimate |
| Feature roles | any feature that helps is welcome | treatment, confounders, and things you must **not** control for (Day 16) |
| Checkable? | yes: a holdout set | not directly: the counterfactual is never observed |

Machine learning is built for the first column. Days 51–54 are about borrowing its power for the second without inheriting its blind spots.

---

## 51.2 A great predictor, a biased effect (50 min)

NorthMart's targeting model sends $10-off coupons mostly to deal-seekers and lapsing customers. Lapsing customers spend less, so the people who get coupons would have spent less anyway.

Create `day51/prediction_vs_causation.py`:

```python
# day51/prediction_vs_causation.py   (python -m day51.prediction_vs_causation)
"""
NorthMart's targeting model sends $10-off coupons. Did they raise next-month spend?
A model that PREDICTS spend very well is not automatically a model that measures the coupon's effect.
"""
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import reg

full = coupons()
df = public(full).to_pandas()
t = truth(full)
print(f"{len(df):,} customers, {len(X_COLS)} features; {df.coupon.mean():.0%} got a coupon")

naive = df.spend[df.coupon == 1].mean() - df.spend[df.coupon == 0].mean()
ols = smf.ols("spend ~ coupon + " + " + ".join(X_COLS), df).fit()

model = reg()
r2 = cross_val_score(model, df[X_COLS + ["coupon"]], df.spend, cv=5, scoring="r2").mean()
model.fit(df[X_COLS + ["coupon"]], df.spend)
s_learner = (model.predict(df[X_COLS + ["coupon"]].assign(coupon=1))
             - model.predict(df[X_COLS + ["coupon"]].assign(coupon=0))).mean()
rank = np.argsort(-model.feature_importances_).tolist().index(len(X_COLS)) + 1

print(f"\nnaive difference (coupon vs none):          {naive:+6.1f}")
print(f"OLS with all 24 features, linear:          {ols.params['coupon']:+6.1f}  (R2 {ols.rsquared:.2f})")
print(f"LightGBM 'S-learner' (predict with/without): {s_learner:+6.1f}  (cross-validated R2 {r2:.2f})")
print(f"   the coupon ranks {rank} of {len(X_COLS) + 1} in the model's feature importance")
print(f"\nTruth: ATE {t['ATE']:.1f}, ATT {t['ATT']:.1f} ($ per customer)")
```

```bash
python -m day51.prediction_vs_causation
```

```
20,000 customers, 24 features; 51% got a coupon

naive difference (coupon vs none):            +0.9
OLS with all 24 features, linear:           +13.4  (R2 0.63)
LightGBM 'S-learner' (predict with/without):  +14.9  (cross-validated R2 0.68)
   the coupon ranks 10 of 25 in the model's feature importance

Truth: ATE 16.5, ATT 20.3 ($ per customer)
```

- The naive comparison says coupons do nothing: confounding hides almost the whole effect.
- Linear OLS gets most of the way, but its straight lines can't remove all the confounding.
- LightGBM, a **better predictor** (R² 0.68 vs 0.63), is still $1.6 short. The "S-learner" (one model, predict everyone twice: with and without the coupon) is the most natural way to use a predictive model for effects, and it's biased.

---

## 51.3 Why: regularisation and confounding (45 min)

**Concept.** A predictive model trades a little bias for a lot less variance: that's what regularisation (shrinkage, early stopping, limited tree depth) does. It's the right trade for prediction. But the bias lands wherever it costs least predictive accuracy, and the coupon is a modest predictor of spend (rank 10 of 25), so its effect gets shrunk.

There's a second problem. Coupon and confounders are correlated (deal-seekers get coupons). When the model regularises, it can credit part of the coupon's effect to the confounders, or part of the confounders' effect to the coupon. Nothing in the fitting process cares which, because predictions are equally good either way.

Chernozhukov et al. (2018) call this **regularisation bias**, and show it doesn't disappear with more data at the rate needed for valid confidence intervals. Their fix, **double/debiased machine learning (DML)**, is tomorrow.

---

## 51.4 Is it just confounding? (40 min)

**Try it.** Fit the same S-learner on `coupons(randomized=True)`: same customers, but coupons given by a coin flip. There's no confounding. Is the S-learner still biased? Create `day51/s_learner_rct.py`.

<details>
<summary>Solution</summary>

```python
# day51/s_learner_rct.py   (python -m day51.s_learner_rct)
from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import reg

full = coupons(randomized=True)
df = public(full).to_pandas()
m = reg().fit(df[X_COLS + ["coupon"]], df.spend)
s = (m.predict(df[X_COLS + ["coupon"]].assign(coupon=1)) - m.predict(df[X_COLS + ["coupon"]].assign(coupon=0))).mean()
diff = df.spend[df.coupon == 1].mean() - df.spend[df.coupon == 0].mean()
print(f"Randomised coupons: difference in means {diff:.1f}, S-learner {s:.1f}, truth {truth(full)['ATE']:.1f}")
```

```
Randomised coupons: difference in means 16.3, S-learner 15.4, truth 16.5
```

Still biased: $15.4 against $16.3 from a plain difference in means on the same data. That's pure regularisation bias: the trees shrink a weak predictor's effect even when there's nothing to confound it. In an experiment, the difference in means (or Day 24's regression) beats a fancy model.

</details>

---

## 51.5 Day 51 mini-project: is a better predictor a better estimator? (75 min)

Fit LightGBM S-learners of three sizes on 10 simulated datasets. Score each on held-out R² and on the error of its ATE. Create `day51/better_predictor.py` (about a minute):

```python
# day51/better_predictor.py   (python -m day51.better_predictor)
"""
Mini-project: is a better predictor a better effect estimator? Fit LightGBM S-learners of increasing
complexity on 10 simulated datasets. Score each on out-of-sample R2 (prediction) and on the error of its
ATE (causal). The coupon is one feature among 25, and the trees spend their splits where the
predictive payoff is largest, which isn't the coupon.
"""
import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import reg

settings = {"shallow (50 trees, 4 leaves)": dict(n_estimators=50, num_leaves=4),
            "medium (300 trees, 15 leaves)": dict(n_estimators=300, num_leaves=15),
            "deep (1000 trees, 63 leaves)": dict(n_estimators=1000, num_leaves=63, min_child_samples=10)}
rows = []
for name, kw in settings.items():
    r2s, errs = [], []
    for s in range(10):
        full = coupons(seed=700 + s)
        df = public(full).to_pandas()
        tr, te = train_test_split(df, test_size=0.3, random_state=s)
        m = reg(**kw).fit(tr[X_COLS + ["coupon"]], tr.spend)
        r2s.append(r2_score(te.spend, m.predict(te[X_COLS + ["coupon"]])))
        ate = (m.predict(df[X_COLS + ["coupon"]].assign(coupon=1))
               - m.predict(df[X_COLS + ["coupon"]].assign(coupon=0))).mean()
        errs.append(ate - truth(full)["ATE"])
    rows.append({"model": name, "test R2": round(np.mean(r2s), 3), "ATE average error": round(np.mean(errs), 1),
                 "ATE error SD": round(np.std(errs), 1)})
with pl.Config(tbl_width_chars=100, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
```

```bash
python -m day51.better_predictor
```

```
│ model                         ┆ test R2 ┆ ATE average error ┆ ATE error SD │
│ shallow (50 trees, 4 leaves)  ┆ 0.524   ┆ -13.2             ┆ 0.9          │
│ medium (300 trees, 15 leaves) ┆ 0.671   ┆ -1.9              ┆ 0.7          │
│ deep (1000 trees, 63 leaves)  ┆ 0.652   ┆ -1.8              ┆ 0.6          │
```

- The shallow model loses most of the effect (−$13 of $16.5): it barely uses the coupon at all.
- The medium and deep models are similar predictors and similarly biased (about −$1.9), with a small SD. The bias is **systematic**, so more data won't fix it, and a confidence interval around it would be confidently wrong.
- Model selection by R² doesn't select for good effect estimates.

Write two sentences in `day51/NOTES.md`: why would a marketing team that judges its model by R² end up underestimating what its coupons do?

---

## 51.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 51: coupon simulator, prediction vs causation, regularisation bias"
git push
```

### Day 51 checklist

- [ ] I can explain the difference between predicting an outcome and estimating an effect
- [ ] I can estimate an effect with an S-learner, and explain why it's biased even in an experiment
- [ ] I know that a better R² doesn't mean a better effect estimate
- [ ] I can explain regularisation bias in plain language

---

# DAY 52 — Neyman Orthogonality and Cross-Fitting

| Time | Block |
|---|---|
| 0:00 – 0:40 | 52.1 Frisch–Waugh–Lovell, again |
| 0:40 – 1:30 | 52.2 Partialling out with machine learning |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 52.3 Two ideas that make it work |
| 2:30 – 3:10 | 52.4 How many folds? |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 52.5 Day 52 mini-project: break orthogonality, break cross-fitting |
| 4:40 – 5:00 | 52.6 Save your work with Git |

---

## 52.1 Frisch–Waugh–Lovell, again (40 min)

Every economist has seen the Frisch–Waugh–Lovell theorem: the coefficient on D in a regression of Y on D and X equals the coefficient from regressing **(Y minus its prediction from X)** on **(D minus its prediction from X)**. "Partial X out of both, then compare what's left."

Robinson (1988) noticed that the predictions don't have to be linear. In the **partially linear model**

```
spend = θ · coupon + g(X) + error,         coupon = m(X) + error
```

g and m can be any functions. Predict spend from X (call it l(X) = E[spend | X]), predict the coupon from X (m(X) = E[coupon | X]), and regress residual on residual. Machine learning is exactly a tool for estimating unknown functions like l and m. They're called **nuisance functions**: we need them, but we don't care about them.

---

## 52.2 Partialling out with machine learning (50 min)

Create `day52/partialling_out.py`:

```python
# day52/partialling_out.py   (python -m day52.partialling_out)
"""
Frisch-Waugh-Lovell, then Robinson (1988), then double machine learning:
  1. predict spend from the features (no coupon):  l(X) = E[spend | X]
  2. predict the coupon from the features:          m(X) = E[coupon | X]
  3. regress (spend - l(X)) on (coupon - m(X)).
With linear predictions this reproduces OLS exactly. With LightGBM and cross-fitting, it's DML.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg, variance_weighted

full = coupons()
df = public(full).to_pandas()
X, y, d = df[X_COLS], df.spend, df.coupon


def residual_on_residual(y_res, d_res):
    fit = sm.OLS(y_res, d_res).fit(cov_type="HC1")
    return fit.params.iloc[0], fit.bse.iloc[0]


ols = smf.ols("spend ~ coupon + " + " + ".join(X_COLS), df).fit()
lin = residual_on_residual(y - LinearRegression().fit(X, y).predict(X), d - LinearRegression().fit(X, d).predict(X))
print(f"OLS coefficient:                          {ols.params['coupon']:.2f}")
print(f"partialling out with linear models:       {lin[0]:.2f}   <- identical (Frisch-Waugh-Lovell)")

l_hat = cross_val_predict(reg(), X, y, cv=5)
m_hat = cross_val_predict(clf(), X, d, cv=5, method="predict_proba")[:, 1]
dml = residual_on_residual(y - l_hat, d - m_hat)
print(f"partialling out with LightGBM, 5 folds:   {dml[0]:.2f}  (SE {dml[1]:.2f})")
print(f"Target: {variance_weighted(full['_tau'].to_numpy(), full['_ps'].to_numpy()):.2f} "
      "(the variance-weighted effect, explained on Day 53)")
print(f"\nNuisance models: spend predicted with R2 {1 - np.var(y - l_hat) / np.var(y):.2f}; "
      f"predicted coupon probabilities range from {np.percentile(m_hat, 5):.2f} to {np.percentile(m_hat, 95):.2f} "
      "(5th to 95th percentile)")
```

```bash
python -m day52.partialling_out
```

```
OLS coefficient:                          13.41
partialling out with linear models:       13.41   <- identical (Frisch-Waugh-Lovell)
partialling out with LightGBM, 5 folds:   16.08  (SE 0.56)
Target: 16.39 (the variance-weighted effect, explained on Day 53)

Nuisance models: spend predicted with R2 0.66; predicted coupon probabilities range from 0.18 to 0.86 (5th to 95th percentile)
```

Linear partialling-out reproduces OLS to the cent. Replacing the linear predictions with LightGBM moves the estimate from $13.4 to $16.1, within half a standard error of its target. That's **double machine learning** in five lines.

`cross_val_predict(..., cv=5)` is doing something important: every customer's prediction comes from a model trained on the **other** four-fifths of customers.

---

## 52.3 Two ideas that make it work (45 min)

**1. Neyman orthogonality: residualise BOTH spend and coupon.** If you only residualise spend, any error in l(X) that correlates with the coupon goes straight into θ. Residualising the coupon too makes θ insensitive to small errors in either nuisance: errors enter only as a **product** of the two models' errors. Two decent models, each wrong by a little, give a much smaller error in θ. That's the "double" in double ML, and the same property made AIPW doubly robust on Day 36.

**2. Cross-fitting: never predict a customer with a model that saw them.** Flexible models overfit: in-sample, they partly memorise each customer's outcome, so residuals are too small, and in a way that correlates with the coupon. Splitting the data into K folds, fitting on K−1 and predicting the held-out fold (then rotating), removes this. It's the Day 36 cross-fitting, now essential.

With both, θ̂ is approximately normal around the truth with the usual standard error, **as long as** both nuisance models are reasonably accurate (formally: their errors shrink faster than n^(−1/4)).

---

## 52.4 How many folds? (40 min)

**Try it.** Rerun the LightGBM partialling-out with 2, 5 and 10 folds. Create `day52/folds.py`.

<details>
<summary>Solution</summary>

```python
# day52/folds.py   (python -m day52.folds)
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg

df = public(coupons()).to_pandas()
X, y, d = df[X_COLS], df.spend, df.coupon
for k in [2, 5, 10]:
    l_hat = cross_val_predict(reg(), X, y, cv=k)
    m_hat = cross_val_predict(clf(), X, d, cv=k, method="predict_proba")[:, 1]
    fit = sm.OLS(y - l_hat, d - m_hat).fit(cov_type="HC1")
    print(f"{k:>2} folds: {fit.params.iloc[0]:.2f} (SE {fit.bse.iloc[0]:.2f})")
```

```
 2 folds: 15.89 (SE 0.56)
 5 folds: 16.08 (SE 0.56)
10 folds: 16.20 (SE 0.56)
```

More folds means each nuisance model trains on more data (50%, 80%, 90%), so it's a bit more accurate, and the estimate moves a little towards the target. 5 is the usual default; with small samples, use 10.

</details>

---

## 52.5 Day 52 mini-project: break orthogonality, break cross-fitting (75 min)

Three estimators on 40 datasets of 5,000 customers. Each estimate's error is divided by its own standard error: a valid estimator gives a standard normal, with 95% of 95% CIs covering the target. Create `day52/orthogonality_mc.py` (about 5 minutes):

```python
# day52/orthogonality_mc.py   (python -m day52.orthogonality_mc)
"""
Mini-project: why BOTH residuals, and why cross-fitting? 40 simulated datasets of 5,000 customers.
  A. partial out spend only:  regress (spend - l(X)) on the raw coupon      (not orthogonal)
  B. partial out both, but predict each customer with a model trained on that same customer
     (an overfitting forest, in-sample)                                       (no cross-fitting)
  C. partial out both, out-of-fold predictions                                (DML)
Errors are divided by each estimate's standard error: a well-behaved estimator gives a standard normal.
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg, variance_weighted


def rr(y_res, d_res):
    fit = sm.OLS(y_res, d_res).fit(cov_type="HC1")
    return float(np.asarray(fit.params)[0]), float(np.asarray(fit.bse)[0])


z = {"A. outcome residual only": [], "B. both residuals, no cross-fitting": [], "C. DML (cross-fitted)": []}
bias = {k: [] for k in z}
for s in range(40):
    full = coupons(n=5000, seed=1000 + s)
    df = public(full).to_pandas()
    X, y, d = df[X_COLS], df.spend.to_numpy(), df.coupon.to_numpy().astype(float)
    target = variance_weighted(full["_tau"].to_numpy(), full["_ps"].to_numpy())
    l_cf = cross_val_predict(reg(), X, y, cv=5)
    m_cf = cross_val_predict(clf(), X, d, cv=5, method="predict_proba")[:, 1]
    l_in = RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, y).predict(X)
    m_in = RandomForestClassifier(200, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, d).predict_proba(X)[:, 1]
    for k, (yr, dr) in zip(z, [(y - l_cf, d - d.mean()), (y - l_in, d - m_in), (y - l_cf, d - m_cf)]):
        est, se = rr(yr, dr)
        z[k].append((est - target) / se)
        bias[k].append(est - target)

rows = [{"estimator": k, "average error ($)": round(np.mean(bias[k]), 2),
         "average error / SE": round(np.mean(v), 2), "95% CIs covering the target": f"{np.mean(np.abs(v) < 1.96):.0%}"}
        for k, v in z.items()]
with pl.Config(tbl_width_chars=120, fmt_str_lengths=40):
    print(pl.DataFrame(rows))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
grid = np.linspace(-6, 6, 200)
for ax, (k, v) in zip(axes, z.items()):
    ax.hist(np.clip(v, -6, 6), bins=np.linspace(-6, 6, 25), density=True, color="#2563eb", alpha=0.7)
    ax.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), color="k", lw=1)
    ax.set_title(k, fontsize=9)
    ax.set_xlabel("(estimate - target) / SE")
fig.tight_layout()
fig.savefig("figures/day52_orthogonality.png", dpi=120)
print("Saved figures/day52_orthogonality.png")
```

```bash
python -m day52.orthogonality_mc
```

```
│ estimator                           ┆ average error ($) ┆ average error / SE ┆ 95% CIs covering the target │
│ A. outcome residual only            ┆ -2.81             ┆ -2.66              ┆ 20%                         │
│ B. both residuals, no cross-fitting ┆ -2.53             ┆ -2.52              ┆ 32%                         │
│ C. DML (cross-fitted)               ┆ -0.47             ┆ -0.41              ┆ 92%                         │
Saved figures/day52_orthogonality.png
```

Open the figure: A and B are shifted far to the left of the standard normal curve; C sits close to it. Without orthogonality, 95% intervals cover the target 20% of the time; without cross-fitting, 32%. With both, 92%. This reproduces the famous figure in Chernozhukov et al. (2018).

---

## 52.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 52: partialling out, orthogonality, cross-fitting, Monte Carlo"
git push
```

### Day 52 checklist

- [ ] I can state the Frisch–Waugh–Lovell theorem and Robinson's partially linear model
- [ ] I can estimate an effect by cross-fitted partialling-out with LightGBM
- [ ] I can explain why both residuals are needed (orthogonality) and why cross-fitting is needed
- [ ] I can show by simulation that skipping either one breaks the confidence intervals

---

# DAY 53 — DoubleML: Partially Linear Models

| Time | Block |
|---|---|
| 0:00 – 0:30 | 53.1 The DoubleML package |
| 0:30 – 1:40 | 53.2 Three kinds of nuisance learners |
| 1:40 – 1:55 | Break |
| 1:55 – 2:45 | 53.3 What PLR's single number means |
| 2:45 – 3:20 | 53.4 ML can't find a confounder that isn't there |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 53.5 Day 53 mini-project: does the answer depend on the split? |
| 4:40 – 5:00 | 53.6 Save your work with Git |

---

## 53.1 The DoubleML package (30 min)

**DoubleML** (Bach, Chernozhukov, Kurz & Spindler) implements Day 52's estimator and its relatives with a common interface:

1. Describe the data: `DoubleMLData(df, y_col, d_cols, x_cols)`.
2. Pick a model class: `DoubleMLPLR` (partially linear, today), `DoubleMLIRM` and `DoubleMLIIVM` (interactive, tomorrow), and more.
3. Pass any scikit-learn-style learners for the nuisance functions: `ml_l` for E[Y|X], `ml_m` for E[D|X].
4. `.fit()`, then read `.coef`, `.se`, `.confint()`.

One practical detail: DoubleML draws its cross-fitting folds from numpy's global random generator, so each script calls `np.random.seed(...)` first. Without it, every run gives slightly different numbers.

---

## 53.2 Three kinds of nuisance learners (70 min)

Create `day53/plr.py` (about 5 minutes; the random forest is slow):

```python
# day53/plr.py   (python -m day53.plr)
"""
The DoubleML package: the partially linear regression (PLR) model  spend = theta * coupon + g(X) + error,
estimated by orthogonal scores and cross-fitting, with three kinds of nuisance learners, and repeated
cross-fitting (n_rep) so the answer doesn't depend on one random split.
"""
import doubleml as dml
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import reg, variance_weighted

np.random.seed(53)  # DoubleML draws its cross-fitting folds from numpy's global RNG

full = coupons()
df = public(full).to_pandas()
data = dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], y_col="spend", d_cols="coupon", x_cols=X_COLS)
print(data)

learners = {
    "lasso": make_pipeline(StandardScaler(), LassoCV(cv=3)),
    "random forest": RandomForestRegressor(300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0),
    "LightGBM": reg(),
}
rows = []
for name, learner in learners.items():
    plr = dml.DoubleMLPLR(data, ml_l=learner, ml_m=learner, n_folds=5, n_rep=3)
    plr.fit()
    rmse = plr.evaluate_learners()
    rows.append({"learners": name, "estimate": round(plr.coef[0], 2), "SE": round(plr.se[0], 2),
                 "95% CI": f"[{plr.confint().iloc[0, 0]:.1f}, {plr.confint().iloc[0, 1]:.1f}]",
                 "RMSE of l(X)": round(float(np.mean(rmse["ml_l"])), 1),
                 "RMSE of m(X)": round(float(np.mean(rmse["ml_m"])), 3)})
with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
print(f"Target: {variance_weighted(full['_tau'].to_numpy(), full['_ps'].to_numpy()):.2f}")
```

```bash
python -m day53.plr
```

```
│ learners      ┆ estimate ┆ SE   ┆ 95% CI       ┆ RMSE of l(X) ┆ RMSE of m(X) │
│ lasso         ┆ 13.4     ┆ 0.6  ┆ [12.2, 14.6] ┆ 38.6         ┆ 0.459        │
│ random forest ┆ 16.33    ┆ 0.58 ┆ [15.2, 17.5] ┆ 38.0         ┆ 0.457        │
│ LightGBM      ┆ 16.16    ┆ 0.56 ┆ [15.1, 17.3] ┆ 36.5         ┆ 0.457        │
Target: 16.39
```

- With **lasso** nuisance models, DML gives exactly OLS's answer ($13.4): linear models can't capture the confounding, and no amount of orthogonality fixes a nuisance model that's systematically wrong.
- **Random forest** and **LightGBM** both land within half a standard error of the target.
- `evaluate_learners()` reports each nuisance model's out-of-fold RMSE. LightGBM predicts spend best (36.5), and that's the honest way to choose between learners: compare **nuisance** prediction accuracy, never the effect estimates themselves (choosing the learner that gives the answer you like is fishing, Day 26).
- `n_rep=3` repeats cross-fitting with three different splits; more in 53.5.

---

## 53.3 What PLR's single number means (50 min)

**Concept.** PLR assumes one θ for everyone. When effects differ (they do: −$26 to +$69), what does θ̂ estimate? Not the ATE: a **weighted** average of customers' effects, weighting each by e(X)(1 − e(X)), the variance of their coupon given their features. Customers whose coupon was a coin flip count most; customers the targeting model was nearly certain about count least.

Create `day53/what_plr_estimates.py`:

```python
# day53/what_plr_estimates.py   (python -m day53.what_plr_estimates)
"""
When effects differ across customers, PLR's single theta is NOT the ATE. It weights each customer's
effect by e(X)(1 - e(X)): customers whose coupon was hardest to predict count most, customers the
targeting model was nearly sure about (e near 0 or 1) count almost nothing.
"""
import numpy as np
import polars as pl

from causalkit.coupons import coupons, truth
from causalkit.learners import variance_weighted

full = coupons()
tau, ps = full["_tau"].to_numpy(), full["_ps"].to_numpy()
t = truth(full)
print(f"ATE (everyone):                     {t['ATE']:.2f}")
print(f"ATT (customers who got a coupon):   {t['ATT']:.2f}")
print(f"PLR target (variance-weighted):     {variance_weighted(tau, ps):.2f}")

w = ps * (1 - ps)
bands = (full.with_columns(weight=pl.Series(w / w.sum()), ps_band=pl.Series(np.digitize(ps, [0.2, 0.4, 0.6, 0.8])))
         .group_by("ps_band").agg(customers=pl.len(), share_of_weight=pl.col("weight").sum().round(3),
                                  average_effect=pl.col("_tau").mean().round(1)).sort("ps_band")
         .with_columns(propensity=pl.col("ps_band").replace_strict(
             {0: "< 0.2", 1: "0.2-0.4", 2: "0.4-0.6", 3: "0.6-0.8", 4: ">= 0.8"})).drop("ps_band"))
print(bands.select("propensity", "customers", "share_of_weight", "average_effect"))
```

```bash
python -m day53.what_plr_estimates
```

```
ATE (everyone):                     16.51
ATT (customers who got a coupon):   20.26
PLR target (variance-weighted):     16.39
│ propensity ┆ customers ┆ share_of_weight ┆ average_effect │
│ < 0.2      ┆ 1360      ┆ 0.042           ┆ -2.2           │
│ 0.2-0.4    ┆ 5397      ┆ 0.275           ┆ 8.5            │
│ 0.4-0.6    ┆ 6146      ┆ 0.369           ┆ 16.9           │
│ 0.6-0.8    ┆ 5014      ┆ 0.256           ┆ 23.9           │
│ >= 0.8     ┆ 2083      ┆ 0.058           ┆ 30.4           │
```

Here the PLR target happens to be close to the ATE ($16.39 vs $16.51), because the down-weighted customers at both ends (effects −2.2 and +30.4) roughly cancel. That's luck, not a property of PLR. The customers the targeting model was surest about (propensity ≥ 0.8) have the biggest effects (+30.4) and only 6% of the weight. If the question is "what did the coupons we sent achieve?" (the ATT, $20.26), PLR is the wrong tool. Tomorrow's IRM targets the ATE or ATT directly. PLR is the right tool when the effect is plausibly constant, or the treatment is continuous (Days 58–60).

---

## 53.4 ML can't find a confounder that isn't there (35 min)

**Try it.** Drop `deal_share` from the features and rerun PLR with LightGBM. Create `day53/missing_confounder.py`.

<details>
<summary>Solution</summary>

```python
# day53/missing_confounder.py   (python -m day53.missing_confounder)
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg, variance_weighted

full = coupons()
df = public(full).to_pandas()
for name, cols in [("all features", X_COLS), ("without deal_share", [c for c in X_COLS if c != "deal_share"])]:
    np.random.seed(53)
    plr = dml.DoubleMLPLR(dml.DoubleMLData(df[cols + ["coupon", "spend"]], "spend", "coupon", cols),
                          ml_l=reg(), ml_m=clf(), n_folds=5).fit()
    print(f"{name:<20} {plr.coef[0]:.2f} (SE {plr.se[0]:.2f})")
print(f"Target {variance_weighted(full['_tau'].to_numpy(), full['_ps'].to_numpy()):.2f}")
```

```
all features         16.12 (SE 0.56)
without deal_share   19.87 (SE 0.54)
Target 16.39
```

Without `deal_share` the estimate rises by $3.75, and the standard error even shrinks a little. Deal-seekers get more coupons and respond more, so leaving them out makes coupons look better than they are. Flexible learners and orthogonal scores remove **regularisation** bias; they can't remove **omitted-variable** bias. Unconfoundedness is still the assumption doing the work (Day 32).

</details>

---

## 53.5 Day 53 mini-project: does the answer depend on the split? (65 min)

Cross-fitting splits customers into folds at random. Run PLR with 20 different splits and look at the spread. Create `day53/split_robustness.py` (about a minute and a half):

```python
# day53/split_robustness.py   (python -m day53.split_robustness)
"""
Mini-project: cross-fitting splits the data at random, so the estimate depends a little on the split.
Run PLR with 20 different splits (n_rep=20) and look at the spread. DoubleML reports the median
across splits, with a standard error that includes the between-split variation.
"""
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg

np.random.seed(53)  # DoubleML draws its cross-fitting folds from numpy's global RNG

df = public(coupons()).to_pandas()
data = dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS)
plr = dml.DoubleMLPLR(data, ml_l=reg(), ml_m=clf(), n_folds=5, n_rep=20)
plr.fit()
each = plr.all_coef.ravel()
print(f"20 splits: estimates from {each.min():.2f} to {each.max():.2f} (SD {each.std():.2f})")
print(f"Median across splits: {plr.coef[0]:.2f}  (SE {plr.se[0]:.2f}, compared with a typical single-split SE "
      f"{np.median(plr.all_se):.2f})")
```

```bash
python -m day53.split_robustness
```

```
20 splits: estimates from 15.98 to 16.34 (SD 0.09)
Median across splits: 16.16  (SE 0.56, compared with a typical single-split SE 0.56)
```

Different splits move the estimate by up to $0.36, with an SD of $0.09: small compared with the standard error ($0.56), so the split isn't driving the answer. DoubleML reports the **median** over splits and adds the between-split spread to the standard error (here it's too small to change the second decimal). With smaller samples or noisier learners, the spread can be much larger; always use `n_rep` of 5 or more for a number you'll report.

Report: "DML with LightGBM, 5 folds, median over 20 splits". Anyone rerunning your analysis with a different seed should get the same answer to the precision you report.

---

## 53.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 53: DoubleML PLR, learners, what PLR estimates, repeated cross-fitting"
git push
```

### Day 53 checklist

- [ ] I can fit DoubleMLPLR with different nuisance learners and compare them by nuisance RMSE
- [ ] I can explain what PLR's single coefficient means when effects differ across customers
- [ ] I know that ML removes regularisation bias, not omitted-variable bias
- [ ] I use repeated cross-fitting and report the median over splits

---

# DAY 54 — DoubleML: Interactive Models and IV

| Time | Block |
|---|---|
| 0:00 – 0:40 | 54.1 The interactive regression model |
| 0:40 – 1:30 | 54.2 ATE and ATT with IRM |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 54.3 A confounder that was never stored, and a random holdout |
| 2:40 – 3:20 | 54.4 When there's nothing hidden |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 54.5 Day 54 mini-project: what do features buy an IV? |
| 4:40 – 5:00 | 54.6 Save your work with Git |

---

## 54.1 The interactive regression model (40 min)

**Concept.** The **interactive regression model (IRM)** drops PLR's constant-effect assumption: spend can depend on the coupon and the features in any way. Its nuisance functions are:

- g(0, X) and g(1, X): expected spend without and with a coupon, for customers with features X;
- m(X): the propensity score.

These are combined in the **AIPW score** of Day 36, now with ML models and cross-fitting. Everything you learned there still holds: the estimate is doubly robust; it targets the **ATE** (`score="ATE"`) or the **ATT** (`score="ATTE"`) directly; and propensities close to 0 or 1 make it noisy, so `trimming_threshold` clips them (Day 35).

---

## 54.2 ATE and ATT with IRM (50 min)

Create `day54/irm.py`:

```python
# day54/irm.py   (python -m day54.irm)
"""
The interactive regression model (IRM): no 'theta x coupon' restriction. Fit spend separately for
couponed and non-couponed customers, plus a propensity model, and combine them in the doubly robust
(AIPW) score of Day 36, now with machine-learning nuisances and cross-fitting.
Targets the ATE (score="ATE") or the ATT (score="ATTE") directly.
"""
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

np.random.seed(54)  # DoubleML draws its cross-fitting folds from numpy's global RNG

full = coupons()
df = public(full).to_pandas()
t = truth(full)
data = dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS)

for score, target in [("ATE", t["ATE"]), ("ATTE", t["ATT"])]:
    irm = dml.DoubleMLIRM(data, ml_g=reg(), ml_m=clf(), score=score, n_folds=5, n_rep=3,
                          trimming_threshold=0.02)
    irm.fit()
    lo, hi = irm.confint().iloc[0]
    print(f"IRM {score:<4} {irm.coef[0]:6.2f}  (SE {irm.se[0]:.2f})  95% CI [{lo:.1f}, {hi:.1f}]   truth {target:.2f}")
```

```bash
python -m day54.irm
```

```
IRM ATE   16.14  (SE 0.60)  95% CI [15.0, 17.3]   truth 16.51
IRM ATTE  19.03  (SE 0.71)  95% CI [17.6, 20.4]   truth 20.26
```

The ATE interval contains the truth. The ATT estimate is $1.2 low and its interval just misses the truth ($20.4 vs $20.26): the customers who got coupons include many with propensities near 1, where the non-couponed comparisons are scarce. With a well-targeted campaign, the ATT is always the harder estimand (Day 35's overlap lesson).

The answer to "what did the coupons we sent achieve?" is about $19–20 each, against an ATE of $16.5: the targeting model already sends coupons to customers who respond more than average.

---

## 54.3 A confounder that was never stored, and a random holdout (55 min)

The targeting model also used a real-time **browsing signal** (customers who had just browsed a lot were more likely to get a coupon) that was never written to the data warehouse. Browsers also spend more. That's an unobserved confounder, so IRM, like everything in Days 31–36, will be biased.

But last month, the marketing team held out a **random half** of customers: they were ineligible for coupons. Random eligibility is an instrument (Day 37). Ineligible customers can't get a coupon at all (**one-sided non-compliance**), so the compliers are exactly the customers the targeting model picks when eligible: the LATE is the effect for targeted customers.

DoubleML's **interactive IV model (IIVM)** is the IV version of IRM, with ML nuisance models for spend, for the coupon, and for eligibility. Create `day54/hidden_and_iv.py`:

```python
# day54/hidden_and_iv.py   (python -m day54.hidden_and_iv)
"""
The targeting model also used a real-time browsing signal that was never stored: an unobserved confounder.
But last month NorthMart held out a random half of customers (ineligible for coupons). Random eligibility
is an instrument (Day 37). DoubleML's interactive IV model (IIVM) estimates the LATE with ML nuisances.
With one-sided non-compliance (ineligible customers can't get a coupon), the compliers are exactly the
customers the targeting model picks, so the LATE is the effect for targeted customers.
"""
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

np.random.seed(54)  # DoubleML draws its cross-fitting folds from numpy's global RNG

full = coupons(holdout=True, hidden=1.0)
df = public(full).to_pandas()
t = truth(full)
print(f"{df.eligible.mean():.0%} eligible; coupon rate {df.coupon[df.eligible == 1].mean():.0%} among eligible, "
      f"{df.coupon[df.eligible == 0].mean():.0%} among ineligible")

irm = dml.DoubleMLIRM(dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS),
                      ml_g=reg(), ml_m=clf(), score="ATTE", n_folds=5, trimming_threshold=0.02).fit()
print(f"IRM ATT (assumes no hidden confounders): {irm.coef[0]:6.2f}  (SE {irm.se[0]:.2f})   truth ATT {t['ATT']:.2f}")

wald = ((df.spend[df.eligible == 1].mean() - df.spend[df.eligible == 0].mean())
        / (df.coupon[df.eligible == 1].mean() - df.coupon[df.eligible == 0].mean()))
print(f"Wald estimate (no features):             {wald:6.2f}")

iv_data = dml.DoubleMLData(df[X_COLS + ["eligible", "coupon", "spend"]], "spend", "coupon", X_COLS, z_cols="eligible")
iivm = dml.DoubleMLIIVM(iv_data, ml_g=reg(), ml_m=clf(), ml_r=clf(), n_folds=5, n_rep=3,
                        subgroups={"always_takers": False, "never_takers": True})
iivm.fit()
lo, hi = iivm.confint().iloc[0]
print(f"IIVM LATE:                               {iivm.coef[0]:6.2f}  (SE {iivm.se[0]:.2f})  95% CI [{lo:.1f}, {hi:.1f}]"
      f"   truth LATE {t['LATE']:.2f}")
```

```bash
python -m day54.hidden_and_iv
```

```
50% eligible; coupon rate 50% among eligible, 0% among ineligible
IRM ATT (assumes no hidden confounders):  35.62  (SE 0.79)   truth ATT 19.70
Wald estimate (no features):              18.34
IIVM LATE:                                19.60  (SE 1.35)  95% CI [17.0, 22.2]   truth LATE 19.76
```

IRM reports $35.62 per coupon, **80% too high**, with a tight interval. The browsing signal makes coupon recipients look like big responders. IIVM, using the random holdout, gets $19.60, within $0.20 of the truth. The `subgroups` argument tells IIVM there are no always-takers (no ineligible customer can get a coupon), which makes it more precise.

The holdout cost the marketing team half a month of coupons for half its customers. It bought an answer that doesn't depend on what was stored.

---

## 54.4 When there's nothing hidden (40 min)

**Try it.** Rerun with `coupons(holdout=True, hidden=0.0)`: the same holdout, but no browsing signal. Compare IRM's ATT with IIVM's LATE. Create `day54/no_hidden.py`.

<details>
<summary>Solution</summary>

```python
# day54/no_hidden.py   (python -m day54.no_hidden)
import doubleml as dml
import numpy as np

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

full = coupons(holdout=True, hidden=0.0)
df = public(full).to_pandas()
t = truth(full)
np.random.seed(54)
irm = dml.DoubleMLIRM(dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS),
                      ml_g=reg(), ml_m=clf(), score="ATTE", n_folds=5, trimming_threshold=0.02).fit()
iivm = dml.DoubleMLIIVM(dml.DoubleMLData(df[X_COLS + ["eligible", "coupon", "spend"]], "spend", "coupon", X_COLS,
                                         z_cols="eligible"), reg(), clf(), clf(), n_folds=5,
                        subgroups={"always_takers": False, "never_takers": True}).fit()
print(f"IRM ATT   {irm.coef[0]:.2f} (SE {irm.se[0]:.2f})   truth {t['ATT']:.2f}")
print(f"IIVM LATE {iivm.coef[0]:.2f} (SE {iivm.se[0]:.2f})   truth {t['LATE']:.2f}")
```

```
IRM ATT   20.47 (SE 0.65)   truth 20.20
IIVM LATE 21.44 (SE 1.04)   truth 20.26
```

Both are right (IIVM is 1.2 SE high, within noise), and IRM is more precise. When nothing is hidden, the holdout is an expensive way to get a noisier answer, but you can only know nothing is hidden with a holdout to compare against. Comparing the two estimates is a check of the no-hidden-confounders assumption: they agree here and disagree by $16 in the previous section.

</details>

---

## 54.5 Day 54 mini-project: what do features buy an IV? (65 min)

The random holdout identifies the effect without any features (the Wald estimator). Features add precision, as CUPED did on Day 25. How much, and do you need ML for it? Create `day54/iv_precision.py` (about a minute):

```python
# day54/iv_precision.py   (python -m day54.iv_precision)
"""
Mini-project: the random holdout identifies the effect on its own (Wald / 2SLS without features).
What do the 24 features add? Same estimand, smaller standard error, like CUPED on Day 25.
Compare 2SLS without features, 2SLS with linear features, and IIVM with LightGBM, on 10 datasets.
"""
import doubleml as dml
import numpy as np
import polars as pl
from linearmodels.iv import IV2SLS

from causalkit.coupons import X_COLS, coupons, public, truth
from causalkit.learners import clf, reg

np.random.seed(54)  # DoubleML draws its cross-fitting folds from numpy's global RNG

rows = {"2SLS, no features": [], "2SLS, linear features": [], "IIVM, LightGBM": []}
ses = {k: [] for k in rows}
for s in range(10):
    full = coupons(seed=1500 + s, holdout=True, hidden=1.0)
    df = public(full).to_pandas().assign(const=1.0)
    late = truth(full)["LATE"]
    for k, exog in [("2SLS, no features", ["const"]), ("2SLS, linear features", ["const"] + X_COLS)]:
        fit = IV2SLS(df.spend, df[exog], df.coupon, df.eligible).fit(cov_type="robust")
        rows[k].append(fit.params["coupon"] - late)
        ses[k].append(fit.std_errors["coupon"])
    data = dml.DoubleMLData(df[X_COLS + ["eligible", "coupon", "spend"]], "spend", "coupon", X_COLS, z_cols="eligible")
    iivm = dml.DoubleMLIIVM(data, reg(), clf(), clf(), n_folds=5,
                            subgroups={"always_takers": False, "never_takers": True}).fit()
    rows["IIVM, LightGBM"].append(iivm.coef[0] - late)
    ses["IIVM, LightGBM"].append(iivm.se[0])
print(pl.DataFrame([{"estimator": k, "average error": round(np.mean(v), 2), "SD of error": round(np.std(v), 2),
                     "average SE": round(np.mean(ses[k]), 2)} for k, v in rows.items()]))
```

```bash
python -m day54.iv_precision
```

```
│ estimator             ┆ average error ┆ SD of error ┆ average SE │
│ 2SLS, no features     ┆ -0.7          ┆ 2.43        ┆ 1.94       │
│ 2SLS, linear features ┆ -0.18         ┆ 1.31        ┆ 1.34       │
│ IIVM, LightGBM        ┆ -0.18         ┆ 1.27        ┆ 1.33       │
```

Features cut the standard error by about 30% (from $1.94 to $1.33), worth twice as many customers. Here linear features do as well as LightGBM: for **precision** in a randomised design, a linear adjustment captures most of the gain. ML earns its keep when it's needed to remove **bias** (Days 51–53), not just noise.

---

## 54.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 54: DoubleML IRM (ATE, ATT), hidden confounding, IIVM with a random holdout"
git push
```

**Coming next (Days 55–57).** One effect per question is not enough to decide who gets a coupon. Days 55–57 estimate an effect **per customer**, and show how to check whether those estimates mean anything.

### Day 54 checklist

- [ ] I can fit DoubleMLIRM for the ATE and the ATT, with trimming
- [ ] I know that the ATT is harder to estimate when targeting is sharp
- [ ] I can use a random holdout as an instrument with DoubleMLIIVM
- [ ] I can compare IRM and IIVM to check for hidden confounding
- [ ] I know when features buy precision and when ML is needed

