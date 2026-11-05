# Module 6 · Causal Machine Learning & Heterogeneity
## Days 58–60: Price Elasticities and the Capstone

**Goal by end of Day 60:** Students can explain why prices set by a pricing team are endogenous, estimate a price elasticity with fixed effects, an instrument and DML, estimate elasticities for segments and segment × category cells, cross-check two estimators that rest on different assumptions, and turn elasticities and margins into a pricing decision with an honest "test first" category.

---

# DAY 58 — Price Elasticity Estimation

| Time | Block |
|---|---|
| 0:00 – 0:40 | 58.1 Why prices are endogenous |
| 0:40 – 1:30 | 58.2 Regressions that get it wrong |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 58.3 Supplier costs as an instrument |
| 2:30 – 3:20 | 58.4 DML for elasticities |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 58.5 Day 58 mini-project: when the drivers aren't recorded |
| 4:40 – 5:00 | 58.6 Save your work with Git |

---

## 58.1 Why prices are endogenous (40 min)

The **price elasticity of demand** is the % change in units sold for a 1% change in price. With elasticity −2.5, a 1% price rise loses 2.5% of units. In a regression of log units on log price, it's the slope.

The trouble: NorthMart's prices aren't random. The pricing team raises prices in weeks it expects to be busy (warm weather for snacks, holidays, local events, competitors charging more) and passes on supplier cost changes. So high prices and high demand show up together, and a naive regression mistakes "busy weeks get higher prices" for "higher prices don't hurt sales".

Create `causalkit/demand.py` (provided; read only the docstrings):

<details>
<summary>causalkit/demand.py (provided; click to expand)</summary>

```python
"""
NorthMart weekly demand (simulated). Provided for Module 6: read the docstrings, not the code,
until the Day 60 capstone is done.

80 stores in four shopper segments, five categories, 104 weeks. The pricing team sets each week's
category price from supplier costs AND from its own demand forecast: when it expects a busy week
(warm weather, a holiday, a local event, competitors charging more) it raises prices. That makes
prices and demand move together for reasons that have nothing to do with how customers respond to price.

    from causalkit.demand import demand, truth, SEGMENTS, CATEGORIES

    df = demand()                      # 41,600 store-category-weeks
    df = demand(hidden=1.0)            # the pricing team also reacts to signals that were never recorded
    truth(df)                          # true elasticity for each segment x category (for checking only)

Columns an analyst has:
    store_id, segment, category, week, price ($), units, log_price, log_units,
    cost_index (supplier cost, 1.0 = normal), promo (1 = on promotion),
    temperature (deg C), holiday (1 = holiday week), local_event (1 = event near the store),
    competitor_index (competitors' price level, 1.0 = normal), margin (gross margin share at the base price)
Columns starting with '_' are the simulator's hidden truth: _elasticity.
"""
import numpy as np
import polars as pl

SEGMENTS = ["urban premium", "suburban family", "student", "rural value"]
CATEGORIES = ["fresh", "pantry", "household", "personal care", "snacks"]
SEG_BASE = {"urban premium": -1.4, "suburban family": -2.6, "student": -3.6, "rural value": -3.2}
CAT_MULT = {"fresh": 1.0, "pantry": 0.9, "household": 1.1, "personal care": 0.8, "snacks": 1.25}
MARGIN = {"fresh": 0.28, "pantry": 0.22, "household": 0.30, "personal care": 0.35, "snacks": 0.30}
BASE_PRICE = {"fresh": 6.0, "pantry": 4.0, "household": 9.0, "personal care": 8.0, "snacks": 3.5}


def elasticity(segment: str, category: str) -> float:
    return round(SEG_BASE[segment] * CAT_MULT[category], 3)


def demand(n_stores: int = 80, weeks: int = 104, seed: int = 58, hidden: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    seg = np.array(SEGMENTS)[np.arange(n_stores) % 4]
    size = rng.lognormal(0, 0.3, n_stores)
    t = np.arange(1, weeks + 1)
    temp_wk = 10 + 12 * np.sin(2 * np.pi * (t - 15) / 52) + rng.normal(0, 3, weeks)
    holiday_wk = np.isin(t % 52, [0, 1, 13, 26, 35, 47, 51]).astype(float)
    rows = []
    for c_i, cat in enumerate(CATEGORIES):
        cost_wk = np.exp(np.cumsum(rng.normal(0, 0.02, weeks)) - 0.0)      # supplier cost drifts
        comp_wk = np.exp(rng.normal(0, 0.04, weeks))
        for s in range(n_stores):
            e = elasticity(seg[s], cat)
            temp = temp_wk + rng.normal(0, 1.5, weeks)
            event = (rng.random(weeks) < 0.15).astype(float)
            promo = (rng.random(weeks) < 0.12).astype(float)
            cost = cost_wk * np.exp(rng.normal(0, 0.05, weeks))
            comp = comp_wk * np.exp(rng.normal(0, 0.06, weeks))
            unrecorded = rng.normal(0, 1, weeks)
            # demand drivers, nonlinear: what the pricing team forecasts from
            shifter = (0.25 * np.tanh((temp - 12) / 6) * (1 if cat in ("snacks", "fresh") else 0.3)
                       + 0.30 * holiday_wk + 0.40 * event + 1.5 * np.log(comp) * (1 + (seg[s] == "student"))
                       + 0.1 * np.sin(2 * np.pi * t / 52 + c_i))
            forecast = shifter + 0.15 * hidden * unrecorded
            log_p = (np.log(BASE_PRICE[cat]) + 0.7 * np.log(cost) + 0.6 * forecast + 0.15 * forecast**2
                     - 0.20 * promo + rng.normal(0, 0.12, weeks))
            log_q = (np.log(400 * size[s]) + shifter + 0.15 * hidden * unrecorded
                     + e * (log_p - np.log(BASE_PRICE[cat])) + 0.10 * promo + rng.normal(0, 0.12, weeks))
            units = rng.poisson(np.exp(log_q))
            rows.append(pl.DataFrame({
                "store_id": f"T{s + 1:02d}", "segment": seg[s], "category": cat, "week": t,
                "price": np.round(np.exp(log_p), 2), "units": units,
                "cost_index": np.round(cost, 4), "promo": promo.astype(np.int64),
                "temperature": np.round(temp, 1), "holiday": holiday_wk.astype(np.int64),
                "local_event": event.astype(np.int64), "competitor_index": np.round(comp, 4),
                "margin": MARGIN[cat], "_elasticity": e,
            }))
    df = pl.concat(rows).filter(pl.col("units") > 0)
    return df.with_columns(log_price=pl.col("price").log(), log_units=pl.col("units").log()).select(
        "store_id", "segment", "category", "week", "price", "units", "log_price", "log_units", "cost_index", "promo",
        "temperature", "holiday", "local_event", "competitor_index", "margin", "_elasticity")


def truth(df: pl.DataFrame) -> pl.DataFrame:
    return (df.group_by("segment", "category").agg(elasticity=pl.col("_elasticity").first(),
                                                    margin=pl.col("margin").first())
            .sort("segment", "category"))


def public(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

The confounders here are the **demand drivers**: temperature, holidays, local events, competitors' prices. They're all recorded. Supplier cost is a second lever: it moves prices but not demand.

---

## 58.2 Regressions that get it wrong (50 min)

Create `day58/naive_elasticity.py`:

```python
# day58/naive_elasticity.py   (python -m day58.naive_elasticity)
"""
Price elasticity: the % change in units sold for a 1% change in price. In a log-log regression of
units on price it's the slope. But NorthMart's pricing team raises prices in weeks it expects to be busy,
so high prices and high demand go together for reasons that have nothing to do with customers' response.
"""
import pyfixest as pf

from causalkit.demand import demand, public

full = demand()
df = public(full).to_pandas()
print(f"{len(df):,} store-category-weeks: {df.store_id.nunique()} stores, {df.category.nunique()} categories, "
      f"{df.week.nunique()} weeks")
print(df.head(3).to_string(index=False))

for name, fml in [("pooled, no controls", "log_units ~ log_price"),
                  ("store x category fixed effects", "log_units ~ log_price | store_id^category"),
                  ("+ week fixed effects", "log_units ~ log_price | store_id^category + week"),
                  ("+ linear demand controls", "log_units ~ log_price + promo + temperature + holiday + local_event"
                                               " + competitor_index | store_id^category + week")]:
    fit = pf.feols(fml, df, vcov={"CRV1": "store_id"})
    print(f"{name:<34} elasticity {fit.coef()['log_price']:+.2f}  (SE {fit.se()['log_price']:.2f})")
print(f"Truth: average elasticity {full['_elasticity'].mean():.2f} (from {full['_elasticity'].min():.2f} "
      f"to {full['_elasticity'].max():.2f} across segments and categories)")
```

```bash
python -m day58.naive_elasticity
```

```
41,600 store-category-weeks: 80 stores, 5 categories, 104 weeks
...
pooled, no controls                elasticity -0.38  (SE 0.03)
store x category fixed effects     elasticity -2.04  (SE 0.09)
+ week fixed effects               elasticity -2.14  (SE 0.09)
+ linear demand controls           elasticity -2.51  (SE 0.09)
Truth: average elasticity -2.73 (from -4.50 to -1.12 across segments and categories)
```

- The pooled regression is nonsense: it mostly compares expensive categories with cheap ones.
- Store × category fixed effects compare each store-category with itself over time: much better, but still 25% too small in magnitude, because busy weeks get higher prices.
- Week fixed effects remove what's common to every store (national holidays, national weather), and linear controls remove more. But the pricing rule is nonlinear (it reacts more than proportionally to very busy weeks) and differs by segment, so a linear adjustment leaves bias behind.

An elasticity of −2.1 instead of −2.7 is not a detail: on Day 60 it decides whether a price rise looks profitable.

---

## 58.3 Supplier costs as an instrument (45 min)

**Concept.** Supplier costs shift prices (relevance) and plausibly affect units sold only through the price (exclusion: customers don't see supplier costs). That's Day 37's instrument, now for a continuous treatment. Create `day58/iv_elasticity.py`:

```python
# day58/iv_elasticity.py   (python -m day58.iv_elasticity)
"""
Supplier costs move prices but don't move customers' demand: a classic instrument (Day 37) for price.
"""
import pyfixest as pf

from causalkit.demand import demand, public

full = demand()
df = public(full).to_pandas()
first = pf.feols("log_price ~ cost_index + promo | store_id^category + week", df, vcov={"CRV1": "store_id"})
print(f"First stage: a 10% higher supplier cost raises the price by "
      f"{first.coef()['cost_index'] * 10:.1f}% (cost_index is 1.0 at normal cost); "
      f"t = {first.tstat()['cost_index']:.0f}")
iv = pf.feols("log_units ~ promo | store_id^category + week | log_price ~ cost_index", df, vcov={"CRV1": "store_id"})
print(f"2SLS elasticity: {iv.coef()['log_price']:+.2f}  (SE {iv.se()['log_price']:.2f})")
print(f"Truth: average elasticity {full['_elasticity'].mean():.2f}")
```

```bash
python -m day58.iv_elasticity
```

```
First stage: a 10% higher supplier cost raises the price by 6.2% (cost_index is 1.0 at normal cost); t = 75
2SLS elasticity: -2.82  (SE 0.10)
Truth: average elasticity -2.73
```

A strong first stage (t = 75) and an estimate within one standard error of the truth. The instrument doesn't need the demand drivers at all; its assumption is the exclusion restriction instead. (Could a supplier cost rise also signal a shortage that customers notice? That's the kind of argument Day 38 asked for.)

---

## 58.4 DML for elasticities (50 min)

**Concept.** The demand drivers are recorded, so selection on observables holds: DML's partially linear model applies with log price as a **continuous** treatment. Residualise log units and log price on the drivers with LightGBM, regress residual on residual. Two panel details:

- Remove each store × category's average first: a store's normal sales level isn't something the ML model should have to learn.
- Cluster standard errors by store (Day 41): weeks of the same store aren't independent. DoubleML does this with `cluster_cols`.

Create `day58/dml_elasticity.py`. Its `prepare` and `dml_elasticity` functions are reused on Days 59–60:

```python
# day58/dml_elasticity.py   (python -m day58.dml_elasticity)
"""
The pricing team's rule depends on things that ARE recorded (temperature, holidays, local events,
competitors' prices), in nonlinear ways. That's selection on observables, so DML's partially linear
model applies: residualise log units and log price on the demand drivers with ML, then regress
residual on residual. Store x category averages are removed first (a store's normal sales level), and
standard errors are clustered by store: weeks of the same store are not independent (Day 41).
"""
import doubleml as dml
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causalkit.demand import demand, public
from causalkit.learners import reg

DRIVERS = ["promo", "temperature", "holiday", "local_event", "competitor_index", "week", "cat", "seg"]


def prepare(df):
    """Subtract each store x category's average log units and log price; code segment and category."""
    g = df.groupby(["store_id", "category"])
    return df.assign(y=df.log_units - g.log_units.transform("mean"), d=df.log_price - g.log_price.transform("mean"),
                     cat=df.category.astype("category").cat.codes, seg=df.segment.astype("category").cat.codes,
                     store=df.store_id.str[1:].astype(int))


def dml_elasticity(df, learner=None, drivers=DRIVERS):
    np.random.seed(58)                                   # DoubleML draws its folds from numpy's global RNG
    data = dml.DoubleMLData(df[drivers + ["y", "d", "store"]], "y", "d", drivers, cluster_cols="store")
    learner = learner or reg(n_estimators=500, num_leaves=31, min_child_samples=20)
    return dml.DoubleMLPLR(data, ml_l=learner, ml_m=learner, n_folds=5).fit()


if __name__ == "__main__":
    full = demand()
    df = prepare(public(full).to_pandas())
    for name, learner in [("lasso", make_pipeline(StandardScaler(), LassoCV(cv=3))), ("LightGBM", None)]:
        fit = dml_elasticity(df, learner)
        print(f"DML with {name:<8} elasticity {fit.coef[0]:+.2f}  (SE {fit.se[0]:.2f})")
    print(f"Truth: average elasticity {full['_elasticity'].mean():.2f}")
```

```bash
python -m day58.dml_elasticity
```

```
DML with lasso    elasticity -2.49  (SE 0.10)
DML with LightGBM elasticity -2.68  (SE 0.09)
Truth: average elasticity -2.73
```

With lasso (linear) nuisance models, DML is no better than the linear-controls regression: the nuisance models can't capture the pricing rule. With LightGBM, it's within one standard error of the truth. **DML is only as good as its nuisance models.**

---

## 58.5 Day 58 mini-project: when the drivers aren't recorded (65 min)

`demand(hidden=1.0)`: the pricing team also reacts to signals it never wrote down (a store manager's call about a local road closure, say). Compare DML with 2SLS. Create `day58/hidden_drivers.py`.

<details>
<summary>Solution</summary>

```python
# day58/hidden_drivers.py   (python -m day58.hidden_drivers)
import pyfixest as pf

from causalkit.demand import demand, public
from day58.dml_elasticity import dml_elasticity, prepare

full = demand(hidden=1.0)
df = public(full).to_pandas()
fit = dml_elasticity(prepare(df))
iv = pf.feols("log_units ~ promo | store_id^category + week | log_price ~ cost_index", df, vcov={"CRV1": "store_id"})
print(f"DML  {fit.coef[0]:+.2f} (SE {fit.se[0]:.2f})")
print(f"2SLS {iv.coef()['log_price']:+.2f} (SE {iv.se()['log_price']:.2f})")
print(f"Truth {full['_elasticity'].mean():.2f}")
```

```
DML  -2.13 (SE 0.09)
2SLS -2.64 (SE 0.09)
Truth -2.73
```

DML falls back to −2.13: no ML model can adjust for a signal that isn't in the data (Day 36's "missing confounder" lesson). The instrument doesn't care: supplier costs are still unrelated to the unrecorded signal. When you can't be sure the pricing team's inputs are all recorded, an instrument is the safer estimator, if a credible one exists.

</details>

Ask in `day58/NOTES.md`: what would you need to know about NorthMart's pricing process to trust the DML estimate? Who would you ask?

---

## 58.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 58: demand simulator, endogenous prices, cost instrument, DML elasticity"
git push
```

### Day 58 checklist

- [ ] I can explain why prices set by a pricing team are endogenous
- [ ] I can estimate an elasticity with fixed effects, and say what they do and don't remove
- [ ] I can estimate an elasticity with a cost instrument and check the first stage
- [ ] I can estimate an elasticity with DML, with store-clustered standard errors
- [ ] I can say which assumption each estimator rests on, and when each one fails

---

# DAY 59 — Segment-Level Elasticities

| Time | Block |
|---|---|
| 0:00 – 0:30 | 59.1 One elasticity is not a pricing strategy |
| 0:30 – 1:30 | 59.2 Elasticities by segment and by cell |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 59.3 One model for all cells: LinearDML |
| 2:40 – 3:20 | 59.4 Additive or interacted? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 59.5 Day 59 mini-project: two estimates per cell |
| 4:40 – 5:00 | 59.6 Save your work with Git |

---

## 59.1 One elasticity is not a pricing strategy (30 min)

NorthMart's four shopper segments differ: urban premium shoppers barely notice a price change, while students and rural value shoppers switch to cheaper stores or brands. Categories differ too: snacks are easy to skip, personal care isn't. The decision on Day 60 is made cell by cell (segment × category), so the elasticity must be too.

This is Days 55–57's heterogeneity question with a continuous treatment. Two approaches today:

1. **Split and estimate**: run Day 58's DML separately in each segment and each cell.
2. **One model**: EconML's `LinearDML` makes the elasticity a linear function of segment and category features, θ(x) = x′β, with ML nuisance models for everything else.

---

## 59.2 Elasticities by segment and by cell (60 min)

Create `day59/segment_elasticities.py` (about a minute):

```python
# day59/segment_elasticities.py   (python -m day59.segment_elasticities)
"""
One elasticity for all of NorthMart hides what pricing needs: students and rural shoppers react very
differently from urban premium shoppers. Run Day 58's DML separately for each segment, then for each
segment x category cell (20 cells of about 2,000 store-weeks each).
"""
import polars as pl

from causalkit.demand import demand, public, truth
from day58.dml_elasticity import dml_elasticity, prepare

full = demand()
df = prepare(public(full).to_pandas())
true = {(r["segment"], r["category"]): r["elasticity"] for r in truth(full).iter_rows(named=True)}

rows = []
for seg, s in df.groupby("segment"):
    fit = dml_elasticity(s, drivers=["promo", "temperature", "holiday", "local_event", "competitor_index", "week", "cat"])
    rows.append({"segment": seg, "DML": round(fit.coef[0], 2), "SE": round(fit.se[0], 2),
                 "truth": round(sum(v for (sg, _), v in true.items() if sg == seg) / 5, 2)})
print(pl.DataFrame(rows))

cells = []
for (seg, cat), s in df.groupby(["segment", "category"]):
    fit = dml_elasticity(s, drivers=["promo", "temperature", "holiday", "local_event", "competitor_index", "week"])
    cells.append({"segment": seg, "category": cat, "DML": round(fit.coef[0], 2), "SE": round(fit.se[0], 2),
                  "truth": true[seg, cat]})
cells = pl.DataFrame(cells).with_columns(error=(pl.col("DML") - pl.col("truth")).round(2))
cells.write_csv("data/day59_cell_elasticities.csv")
with pl.Config(tbl_rows=20):
    print(cells)
print(f"Cells whose 95% CI contains the truth: {((cells['error'].abs()) <= 1.96 * cells['SE']).sum()} of 20")
```

```bash
python -m day59.segment_elasticities
```

```
│ segment         ┆ DML   ┆ SE   ┆ truth │
│ rural value     ┆ -3.17 ┆ 0.02 ┆ -3.23 │
│ student         ┆ -3.53 ┆ 0.02 ┆ -3.64 │
│ suburban family ┆ -2.57 ┆ 0.02 ┆ -2.63 │
│ urban premium   ┆ -1.33 ┆ 0.01 ┆ -1.41 │
│ segment         ┆ category      ┆ DML   ┆ SE   ┆ truth ┆ error │
│ rural value     ┆ fresh         ┆ -3.07 ┆ 0.03 ┆ -3.2  ┆ 0.13  │
│ rural value     ┆ household     ┆ -3.43 ┆ 0.03 ┆ -3.52 ┆ 0.09  │
...
│ urban premium   ┆ snacks        ┆ -1.62 ┆ 0.02 ┆ -1.75 ┆ 0.13  │
Cells whose 95% CI contains the truth: 4 of 20
```

The segments are clearly different (−1.3 to −3.5), and the cells add more: student snacks −4.4, urban personal care −1.1. Every estimate is within 0.15 of the truth.

But look at the last line: only 4 of 20 confidence intervals contain the truth. Every error has the **same sign**: DML is slightly too close to zero everywhere (by 0.08 on average, about 3%). The standard errors (0.02–0.03) only measure sampling noise. They don't include the small bias left by imperfect nuisance models: errors in the price model act like measurement error in price, and measurement error pulls slopes towards zero. It's small here because the data include a lot of price variation the drivers don't explain. With less, it would be bigger.

**Lesson: a precise DML estimate can still be slightly biased, and its confidence interval won't tell you.** Tomorrow's decision rule takes that into account.

---

## 59.3 One model for all cells: LinearDML (55 min)

Create `day59/linear_dml.py`:

```python
# day59/linear_dml.py   (python -m day59.linear_dml)
"""
Twenty separate DML fits are fine for 20 cells, but not for 2,000. EconML's LinearDML fits ONE model:
the elasticity is a linear function of segment and category features, theta(x) = x' beta,
while the nuisance models still use all the demand drivers (W). Here x = segment and category dummies,
so the elasticity is 'segment effect + category effect', with confidence intervals for every cell.
"""
import numpy as np
import pandas as pd
import polars as pl
from econml.dml import LinearDML

from causalkit.demand import demand, public, truth
from causalkit.learners import reg
from day58.dml_elasticity import prepare

full = demand()
df = prepare(public(full).to_pandas())
X = pd.get_dummies(df[["segment", "category"]], drop_first=True).astype(float)
W = df[["promo", "temperature", "holiday", "local_event", "competitor_index", "week", "cat", "seg"]]
learner = reg(n_estimators=500, num_leaves=31, min_child_samples=20)
est = LinearDML(model_y=learner, model_t=learner, cv=5, random_state=0)
est.fit(df.y, df.d, X=X, W=W, groups=df.store)              # groups: folds keep each store together
print(est.summary(feature_names=list(X.columns)).tables[0])

cells = truth(full).to_pandas()
Xc = pd.get_dummies(cells[["segment", "category"]], drop_first=True).astype(float)[X.columns]
pt = est.effect(Xc)
lo, hi = est.effect_interval(Xc, alpha=0.05)
out = pl.DataFrame({"segment": cells.segment, "category": cells.category, "LinearDML": np.round(pt, 2),
                    "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(lo, hi)], "truth": cells.elasticity})
with pl.Config(tbl_rows=20, tbl_width_chars=100, fmt_str_lengths=30):
    print(out)
print(f"Mean absolute error {np.mean(np.abs(pt - cells.elasticity)):.2f}")
```

```bash
python -m day59.linear_dml
```

```
                        point_estimate stderr  zstat  pvalue ci_lower ci_upper
segment_student                 -0.374  0.019 -20.103    0.0    -0.41   -0.337
segment_suburban family          0.586  0.017   33.94    0.0    0.552     0.62
segment_urban premium            1.826  0.017 105.117    0.0    1.792     1.86
category_household              -0.295  0.019 -15.791    0.0   -0.332   -0.259
category_pantry                   0.25  0.019  13.296    0.0    0.213    0.287
category_personal care           0.491  0.019  26.011    0.0    0.454    0.528
category_snacks                 -0.654  0.021 -31.507    0.0   -0.695   -0.613
│ segment         ┆ category      ┆ LinearDML ┆ 95% CI         ┆ truth │
│ urban premium   ┆ personal care ┆ -0.81     ┆ [-0.85, -0.78] ┆ -1.12 │
│ urban premium   ┆ snacks        ┆ -1.96     ┆ [-2.00, -1.92] ┆ -1.75 │
...
Mean absolute error 0.11
```

The coefficients read as differences from the base cell (rural value, fresh): urban premium shoppers are 1.83 less elastic, snacks 0.65 more. One model, with confidence intervals for every cell, and it scales to thousands of cells, where splitting wouldn't. `groups=df.store` keeps each store's weeks in the same cross-fitting fold.

But the error is bigger than splitting (0.11 vs about 0.08), and the worst cells are off by 0.3: urban premium personal care is estimated at −0.81 against a true −1.12.

---

## 59.4 Additive or interacted? (40 min)

**Concept.** The model above is **additive**: segment effect + category effect. But true elasticities are **multiplicative**: a segment's sensitivity scales every category's (students are 2.6 times as elastic as urban premium shoppers in every category). An additive model can't represent that, however much data it has. The CATE model's functional form matters, just like the outcome model's did on Day 31.

**Try it.** Refit `LinearDML` with one dummy per segment × category cell (19 dummies plus the base cell) instead of separate segment and category dummies. Create `day59/interactions.py`.

<details>
<summary>Solution</summary>

```python
# day59/interactions.py   (python -m day59.interactions)
import numpy as np
import pandas as pd
from econml.dml import LinearDML

from causalkit.demand import demand, public, truth
from causalkit.learners import reg
from day58.dml_elasticity import prepare

full = demand()
df = prepare(public(full).to_pandas())
cell = df.segment + " / " + df.category
X = pd.get_dummies(cell, drop_first=True).astype(float)
W = df[["promo", "temperature", "holiday", "local_event", "competitor_index", "week", "cat", "seg"]]
learner = reg(n_estimators=500, num_leaves=31, min_child_samples=20)
est = LinearDML(model_y=learner, model_t=learner, cv=5, random_state=0).fit(df.y, df.d, X=X, W=W, groups=df.store)
cells = truth(full).to_pandas()
Xc = pd.get_dummies(cells.segment + " / " + cells.category).astype(float).reindex(columns=X.columns, fill_value=0)
print(f"LinearDML with a separate elasticity per cell: mean absolute error "
      f"{np.mean(np.abs(est.effect(Xc) - cells.elasticity)):.2f}")
```

```
LinearDML with a separate elasticity per cell: mean absolute error 0.06
```

With a free elasticity per cell, the error falls to 0.06, a bit better than splitting (the nuisance models now learn from all 41,600 rows). With thousands of cells you can't afford a dummy each; then the choice is between a structured model (segment × category interactions, or log-elasticities that add up) and a causal forest, which learns the interactions itself.

</details>

---

## 59.5 Day 59 mini-project: two estimates per cell (65 min)

DML assumes the pricing team only reacted to recorded drivers. The cost instrument assumes supplier costs affect sales only through price. Different assumptions, so if both estimates agree in a cell, the answer doesn't rest on one assumption alone. Create `day59/iv_crosscheck.py`:

```python
# day59/iv_crosscheck.py   (python -m day59.iv_crosscheck)
"""
Mini-project: two estimates per cell that rest on DIFFERENT assumptions.
  DML (Day 59):  the pricing team only reacted to recorded demand drivers (selection on observables).
  2SLS (Day 58): supplier costs affect units only through price (exclusion).
If they agree, neither assumption is doing all the work alone. Run day59.segment_elasticities first.
"""
import numpy as np
import polars as pl
import pyfixest as pf

from causalkit.demand import demand, public

full = demand()
df = public(full).to_pandas()
dml_cells = pl.read_csv("data/day59_cell_elasticities.csv")
rows = []
for r in dml_cells.iter_rows(named=True):
    s = df[(df.segment == r["segment"]) & (df.category == r["category"])]
    iv = pf.feols("log_units ~ promo | store_id + week | log_price ~ cost_index", s, vcov={"CRV1": "store_id"})
    rows.append({"segment": r["segment"], "category": r["category"], "DML": r["DML"],
                 "2SLS": round(iv.coef()["log_price"], 2), "2SLS SE": round(iv.se()["log_price"], 2),
                 "truth": r["truth"]})
out = pl.DataFrame(rows)
with pl.Config(tbl_rows=20):
    print(out)
iv_err = out["2SLS"] - out["truth"]
dml_err = out["DML"] - out["truth"]
print(f"Average error: DML {dml_err.mean():+.3f}, 2SLS {iv_err.mean():+.3f}")
print(f"Typical size of error: DML {np.sqrt((dml_err**2).mean()):.3f}, 2SLS {np.sqrt((iv_err**2).mean()):.3f}")
print(f"2SLS 95% CIs containing the truth: {(iv_err.abs() <= 1.96 * out['2SLS SE']).sum()} of 20")
```

```bash
python -m day59.iv_crosscheck
```

```
│ segment         ┆ category      ┆ DML   ┆ 2SLS  ┆ 2SLS SE ┆ truth │
│ rural value     ┆ fresh         ┆ -3.07 ┆ -3.32 ┆ 0.13    ┆ -3.2  │
│ student         ┆ snacks        ┆ -4.36 ┆ -4.86 ┆ 0.23    ┆ -4.5  │
│ suburban family ┆ personal care ┆ -2.03 ┆ -2.51 ┆ 0.19    ┆ -2.08 │
...
Average error: DML +0.077, 2SLS -0.036
Typical size of error: DML 0.086, 2SLS 0.167
2SLS 95% CIs containing the truth: 19 of 20
```

The two estimators fail in opposite ways:

| | DML | 2SLS |
|---|---|---|
| Bias | small, always towards zero (+0.08) | none on average |
| Noise | small (SE 0.02–0.03) | large (SE 0.08–0.23) |
| Honest CI? | no: 4 of 20 cover | yes: 19 of 20 cover |
| Typical error | 0.09 | 0.17 |

DML is usually closer; 2SLS is honest about its uncertainty. Day 60 uses both: DML as the point estimate, the 2SLS interval as the range of plausible values.

---

## 59.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 59: segment and cell elasticities, LinearDML, IV cross-check"
git push
```

### Day 59 checklist

- [ ] I can estimate elasticities by segment and by segment × category cell with DML
- [ ] I know that DML's confidence intervals don't include the small bias left by its nuisance models
- [ ] I can fit LinearDML with segment and category features, and read its coefficients
- [ ] I can explain why an additive CATE model misses multiplicative heterogeneity
- [ ] I cross-check cell estimates with an estimator that rests on a different assumption

---

# DAY 60 — Capstone: The Price Elasticity Engine

| Time | Block |
|---|---|
| 0:00 – 0:30 | 60.1 The brief |
| 0:30 – 1:30 | 60.2 Step 1: an elasticity for every cell, two ways |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 60.3 Step 2: from elasticities to profit |
| 2:45 – 3:00 | Break |
| 3:00 – 4:20 | 60.4 Step 3: the pricing memo |
| 4:20 – 5:00 | 60.5 Answer key, final commit and Module 6 retrospective |

---

## 60.1 The brief (30 min)

NorthMart's category directors want to raise prices by **5%** to recover rising costs. The CFO asks: **which shopper segments, in which categories, can absorb a 5% rise without losing gross profit?**

The arithmetic. With gross margin m (as a share of the current price) and elasticity e, a 5% rise:

- changes units by a factor of 1.05^e (with e = −3, units fall by 14%);
- raises the profit per unit from m to m + 0.05 (as a share of the old price).

So gross profit changes by `1.05^e × (m + 0.05) / m − 1`. Setting it to zero gives each category's **break-even elasticity**: a fresh category (m = 28%) can absorb a 5% rise only if its elasticity is above −3.37; personal care (m = 35%) only above −2.74. Low-margin categories can absorb more, because the extra 5% is a bigger share of what they earn per unit.

Generate a new year of data:

```bash
python data/make_demand_scenario.py
```

```python
"""
Capstone data for Module 6: a new year of NorthMart weekly demand. Provided.

    python data/make_demand_scenario.py

Writes data/scenario/weekly_demand.csv (what the analyst gets) and data/truth/elasticities.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.demand import demand, public, truth  # noqa: E402

full = demand(seed=60)
root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
(root / "truth").mkdir(exist_ok=True)
public(full).write_csv(root / "scenario" / "weekly_demand.csv")
truth(full).write_csv(root / "truth" / "elasticities.csv")
print(f"Wrote {full.height:,} store-category-weeks to data/scenario/weekly_demand.csv")
```

**Don't open `data/truth/` until your memo is committed.**

---

## 60.2 Step 1: an elasticity for every cell, two ways (60 min)

Create `day60/estimates.py` (about a minute):

```python
# day60/estimates.py   (python -m day60.estimates)
"""Capstone step 1: an elasticity for every segment x category cell, two ways (Day 59)."""
import pandas as pd
import polars as pl
import pyfixest as pf

from day58.dml_elasticity import dml_elasticity, prepare

DRIVERS = ["promo", "temperature", "holiday", "local_event", "competitor_index", "week"]
raw = pd.read_csv("data/scenario/weekly_demand.csv")
df = prepare(raw)
rows = []
for (seg, cat), s in df.groupby(["segment", "category"]):
    fit = dml_elasticity(s, drivers=DRIVERS)
    iv = pf.feols("log_units ~ promo | store_id + week | log_price ~ cost_index", s, vcov={"CRV1": "store_id"})
    rows.append({"segment": seg, "category": cat, "margin": s.margin.iloc[0],
                 "annual gross profit ($k)": round((s.price * s.units * s.margin).sum() / 2 / 1000, 1),
                 "DML": round(fit.coef[0], 3), "DML SE": round(fit.se[0], 3),
                 "2SLS": round(iv.coef()["log_price"], 3), "2SLS SE": round(iv.se()["log_price"], 3)})
out = pl.DataFrame(rows)
out.write_csv("data/day60_estimates.csv")
with pl.Config(tbl_rows=20, tbl_width_chars=130):
    print(out)
```

```bash
python -m day60.estimates
```

```
│ segment         ┆ category      ┆ margin ┆ annual gross profit ($k) ┆ DML    ┆ DML SE ┆ 2SLS   ┆ 2SLS SE │
│ rural value     ┆ fresh         ┆ 0.28   ┆ 649.6                    ┆ -3.054 ┆ 0.025  ┆ -3.084 ┆ 0.101   │
│ rural value     ┆ household     ┆ 0.3    ┆ 1328.2                   ┆ -3.463 ┆ 0.022  ┆ -3.422 ┆ 0.145   │
...
│ urban premium   ┆ snacks        ┆ 0.3    ┆ 422.0                    ┆ -1.692 ┆ 0.03   ┆ -1.783 ┆ 0.12    │
```

---

## 60.3 Step 2: from elasticities to profit (60 min)

Create `day60/decision.py`. Its decision rule follows Day 59: DML is the best guess, but its interval is too narrow; the 2SLS interval is honest. A cell is a clear **raise** only if profit rises at the DML estimate *and* at the most elastic end of the 2SLS interval; a clear **don't** if it falls at both the DML estimate and the least elastic end; everything else is **test first**.

```python
# day60/decision.py   (python -m day60.decision)
"""
Capstone step 2: can each cell absorb a 5% price rise?
With gross margin m (share of the current price) and elasticity e, a 5% rise changes gross profit by
    1.05**e * (m + 0.05) / m - 1
(units fall by 1.05**e; each unit now earns m + 0.05 of the old price instead of m).
Decision rule, using DML as the estimate and the 2SLS 95% interval as an honest range (Day 59):
    raise      if profit rises at the DML estimate AND at the elastic end of the 2SLS interval
    don't      if profit falls at the DML estimate AND at the inelastic end of the 2SLS interval
    test first otherwise
"""
import numpy as np
import polars as pl


def profit_change(e, m):
    return 1.05**e * (m + 0.05) / m - 1


if __name__ == "__main__":
    est = pl.read_csv("data/day60_estimates.csv")
    rows = []
    for r in est.iter_rows(named=True):
        m = r["margin"]
        breakeven = np.log(m / (m + 0.05)) / np.log(1.05)
        lo, hi = r["2SLS"] - 1.96 * r["2SLS SE"], r["2SLS"] + 1.96 * r["2SLS SE"]
        at_dml, worst, best = profit_change(r["DML"], m), profit_change(lo, m), profit_change(hi, m)
        verdict = "raise" if at_dml > 0 and worst > 0 else "don't" if at_dml < 0 and best < 0 else "test first"
        rows.append({"segment": r["segment"], "category": r["category"], "break-even elasticity": round(breakeven, 2),
                     "DML": r["DML"], "profit change": f"{at_dml:+.1%}", "range (2SLS CI)": f"{worst:+.1%} to {best:+.1%}",
                     "$k per year": round(at_dml * r["annual gross profit ($k)"], 1), "verdict": verdict})
    out = pl.DataFrame(rows)
    out.write_csv("data/day60_decisions.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=150, fmt_str_lengths=30):
        print(out)
    summary = out.group_by("segment").agg(raise_=(pl.col("verdict") == "raise").sum(),
                                           dont=(pl.col("verdict") == "don't").sum(),
                                           test=(pl.col("verdict") == "test first").sum(),
                                           gain_if_raise_only=pl.when(pl.col("verdict") == "raise")
                                           .then(pl.col("$k per year")).otherwise(0).sum().round(1)).sort("segment")
    print(summary.rename({"raise_": "raise", "dont": "don't", "test": "test first",
                          "gain_if_raise_only": "$k/yr from 'raise' cells"}))
    print(f"Raising every cell: {out['$k per year'].sum():+.1f} $k/yr;  raising only 'raise' cells: "
          f"{out.filter(pl.col('verdict') == 'raise')['$k per year'].sum():+.1f} $k/yr")
```

```bash
python -m day60.decision
```

```
│ segment         ┆ category      ┆ break-even elasticity ┆ DML    ┆ profit change ┆ range (2SLS CI)  ┆ $k per year ┆ verdict    │
│ rural value     ┆ fresh         ┆ -3.37                 ┆ -3.054 ┆ +1.5%         ┆ +0.4% to +2.4%   ┆ 10.0        ┆ raise      │
│ rural value     ┆ household     ┆ -3.16                 ┆ -3.463 ┆ -1.5%         ┆ -2.6% to +0.1%   ┆ -19.5       ┆ test first │
│ rural value     ┆ pantry        ┆ -4.2                  ┆ -2.832 ┆ +6.9%         ┆ +4.5% to +8.2%   ┆ 28.0        ┆ raise      │
│ rural value     ┆ personal care ┆ -2.74                 ┆ -2.444 ┆ +1.4%         ┆ -0.9% to +1.9%   ┆ 19.4        ┆ test first │
│ rural value     ┆ snacks        ┆ -3.16                 ┆ -3.936 ┆ -3.7%         ┆ -7.3% to -3.1%   ┆ -15.5       ┆ don't      │
│ student         ┆ fresh         ┆ -3.37                 ┆ -3.522 ┆ -0.8%         ┆ -2.6% to +0.2%   ┆ -4.5        ┆ test first │
│ student         ┆ household     ┆ -3.16                 ┆ -3.811 ┆ -3.1%         ┆ -5.3% to -3.0%   ┆ -39.2       ┆ don't      │
│ student         ┆ pantry        ┆ -4.2                  ┆ -3.114 ┆ +5.4%         ┆ +3.2% to +6.5%   ┆ 20.5        ┆ raise      │
│ student         ┆ personal care ┆ -2.74                 ┆ -2.828 ┆ -0.4%         ┆ -2.2% to +0.3%   ┆ -5.6        ┆ test first │
│ student         ┆ snacks        ┆ -3.16                 ┆ -4.37  ┆ -5.7%         ┆ -7.7% to -4.5%   ┆ -23.1       ┆ don't      │
│ suburban family ┆ fresh         ┆ -3.37                 ┆ -2.45  ┆ +4.6%         ┆ +2.2% to +4.8%   ┆ 30.4        ┆ raise      │
│ suburban family ┆ household     ┆ -3.16                 ┆ -2.783 ┆ +1.9%         ┆ -0.6% to +1.4%   ┆ 23.8        ┆ test first │
│ suburban family ┆ pantry        ┆ -4.2                  ┆ -2.259 ┆ +9.9%         ┆ +9.1% to +11.9%  ┆ 39.9        ┆ raise      │
│ suburban family ┆ personal care ┆ -2.74                 ┆ -2.032 ┆ +3.5%         ┆ +2.2% to +4.8%   ┆ 47.2        ┆ raise      │
│ suburban family ┆ snacks        ┆ -3.16                 ┆ -3.153 ┆ +0.0%         ┆ -0.4% to +2.0%   ┆ 0.1         ┆ test first │
│ urban premium   ┆ fresh         ┆ -3.37                 ┆ -1.372 ┆ +10.2%        ┆ +8.1% to +11.6%  ┆ 70.7        ┆ raise      │
│ urban premium   ┆ household     ┆ -3.16                 ┆ -1.485 ┆ +8.5%         ┆ +7.8% to +9.5%   ┆ 99.7        ┆ raise      │
│ urban premium   ┆ pantry        ┆ -4.2                  ┆ -1.262 ┆ +15.4%        ┆ +14.3% to +16.8% ┆ 59.1        ┆ raise      │
│ urban premium   ┆ personal care ┆ -2.74                 ┆ -1.112 ┆ +8.3%         ┆ +7.2% to +10.0%  ┆ 104.8       ┆ raise      │
│ urban premium   ┆ snacks        ┆ -3.16                 ┆ -1.692 ┆ +7.4%         ┆ +5.7% to +8.2%   ┆ 31.3        ┆ raise      │
│ segment         ┆ raise ┆ don't ┆ test first ┆ $k/yr from 'raise' cells │
│ rural value     ┆ 2     ┆ 1     ┆ 2          ┆ 38.0                     │
│ student         ┆ 1     ┆ 2     ┆ 2          ┆ 20.5                     │
│ suburban family ┆ 3     ┆ 0     ┆ 2          ┆ 117.5                    │
│ urban premium   ┆ 5     ┆ 0     ┆ 0          ┆ 365.6                    │
Raising every cell: +477.5 $k/yr;  raising only 'raise' cells: +541.6 $k/yr
```

The answer to the CFO's question by segment:

- **Urban premium** can absorb 5% everywhere: +$366k a year.
- **Suburban families** in fresh, pantry and personal care; household and snacks are too close to call.
- **Students and rural value shoppers** only in low-margin pantry (and rural fresh); household and snacks lose money.

A blanket 5% rise earns $64k a year **less** than the targeted one, because it loses money in five cells.

---

## 60.4 Step 3: the pricing memo (80 min)

Write `docs/PRICE_RISE_DECISION.md`, one page, for the CFO:

1. **Recommendation** (which cells to raise now, which not, which to test).
2. **Why we didn't use last year's simple regression** (one paragraph; the fixed-effects estimate of −2.1 against −2.7).
3. **Results** (the 20-cell table, condensed to segment level, with dollars).
4. **How sure are we?** (two estimators, what each assumes, the "test first" cells).
5. **Next steps** (how to settle the test-first cells).

<details>
<summary>Model answer for sections 1, 4 and 5</summary>

> **Recommendation.** Raise prices 5% in 11 of 20 segment-category cells: every category for urban premium shoppers; fresh, pantry and personal care for suburban families; pantry for students; pantry and fresh for rural value shoppers. Expected gain: about $540k of gross profit a year. Don't raise household or snacks prices for students, or snacks for rural value shoppers: those shoppers would buy enough less to cost about $78k a year. Test the remaining six cells before deciding.
>
> **How sure are we?** Each elasticity was estimated twice, with methods that rely on different assumptions: one assumes we recorded everything the pricing team reacts to; the other uses supplier cost changes, which customers don't see. They agree closely in most cells. We only call a cell "raise" or "don't" if the decision holds across the plausible range of both. Six cells (household and personal care for rural value shoppers, fresh and personal care for students, household and snacks for suburban families) sit too close to the break-even point to call from historical data.
>
> **Next steps.** For the test-first cells, run a 6-week price test (Module 3): raise prices by 5% in a random half of the stores in each segment and compare. With about 20 stores per segment, that settles whether each cell is above or below break-even. Revisit the "don't" cells if costs keep rising: the break-even point moves with the margin.

</details>

---

## 60.5 Answer key, final commit and Module 6 retrospective (40 min)

After the memo is committed, compare with the truth. Create `day60/answer_key.py`:

```python
# day60/answer_key.py   (python -m day60.answer_key)
"""Instructor check, after the memo: true elasticities, true profit changes, and how the verdicts fared."""
import polars as pl

from day60.decision import profit_change

truth = pl.read_csv("data/truth/elasticities.csv")
dec = pl.read_csv("data/day60_decisions.csv")
est = pl.read_csv("data/day60_estimates.csv").select("segment", "category", "annual gross profit ($k)")
out = (dec.join(truth, on=["segment", "category"]).join(est, on=["segment", "category"])
       .with_columns(true_change=pl.struct("elasticity", "margin").map_elements(
           lambda r: profit_change(r["elasticity"], r["margin"]), return_dtype=pl.Float64)))
out = out.with_columns(true_dollars=(pl.col("true_change") * pl.col("annual gross profit ($k)")).round(1),
                       right=pl.when(pl.col("verdict") == "test first").then(pl.lit("-"))
                       .when((pl.col("verdict") == "raise") == (pl.col("true_change") > 0)).then(pl.lit("yes"))
                       .otherwise(pl.lit("NO")))
with pl.Config(tbl_rows=20, tbl_width_chars=140):
    print(out.select("segment", "category", "elasticity", "DML", pl.col("true_change").map_elements(
        lambda v: f"{v:+.1%}", return_dtype=pl.String).alias("true profit change"), "true_dollars", "verdict", "right"))
print(f"True gain from raising only the 'raise' cells: "
      f"{out.filter(pl.col('verdict') == 'raise')['true_dollars'].sum():+.1f} $k/yr; "
      f"from raising every cell: {out['true_dollars'].sum():+.1f} $k/yr; "
      f"best possible (raise exactly the profitable cells): {out.filter(pl.col('true_dollars') > 0)['true_dollars'].sum():+.1f} $k/yr")
```

```bash
python -m day60.answer_key
```

```
│ segment         ┆ category      ┆ elasticity ┆ DML    ┆ true profit change ┆ true_dollars ┆ verdict    ┆ right │
│ rural value     ┆ fresh         ┆ -3.2       ┆ -3.054 ┆ +0.8%              ┆ 5.3          ┆ raise      ┆ yes   │
│ rural value     ┆ household     ┆ -3.52      ┆ -3.463 ┆ -1.7%              ┆ -23.2        ┆ test first ┆ -     │
...
│ suburban family ┆ snacks        ┆ -3.25      ┆ -3.153 ┆ -0.4%              ┆ -1.9         ┆ test first ┆ -     │
...
True gain from raising only the 'raise' cells: +517.0 $k/yr; from raising every cell: +416.5 $k/yr; best possible (raise exactly the profitable cells): +547.6 $k/yr
```

Every "raise" and "don't" verdict was right. The test-first cells were genuinely close: four of the six would have lost money (suburban snacks by just 0.4%), two would have made a little. The targeted plan earns $517k of the $548k possible, and $101k more than a blanket rise. The DML point estimates alone would have called suburban snacks a (tiny) gain, wrongly; the cross-check kept it in "test first".

```bash
git add .
git commit -m "Day 60: price elasticity engine capstone: cell elasticities, profit, memo"
git push
```

**What you learned in Module 6.**

| Day | Idea | The simulator lesson |
|---|---|---|
| 51 | Prediction ≠ causation | The best predictor underestimated the coupon effect, even in an experiment |
| 52 | Orthogonality, cross-fitting | Without either, 95% CIs covered 20–32% of the time; with both, 92% |
| 53 | DoubleML PLR | Linear nuisances gave OLS's answer; LightGBM within half an SE |
| 54 | IRM and IIVM | A hidden browsing signal nearly doubled the ATT; the random holdout recovered it |
| 55 | Meta-learners | DR-learner best with enough data and a simple final model; S-learner best at 2,000 |
| 56 | Causal forests | Forest group averages were shrunk; DR-score GATEs were within $1.5 |
| 57 | Validation | On an experiment, BLP and calibration flagged the T-learner's noise and the forest's compression |
| 58 | Elasticities | Fixed effects: −2.1; cost IV −2.8; DML −2.7; truth −2.7 |
| 59 | Segment elasticities | DML precise but 3% biased; 2SLS noisy but honest; use both |
| 60 | Capstone | Targeted rise +$517k vs blanket +$417k; no wrong "raise" or "don't" verdicts |

**Coming next: Module 7, Sensitivity, Robustness & Decisions.** Every module so far ended with an assumption you couldn't test. Module 7 asks how wrong the assumption would have to be to change the decision: sensitivity analysis for unobserved confounding, specification curves, and decision analysis under uncertainty.

### Day 60 checklist

- [ ] I computed each cell's break-even elasticity from its margin
- [ ] I estimated every cell's elasticity with two estimators that rest on different assumptions
- [ ] I made "raise", "don't" and "test first" calls that account for DML's narrow intervals
- [ ] I put dollars on the decision and compared it with a blanket rise
- [ ] I wrote a memo whose numbers match the table

