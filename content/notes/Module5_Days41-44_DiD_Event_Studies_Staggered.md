# Module 5 · Quasi-Experiments: DiD, Synthetic Control, RDD
## Days 41–44: Difference-in-Differences, Event Studies and Staggered Rollouts

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–4
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 44:** Students can estimate a difference-in-differences with fixed-effects regression and correctly clustered standard errors, read an event study and know what a passed pre-trend test does and doesn't prove, explain why two-way fixed effects fails with staggered rollouts, and estimate staggered effects with Callaway–Sant'Anna and imputation estimators.

---

### Where we are

Module 4 needed either every confounder measured, or a lucky random nudge. Module 5 uses a third kind of leverage: **structure in time and in rules**.

| Days | Design | Leverage | Key assumption |
|---|---|---|---|
| 41–44 | Difference-in-differences | a group that wasn't treated, followed over the same weeks | parallel trends |
| 45–47 | Synthetic control and Bayesian time series | a weighted mix of untreated units that tracked the treated one | the mix keeps tracking it |
| 48 | Regression discontinuity | a rule with a cutoff | nothing else jumps at the cutoff |
| 49–50 | Placebo tests, then the capstone | | |

### Three simulators

All three are provided. **Read only the docstrings**: the code contains the capstone answers.

| File | What it simulates | Used on |
|---|---|---|
| `causalkit/stores.py` | 240 stores × 60 weeks: a regional campaign, and a new store format rolled out in waves | Days 41–44, 49 |
| `causalkit/regions.py` | 30 regions × 104 weeks: one region runs an 8-week TV campaign | Days 45–47, 49, 50 |
| `causalkit/tiers.py` | 30,000 customers: Gold status starts at $1,000 of spending | Days 48–49 |

<details>
<summary>causalkit/stores.py (provided; click to expand)</summary>

```python
"""
NorthMart store panel (simulated). Provided for Module 5: read the docstrings, not the code,
until the Day 50 capstone is done.

    from causalkit.stores import campaign, rollout, truth

    df = campaign()                  # 240 stores x 60 weeks; stores in two regions get a campaign from week 40
    df = campaign(trend_gap=0.15)    # same, but treated stores were already growing faster
    df = rollout()                   # a new store format, adopted in waves (weeks 15, 30, 45) or never
    truth(df)                        # true average effects (for checking only)

Columns an analyst has:
    store_id, region, week, sales ($ thousands per week),
    treated_group (1 = store is ever treated), cohort (first treated week; 0 = never),
    post (1 = week >= start; campaign only), d (1 = treated in this week)
Columns starting with '_' are the simulator's hidden truth: _y0 (sales without treatment), _tau (effect).
"""
import numpy as np
import polars as pl

REGIONS = ["BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS"]


def _base(n_stores: int, weeks: int, rng: np.random.Generator, rho: float = 0.7, region_sd: float = 0.0):
    """Store effects, shared seasonality, regional shocks and AR(1) store noise."""
    region = rng.choice(REGIONS, n_stores)
    store_fe = rng.normal(100, 20, n_stores)
    t = np.arange(1, weeks + 1)
    week_fe = 4 * np.sin(2 * np.pi * t / 52) + 0.05 * t + rng.normal(0, 1.0, weeks)
    region_shock = {r: np.cumsum(rng.normal(0, 1, weeks)) * region_sd for r in REGIONS}
    e = np.zeros((n_stores, weeks))
    u = rng.normal(0, 2.5, (n_stores, weeks))
    e[:, 0] = u[:, 0] / np.sqrt(1 - rho**2)
    for w in range(1, weeks):
        e[:, w] = rho * e[:, w - 1] + u[:, w]
    y0 = store_fe[:, None] + week_fe[None, :] + np.vstack([region_shock[r] for r in region]) + e
    return region, t, y0


def _frame(region, t, y0, tau, d, cohort, extra=None) -> pl.DataFrame:
    n, w = y0.shape
    df = pl.DataFrame({
        "store_id": np.repeat([f"S{i:03d}" for i in range(1, n + 1)], w),
        "region": np.repeat(region, w),
        "week": np.tile(t, n).astype(np.int64),
        "sales": np.round((y0 + tau * d).ravel(), 2),
        "treated_group": np.repeat((cohort > 0).astype(np.int64), w),
        "cohort": np.repeat(cohort, w).astype(np.int64),
        "d": d.ravel().astype(np.int64),
        "_y0": y0.ravel(),
        "_tau": (tau * d).ravel(),
    })
    return df.with_columns(**(extra or {}))


def campaign(n_stores: int = 240, weeks: int = 60, start: int = 40, seed: int = 41,
             effect: float = 3.0, trend_gap: float = 0.0, treated_regions=("ON", "QC"),
             region_sd: float = 0.0) -> pl.DataFrame:
    """A regional campaign: every store in treated_regions is treated from week `start`.
    effect:    true sales lift, $ thousands per store-week (constant over time)
    trend_gap: extra weekly growth of treated stores' sales, before AND after the campaign
               (violates parallel trends when > 0)
    region_sd: size of week-to-week shocks shared by all stores in a region (random walk; 0 = none)"""
    rng = np.random.default_rng(seed)
    region, t, y0 = _base(n_stores, weeks, rng, region_sd=region_sd)
    g = np.isin(region, treated_regions)
    y0 = y0 + trend_gap * g[:, None] * (t[None, :] - start)
    d = g[:, None] & (t[None, :] >= start)
    tau = np.full(y0.shape, effect)
    cohort = np.where(g, start, 0)
    return _frame(region, t, y0, tau, d, cohort,
                  {"post": (pl.col("week") >= start).cast(pl.Int64)}).select(
        "store_id", "region", "week", "sales", "treated_group", "cohort", "post", "d", "_y0", "_tau")


def rollout(n_stores: int = 240, weeks: int = 60, seed: int = 47, cohorts=(15, 30, 45),
            growth: float = 0.4, cap: int = 20, dynamic: bool = True) -> pl.DataFrame:
    """A new store format adopted in waves. A quarter of stores adopt in each cohort week; a quarter never do.
    The effect grows by `growth` per week since adoption up to `cap` weeks (dynamic=True), and is larger
    for earlier adopters (stores that volunteered first were better suited to the format).
    dynamic=False gives the same average effect but constant over time."""
    rng = np.random.default_rng(seed)
    region, t, y0 = _base(n_stores, weeks, rng)
    groups = np.array(list(cohorts) + [0])
    cohort = groups[rng.permutation(np.arange(n_stores) % len(groups))]
    rel = t[None, :] - cohort[:, None]
    d = (cohort[:, None] > 0) & (rel >= 0)
    suit = np.select([cohort == cohorts[0], cohort == cohorts[1]], [1.5, 1.0], 0.6)[:, None]
    if dynamic:
        tau = suit * growth * np.minimum(rel + 1, cap)
    else:
        tau = suit * growth * np.minimum(np.maximum(rel, 0) + 1, cap).mean() * np.ones_like(y0)
    tau = np.where(d, tau, 0.0)
    return _frame(region, t, y0, tau, d, cohort)


def truth(df: pl.DataFrame) -> dict:
    """True ATT over treated store-weeks, and by cohort."""
    tr = df.filter(pl.col("d") == 1)
    out = {"ATT": round(tr["_tau"].mean(), 3)}
    for c in sorted(tr["cohort"].unique().to_list()):
        out[f"ATT_cohort_{c}"] = round(tr.filter(pl.col("cohort") == c)["_tau"].mean(), 3)
    return out


def public(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

### New folders and packages

```
causal-decision-lab/
├── causalkit/
│   ├── stores.py, regions.py, tiers.py   # provided simulators
│   ├── did.py                            # Days 43–44: staggered DiD by hand
│   └── synth.py                          # Days 45–49: synthetic control, SDID, placebos
├── data/make_campaign_scenario.py        # Day 50 capstone data
└── day41/ ... day50/
```

Add to `requirements.txt`, then `pip install -r requirements.txt`:

```text
# Module 5: quasi-experiments
pyfixest==0.60.0
wildboottest==0.3.2
rdrobust==2.1.1
rddensity==3.0
causalpy==0.9.0
pymc==5.28.5
```

**One change to an earlier pin.** PyMC compiles models with a library (numba) that doesn't support numpy 2.5 yet. Change the numpy line at the top of `requirements.txt` to:

```text
numpy==2.4.6           # not 2.5 yet: PyMC (Day 47) needs numpy < 2.5
```

Every Module 1–4 result is identical under numpy 2.4.6; we checked.

```bash
mkdir -p day41 day42 day43 day44 day45 day46 day47 day48 day49 day50
```

---

# DAY 41 — Difference-in-Differences Basics

| Time | Block |
|---|---|
| 0:00 – 0:40 | 41.1 Two bad comparisons and one good one |
| 0:40 – 1:20 | 41.2 DiD from four averages |
| 1:20 – 1:35 | Break |
| 1:35 – 2:20 | 41.3 DiD as a fixed-effects regression |
| 2:20 – 3:10 | 41.4 Standard errors: the part most people get wrong |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 41.5 Day 41 mini-project: fake campaigns |
| 4:40 – 5:00 | 41.6 Save your work with Git |

---

## 41.1 Two bad comparisons and one good one (40 min)

NorthMart ran a campaign in every Ontario and Quebec store from week 40. Two comparisons come to mind first, and both mislead:

| Comparison | What it mixes in |
|---|---|
| Treated stores vs other stores, **after** week 40 | permanent differences between stores (Ontario stores are bigger) |
| Treated stores, **after vs before** week 40 | everything that changed for everyone (season, the economy) |

**Difference-in-differences (DiD)** subtracts one from the other: the treated stores' change, minus the other stores' change. Permanent store differences cancel in each change; common shocks cancel in the difference. What remains is the effect, **if** the treated stores would have changed like the others without the campaign. That's **parallel trends**, the assumption the whole design rests on. It's about trends, not levels: Ontario stores can be bigger, as long as they'd have *grown* the same.

---

## 41.2 DiD from four averages (40 min)

Create `causalkit/stores.py` from the provided file above, then `day41/two_by_two.py`:

```python
# day41/two_by_two.py   (python -m day41.two_by_two)
"""
NorthMart ran a campaign in every Ontario and Quebec store from week 40.
Three ways to estimate its effect from four averages; only one removes both store and time differences.
"""
import polars as pl

from causalkit.stores import campaign, public, truth

full = campaign()
df = public(full)
print(df.head(3))

means = (df.group_by("treated_group", "post").agg(pl.col("sales").mean().round(2))
         .pivot(on="post", index="treated_group", values="sales").sort("treated_group")
         .select("treated_group", pl.col("0").alias("before (weeks 1-39)"), pl.col("1").alias("after (weeks 40-60)")))
print(means)

m = {(g, p): v for g, p, v in df.group_by("treated_group", "post").agg(pl.col("sales").mean()).iter_rows()}
after_only = m[1, 1] - m[0, 1]
before_after = m[1, 1] - m[1, 0]
did = (m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0])
print(f"\ntreated vs control, after only:   {after_only:+.2f}   (mixes in store differences)")
print(f"treated, after vs before:        {before_after:+.2f}   (mixes in what changed for everyone)")
print(f"difference-in-differences:       {did:+.2f}")
print(f"Truth: {truth(full)['ATT']:+.2f} ($ thousands per store-week)")
```

```bash
python -m day41.two_by_two
```

```
│ treated_group ┆ before (weeks 1-39) ┆ after (weeks 40-60) │
│ 0             ┆ 100.85              ┆ 101.17              │
│ 1             ┆ 103.54              ┆ 106.6               │

treated vs control, after only:   +5.43   (mixes in store differences)
treated, after vs before:        +3.06   (mixes in what changed for everyone)
difference-in-differences:       +2.74
Truth: +3.00 ($ thousands per store-week)
```

Sales are in $ thousands per store per week. The after-only comparison nearly doubles the effect; DiD lands within 0.3 of the truth.

---

## 41.3 DiD as a fixed-effects regression (45 min)

**Concept.** The four-average DiD is a regression: `sales ~ treated_group * post`, where the interaction coefficient is the DiD. The general form, which works for any number of groups and periods, is the **two-way fixed effects (TWFE)** regression:

```
sales_it = store effect_i + week effect_t + β · d_it + error
```

`d_it` is 1 when store i is treated in week t. Store effects absorb permanent differences; week effects absorb common shocks. We use **pyfixest**, a fast fixed-effects package with R's `fixest` syntax: variables after `|` are absorbed as fixed effects.

---

## 41.4 Standard errors: the part most people get wrong (50 min)

Create `day41/did_regression.py`:

```python
# day41/did_regression.py   (python -m day41.did_regression)
"""
The same DiD as a regression: store fixed effects + week fixed effects + the treatment dummy (TWFE).
Same estimate every time; four different standard errors.
"""
import pyfixest as pf

from causalkit.stores import REGIONS, campaign, public, truth

full = campaign()
df = public(full).to_pandas()
df["region_id"] = df.region.map({r: i for i, r in enumerate(REGIONS)}).astype("int64")

simple = pf.feols("sales ~ treated_group * post", df, vcov={"CRV1": "store_id"})
print(f"2x2 regression (group x post): {simple.coef()['treated_group:post']:.2f}")
for name, vcov in [("i.i.d.", "iid"), ("clustered by store", {"CRV1": "store_id"}),
                   ("clustered by region (8)", {"CRV1": "region_id"})]:
    fit = pf.feols("sales ~ d | store_id + week", df, vcov=vcov)
    print(f"TWFE, {name:<24} {fit.coef()['d']:.2f}  SE {fit.se()['d']:.2f}  p = {fit.pvalue()['d']:.4f}")
wild = fit.wildboottest(param="d", reps=9999, seed=1)          # uses the region clusters of the last fit
print(f"TWFE, wild cluster bootstrap by region:     p = {wild['Pr(>|t|)']:.4f}")
print(f"Truth: {truth(full)['ATT']:.2f}")
```

```bash
python -m day41.did_regression
```

```
2x2 regression (group x post): 2.74
TWFE, i.i.d.                   2.74  SE 0.13  p = 0.0000
TWFE, clustered by store       2.74  SE 0.30  p = 0.0000
TWFE, clustered by region (8)  2.74  SE 0.18  p = 0.0000
TWFE, wild cluster bootstrap by region:     p = 0.0156
Truth: 3.00
```

Same estimate, very different uncertainty. Why?

- A store's sales this week are strongly correlated with its sales last week (**serial correlation**). The i.i.d. formula treats 14,400 store-weeks as 14,400 independent observations; really there are 240 stores, each giving one noisy before-after change. Its SE is **less than half** the right size.
- **Clustering by store** lets each store's errors be correlated over time. It's the minimum for any panel.
- The campaign was assigned by **region**. If regions have their own shocks, the right cluster is the region, but there are only 8 regions, and clustered SEs with few clusters are too small (notice: 0.18 < 0.30).
- The **wild cluster bootstrap** (Cameron, Gelbach & Miller, 2008) is the standard fix for few clusters: resample the region-level residuals with random sign flips.

Rule: **cluster at the level at which treatment was assigned**, and use the wild cluster bootstrap when there are fewer than about 40 clusters.

**Try it.** Estimate the DiD using only weeks 39 and 40 (one week before, one week after). Create `day41/one_week.py`. What happens to the estimate and its SE?

<details>
<summary>Solution</summary>

```python
# day41/one_week.py   (python -m day41.one_week)
import pyfixest as pf

from causalkit.stores import campaign, public

df = public(campaign()).to_pandas()
for name, d in [("weeks 39 and 40 only", df[df.week.isin([39, 40])]), ("all 60 weeks", df)]:
    fit = pf.feols("sales ~ d | store_id + week", d, vcov={"CRV1": "store_id"})
    print(f"{name:<22} {fit.coef()['d']:.2f}  (SE {fit.se()['d']:.2f})")
```

```
weeks 39 and 40 only   3.86  (SE 0.43)
all 60 weeks           2.74  (SE 0.30)
```

One week either side gives a noisier estimate (SE 0.43 vs 0.30): each store's change rests on two noisy weeks. More periods average the noise away, but only if parallel trends hold over the longer window. Choosing the window is a trade-off between noise and how far you trust the trend assumption.

</details>

---

## 41.5 Day 41 mini-project: fake campaigns (75 min)

Bertrand, Duflo & Mullainathan (2004) showed that most published DiD papers of the time over-rejected badly. Repeat their experiment: give two random regions a **fake** campaign in data where nothing happened, 200 times, and count how often each method calls it significant. Two worlds: store noise only, and stores that also share regional shocks (`region_sd=0.1`).

Create `day41/clustered_se.py` (it takes about 4 minutes):

```python
# day41/clustered_se.py   (python -m day41.clustered_se)
"""
Mini-project (Bertrand, Duflo & Mullainathan, 2004): give two random regions a FAKE campaign in data where
nothing happened, 200 times. A correct 5% test rejects about 5% of the time. Which standard error does?
Two worlds: store noise only, and stores that also share regional shocks.
"""
import numpy as np
import polars as pl
import pyfixest as pf

from causalkit.stores import REGIONS, campaign, public

methods = ["i.i.d.", "clustered by store", "clustered by region", "wild bootstrap by region"]
rows = []
for world, region_sd in [("store noise only", 0.0), ("plus regional shocks", 0.1)]:
    rng = np.random.default_rng(0)
    reject = dict.fromkeys(methods, 0)
    n = 200
    for s in range(n):
        fake = tuple(rng.choice(REGIONS, 2, replace=False))
        df = public(campaign(seed=7000 + s, effect=0.0, treated_regions=fake, region_sd=region_sd)).to_pandas()
        df["region_id"] = df.region.map({r: i for i, r in enumerate(REGIONS)}).astype("int64")
        for name, vcov in zip(methods[:3], ["iid", {"CRV1": "store_id"}, {"CRV1": "region_id"}]):
            fit = pf.feols("sales ~ d | store_id + week", df, vcov=vcov)
            reject[name] += fit.pvalue()["d"] < 0.05
        reject[methods[3]] += fit.wildboottest(param="d", reps=999, seed=s)["Pr(>|t|)"] < 0.05
    rows.append({"world": world} | {m: f"{r / n:.1%}" for m, r in reject.items()})
print("False positive rate of a 5% test (should be 5%):")
with pl.Config(tbl_width_chars=140, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day41.clustered_se
```

```
False positive rate of a 5% test (should be 5%):
│ world                ┆ i.i.d. ┆ clustered by store ┆ clustered by region ┆ wild bootstrap by region │
│ store noise only     ┆ 38.0%  ┆ 8.0%               ┆ 15.0%               ┆ 2.5%                     │
│ plus regional shocks ┆ 57.0%  ┆ 24.0%              ┆ 19.0%               ┆ 2.5%                     │
```

- i.i.d. standard errors find a "significant" effect of a campaign that never happened 38–57% of the time.
- Store clustering is nearly right when there are no regional shocks (8%), and badly wrong when there are (24%).
- Region clustering with 8 clusters is wrong in both worlds.
- The wild cluster bootstrap is conservative (2.5%) but never over-rejects.

In real data you don't know whether regional shocks exist, so assume they do.

---

## 41.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 41: store simulator, 2x2 DiD, TWFE, clustered SEs, wild bootstrap"
git push
```

### Day 41 checklist

- [ ] I can explain the two bad comparisons DiD fixes, and state parallel trends
- [ ] I can compute a 2×2 DiD from four averages and as a TWFE regression with pyfixest
- [ ] I cluster standard errors at the level of treatment assignment
- [ ] I use the wild cluster bootstrap when there are few clusters
- [ ] I can show by simulation which standard errors give a 5% false positive rate

---

# DAY 42 — Event Studies and Pre-Trends

| Time | Block |
|---|---|
| 0:00 – 0:40 | 42.1 One coefficient per week |
| 0:40 – 1:40 | 42.2 Reading an event study |
| 1:40 – 1:55 | Break |
| 1:55 – 2:50 | 42.3 What a passed pre-trend test proves |
| 2:50 – 3:20 | 42.4 How big a trend would it take? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 42.5 Day 42 mini-project: detrending |
| 4:40 – 5:00 | 42.6 Save your work with Git |

---

## 42.1 One coefficient per week (40 min)

**Concept.** Instead of one DiD coefficient, estimate one for each week **relative to the start**: −39, −38, …, −2, 0, 1, …, 20. Week −1 is the reference (set to zero). This is an **event study**:

- **Leads** (weeks before the start) should be about zero if the groups were on parallel paths before the campaign. They are the main visual evidence for parallel trends.
- **Lags** (weeks after) show how the effect evolves: immediate, growing, or fading.

In pyfixest: `i(rel_week, treated_group, ref=-1)` creates the interaction of each relative week with the treated group.

---

## 42.2 Reading an event study (60 min)

Create `day42/event_study.py`. It also defines `pretrend_test`, a one-number test of whether treated stores were already growing faster before week 40:

```python
# day42/event_study.py   (python -m day42.event_study)
"""
An event study: one DiD coefficient per week relative to the campaign start (week 39 = reference).
Leads (weeks before) test parallel trends; lags (weeks after) show how the effect evolves.
Two datasets: parallel trends hold, and treated stores were already growing faster.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyfixest as pf

from causalkit.stores import campaign, public


def event_study(df):
    """Fit the event study; return relative weeks, the tidy table and the fitted model."""
    df = df.assign(rel_week=df.week - 40)
    es = pf.feols("sales ~ i(rel_week, treated_group, ref=-1) | store_id + week", df, vcov={"CRV1": "store_id"})
    tidy = es.tidy()
    rel = np.array([int(name.split("::")[1].split(":")[0]) for name in tidy.index])
    return rel, tidy, es


def pretrend_test(df):
    """A one-number pre-trend test: in the weeks BEFORE the campaign, did treated stores' sales
    grow faster than the others'? Returns the extra weekly growth and its p-value."""
    pre = df[df.week < 40]
    fit = pf.feols("sales ~ treated_group:week | store_id + week", pre, vcov={"CRV1": "store_id"})
    return fit.coef().iloc[0], fit.pvalue().iloc[0]


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, gap) in zip(axes, [("parallel trends hold", 0.0),
                                       ("treated stores already growing faster", 0.15)]):
        df = public(campaign(trend_gap=gap)).to_pandas()
        rel, tidy, es = event_study(df)
        lead = rel < -1
        slope, p = pretrend_test(df)
        print(f"{title}:")
        print(f"   average lead (should be 0): {tidy.Estimate[lead].mean():+.2f}   "
              f"average lag: {tidy.Estimate[~lead].mean():+.2f}")
        print(f"   pre-trend test: treated stores grew {slope:+.3f} $k/week faster before week 40, p = {p:.4f}")
        ax.errorbar(rel, tidy.Estimate, yerr=1.96 * tidy["Std. Error"], fmt="o", ms=3, color="#2563eb")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(-0.5, color="#dc2626", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("weeks relative to campaign start")
    axes[0].set_ylabel("effect on weekly sales ($k)")
    fig.tight_layout()
    fig.savefig("figures/day42_event_study.png", dpi=120)
    print("Saved figures/day42_event_study.png")
```

```bash
python -m day42.event_study
```

```
parallel trends hold:
   average lead (should be 0): +0.53   average lag: +3.25
   pre-trend test: treated stores grew -0.019 $k/week faster before week 40, p = 0.2263
treated stores already growing faster:
   average lead (should be 0): -2.40   average lag: +4.90
   pre-trend test: treated stores grew +0.131 $k/week faster before week 40, p = 0.0000
Saved figures/day42_event_study.png
```

Open the figure. Left: leads flat around zero, then a jump at week 0. Right: a straight upward line through both periods. There's no visible jump at week 0 at all; the "effect" DiD reports is mostly the trend continuing. **Always plot the event study before reporting a DiD.**

Why are the left panel's leads around +0.5, not 0? Every coefficient is measured relative to one week (week −1), so that week's noise shifts all of them together. Read the *shape*, not the level of individual dots.

---

## 42.3 What a passed pre-trend test proves (55 min)

**Concept.** A pre-trend test that **fails** is strong evidence against parallel trends. A test that **passes** is weak evidence for it: it may simply lack power to detect a small trend, and a small trend can still bias the estimate (Roth, 2022).

Create `day42/pretrend_power.py`:

```python
# day42/pretrend_power.py   (python -m day42.pretrend_power)
"""
Passing a pre-trend test is not proof of parallel trends (Roth, 2022).
Small trend gaps, 40 datasets each: how often does the pre-test catch the gap, and how biased is DiD
in the datasets that PASS it?
"""
import numpy as np
import polars as pl
import pyfixest as pf

from causalkit.stores import campaign, public
from day42.event_study import pretrend_test

rows = []
for gap in [0.0, 0.02, 0.05, 0.10]:
    caught, errors_passed, errors_all = 0, [], []
    for s in range(40):
        full = campaign(seed=8000 + s, trend_gap=gap)
        df = public(full).to_pandas()
        p = pretrend_test(df)[1]
        err = pf.feols("sales ~ d | store_id + week", df).coef()["d"] - 3.0
        errors_all.append(err)
        if p < 0.05:
            caught += 1
        else:
            errors_passed.append(err)
    rows.append({"trend gap ($k/week)": gap, "pre-test rejects": f"{caught / 40:.0%}",
                 "DiD bias, all datasets": round(np.mean(errors_all), 2),
                 "DiD bias, datasets that pass": round(np.mean(errors_passed), 2) if errors_passed else None})
with pl.Config(tbl_width_chars=120):
    print(pl.DataFrame(rows))
```

```bash
python -m day42.pretrend_power
```

```
│ trend gap ($k/week) ┆ pre-test rejects ┆ DiD bias, all datasets ┆ DiD bias, datasets that pass │
│ 0.0                 ┆ 2%               ┆ 0.03                   ┆ 0.02                         │
│ 0.02                ┆ 22%              ┆ 0.63                   ┆ 0.64                         │
│ 0.05                ┆ 88%              ┆ 1.53                   ┆ 1.49                         │
│ 0.1                 ┆ 100%             ┆ 3.03                   ┆ null                         │
```

A trend gap of $0.02k a week is caught only 22% of the time, and when it passes, it biases the estimate by $0.64k: **21% of the true effect**, with a clean-looking event study. "We tested for pre-trends and found none" is a statement about power as much as about trends.

---

## 42.4 How big a trend would it take? (30 min)

**Concept.** A more useful question than "did the test pass?" is: **how large would a trend gap have to be to explain the whole estimate, and can the data rule that out?** With a linear trend gap δ, TWFE's bias is δ × (average post week − average pre week). (This is the idea behind the HonestDiD approach of Rambachan & Roth, 2023.)

**Try it.** Compute that "breakdown" trend for the parallel-trends dataset and compare it with the 95% CI of the pre-period trend gap. Create `day42/breakdown_trend.py`.

<details>
<summary>Solution</summary>

```python
# day42/breakdown_trend.py   (python -m day42.breakdown_trend)
import pyfixest as pf

from causalkit.stores import campaign, public
from day42.event_study import pretrend_test

df = public(campaign()).to_pandas()
est = pf.feols("sales ~ d | store_id + week", df).coef()["d"]
distance = df[df.week >= 40].week.mean() - df[df.week < 40].week.mean()
slope, p = pretrend_test(df)
pre = df[df.week < 40]
se = pf.feols("sales ~ treated_group:week | store_id + week", pre, vcov={"CRV1": "store_id"}).se().iloc[0]
print(f"DiD estimate {est:.2f}; post weeks are on average {distance:.1f} weeks after pre weeks")
print(f"A trend gap of {est / distance:.3f} $k/week would explain the whole estimate")
print(f"Pre-period trend gap: {slope:+.3f}, 95% CI [{slope - 1.96 * se:+.3f}, {slope + 1.96 * se:+.3f}]")
```

```
DiD estimate 2.74; post weeks are on average 30.0 weeks after pre weeks
A trend gap of 0.091 $k/week would explain the whole estimate
Pre-period trend gap: -0.019, 95% CI [-0.049, +0.011]
```

To explain away the whole effect, treated stores would need to grow $0.091k a week faster, eight times the upper end of the pre-period CI (+0.011). Report it as: "the effect survives any trend gap up to 0.09; the pre-period data rule out gaps above 0.01". That is a much stronger statement than "the pre-trend test passed".

</details>

---

## 42.5 Day 42 mini-project: detrending (65 min)

When pre-trends are clearly not parallel, a common fix is to give the treated group its own linear trend. Create `day42/detrend.py`:

```python
# day42/detrend.py   (python -m day42.detrend)
"""
Mini-project: when the pre-trends are clearly not parallel, a common fix is to let each group have its
own linear trend. It works if the gap really is a straight line; it costs precision; and if the EFFECT
itself grew over time, the trend term would absorb part of it.
"""
import polars as pl
import pyfixest as pf

from causalkit.stores import campaign, public, truth

rows = []
for title, kw in [("parallel trends", {}), ("linear trend gap 0.15", {"trend_gap": 0.15})]:
    full = campaign(**kw)
    df = public(full).to_pandas()
    plain = pf.feols("sales ~ d | store_id + week", df, vcov={"CRV1": "store_id"})
    trend = pf.feols("sales ~ d + treated_group:week | store_id + week", df, vcov={"CRV1": "store_id"})
    rows.append({"data": title, "truth": truth(full)["ATT"],
                 "DiD": round(plain.coef()["d"], 2), "DiD + group trend": round(trend.coef()["d"], 2),
                 "SE without / with trend": f"{plain.se()['d']:.2f} / {trend.se()['d']:.2f}"})
with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day42.detrend
```

```
│ data                  ┆ truth ┆ DiD  ┆ DiD + group trend ┆ SE without / with trend │
│ parallel trends       ┆ 3.0   ┆ 2.74 ┆ 3.32              ┆ 0.30 / 0.52             │
│ linear trend gap 0.15 ┆ 3.0   ┆ 7.24 ┆ 3.32              ┆ 0.30 / 0.52             │
```

The group trend removes the bias completely here, because the gap really is a straight line, and costs 70% more standard error. Two warnings to write in `day42/NOTES.md`:

1. A linear trend fitted through the **whole** period also fits the post-period. If the campaign's effect grew week by week, the trend term would absorb part of the effect and the estimate would be too small.
2. Real trend gaps are rarely straight lines. If you need a trend term, say so prominently: the result now rests on a functional-form assumption the reader should judge.

---

## 42.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 42: event studies, pre-trend power, breakdown trend, detrending"
git push
```

### Day 42 checklist

- [ ] I can estimate and plot an event study, and read leads and lags
- [ ] I know that a passed pre-trend test may just lack power, and that small trends still bias DiD
- [ ] I can compute the trend gap that would explain away an estimate, and compare it with the data
- [ ] I can add a group-specific trend, and explain its costs and risks

---

# DAY 43 — Staggered Adoption: What Breaks TWFE

| Time | Block |
|---|---|
| 0:00 – 0:40 | 43.1 Rollouts happen in waves |
| 0:40 – 1:30 | 43.2 Look at the design |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 43.3 The TWFE coefficient goes wrong |
| 2:40 – 3:20 | 43.4 Why: forbidden comparisons |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 43.5 Day 43 mini-project: four 2×2s by hand |
| 4:40 – 5:00 | 43.6 Save your work with Git |

---

## 43.1 Rollouts happen in waves (40 min)

Day 41's campaign started everywhere at once. Most real changes don't: a new store format, a pricing system or an app feature is rolled out to some stores in spring, more in summer, the rest in autumn, and some never. That's **staggered adoption**.

Until about 2018, everyone estimated it with Day 41's TWFE regression. Then several papers (de Chaisemartin & D'Haultfœuille 2020; Goodman-Bacon 2021; Sun & Abraham 2021; Callaway & Sant'Anna 2021; Borusyak, Jaravel & Spiess 2024) showed that TWFE can be badly wrong, even with the wrong sign, whenever the effect differs across waves or changes over time. It nearly always does.

---

## 43.2 Look at the design (50 min)

`rollout()` simulates NorthMart's new store format: a quarter of stores adopt at week 15, a quarter at 30, a quarter at 45, a quarter never. The effect grows by $0.4k a week after adoption, for up to 20 weeks, and it's larger for early adopters (they volunteered because they suited the format).

Create `day43/rollout_overview.py`:

```python
# day43/rollout_overview.py   (python -m day43.rollout_overview)
"""
NorthMart's new store format was rolled out in three waves (weeks 15, 30, 45); a quarter of stores never
got it. Look at the design and at average sales by cohort before estimating anything.
"""
import matplotlib.pyplot as plt
import polars as pl

from causalkit.stores import public, rollout, truth

full = rollout()
df = public(full)
print(df.group_by("cohort").agg(stores=pl.col("store_id").n_unique()).sort("cohort"))
print("True effects:", truth(full))

avg = df.group_by("cohort", "week").agg(pl.col("sales").mean()).sort("week")
fig, ax = plt.subplots(figsize=(9, 4))
for c, colour in zip([0, 15, 30, 45], ["#64748b", "#2563eb", "#16a34a", "#d97706"]):
    s = avg.filter(pl.col("cohort") == c)
    ax.plot(s["week"], s["sales"], color=colour, label="never" if c == 0 else f"adopted week {c}")
    if c:
        ax.axvline(c, color=colour, ls=":", lw=1)
ax.set_xlabel("week")
ax.set_ylabel("average sales ($k)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day43_rollout.png", dpi=120)
print("Saved figures/day43_rollout.png")
```

```bash
python -m day43.rollout_overview
```

```
│ cohort ┆ stores │
│ 0      ┆ 60     │
│ 15     ┆ 60     │
│ 30     ┆ 60     │
│ 45     ┆ 60     │
True effects: {'ATT': 6.91, 'ATT_cohort_15': 9.522, 'ATT_cohort_30': 5.548, 'ATT_cohort_45': 2.04}
Saved figures/day43_rollout.png
```

The **ATT** here is the average effect over all treated store-weeks: $6.91k. Early adopters have been treated longest and suit the format best, so their average effect is largest.

---

## 43.3 The TWFE coefficient goes wrong (55 min)

Create `day43/twfe_bias.py`:

```python
# day43/twfe_bias.py   (python -m day43.twfe_bias)
"""
The standard regression for a staggered rollout, sales ~ d | store + week, run on two versions of the
same rollout: effects that are constant over time, and effects that grow with time since adoption.
"""
import pyfixest as pf

from causalkit.stores import public, rollout, truth

for title, dynamic in [("effects constant over time", False), ("effects grow after adoption", True)]:
    full = rollout(dynamic=dynamic)
    fit = pf.feols("sales ~ d | store_id + week", public(full).to_pandas(), vcov={"CRV1": "store_id"})
    t = truth(full)
    print(f"{title:<32} TWFE {fit.coef()['d']:5.2f} (SE {fit.se()['d']:.2f})   true ATT {t['ATT']:5.2f}   "
          f"by cohort: {t['ATT_cohort_15']:.2f} / {t['ATT_cohort_30']:.2f} / {t['ATT_cohort_45']:.2f}")
```

```bash
python -m day43.twfe_bias
```

```
effects constant over time       TWFE  4.13 (SE 0.23)   true ATT  4.74   by cohort: 6.03 / 4.02 / 2.41
effects grow after adoption      TWFE  3.67 (SE 0.29)   true ATT  6.91   by cohort: 9.52 / 5.55 / 2.04
```

With effects that grow over time, TWFE reports $3.67k against a true $6.91k: **less than 55% of the truth**, with a standard error that makes it look precise. Even with effects constant over time (but different across waves), it's off by 13%.

---

## 43.4 Why: forbidden comparisons (40 min)

**Concept.** Goodman-Bacon (2021) proved that the TWFE coefficient is a weighted average of **every 2×2 DiD** in the data:

| 2×2 comparison | Control group | OK? |
|---|---|---|
| early adopters vs never-treated | never treated | yes |
| late adopters vs never-treated | never treated | yes |
| late adopters vs early adopters, **before** the late ones adopt | not yet treated | yes |
| late adopters vs early adopters, **after** the early ones adopted | **already treated** | **no** |

The last one is the "forbidden comparison". The early adopters' sales are still rising because *their* effect is still growing. Using them as a control subtracts that growth from the late adopters' effect. If the early adopters' effect grows fast enough, the 2×2 can be negative.

**Try it.** Drop the never-treated stores and rerun TWFE on adopters only. Create `day43/no_never_treated.py`. What happens, and why?

<details>
<summary>Solution</summary>

```python
# day43/no_never_treated.py   (python -m day43.no_never_treated)
import polars as pl
import pyfixest as pf

from causalkit.stores import public, rollout, truth

full = rollout()
for name, d in [("all stores", full), ("adopters only", full.filter(pl.col("cohort") > 0))]:
    fit = pf.feols("sales ~ d | store_id + week", public(d).to_pandas(), vcov={"CRV1": "store_id"})
    print(f"{name:<14} TWFE {fit.coef()['d']:.2f}   true ATT {truth(d)['ATT']:.2f}")
```

```
all stores     TWFE 3.67   true ATT 6.91
adopters only  TWFE 1.54   true ATT 6.91
```

Without never-treated stores, the only comparisons left use other adopters as controls, many of them already treated. TWFE falls to $1.54k, less than a quarter of the truth. Rollouts where every unit eventually adopts are common (a company-wide system change), and they are exactly where TWFE does worst.

</details>

---

## 43.5 Day 43 mini-project: four 2×2s by hand (65 min)

Create `causalkit/did.py` with the `_wide` helper and `two_by_two` (the rest of the file is for Day 44):

<details>
<summary>causalkit/did.py (complete, built Days 43–44; click to expand)</summary>

```python
"""Staggered difference-in-differences by hand (Days 43-44)."""
import numpy as np
import polars as pl


def _wide(df: pl.DataFrame, y: str = "sales"):
    w = df.pivot(on="week", index="store_id", values=y).sort("store_id")
    cohorts = df.group_by("store_id").agg(pl.col("cohort").first()).sort("store_id")["cohort"].to_numpy()
    weeks = np.array([int(c) for c in w.columns[1:]])
    return w.drop("store_id").to_numpy(), cohorts, weeks


def two_by_two(df: pl.DataFrame, treated_cohort: int, control_cohort: int, pre: tuple, post: tuple) -> float:
    """A single 2x2 DiD between two cohorts over two windows of weeks (inclusive ranges)."""
    Y, c, weeks = _wide(df)
    def m(cohort, win):
        cols = (weeks >= win[0]) & (weeks <= win[1])
        return Y[c == cohort][:, cols].mean()
    return (m(treated_cohort, post) - m(treated_cohort, pre)) - (m(control_cohort, post) - m(control_cohort, pre))


def att_gt(df: pl.DataFrame, control: str = "never", y: str = "sales") -> pl.DataFrame:
    """Callaway & Sant'Anna (2021) group-time effects, no covariates.
    ATT(g, t) = change in cohort g from week g-1 to week t, minus the same change for the controls:
    never-treated stores (control="never") or stores not yet treated by week t (control="notyet")."""
    Y, c, weeks = _wide(df, y)
    rows = []
    for g in sorted(set(c[c > 0])):
        base = np.flatnonzero(weeks == g - 1)[0]
        for k, t in enumerate(weeks):
            if t == g - 1:
                continue
            ctrl = (c == 0) if control == "never" else ((c == 0) | (c > max(t, g - 1)))
            if ctrl.sum() == 0:
                continue
            diff_g = Y[c == g, k] - Y[c == g, base]
            diff_c = Y[ctrl, k] - Y[ctrl, base]
            rows.append({"cohort": int(g), "week": int(t), "rel": int(t - g),
                         "att": diff_g.mean() - diff_c.mean(), "n_treated": int((c == g).sum())})
    return pl.DataFrame(rows)


def aggregate(gt: pl.DataFrame, how: str = "simple") -> pl.DataFrame | float:
    """simple: average of post-treatment ATT(g,t), weighted by cohort size (= the ATT over treated store-weeks).
    event:  average by weeks since adoption (an event study).   group: average per cohort."""
    post = gt.filter(pl.col("rel") >= 0)
    if how == "simple":
        return float((post["att"] * post["n_treated"]).sum() / post["n_treated"].sum())
    by = "rel" if how == "event" else "cohort"
    src = gt if how == "event" else post
    return (src.group_by(by).agg((pl.col("att") * pl.col("n_treated")).sum() / pl.col("n_treated").sum())
            .rename({"att": "att"}).sort(by))


def bootstrap_se(df: pl.DataFrame, estimator, reps: int = 200, seed: int = 0, **kw) -> float:
    """Store-level (cluster) bootstrap: resample whole stores with replacement."""
    rng = np.random.default_rng(seed)
    ids = df["store_id"].unique().sort().to_list()
    out = []
    for _ in range(reps):
        pick = rng.choice(ids, len(ids))
        sample = pl.concat([df.filter(pl.col("store_id") == s).with_columns(store_id=pl.lit(f"{s}_{i}"))
                            for i, s in enumerate(pick)])
        out.append(estimator(sample, **kw))
    return float(np.std(out, ddof=1))
```

</details>

Create `day43/forbidden_comparisons.py`:

```python
# day43/forbidden_comparisons.py   (python -m day43.forbidden_comparisons)
"""
Mini-project: Goodman-Bacon (2021) showed that TWFE is a weighted average of every possible 2x2 DiD
in the data. Compute four of them by hand. The last two use stores that were ALREADY treated as the
control group: their own growing effect is subtracted from the late adopters' effect.
"""
from causalkit.did import two_by_two
from causalkit.stores import public, rollout

df = public(rollout())
comparisons = [
    ("wave 1 (wk 15) vs never, weeks 1-14 vs 15-60", 15, 0, (1, 14), (15, 60)),
    ("wave 3 (wk 45) vs never, weeks 1-44 vs 45-60", 45, 0, (1, 44), (45, 60)),
    ("wave 2 (wk 30) vs wave 1, weeks 15-29 vs 30-44", 30, 15, (15, 29), (30, 44)),
    ("wave 3 (wk 45) vs wave 1, weeks 30-44 vs 45-60", 45, 15, (30, 44), (45, 60)),
]
for name, g, c, pre, post in comparisons:
    print(f"{name:<50} {two_by_two(df, g, c, pre, post):+6.2f}")
print("\nTrue average effects: wave 1 9.52 (weeks 15-60); wave 2 3.20 (weeks 30-44); wave 3 2.04 (weeks 45-60)")
```

```bash
python -m day43.forbidden_comparisons
```

```
wave 1 (wk 15) vs never, weeks 1-14 vs 15-60       +10.04
wave 3 (wk 45) vs never, weeks 1-44 vs 45-60        +1.41
wave 2 (wk 30) vs wave 1, weeks 15-29 vs 30-44      -3.28
wave 3 (wk 45) vs wave 1, weeks 30-44 vs 45-60      +1.19

True average effects: wave 1 9.52 (weeks 15-60); wave 2 3.20 (weeks 30-44); wave 3 2.04 (weeks 45-60)
```

The third line is the headline: wave 2's true effect over weeks 30–44 is **+3.20**, but compared with wave 1 (whose effect was still growing by $0.6k a week), it comes out **−3.28**. The sign flips. The fourth is less wrong because wave 1's effect had stopped growing (capped at 20 weeks) by week 35.

The clean comparisons (lines 1–2) are close to the truth. Tomorrow's estimators use **only** clean comparisons.

---

## 43.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 43: staggered rollout, TWFE bias, forbidden comparisons"
git push
```

### Day 43 checklist

- [ ] I can describe a staggered rollout and its cohorts
- [ ] I can show that TWFE is biased when effects differ across cohorts or grow over time
- [ ] I can explain the Goodman-Bacon decomposition and what makes a comparison "forbidden"
- [ ] I know TWFE is worst when every unit eventually adopts

---

# DAY 44 — Modern DiD Estimators

| Time | Block |
|---|---|
| 0:00 – 0:30 | 44.1 The fix: only clean comparisons |
| 0:30 – 1:30 | 44.2 Callaway & Sant'Anna by hand |
| 1:30 – 1:45 | Break |
| 1:45 – 2:35 | 44.3 The imputation estimator |
| 2:35 – 3:20 | 44.4 Never-treated or not-yet-treated controls? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 44.5 Day 44 mini-project: event studies, done right |
| 4:40 – 5:00 | 44.6 Save your work with Git |

---

## 44.1 The fix: only clean comparisons (30 min)

All the modern estimators share one idea: **never use an already-treated unit as a control**. They differ in how they build the comparison:

| Estimator | Idea | In this course |
|---|---|---|
| Callaway & Sant'Anna (2021) | a separate 2×2 for every (cohort, week), then average | by hand, `causalkit/did.py` |
| Borusyak, Jaravel & Spiess (2024); Gardner (2022) | fit store and week effects on untreated observations only; impute each treated store-week's counterfactual | by hand, then `pf.did2s` |
| Sun & Abraham (2021) | an event study with separate coefficients for every cohort × event time | `pf.event_study(estimator="saturated")` |

---

## 44.2 Callaway & Sant'Anna by hand (60 min)

**Concept.** For cohort g (stores that adopted in week g) and any week t, compare their change from week g−1 (the last week before adoption) to week t with the same change for never-treated stores:

```
ATT(g, t) = [mean sales of cohort g in week t − in week g−1] − [the same for never-treated stores]
```

Each ATT(g,t) is a clean 2×2. Then **aggregate** them for the question at hand: one overall ATT, one per cohort, or one per week since adoption.

Type `att_gt`, `aggregate` and `bootstrap_se` into `causalkit/did.py`. Create `day44/callaway_santanna.py`:

```python
# day44/callaway_santanna.py   (python -m day44.callaway_santanna)
"""
Callaway & Sant'Anna (2021): never use an already-treated store as a control.
Estimate one clean 2x2 for every (cohort, week), then average them in whichever way the question needs.
"""
import polars as pl

from causalkit.did import aggregate, att_gt, bootstrap_se
from causalkit.stores import public, rollout, truth

full = rollout()
df = public(full)
t = truth(full)

gt = att_gt(df, control="never")
print(f"{gt.height} group-time effects, e.g.:")
print(gt.filter(pl.col("rel").is_in([-5, 0, 5, 10])).sort("cohort", "rel").with_columns(pl.col("att").round(2)))

simple = aggregate(gt, "simple")
se = bootstrap_se(df, lambda d: aggregate(att_gt(d), "simple"), reps=100)
print(f"\nOverall ATT: {simple:.2f} (bootstrap SE {se:.2f})   truth {t['ATT']:.2f}")
print("By cohort:")
by_cohort = aggregate(gt, "group").with_columns(
    pl.col("att").round(2),
    truth=pl.col("cohort").map_elements(lambda c: t[f"ATT_cohort_{c}"], return_dtype=pl.Float64).round(2))
print(by_cohort)
```

```bash
python -m day44.callaway_santanna
```

```
177 group-time effects, e.g.:
│ cohort ┆ week ┆ rel ┆ att   ┆ n_treated │
│ 15     ┆ 10   ┆ -5  ┆ 0.33  ┆ 60        │
│ 15     ┆ 15   ┆ 0   ┆ 0.72  ┆ 60        │
│ 15     ┆ 20   ┆ 5   ┆ 4.28  ┆ 60        │
│ 15     ┆ 25   ┆ 10  ┆ 6.06  ┆ 60        │
...
Overall ATT: 6.97 (bootstrap SE 0.42)   truth 6.91
By cohort:
│ cohort ┆ att  ┆ truth │
│ 15     ┆ 9.63 ┆ 9.52  │
│ 30     ┆ 5.59 ┆ 5.55  │
│ 45     ┆ 2.01 ┆ 2.04  │
```

$6.97k against a true $6.91k, where TWFE said $3.67k. Every cohort is right too. The negative `rel` rows are pre-period placebo comparisons: a built-in pre-trend check.

The bootstrap resamples **whole stores** (Day 41: cluster at the level of treatment assignment). It takes a minute.

---

## 44.3 The imputation estimator (50 min)

**Concept.** Fit the TWFE model **without** the treatment, using only untreated store-weeks (never-treated stores, and adopters before they adopted). That model has never seen an effect, so it predicts each treated store-week's sales *without* the new format. The effect is actual minus predicted, averaged.

Create `day44/imputation.py`:

```python
# day44/imputation.py   (python -m day44.imputation)
"""
The imputation idea (Borusyak, Jaravel & Spiess 2024; Gardner's two-stage DiD is the same estimate):
1. fit store and week effects on UNTREATED store-weeks only (never-treated, and adopters before adoption);
2. predict every treated store-week's sales without the new format;
3. the effect is actual minus predicted, averaged over treated store-weeks.
By hand, then with pyfixest's did2s (which also gives a correct standard error).
"""
import pyfixest as pf

from causalkit.stores import public, rollout, truth

full = rollout()
df = public(full).to_pandas()

untreated = df[df.d == 0]
stage1 = pf.feols("sales ~ 1 | store_id + week", untreated)
treated = df[df.d == 1].copy()
treated["predicted_y0"] = stage1.predict(newdata=treated)
by_hand = (treated.sales - treated.predicted_y0).mean()

gardner = pf.did2s(df.assign(d=df.d.astype(float)), yname="sales", first_stage="~ 0 | store_id + week",
                   second_stage="~ 0 + d", treatment="d", cluster="store_id")
print(f"Imputation by hand:   {by_hand:.2f}")
print(f"pyfixest did2s:       {gardner.coef()['d']:.2f}  (SE {gardner.se()['d']:.2f})")
print(f"Truth:                {truth(full)['ATT']:.2f}")
```

```bash
python -m day44.imputation
```

```
Imputation by hand:   7.02
pyfixest did2s:       7.02  (SE 0.33)
Truth:                6.91
```

Same number by hand and from the library; the library adds a standard error that accounts for the first stage having been estimated. Imputation uses more of the pre-period than Callaway & Sant'Anna (which relies on week g−1 alone as the baseline), so it is usually more precise: SE 0.33 vs 0.42 here.

---

## 44.4 Never-treated or not-yet-treated controls? (45 min)

Callaway & Sant'Anna can use either never-treated stores, or every store not yet treated by week t. Not-yet-treated gives more controls (more precision), but assumes that stores' timing of adoption is unrelated to their trends, and that nobody changed behaviour in anticipation.

**Try it.** Compare `att_gt(df, control="never")` and `control="notyet"`. Create `day44/not_yet_treated.py`.

<details>
<summary>Solution</summary>

```python
# day44/not_yet_treated.py   (python -m day44.not_yet_treated)
from causalkit.did import aggregate, att_gt
from causalkit.stores import public, rollout, truth

full = rollout()
df = public(full)
for control in ["never", "notyet"]:
    gt = att_gt(df, control=control)
    print(f"controls = {control:<7} ATT {aggregate(gt):.2f}   by cohort "
          f"{[round(x, 2) for x in aggregate(gt, 'group')['att']]}")
print(f"truth ATT {truth(full)['ATT']:.2f}")
```

```
controls = never   ATT 6.97   by cohort [9.63, 5.59, 2.01]
controls = notyet  ATT 6.77   by cohort [9.27, 5.52, 2.01]
truth ATT 6.91
```

Both are close; cohort 45 is identical, because by week 45 no other store is still waiting to adopt. With a small never-treated group, not-yet-treated controls are often the only practical choice. When every store eventually adopts, they are the *only* choice (the last cohort has no clean comparisons after it adopts).

</details>

---

## 44.5 Day 44 mini-project: event studies, done right (65 min)

How does the effect evolve with weeks since adoption? Compare three event studies with the truth. Create `day44/event_study_modern.py`:

```python
# day44/event_study_modern.py   (python -m day44.event_study_modern)
"""
Mini-project: how does the effect evolve with weeks since adoption? Three event studies against the truth:
  TWFE with relative-week dummies (the common, contaminated version),
  Callaway & Sant'Anna aggregated by event time,
  pyfixest's saturated (Sun & Abraham-style) event study, which interacts event time with cohort.
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf

from causalkit.did import aggregate, att_gt
from causalkit.stores import public, rollout

full = rollout()
df = public(full)
pdf = df.to_pandas()

true_es = (full.filter(pl.col("cohort") > 0).with_columns(rel=pl.col("week") - pl.col("cohort"))
           .group_by("rel").agg(truth=pl.col("_tau").mean()))

pdf["rel"] = np.where(pdf.cohort > 0, pdf.week - pdf.cohort, -1)       # never-treated sit at the reference
twfe = pf.feols("sales ~ i(rel, ref=-1) | store_id + week", pdf, vcov={"CRV1": "store_id"}).tidy()
twfe_es = pl.DataFrame({"rel": [int(float(n.split("::")[1])) for n in twfe.index],
                        "TWFE dummies": twfe.Estimate.to_numpy()})

cs_es = aggregate(att_gt(df), "event").rename({"att": "Callaway-Sant'Anna"})

sat = pf.event_study(pdf, "sales", "store_id", "week", "cohort", estimator="saturated", cluster="store_id")
agg = sat.aggregate()
sat_es = pl.DataFrame({"rel": agg.index.astype(float).astype(int).to_numpy(), "saturated": agg.Estimate.astype(float).to_numpy()})

es = (true_es.join(twfe_es, on="rel", how="left").join(cs_es, on="rel", how="left")
      .join(sat_es, on="rel", how="left").sort("rel"))
show = es.filter(pl.col("rel").is_in([-10, -5, 0, 5, 10, 20, 30, 40])).with_columns(pl.exclude("rel").round(2))
with pl.Config(tbl_width_chars=120):
    print(show)

fig, ax = plt.subplots(figsize=(9, 4))
for col, style in [("truth", "k-"), ("TWFE dummies", "o"), ("Callaway-Sant'Anna", "s"), ("saturated", "^")]:
    s = es.filter(pl.col("rel").is_between(-14, 44))
    ax.plot(s["rel"], s[col], style, ms=3, label=col)
ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
ax.set_xlabel("weeks since adoption")
ax.set_ylabel("effect on weekly sales ($k)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day44_event_studies.png", dpi=120)
print("Saved figures/day44_event_studies.png")
```

```bash
python -m day44.event_study_modern
```

```
│ rel ┆ truth ┆ TWFE dummies ┆ Callaway-Sant'Anna ┆ saturated │
│ -10 ┆ 0.0   ┆ -0.16        ┆ -0.41              ┆ -0.41     │
│ -5  ┆ 0.0   ┆ 0.48         ┆ 0.42               ┆ 0.42      │
│ 0   ┆ 0.41  ┆ 0.72         ┆ 0.71               ┆ 0.71      │
│ 5   ┆ 2.48  ┆ 3.06         ┆ 2.96               ┆ 2.96      │
│ 10  ┆ 4.55  ┆ 4.55         ┆ 4.2                ┆ 4.2       │
│ 20  ┆ 10.0  ┆ 10.64        ┆ 10.69              ┆ 10.69     │
│ 30  ┆ 10.0  ┆ 10.49        ┆ 9.96               ┆ 9.96      │
│ 40  ┆ 12.0  ┆ 12.07        ┆ 11.67              ┆ 11.67     │
Saved figures/day44_event_studies.png
```

Three observations:

1. Callaway & Sant'Anna and the saturated estimator give **identical** event studies here: with never-treated controls and no covariates, they are the same estimator written two ways.
2. The TWFE event study with a full set of relative-week dummies is also close in this simulation. Most of TWFE's damage came from collapsing everything into **one** coefficient (Day 43). But Sun & Abraham proved that the dummy version can still be contaminated when cohorts' effect paths differ, and nothing in the output tells you whether that has happened. Use the clean estimators by default.
3. The effect builds up over 20 weeks. A study that stopped 5 weeks after adoption would report a third of the eventual effect.

Write in `day44/NOTES.md`: for a rollout you know at work (or a plausible one), which estimator would you use, which controls, and what's the estimand?

---

## 44.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 44: Callaway-Sant'Anna, imputation (did2s), modern event studies"
git push
```

**Coming next (Days 45–47).** DiD needs many treated units, or at least a control group on a parallel path. What if only **one** region ran the campaign, and it's unlike the average region? Synthetic control builds a better comparison from a weighted mix of regions.

### Day 44 checklist

- [ ] I can estimate group-time ATTs by hand and aggregate them overall, by cohort and by event time
- [ ] I can explain and run the imputation estimator (did2s)
- [ ] I can choose between never-treated and not-yet-treated controls
- [ ] I use clean event studies, and know why the TWFE-dummy version is risky

