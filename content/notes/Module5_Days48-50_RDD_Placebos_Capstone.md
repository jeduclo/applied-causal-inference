# Module 5 · Quasi-Experiments: DiD, Synthetic Control, RDD
## Days 48–50: Regression Discontinuity, Placebo Tests and the Capstone

**Goal by end of Day 50:** Students can estimate sharp and fuzzy regression discontinuities with local linear regression and rdrobust, run the falsification checks every quasi-experiment needs (placebo dates, placebo units, placebo cutoffs, covariate continuity, density tests), and evaluate a real campaign with several designs, reaching a decision with money and uncertainty attached.

---

# DAY 48 — Regression Discontinuity

| Time | Block |
|---|---|
| 0:00 – 0:40 | 48.1 A rule with a cutoff |
| 0:40 – 1:20 | 48.2 The RD plot |
| 1:20 – 1:35 | Break |
| 1:35 – 2:40 | 48.3 Estimating the jump |
| 2:40 – 3:20 | 48.4 Bandwidths and polynomials |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 48.5 Day 48 mini-project: a fuzzy discontinuity |
| 4:40 – 5:00 | 48.6 Save your work with Git |

---

## 48.1 A rule with a cutoff (40 min)

NorthMart customers who spent $1,000 or more last year get **Gold** status: free shipping and priority support. Does Gold raise next-year spending?

Comparing Gold and non-Gold customers is hopeless: Gold customers were already the big spenders. But customers who spent $990 and $1,010 are almost identical; only one has Gold. If nothing else changes at exactly $1,000, any jump in next-year spending at the cutoff is the effect of Gold. That's a **regression discontinuity (RD)** design (Thistlethwaite & Campbell, 1960). The variable that decides treatment (last year's spend) is the **running variable**.

| | Sharp RD | Fuzzy RD |
|---|---|---|
| Rule | everyone above the cutoff is treated; nobody below | crossing the cutoff makes treatment more likely |
| Estimate | the jump in the outcome | jump in outcome ÷ jump in treatment (an IV, Day 37) |
| Estimand | effect **at the cutoff** | effect for compliers **at the cutoff** |

The estimand is narrow: the effect for customers near $1,000, not for a customer who spends $5,000. Keep that in every sentence you write about an RD.

Create `causalkit/tiers.py` (provided; read only the docstrings):

<details>
<summary>causalkit/tiers.py (provided; click to expand)</summary>

```python
"""
NorthMart Gold tier (simulated). Provided for Module 5: read the docstrings, not the code.

Customers whose spending last year reached $1,000 are upgraded to Gold (free shipping, priority support).

    from causalkit.tiers import customers, truth

    df = customers()                    # 30,000 customers; sharp rule: Gold if pre_spend >= 1,000
    df = customers(fuzzy=True)          # managers upgrade some keen customers below the line; some above never activate
    df = customers(manipulation=0.3)    # customers just below $1,000 who know the rule buy their way over it
    truth(df)                           # true effect of Gold at the cutoff (for checking only)

Columns an analyst has:
    customer_id, pre_spend (last year, $; the running variable), age, tenure_years, engaged,
    gold (1 = has Gold status), spend (next year, $; the outcome)
Columns starting with '_' are the simulator's hidden truth.
"""
import numpy as np
import polars as pl

CUTOFF = 1000.0


def customers(n: int = 30_000, seed: int = 54, effect: float = 120.0, fuzzy: bool = False,
              manipulation: float = 0.0) -> pl.DataFrame:
    """effect: true effect of Gold on next-year spend for customers AT the cutoff ($); it grows with pre_spend.
    manipulation: share of customers with organic spend in [$850, $1,000) who top up to just over $1,000.
    They are keen customers: their next-year spend is higher even without Gold."""
    rng = np.random.default_rng(seed)
    engaged = rng.binomial(1, 0.5, n)
    tenure = np.round(rng.gamma(2.0, 2.0, n), 1)
    age = np.clip(rng.normal(45, 13, n), 18, 85).round().astype(int)
    intent = rng.normal(0, 1, n)
    pre = np.exp(rng.normal(6.55 + 0.25 * engaged + 0.03 * tenure + 0.15 * intent, 0.55, n))
    keen = np.zeros(n, bool)
    if manipulation > 0:
        near = (pre >= 850) & (pre < CUTOFF)
        keen = near & (rng.random(n) < manipulation)
        pre = np.where(keen, CUTOFF + rng.uniform(0, 40, n), pre)
    pre = np.round(pre, 2)
    x = (pre - CUTOFF) / 1000
    y0 = (900 + 650 * x - 180 * x**2 + 60 * engaged + 8 * tenure + 60 * intent + 150 * keen
          + rng.normal(0, 220, n))
    tau = effect + 60 * x
    above = pre >= CUTOFF
    if fuzzy:
        keen_share = 1 / (1 + np.exp(-1.5 * intent))           # managers upgrade (and keen customers activate)
        p_below = (0.08 + 0.25 * np.exp(-np.abs(x) * 8)) * 2 * keen_share
        gold = np.where(above, rng.random(n) < 0.6 + 0.4 * keen_share, rng.random(n) < p_below)
    else:
        gold = above
    return pl.DataFrame({
        "customer_id": [f"G{i:05d}" for i in range(1, n + 1)],
        "pre_spend": pre, "age": age, "tenure_years": tenure, "engaged": engaged,
        "gold": gold.astype(np.int64), "spend": np.round(y0 + tau * gold, 2),
        "_y0": y0, "_tau": tau, "_keen": keen,
    })


def truth(df: pl.DataFrame) -> dict:
    """The effect at the cutoff, and the average effect for everyone with Gold."""
    at_cutoff = (df["_tau"] - 60 * (df["pre_spend"] - CUTOFF) / 1000).mean()
    return {"effect_at_cutoff": round(float(at_cutoff), 1),
            "ATT": round(df.filter(pl.col("gold") == 1)["_tau"].mean(), 1)}


def public(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

---

## 48.2 The RD plot (40 min)

Create `day48/rd_plot.py`:

```python
# day48/rd_plot.py   (python -m day48.rd_plot)
"""
Gold status starts at $1,000 of spending. Customers at $990 and $1,010 are nearly identical, except one
has Gold. Plot next-year spend against last year's spend in $25 bins: a jump at $1,000 is the effect.
"""
import matplotlib.pyplot as plt
import polars as pl

from causalkit.tiers import CUTOFF, customers, public

df = public(customers())
print(df.head(3))
print(f"{df.height:,} customers; {df['gold'].mean():.1%} have Gold")

bins = (df.filter(pl.col("pre_spend").is_between(400, 1600))
        .with_columns(bin=(pl.col("pre_spend") // 25) * 25 + 12.5)
        .group_by("bin").agg(pl.col("spend").mean(), pl.col("gold").mean(), n=pl.len()).sort("bin"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, label in [(axes[0], "spend", "next-year spend ($)"), (axes[1], "gold", "share with Gold")]:
    left, right = bins.filter(pl.col("bin") < CUTOFF), bins.filter(pl.col("bin") >= CUTOFF)
    ax.scatter(left["bin"], left[col], s=12, color="#64748b")
    ax.scatter(right["bin"], right[col], s=12, color="#d97706")
    ax.axvline(CUTOFF, color="#dc2626", ls="--", lw=0.8)
    ax.set_xlabel("last year's spend ($)")
    ax.set_ylabel(label)
fig.tight_layout()
fig.savefig("figures/day48_rd_plot.png", dpi=120)
print("Saved figures/day48_rd_plot.png")
```

```bash
python -m day48.rd_plot
```

```
│ customer_id ┆ pre_spend ┆ age ┆ tenure_years ┆ engaged ┆ gold ┆ spend   │
│ G00001      ┆ 1304.93   ┆ 46  ┆ 11.6         ┆ 0       ┆ 1    ┆ 1492.84 │
│ G00002      ┆ 782.06    ┆ 61  ┆ 14.1         ┆ 1       ┆ 0    ┆ 826.5   │
│ G00003      ┆ 391.97    ┆ 30  ┆ 10.5         ┆ 0       ┆ 0    ┆ 574.76  │
30,000 customers; 42.1% have Gold
Saved figures/day48_rd_plot.png
```

Open the figure. Left: next-year spend rises smoothly with last year's spend, then jumps by roughly $120 at $1,000. Right: the share with Gold goes from 0 to 1 at the cutoff, so this is a sharp RD. **Always show the RD plot**: if you can't see a jump, a regression shouldn't convince anyone there is one.

---

## 48.3 Estimating the jump (65 min)

**Concept.** Fit a line on each side of the cutoff, using only customers close to it, and measure the gap between the two lines at $1,000. Two choices:

- **Bandwidth h:** how close is "close"? Narrow = less bias (lines fit better near the cutoff), more noise.
- **Kernel:** a **triangular** kernel weights customers more the closer they are to the cutoff.

**rdrobust** (Calonico, Cattaneo & Titiunik, 2014) picks the bandwidth that minimises mean squared error, and reports a **robust bias-corrected** confidence interval: the MSE-optimal bandwidth leaves some bias, and the robust CI accounts for it. It's the standard RD tool.

Create `day48/local_linear.py`:

```python
# day48/local_linear.py   (python -m day48.local_linear)
"""
Estimating the jump. Three ways:
  a global polynomial (common, fragile),
  a local linear regression by hand: only customers within $h of the cutoff, nearer ones weighted more,
  rdrobust (Calonico, Cattaneo & Titiunik): data-driven bandwidth and a bias-corrected robust CI.
"""
import numpy as np
import statsmodels.formula.api as smf
from rdrobust import rdrobust

from causalkit.tiers import CUTOFF, customers, public, truth

full = customers()
df = public(full).to_pandas()
df["x"] = (df.pre_spend - CUTOFF) / 1000              # distance from the cutoff, $ thousands
df["above"] = (df.x >= 0).astype(int)


def local_linear(d, h):
    """Triangular-kernel local linear regression with separate slopes on each side."""
    near = d[d.x.abs() <= h / 1000].copy()
    near["w"] = 1 - near.x.abs() / (h / 1000)
    fit = smf.wls("spend ~ above * x", near, weights=near.w).fit(cov_type="HC1")
    return fit.params["above"], fit.bse["above"], len(near)


quartic = smf.ols("spend ~ above * (x + I(x**2) + I(x**3) + I(x**4))", df).fit(cov_type="HC1")
print(f"global quartic polynomial, all customers:  {quartic.params['above']:6.1f}  (SE {quartic.bse['above']:.1f})")
est, se, n = local_linear(df, 200)
print(f"local linear, h = $200 ({n:,} customers):   {est:6.1f}  (SE {se:.1f})")

rd = rdrobust(df.spend, df.pre_spend, c=CUTOFF)
h = rd.bws.iloc[0, 0]
print(f"rdrobust: bandwidth ${h:.0f}, {sum(rd.N_h):,} customers")
print(f"   conventional estimate {rd.coef.iloc[0, 0]:6.1f}  95% CI [{rd.ci.iloc[0, 0]:.1f}, {rd.ci.iloc[0, 1]:.1f}]")
print(f"   robust bias-corrected  {rd.coef.iloc[1, 0]:6.1f}  95% CI [{rd.ci.iloc[2, 0]:.1f}, {rd.ci.iloc[2, 1]:.1f}]")
t = truth(full)
print(f"Truth: effect at the cutoff {t['effect_at_cutoff']:.1f}; average for all Gold customers {t['ATT']:.1f}")
```

```bash
python -m day48.local_linear
```

```
global quartic polynomial, all customers:   112.0  (SE 9.6)
local linear, h = $200 (8,050 customers):    124.7  (SE 11.3)
rdrobust: bandwidth $302, 12,065 customers
   conventional estimate  123.9  95% CI [106.0, 141.9]
   robust bias-corrected   123.4  95% CI [101.9, 145.0]
Truth: effect at the cutoff 120.0; average for all Gold customers 158.9
```

Local linear and rdrobust are within $5 of the truth, and the robust CI contains it. Note the last line: Gold is worth $159 on average for **all** Gold customers (bigger spenders gain more), but the RD estimates the $120 effect **at the cutoff**. An RD can't tell you the effect of Gold for a $3,000 customer.

---

## 48.4 Bandwidths and polynomials (40 min)

**Try it.** Estimate the jump by local linear regression with h = $50, $100, $200, $400, $800, and with global polynomials of order 1, 2, 4 and 6 fitted to all customers. Create `day48/bandwidths.py`.

<details>
<summary>Solution</summary>

```python
# day48/bandwidths.py   (python -m day48.bandwidths)
import statsmodels.formula.api as smf

from causalkit.tiers import CUTOFF, customers, public

df = public(customers()).to_pandas()
df["x"] = (df.pre_spend - CUTOFF) / 1000
df["above"] = (df.x >= 0).astype(int)
for h in [50, 100, 200, 400, 800]:
    near = df[df.x.abs() <= h / 1000].copy()
    near["w"] = 1 - near.x.abs() / (h / 1000)
    fit = smf.wls("spend ~ above * x", near, weights=near.w).fit(cov_type="HC1")
    print(f"local linear, h = ${h:<4} {fit.params['above']:6.1f}  (SE {fit.bse['above']:5.1f}, n = {len(near):,})")
for k in [1, 2, 4, 6]:
    terms = " + ".join(f"I(x**{p})" for p in range(1, k + 1))
    fit = smf.ols(f"spend ~ above * ({terms})", df).fit(cov_type="HC1")
    print(f"global polynomial, order {k}:  {fit.params['above']:6.1f}  (SE {fit.bse['above']:5.1f})")
```

```
local linear, h = $50    129.6  (SE  22.4, n = 2,001)
local linear, h = $100   131.4  (SE  15.9, n = 4,051)
local linear, h = $200   124.7  (SE  11.3, n = 8,050)
local linear, h = $400   124.9  (SE   8.1, n = 15,756)
local linear, h = $800   124.6  (SE   6.0, n = 26,255)
global polynomial, order 1:   304.3  (SE  13.2)
global polynomial, order 2:   133.1  (SE   6.2)
global polynomial, order 4:   112.0  (SE   9.6)
global polynomial, order 6:   121.9  (SE  13.2)
```

Local linear estimates are stable across bandwidths; only the SE changes. Global polynomials swing from $304 to $112 depending on the order: a straight line across all customers can't follow the curve, so it puts the curvature into the jump. Gelman & Imbens (2019) recommend against global polynomials of any order in RD; use local linear (or local quadratic) with a data-driven bandwidth, and show that the answer is stable across bandwidths.

</details>

---

## 48.5 Day 48 mini-project: a fuzzy discontinuity (65 min)

In reality, store managers upgrade some keen customers below $1,000, and some customers above it never activate Gold. Crossing $1,000 still makes Gold much more likely, so it's an instrument (Day 37), local to the cutoff. Create `day48/fuzzy_rd.py`:

```python
# day48/fuzzy_rd.py   (python -m day48.fuzzy_rd)
"""
Mini-project: a FUZZY discontinuity. Store managers upgrade some keen customers below $1,000, and some
customers above it never activate Gold. Crossing $1,000 still makes Gold much more likely: an instrument,
exactly like Day 37's invitation. Effect = jump in spend / jump in Gold share (a LATE at the cutoff).
"""
import numpy as np
import statsmodels.formula.api as smf
from rdrobust import rdrobust

from causalkit.tiers import CUTOFF, customers, public, truth


def near_cutoff(df, h=200):
    df = df.assign(x=(df.pre_spend - CUTOFF) / 1000, above=(df.pre_spend >= CUTOFF).astype(int))
    near = df[df.x.abs() <= h / 1000].copy()
    near["w"] = 1 - near.x.abs() / (h / 1000)
    return near


def wald_and_naive(df):
    near = near_cutoff(df)
    first = smf.wls("gold ~ above * x", near, weights=near.w).fit().params["above"]
    reduced = smf.wls("spend ~ above * x", near, weights=near.w).fit().params["above"]
    naive = smf.wls("spend ~ gold + above:x + x", near, weights=near.w).fit().params["gold"]
    return first, reduced, naive


full = customers(fuzzy=True)
df = public(full).to_pandas()
first, reduced, naive = wald_and_naive(df)
print(f"Gold share jumps by {first:.2f} at $1,000; spend jumps by {reduced:.1f}")
print(f"Wald ratio (h = $200):                        {reduced / first:6.1f}")
print(f"Regress spend on gold near the cutoff:        {naive:6.1f}   <- who gets upgraded is not random")
rd = rdrobust(df.spend, df.pre_spend, c=CUTOFF, fuzzy=df.gold)
print(f"rdrobust fuzzy: {rd.coef.iloc[0, 0]:.1f}, robust 95% CI [{rd.ci.iloc[2, 0]:.1f}, {rd.ci.iloc[2, 1]:.1f}]")
print(f"Truth at the cutoff: {truth(full)['effect_at_cutoff']:.1f}")

waldS, naiveS = [], []
for s in range(20):
    f, r, n = wald_and_naive(public(customers(seed=900 + s, fuzzy=True)).to_pandas())
    waldS.append(r / f)
    naiveS.append(n)
print(f"\nOver 20 datasets: Wald ratio {np.mean(waldS):.1f} (SD {np.std(waldS):.1f}); "
      f"naive regression {np.mean(naiveS):.1f} (SD {np.std(naiveS):.1f})")
```

```bash
python -m day48.fuzzy_rd
```

```
Gold share jumps by 0.51 at $1,000; spend jumps by 66.7
Wald ratio (h = $200):                         129.9
Regress spend on gold near the cutoff:         153.3   <- who gets upgraded is not random
rdrobust fuzzy: 131.8, robust 95% CI [75.5, 190.1]
Truth at the cutoff: 120.0

Over 20 datasets: Wald ratio 124.3 (SD 17.4); naive regression 154.6 (SD 7.2)
```

- The fuzzy RD is the Wald estimator from Day 37: reduced-form jump ÷ first-stage jump. It's unbiased on average ($124 over 20 datasets), but noisy: dividing by a jump of 0.5 doubles the noise.
- Regressing spend on Gold near the cutoff is biased by +$35 on average, consistently. Managers upgrade **keen** customers, who would spend more anyway: selection bias, even within $200 of the cutoff.
- The CI is three times wider than in the sharp RD. Fuzziness costs precision, like a weak first stage.

---

## 48.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 48: Gold tier simulator, RD plot, local linear, rdrobust, fuzzy RD"
git push
```

### Day 48 checklist

- [ ] I can explain the RD design, the running variable, and what "effect at the cutoff" means
- [ ] I always show an RD plot with binned means
- [ ] I can estimate the jump by local linear regression and with rdrobust's robust bias-corrected CI
- [ ] I avoid global polynomials, and show stability across bandwidths
- [ ] I can estimate a fuzzy RD as a Wald ratio, and explain why it's noisier

---

# DAY 49 — Placebo and Falsification Tests

| Time | Block |
|---|---|
| 0:00 – 0:30 | 49.1 Try to break your own design |
| 0:30 – 1:15 | 49.2 Placebo dates for DiD |
| 1:15 – 1:30 | Break |
| 1:30 – 2:40 | 49.3 Placebo regions for synthetic control |
| 2:40 – 3:20 | 49.4 Manipulation and the donut |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 49.5 Day 49 mini-project: the RD falsification battery |
| 4:40 – 5:00 | 49.6 Save your work with Git |

---

## 49.1 Try to break your own design (30 min)

Every quasi-experiment rests on an assumption the data can't prove. What the data **can** do is fail tests that the assumption implies. A **placebo test** applies the design where the true effect is known to be zero:

| Placebo | The design applied to... | Expected result |
|---|---|---|
| in time | a fake start date before the real one | no effect |
| in space | units that weren't treated | no effect, or smaller than the real one |
| fake cutoff | a threshold where no rule exists | no jump |
| placebo outcome | an outcome the treatment can't affect (e.g. last year's spend) | no effect |

A design that "finds" effects where there are none can't be trusted where there might be one. Run placebos **before** you look at the real estimate, and report them all, not just the ones that pass.

---

## 49.2 Placebo dates for DiD (45 min)

Create `day49/did_placebos.py`:

```python
# day49/did_placebos.py   (python -m day49.did_placebos)
"""
Placebo-in-time for DiD: keep only the weeks BEFORE the real campaign and pretend it started at week 25.
Nothing happened at week 25, so the estimate should be about 0. A 'placebo effect' means the design
would also have found an effect where there was none.
"""
import pyfixest as pf

from causalkit.stores import campaign, public

for title, gap in [("parallel trends", 0.0), ("treated stores growing faster", 0.15)]:
    df = public(campaign(trend_gap=gap)).to_pandas()
    pre = df[df.week < 40].copy()
    pre["fake_d"] = pre.treated_group * (pre.week >= 25)
    fake = pf.feols("sales ~ fake_d | store_id + week", pre, vcov={"CRV1": "store_id"})
    real = pf.feols("sales ~ d | store_id + week", df, vcov={"CRV1": "store_id"})
    print(f"{title:<30} placebo (week 25): {fake.coef()['fake_d']:+5.2f} (p = {fake.pvalue()['fake_d']:.3f})   "
          f"real estimate: {real.coef()['d']:.2f}")
```

```bash
python -m day49.did_placebos
```

```
parallel trends                placebo (week 25): -0.37 (p = 0.280)   real estimate: 2.74
treated stores growing faster  placebo (week 25): +2.55 (p = 0.000)   real estimate: 7.24
```

With parallel trends, the fake campaign at week 25 finds nothing. With the trend gap, it finds a highly significant "effect" of $2.55k, from a campaign that didn't exist. The placebo uses only pre-campaign data, so it can be run before the real campaign is analysed, or even before it ends.

---

## 49.3 Placebo regions for synthetic control (70 min)

**Concept.** Synthetic control has no standard errors. Abadie, Diamond & Hainmueller (2010) proposed **permutation inference**: pretend each of the 29 untreated regions ran the campaign, build its synthetic control, and compare. The statistic is the ratio of post-period to pre-period RMSPE, which rewards a big post-period gap but discounts regions that were never fitted well. If R07's ratio is the largest of 30, the probability of that by chance is 1/30.

Type `placebo_in_space` into `synth.py`. Create `day49/sc_placebos.py`:

```python
# day49/sc_placebos.py   (python -m day49.sc_placebos)
"""
Synthetic control has no standard errors; inference comes from placebos (Abadie et al., 2010).
In space: pretend each of the other 29 regions ran the campaign. If R07's post/pre misfit ratio is the
largest of 30, the chance of that by luck is 1/30.
In time: pretend R07's campaign started at week 60, using only data before week 80.
"""
import matplotlib.pyplot as plt
import polars as pl

from causalkit.regions import public, regions
from causalkit.synth import placebo_in_space, synthetic_control

df = public(regions())
ranks = placebo_in_space(df, "R07", 80)
print(ranks.head(6).with_columns(pl.col("effect", "rmspe_pre", "ratio").round(2)))
rank = ranks["region"].to_list().index("R07") + 1
print(f"R07 ranks {rank} of {ranks.height}: permutation p-value = {rank}/{ranks.height} = {rank / ranks.height:.3f}")

pre_only = df.filter(pl.col("week") < 80)
fake = synthetic_control(pre_only, "R07", 60)
print(f"\nPlacebo in time (fake start week 60): effect {fake['effect']:+.1f} $k per week, "
      f"post/pre ratio {fake['ratio']:.2f}")

fig, ax = plt.subplots(figsize=(9, 4))
for r in ranks["region"]:
    sub = df if r == "R07" else df.filter(pl.col("region") != "R07")
    sc = synthetic_control(sub, r, 80)
    if r == "R07":
        ax.plot(sc["weeks"], sc["gap"], color="#dc2626", lw=2, label="R07", zorder=3)
    elif sc["rmspe_pre"] < 3 * ranks.filter(pl.col("region") == "R07")["rmspe_pre"][0]:
        ax.plot(sc["weeks"], sc["gap"], color="#94a3b8", lw=0.7)
ax.axvline(80, color="grey", ls="--", lw=0.8)
ax.set_xlabel("week")
ax.set_ylabel("actual minus synthetic ($k)")
ax.set_title("Placebo gaps (regions with a pre-period fit at most 3x worse than R07's)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day49_sc_placebos.png", dpi=120)
print("Saved figures/day49_sc_placebos.png")
```

```bash
python -m day49.sc_placebos
```

```
│ region ┆ is_treated ┆ effect ┆ rmspe_pre ┆ ratio │
│ R07    ┆ true       ┆ 17.33  ┆ 3.86      ┆ 5.41  │
│ R18    ┆ false      ┆ 21.6   ┆ 6.62      ┆ 3.36  │
│ R14    ┆ false      ┆ 17.6   ┆ 6.21      ┆ 3.11  │
│ R04    ┆ false      ┆ -7.83  ┆ 4.54      ┆ 1.9   │
│ R17    ┆ false      ┆ -6.78  ┆ 4.36      ┆ 1.84  │
│ R01    ┆ false      ┆ -29.99 ┆ 17.71     ┆ 1.74  │
R07 ranks 1 of 30: permutation p-value = 1/30 = 0.033

Placebo in time (fake start week 60): effect +5.5 $k per week, post/pre ratio 1.90
Saved figures/day49_sc_placebos.png
```

Read it carefully:

- R07 ranks first: p = 0.033. With 30 regions, that's the smallest p-value possible. Permutation inference needs many units.
- R18 and R14 show placebo "effects" of $17–22k, as big as R07's, from no campaign at all. They are among the most urban regions, so they are hard to match from the remaining donors (R07 is excluded from every placebo's donor pool): their pre-period RMSPE is 60–70% worse than R07's and their post-period drifts. That's why the **ratio**, not the raw effect, is the test statistic: it discounts gaps that come from a poor fit.
- The placebo in time finds $5.5k: not zero, but a third of the real estimate, with a ratio of 1.9 against R07's 5.4.

Open the figure: R07's gap rises far above all placebo gaps at week 80 (regions with poor pre-fit are hidden, as Abadie et al. recommend).

---

## 49.4 Manipulation and the donut (40 min)

**Concept.** RD's key assumption is that customers can't precisely control which side of the cutoff they land on. If customers know the Gold rule, some at $950 will buy a little more in December to reach $1,000. Those who do are the keen ones, so the customers just above the cutoff are no longer comparable with those just below.

`customers(manipulation=0.3)` simulates it: 30% of customers with organic spending of $850–$1,000 top up to just over $1,000.

**Try it.** In the manipulated data, estimate the RD normally, then with a **donut**: drop customers within $20, then $50, of the cutoff. Create `day49/donut.py`.

<details>
<summary>Solution</summary>

```python
# day49/donut.py   (python -m day49.donut)
from rdrobust import rdrobust

from causalkit.tiers import CUTOFF, customers, public

df = public(customers(manipulation=0.3)).to_pandas()
for donut in [0, 20, 50]:
    d = df[(df.pre_spend - CUTOFF).abs() >= donut]
    rd = rdrobust(d.spend, d.pre_spend, c=CUTOFF)
    print(f"donut ${donut:>2}: estimate {rd.coef.iloc[0, 0]:6.1f}, robust CI [{rd.ci.iloc[2, 0]:.1f}, {rd.ci.iloc[2, 1]:.1f}]")
```

```
donut $ 0: estimate  230.8, robust CI [211.2, 262.1]
donut $20: estimate  282.6, robust CI [244.7, 347.5]
donut $50: estimate  159.5, robust CI [110.0, 221.9]
```

Manipulation nearly doubles the estimate (truth $120). A $20 donut makes it worse: the manipulators sit between $1,000 and $1,040, so it removes only half of them and loses the clean customers next to the cutoff. A $50 donut removes all of them and gets much closer, but the estimate is still too high: the keen customers who moved left a hole **below** the cutoff, so the remaining customers there are less keen than they would have been. A donut is a diagnostic, not a cure. Once a rule has been gamed, the design is compromised.

</details>

---

## 49.5 Day 49 mini-project: the RD falsification battery (65 min)

Three checks every RD needs, on clean and manipulated data. Create `day49/rd_falsification.py`:

```python
# day49/rd_falsification.py   (python -m day49.rd_falsification)
"""
Mini-project: three checks every RD needs, run on two datasets:
  clean, and one where customers just below $1,000 who knew the rule bought their way over it.
  1. Density: is there a suspicious pile-up just above the cutoff? (Cattaneo, Jansson & Ma: rddensity)
  2. Covariates: do pre-determined traits jump at the cutoff? They shouldn't.
  3. Placebo cutoffs: is there a 'jump' where no rule exists?
"""
import numpy as np
import polars as pl
from rddensity import rddensity
from rdrobust import rdrobust

from causalkit.tiers import CUTOFF, customers, public, truth

for title, m in [("clean", 0.0), ("customers top up to reach Gold", 0.3)]:
    full = customers(manipulation=m)
    df = public(full).to_pandas()
    rd = rdrobust(df.spend, df.pre_spend, c=CUTOFF)
    dens = rddensity(X=df.pre_spend.to_numpy(), c=CUTOFF)
    print(f"=== {title}")
    print(f"RD estimate {rd.coef.iloc[0, 0]:.1f}, robust CI [{rd.ci.iloc[2, 0]:.1f}, {rd.ci.iloc[2, 1]:.1f}]"
          f"   (truth {truth(full)['effect_at_cutoff']:.1f})")
    print(f"1. density test p-value: {dens.test['p_jk']:.3f}")
    rows = []
    for cov in ["age", "tenure_years", "engaged"]:
        r = rdrobust(df[cov], df.pre_spend, c=CUTOFF)
        rows.append({"check": f"2. jump in {cov}", "estimate": round(r.coef.iloc[0, 0], 3),
                     "robust p": round(r.pv.iloc[2, 0], 3)})
    for c in [700, 850, 1150, 1300]:
        side = df[df.pre_spend < CUTOFF] if c < CUTOFF else df[df.pre_spend >= CUTOFF]
        r = rdrobust(side.spend, side.pre_spend, c=c)
        rows.append({"check": f"3. placebo cutoff ${c}", "estimate": round(r.coef.iloc[0, 0], 1),
                     "robust p": round(r.pv.iloc[2, 0], 3)})
    with pl.Config(tbl_width_chars=100, fmt_str_lengths=40):
        print(pl.DataFrame(rows))
```

```bash
python -m day49.rd_falsification
```

```
=== clean
RD estimate 123.9, robust CI [101.9, 145.0]   (truth 120.0)
1. density test p-value: 0.502
│ 2. jump in age          ┆ 0.725    ┆ 0.227    │
│ 2. jump in tenure_years ┆ 0.061    ┆ 0.493    │
│ 2. jump in engaged      ┆ 0.039    ┆ 0.092    │
│ 3. placebo cutoff $700  ┆ -14.5    ┆ 0.283    │
│ 3. placebo cutoff $850  ┆ -10.9    ┆ 0.87     │
│ 3. placebo cutoff $1150 ┆ 4.6      ┆ 0.798    │
│ 3. placebo cutoff $1300 ┆ -6.2     ┆ 0.744    │
=== customers top up to reach Gold
RD estimate 230.8, robust CI [211.2, 262.1]   (truth 120.0)
1. density test p-value: 0.000
│ 2. jump in age          ┆ 0.921    ┆ 0.099    │
│ 2. jump in tenure_years ┆ 0.025    ┆ 0.739    │
│ 2. jump in engaged      ┆ 0.008    ┆ 0.717    │
│ 3. placebo cutoff $700  ┆ -27.1    ┆ 0.063    │
│ 3. placebo cutoff $850  ┆ 4.3      ┆ 0.766    │
│ 3. placebo cutoff $1150 ┆ 19.2     ┆ 0.407    │
│ 3. placebo cutoff $1300 ┆ 13.8     ┆ 0.536    │
```

1. **Density** (Cattaneo, Jansson & Ma, 2020): in clean data, the number of customers changes smoothly across $1,000 (p = 0.50). With manipulation, the pile-up just above the cutoff is unmistakable (p < 0.001). This is the check that catches it.
2. **Covariates**: no jumps in either dataset. The keen customers who moved differ in *intent*, which isn't in the data. Covariate checks only catch manipulation that shows up in measured traits; passing them doesn't clear the design.
3. **Placebo cutoffs**: no jumps where no rule exists, in either dataset. Each placebo uses customers from one side of the real cutoff only, so the real jump can't leak in.

The clean design passes everything; the manipulated one fails only the density test, and that's enough. Write a short "Falsification" section in `day49/NOTES.md` with the three results for each dataset and what you'd conclude.

---

## 49.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 49: placebo dates, placebo regions, density test, donut RD, covariate checks"
git push
```

### Day 49 checklist

- [ ] I run placebo tests before looking at the real estimate, and report all of them
- [ ] I can run a placebo-in-time test for DiD
- [ ] I can run permutation inference for synthetic control with the RMSPE ratio, and know its minimum p-value
- [ ] I can run a density test, covariate continuity checks and placebo cutoffs for an RD
- [ ] I know that a donut RD is a diagnostic, not a cure for manipulation

---

# DAY 50 — Capstone: Regional Campaign ROI

| Time | Block |
|---|---|
| 0:00 – 0:30 | 50.1 The brief |
| 0:30 – 1:10 | 50.2 Step 1: look before estimating |
| 1:10 – 2:10 | 50.3 Step 2: four estimators and a placebo test |
| 2:10 – 2:25 | Break |
| 2:25 – 3:15 | 50.4 Step 3: a posterior for profit |
| 3:15 – 3:30 | Break |
| 3:30 – 4:30 | 50.5 Step 4: the decision memo |
| 4:30 – 5:00 | 50.6 Answer key, final commit and Module 5 retrospective |

---

## 50.1 The brief (30 min)

NorthMart's marketing team ran an 8-week TV campaign in region **R12**, starting in week 80. It cost **$100k**. NorthMart keeps **30%** of extra sales as margin, so the campaign paid off if it raised sales by more than 100 / 0.30 = **$333k** in total over the 25 post-period weeks (the campaign and the 17 weeks after it, when some of the effect lingers).

The marketing team's own dashboard compares R12 with the average region and reports a strong success. The CFO wants to know: **did the campaign pay for itself, and should NorthMart run it again in other regions at the same cost?**

Generate the data:

```bash
python data/make_campaign_scenario.py
```

```python
"""
Capstone data for Module 5: NorthMart's TV campaign in region R12. Provided.

    python data/make_campaign_scenario.py

Writes data/scenario/regional_sales.csv (what the analyst gets) and data/truth/regional_truth.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.regions import public, regions, truth  # noqa: E402

full = regions(seed=52, treated="R12", treated_size=180, treated_urban=1.8, lift=25)
root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
(root / "truth").mkdir(exist_ok=True)
public(full).write_csv(root / "scenario" / "regional_sales.csv")
(root / "truth" / "regional_truth.json").write_text(json.dumps(truth(full), indent=2))
print(f"Wrote {full.height:,} region-weeks to data/scenario/regional_sales.csv")
```

**Don't open `data/truth/` until your memo is committed.**

---

## 50.2 Step 1: look before estimating (40 min)

Create `day50/diagnostics.py`:

```python
# day50/diagnostics.py   (python -m day50.diagnostics)
"""Capstone step 1: look before estimating. Is R12 like the other regions, before the campaign?"""
import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("data/scenario/regional_sales.csv")
pre = df.filter(pl.col("week") < 80)
size = pre.group_by("region").agg(pl.col("sales").mean()).sort("sales", descending=True)
print(f"R12 average weekly sales before the campaign: {size.filter(pl.col('region') == 'R12')['sales'][0]:.0f} $k; "
      f"rank {size['region'].to_list().index('R12') + 1} of {size.height}")
first_last = (pre.filter(pl.col("week").is_in([1, 79])).pivot(on="week", index="region", values="sales")
              .with_columns(growth=pl.col("79") - pl.col("1")).sort("growth", descending=True))
print(f"R12 sales growth, week 1 to week 79: {first_last.filter(pl.col('region') == 'R12')['growth'][0]:.0f} $k; "
      f"rank {first_last['region'].to_list().index('R12') + 1} of {first_last.height}; "
      f"median region {first_last['growth'].median():.0f}")

others = df.filter(pl.col("region") != "R12").group_by("week").agg(pl.col("sales").mean()).sort("week")
r12 = df.filter(pl.col("region") == "R12").sort("week")
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(r12["week"], r12["sales"], color="#dc2626", label="R12")
ax.plot(others["week"], others["sales"], color="#64748b", label="average of other regions")
ax.axvline(80, color="grey", ls="--", lw=0.8)
ax.set_xlabel("week")
ax.set_ylabel("sales ($k per week)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day50_diagnostics.png", dpi=120)
print("Saved figures/day50_diagnostics.png")
```

```bash
python -m day50.diagnostics
```

```
R12 average weekly sales before the campaign: 713 $k; rank 1 of 30
R12 sales growth, week 1 to week 79: 140 $k; rank 1 of 30; median region 95
Saved figures/day50_diagnostics.png
```

Two warnings before any estimate: R12 is the **biggest** region, and the **fastest-growing**. From Days 45–46: DiD will be biased by the growth gap, and plain synthetic control can't match the level.

---

## 50.3 Step 2: four estimators and a placebo test (60 min)

Create `day50/estimates.py`:

```python
# day50/estimates.py   (python -m day50.estimates)
"""Capstone step 2: four estimators, their fit, and a placebo-in-space p-value for the one that fits."""
import polars as pl

from causalkit.synth import did, placebo_in_space, sdid, synthetic_control

df = pl.read_csv("data/scenario/regional_sales.csv")
WEEKS_POST = df.filter(pl.col("post") == 1)["week"].n_unique()
sc = synthetic_control(df, "R12", 80)
sci = synthetic_control(df, "R12", 80, intercept=True)
rows = [
    {"estimator": "DiD vs average region", "per week": did(df, "R12", 80)["effect"], "pre-period RMSPE": None},
    {"estimator": "synthetic control", "per week": sc["effect"], "pre-period RMSPE": sc["rmspe_pre"]},
    {"estimator": "SC + intercept", "per week": sci["effect"], "pre-period RMSPE": sci["rmspe_pre"]},
    {"estimator": "synthetic DiD", "per week": sdid(df, "R12", 80)["effect"], "pre-period RMSPE": None},
]
table = pl.DataFrame(rows).with_columns(cumulative=pl.col("per week") * WEEKS_POST).with_columns(
    pl.col("per week", "cumulative", "pre-period RMSPE").round(1))
with pl.Config(tbl_width_chars=110, fmt_str_lengths=30):
    print(table)
print("SC + intercept weights:", sci["weights"])

ranks = placebo_in_space(df, "R12", 80, intercept=True)
rank = ranks["region"].to_list().index("R12") + 1
print(f"Placebo in space (SC + intercept): R12 ranks {rank} of {ranks.height}, p = {rank / ranks.height:.3f}")
```

```bash
python -m day50.estimates
```

```
│ estimator             ┆ per week ┆ pre-period RMSPE ┆ cumulative │
│ DiD vs average region ┆ 26.9     ┆ null             ┆ 673.3      │
│ synthetic control     ┆ 75.9     ┆ 31.5             ┆ 1898.0     │
│ SC + intercept        ┆ 17.4     ┆ 5.0              ┆ 434.8      │
│ synthetic DiD         ┆ 11.6     ┆ null             ┆ 290.8      │
SC + intercept weights: {'R29': 0.533, 'R07': 0.264, 'R01': 0.119, 'R18': 0.071, 'R11': 0.012}
Placebo in space (SC + intercept): R12 ranks 1 of 30, p = 0.033
```

- **DiD** ($27k a week) is what the marketing dashboard reports. The diagnostics already told us why it's too high.
- **Plain synthetic control** misses the pre-period by $31.5k a week: discard it.
- **SC with an intercept** fits well ($5k a week) and passes the placebo test (p = 0.033): $17.4k a week, $435k in total.
- **Synthetic DiD**: $11.6k a week, $291k in total.

The two credible estimators disagree, and they fall on **opposite sides of the $333k break-even line**. That's the real finding of this step.

---

## 50.4 Step 3: a posterior for profit (50 min)

Create `day50/causal_impact.py`. It uses the SC + intercept donors as control series (Day 47):

```python
# day50/causal_impact.py   (python -m day50.causal_impact)
"""Capstone step 3: a Bayesian estimate with a posterior for profit. Controls = the SC + intercept donors."""
import numpy as np
import polars as pl

from causalkit.synth import synthetic_control
from day47.causal_impact import causal_impact, summary

MARGIN, COST = 0.30, 100.0
df = pl.read_csv("data/scenario/regional_sales.csv")
controls = list(synthetic_control(df, "R12", 80, intercept=True)["weights"])
result, scale = causal_impact(df, "R12", 80, controls)
s = summary(result, scale)
print(f"Controls: {controls}")
print(f"Average lift {s['average']['mean']:.1f} $k/week [94% HDI {s['average']['low']:.1f}, {s['average']['high']:.1f}]")
print(f"Cumulative lift {s['cumulative']['mean']:.0f} $k [94% HDI {s['cumulative']['low']:.0f}, {s['cumulative']['high']:.0f}]")
profit = MARGIN * (result.post_impact.sum("obs_ind") * scale).values.ravel() - COST
lo, hi = np.percentile(profit, [3, 97])
print(f"Profit {profit.mean():+.1f} $k [94% interval {lo:+.1f}, {hi:+.1f}];  P(profit > 0) = {(profit > 0).mean():.2f}")
```

```bash
python -m day50.causal_impact
```

```
Controls: ['R29', 'R07', 'R01', 'R18', 'R11']
Average lift 11.7 $k/week [94% HDI 8.4, 14.9]
Cumulative lift 293 $k [94% HDI 210, 373]
Profit -12.0 $k [94% interval -35.0, +11.5];  P(profit > 0) = 0.16
```

The Bayesian model agrees with SDID: about $290k of extra sales, a loss of about $12k, and a 16% probability that the campaign paid off.

Then turn every estimate into money. Create `day50/decision.py`:

```python
# day50/decision.py   (python -m day50.decision)
"""
Capstone step 4: from lift to money. The campaign cost $100k; NorthMart keeps 30% of extra sales.
Profit = 0.30 x cumulative lift - 100. Break-even needs a cumulative lift of 100 / 0.30 = 333 $k.
"""
import polars as pl

from causalkit.synth import did, sdid, synthetic_control

MARGIN, COST = 0.30, 100.0
df = pl.read_csv("data/scenario/regional_sales.csv")
weeks = df.filter(pl.col("post") == 1)["week"].n_unique()
estimates = {
    "DiD vs average region": did(df, "R12", 80)["effect"],
    "synthetic control (poor pre-fit)": synthetic_control(df, "R12", 80)["effect"],
    "SC + intercept": synthetic_control(df, "R12", 80, intercept=True)["effect"],
    "synthetic DiD": sdid(df, "R12", 80)["effect"],
}
rows = [{"estimator": k, "cumulative lift ($k)": round(v * weeks), "profit ($k)": round(MARGIN * v * weeks - COST, 1),
         "verdict": "pays off" if MARGIN * v * weeks > COST else "loses money"} for k, v in estimates.items()]
with pl.Config(tbl_width_chars=110, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
print(f"Break-even cumulative lift: {COST / MARGIN:.0f} $k")
```

```bash
python -m day50.decision
```

```
│ estimator                        ┆ cumulative lift ($k) ┆ profit ($k) ┆ verdict     │
│ DiD vs average region            ┆ 673                  ┆ 102.0       ┆ pays off    │
│ synthetic control (poor pre-fit) ┆ 1898                 ┆ 469.4       ┆ pays off    │
│ SC + intercept                   ┆ 435                  ┆ 30.4        ┆ pays off    │
│ synthetic DiD                    ┆ 291                  ┆ -12.8       ┆ loses money │
Break-even cumulative lift: 333 $k
```

Why might SC + intercept be too high? It relies on five donors, two-thirds of the weight on R29 and R07, and R12 is the fastest-growing region, at the edge of what any mix of donors can reproduce. SDID spreads weight over more donors and puts its time weights on recent weeks, which matters when growth is accelerating (Day 46's Try it). Both pass their checks; neither is certain. A good memo says so.

---

## 50.5 Step 4: the decision memo (60 min)

Write `docs/CAMPAIGN_ROI_DECISION.md`, one page, for the CFO:

1. **Recommendation** (two or three sentences).
2. **What happened** (the campaign, its cost, the break-even lift).
3. **Why the dashboard's number is wrong** (R12's size and growth; one picture).
4. **Results** (the four estimators and the Bayesian model, with their checks, and profit for each).
5. **How sure are we?** (where the credible estimates disagree, and what that means for the decision).
6. **Next steps** (how to get a clean answer next time).

<details>
<summary>Model answer for sections 1, 5 and 6</summary>

> **Recommendation.** Don't roll the campaign out to other regions at the current cost. Our best estimate is that it raised R12's sales by about $290k, short of the $333k needed to cover its $100k cost: a loss of roughly $12k, with about a one-in-six chance that it broke even. If marketing wants to continue, the next run should be a randomised geo test at lower cost (below).
>
> **How sure are we?** Two credible methods give different answers: synthetic DiD and the Bayesian model say about $290k of extra sales (a small loss); a synthetic control with an intercept says $435k (a $30k profit). Both fit R12's history well and pass placebo tests. We lean towards the lower estimate because R12 was growing faster than any other region, which the first two methods handle better, but we can't rule out a small profit. What we can rule out is the dashboard's $673k: it compares R12 with regions that were growing much more slowly. Either way, the campaign is not the clear success it was reported to be.
>
> **Next steps.** For the next campaign, randomise: choose 10 similar regions, run the campaign in 5 picked at random, and compare (Module 3). A geo experiment gives a clean answer in the same 25 weeks, at no extra media cost. If only one region can be treated, pick a typical-sized region, not the biggest, so that synthetic control can match it.

</details>

---

## 50.6 Answer key, final commit and Module 5 retrospective (30 min)

After the memo is committed, compare with the truth. Create `day50/answer_key.py`:

```python
# day50/answer_key.py   (python -m day50.answer_key)
"""Instructor check, after the memo: the true lift and the true profit."""
import json
from pathlib import Path

MARGIN, COST = 0.30, 100.0
t = json.loads(Path("data/truth/regional_truth.json").read_text())
print(f"True lift: {t['avg_post_lift']:.1f} $k per post week; cumulative {t['cumulative_lift']:.0f} $k "
      f"({t['campaign_weeks_lift']:.0f} during the 8 campaign weeks)")
print(f"True profit: {MARGIN * t['cumulative_lift'] - COST:+.1f} $k")
```

```bash
python -m day50.answer_key
```

```
True lift: 11.9 $k per post week; cumulative 298 $k (200 during the 8 campaign weeks)
True profit: -10.7 $k
```

The campaign lost about $11k. Synthetic DiD ($291k) and the Bayesian model ($293k) were within $7k of the true cumulative lift; SC + intercept overstated it by 46%, and the dashboard's DiD by 126%. A memo that leaned towards the lower estimates, while admitting the disagreement, got the decision right.

```bash
git add .
git commit -m "Day 50: regional campaign ROI capstone: diagnostics, estimates, posterior, memo"
git push
```

**What you learned in Module 5.**

| Day | Idea | The simulator lesson |
|---|---|---|
| 41 | DiD, clustered SEs | i.i.d. SEs found fake campaigns 38–57% of the time; the wild bootstrap fixed it |
| 42 | Event studies, pre-trends | A pre-test missed a trend gap 78% of the time; it still biased DiD by 21% |
| 43 | Staggered adoption | TWFE reported 53% of the true effect; one 2×2 had the wrong sign |
| 44 | Modern DiD | Callaway–Sant'Anna and imputation got within 2% of the truth |
| 45 | Synthetic control | DiD doubled the effect; synthetic control was within $3k a week |
| 46 | SDID | Synthetic control fails for the biggest region; an intercept or SDID fixes it |
| 47 | Bayesian time series | Without controls, 2.6× the truth; with controls, within $1k a week and a posterior for profit |
| 48 | RD | Local linear within $5 of the truth; global polynomials swung from $112 to $304 |
| 49 | Placebos | A fake campaign "worked" under a trend gap; the density test caught manipulation that covariates missed |
| 50 | Capstone | The dashboard said +$102k; the campaign lost $11k |

**Coming next: Module 6, Causal Machine Learning.** Modules 4 and 5 estimated one average effect. Module 6 asks **for whom** the effect is large: double machine learning, causal forests and uplift models, to target campaigns at the customers they actually move.

### Day 50 checklist

- [ ] I ran diagnostics that predicted which estimators would fail, before estimating
- [ ] I discarded an estimator because of its pre-period fit, not its answer
- [ ] I ran a placebo test and reported a permutation p-value
- [ ] I turned every estimate into profit against a break-even line
- [ ] I wrote a memo that states where credible methods disagree, and what would settle it

