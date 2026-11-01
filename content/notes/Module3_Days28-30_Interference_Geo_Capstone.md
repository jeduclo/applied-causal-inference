# Module 3 · Experimentation & A/B Testing
## Days 28–30: Interference, Switchbacks, Geo Tests and the Free-Shipping Capstone

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 21–27
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 30:** Students can design experiments when units affect each other or when only time or geography can be randomised, and can take a real business test from pre-registered plan to a segment-level launch decision, including a non-inferiority guardrail.

---

### Where we are

Days 21–27 built a complete toolkit for customer-level A/B tests: plans, power, stratified assignment, regression with CUPED, corrections for multiple looks. All of it relied on two assumptions:

1. **No interference** (SUTVA, Day 11): a customer's outcome depends only on their own treatment.
2. **Customers can be randomised individually.**

Day 28 drops the first, Day 29 the second. Day 30 brings everything together.

---

# DAY 28 — Interference and Network Effects

| Time | Block |
|---|---|
| 0:00 – 0:40 | 28.1 When treatment leaks |
| 0:40 – 1:40 | 28.2 Spillover in NorthMart households |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 28.3 Cluster randomisation and its price |
| 2:40 – 3:20 | 28.4 Where interference hides |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 28.5 Day 28 mini-project: measure the spillover |
| 4:45 – 5:00 | 28.6 Save your work with Git |

---

## 28.1 When treatment leaks (40 min)

**Concept.** **Interference** means one unit's treatment changes another unit's outcome. Then "treated vs control" compares two groups that are *both* partly treated, and the difference is not the effect the business cares about: **treat everyone vs treat no one**.

Two kinds, with opposite consequences:

| Kind | Example | Bias of a customer-level test |
|---|---|---|
| **Positive spillover** | a household shares one inbox; a referral email reaches friends | **underestimates**: controls also benefit |
| **Competition** | a marketplace with limited stock or drivers; ad auctions | **overestimates**: treated units take from controls |

The standard fix is to randomise **clusters** that contain the interference: households, cities, store catchments, whole markets.

---

## 28.2 Spillover in NorthMart households (60 min)

The lab's `run()` has a `spillover` argument: the share of a treated customer's lift that reaches the other members of their household. Suppose it's 50%.

Create `day28/household_spillover.py`:

```python
# day28/household_spillover.py   (python -m day28.household_spillover)
"""
Interference: household members share inboxes and shopping trips. If half of the email's lift
spills over to the other members of a treated customer's household, what does each design estimate?
Target: the effect of emailing EVERYONE vs NO ONE (the decision marketing actually faces).
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run, truth

SPILL = 0.5
pop = population()
hh = pop["household_id"].to_numpy()
households, inverse = np.unique(hh, return_inverse=True)
rng = np.random.default_rng(28)

est = {"individual randomisation": [], "household randomisation": []}
for r in range(300):
    z_ind = rng.permutation(np.repeat([0, 1], pop.height // 2))
    z_hh = (rng.random(len(households)) < 0.5)[inverse].astype(np.int8)
    for name, z in [("individual randomisation", z_ind), ("household randomisation", z_hh)]:
        y = run(pop, z, "email_v2", seed=6000 + r, spillover=SPILL)["orders"].to_numpy()
        est[name].append(y[z == 1].mean() - y[z == 0].mean())

target = truth("email_v2")["orders"]
print(pl.DataFrame([{"design": k, "mean estimate": round(np.mean(v), 4), "SD": round(np.std(v), 4),
                     "bias vs truth": f"{np.mean(v) / target - 1:+.0%}"} for k, v in est.items()]))
print(f"True effect of emailing everyone vs no one: {target:+.4f} orders per customer")
```

```bash
python -m day28.household_spillover
```

```
│ design                   ┆ mean estimate ┆ SD     ┆ bias vs truth │
│ individual randomisation ┆ 0.0368        ┆ 0.0147 ┆ -35%          │
│ household randomisation  ┆ 0.0564        ┆ 0.0145 ┆ -1%           │
True effect of emailing everyone vs no one: +0.0570 orders per customer
```

The customer-level test reports an effect 35% too small: two thirds of control customers (67%) share a household with a treated customer and get half the lift. Randomising **whole households** restores the comparison: no control customer lives with a treated one.

**Try it.** Repeat the individual-randomisation test with spillover 0 and 1.0. How does the bias change?

<details>
<summary>Solution</summary>

```python
# day28/full_spillover.py   (python -m day28.full_spillover)
import numpy as np

from causalkit.lab import population, run, truth

pop = population()
rng = np.random.default_rng(283)
for spill in [0.0, 0.5, 1.0]:
    est = []
    for r in range(60):
        z = rng.permutation(np.repeat([0, 1], pop.height // 2))
        y = run(pop, z, "email_v2", seed=9000 + r, spillover=spill)["orders"].to_numpy()
        est.append(y[z == 1].mean() - y[z == 0].mean())
    print(f"spillover {spill:.1f}: average estimate {np.mean(est):.4f}  (truth {truth('email_v2')['orders']:.4f})")
```

```
spillover 0.0: average estimate 0.0566  (truth 0.0570)
spillover 0.5: average estimate 0.0373  (truth 0.0570)
spillover 1.0: average estimate 0.0185  (truth 0.0570)
```

With full spillover, controls who share a household with a treated customer get the whole effect, and the test sees only a third of it. The bias grows with both the spillover strength and the share of controls exposed (which depends on household sizes).

</details>

---

## 28.3 Cluster randomisation and its price (45 min)

**Concept.** When households are randomised, the **household** is the unit: standard errors must be **clustered** by household. How much precision does that cost? It depends on the **intra-cluster correlation (ICC)**, how alike members of a cluster are:

```text
design effect = 1 + (m − 1) × ICC        m = average cluster size
```

The design effect is the factor by which the sample must grow to match an individually randomised test.

Create `day28/cluster_se.py`:

```python
# day28/cluster_se.py   (python -m day28.cluster_se)
"""
Randomising households means the unit of randomisation is the household: standard errors
must be clustered by household. How much does clustering cost here? It depends on the
intra-cluster correlation (ICC): how alike members of the same household are.
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from causalkit.lab import population, run

pop = population()
hh = pop["household_id"].to_numpy()
households, inverse = np.unique(hh, return_inverse=True)
rng = np.random.default_rng(281)
z = (rng.random(len(households)) < 0.5)[inverse].astype(np.int8)
df = run(pop, z, "email_v2", seed=281).with_columns(household_id=pl.Series(hh)).to_pandas()

naive = smf.ols("orders ~ treated", df).fit(cov_type="HC2")
clustered = smf.ols("orders ~ treated", df).fit(cov_type="cluster", cov_kwds={"groups": inverse})
print(f"Effect {naive.params['treated']:+.4f}   naive SE {naive.bse['treated']:.4f}   "
      f"household-clustered SE {clustered.bse['treated']:.4f}")

# ICC: share of variance that is between households (one-way ANOVA estimator)
m = df.groupby("household_id")["orders"].agg(["mean", "size"])
k = m["size"].mean()
between = np.average((m["mean"] - df["orders"].mean()) ** 2, weights=m["size"]) * len(m) / (len(m) - 1)
within = df.groupby("household_id")["orders"].var(ddof=1).mean()
icc = max(0.0, (between - within / k) / (between - within / k + within))
print(f"Average household size {k:.2f}   ICC {icc:.3f}   design effect 1 + (m - 1) x ICC = {1 + (k - 1) * icc:.3f}")
for rho in [0.05, 0.1, 0.3]:
    print(f"  if ICC were {rho:.2f}: design effect {1 + (k - 1) * rho:.2f} -> need {1 + (k - 1) * rho:.0%} of the sample")
```

```
Effect +0.0380   naive SE 0.0139   household-clustered SE 0.0138
Average household size 2.47   ICC 0.000   design effect 1 + (m - 1) x ICC = 1.000
  if ICC were 0.05: design effect 1.07 -> need 107% of the sample
  if ICC were 0.10: design effect 1.15 -> need 115% of the sample
  if ICC were 0.30: design effect 1.44 -> need 144% of the sample
```

In the lab, household members shop independently (ICC ≈ 0), so clustering costs nothing. In real data, household members share income, location and habits; ICCs of 0.1–0.3 are common. For cities (Day 29), ICCs are smaller but clusters are huge, so the design effect is enormous.

> Always cluster standard errors at the level of randomisation, even when you expect the ICC to be small. If it is small, you lose nothing.

---

## 28.4 Where interference hides (40 min)

| Setting | Interference | Cluster to randomise |
|---|---|---|
| Household emails, shared accounts | members influence each other | household |
| Referral and social features | friends influence each other | friendship clusters, or regions |
| Marketplace pricing (sellers, drivers) | treated buyers take supply from controls | market × time (Day 29) |
| Ad budgets and auctions | treated campaigns change prices for controls | budget-split or geo designs |
| Store promotions | customers switch stores | store catchment areas |
| Inventory and delivery capacity | treated orders use shared capacity | warehouse × time |

The question to ask in every plan: *if I treat this customer, whose outcome could change besides theirs?* If the answer isn't "nobody", interference needs a design response.

---

## 28.5 Day 28 mini-project: measure the spillover (70 min)

**Goal.** Cluster randomisation avoids spillover. A **two-stage (saturation) design** *measures* it, which tells you whether the cluster design was needed and how big the network effect is.

- **Stage 1:** randomise households (with 2+ members) into NONE, ONE, or ALL.
- **Stage 2:** in ONE households, randomly choose which member is emailed.

Comparing each group with NONE households gives three effects.

Create `day28/saturation_design.py`:

```python
# day28/saturation_design.py   (python -m day28.saturation_design)
"""
Mini-project: MEASURE the spillover instead of just avoiding it. Two-stage (saturation) design,
households with 2+ members only:
  stage 1: each household is randomised to NONE, ONE (one random member emailed) or ALL
  stage 2: in ONE households, which member is emailed is random
Comparisons against NONE households give the direct, spillover and total effects.
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run

SPILL = 0.5
pop = population()
hh = pop["household_id"].to_numpy()
_, inverse, sizes = np.unique(hh, return_inverse=True, return_counts=True)
multi = sizes[inverse] >= 2
rng = np.random.default_rng(282)

res = {"direct (emailed, in ONE households)": [], "spillover (not emailed, in ONE households)": [],
       "total (everyone, in ALL households)": []}
for r in range(200):
    arm = rng.integers(0, 3, inverse.max() + 1)[inverse]           # 0 = NONE, 1 = ONE, 2 = ALL
    # stage 2: give every customer a random number; in each household, the smallest one is "chosen"
    draw = pl.DataFrame({"hh": inverse, "u": rng.random(len(hh))})
    first = (draw["u"] == draw.select(pl.col("u").min().over("hh")).to_series()).to_numpy()
    z = ((arm == 2) | ((arm == 1) & first)) & multi
    y = run(pop, z.astype(np.int8), "email_v2", seed=7000 + r, spillover=SPILL)["orders"].to_numpy()
    base = y[multi & (arm == 0)].mean()
    res["direct (emailed, in ONE households)"].append(y[multi & (arm == 1) & first].mean() - base)
    res["spillover (not emailed, in ONE households)"].append(y[multi & (arm == 1) & ~first].mean() - base)
    res["total (everyone, in ALL households)"].append(y[multi & (arm == 2)].mean() - base)

rate4 = pop["_rate"].to_numpy()[multi].mean() * 4                  # hidden truth, for checking only
truths = [0.05 * rate4, SPILL * 0.05 * rate4, 0.05 * rate4]          # approximate: lift of 5% for everyone
with pl.Config(fmt_str_lengths=50, tbl_width_chars=110):
    print(pl.DataFrame([{"effect": k, "average estimate": round(np.mean(v), 4), "SD": round(np.std(v), 4),
                         "approximate truth": round(t, 4)} for (k, v), t in zip(res.items(), truths)]))
```

```bash
python -m day28.saturation_design
```

```
│ effect                                     ┆ average estimate ┆ SD     ┆ approximate truth │
│ direct (emailed, in ONE households)        ┆ 0.0562           ┆ 0.0262 ┆ 0.057             │
│ spillover (not emailed, in ONE households) ┆ 0.0279           ┆ 0.0195 ┆ 0.0285            │
│ total (everyone, in ALL households)        ┆ 0.0553           ┆ 0.0177 ┆ 0.057             │
```

All three effects are recovered: the spillover is half the direct effect, as built into the lab. The price is precision: each comparison uses only a third of the households, so standard errors are large. In a real test you'd run it once, to learn the spillover, then design future tests accordingly.

---

## 28.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 28: interference, cluster randomisation, saturation design"
git push
```

### Day 28 checklist

- [ ] I can explain how positive spillover and competition bias customer-level tests
- [ ] I can randomise clusters and cluster standard errors at the randomisation level
- [ ] I can compute a design effect from the ICC and cluster size
- [ ] I can design a two-stage experiment that measures direct, spillover and total effects

---

# DAY 29 — Switchback and Geo Experiments

| Time | Block |
|---|---|
| 0:00 – 0:30 | 29.1 When you can't randomise customers |
| 0:30 – 1:40 | 29.2 Switchback experiments: randomising time |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 29.3 Carryover and wash-out periods |
| 2:40 – 3:20 | 29.4 Geo experiments: randomising places |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 29.5 Day 29 mini-project: a 20-city radio campaign |
| 4:45 – 5:00 | 29.6 Save your work with Git |

---

## 29.1 When you can't randomise customers (30 min)

Some changes reach everyone at once, or can only be bought for whole areas:

| Change | Why customer-level tests fail | Design |
|---|---|---|
| A new delivery-promise algorithm | it runs site-wide; competing for shared capacity | **switchback**: randomise time periods |
| Dynamic pricing for a marketplace | treated and control buyers share the same sellers | switchback, or market × time |
| TV, radio or billboard campaigns | can only be bought by region | **geo experiment**: randomise regions |
| A new warehouse process | one warehouse, one process at a time | switchback by shift or day |

Both designs have **few units** (days, cities), so the principles from Days 21–25 matter even more: stratify or pair, use the pre-period, and quantify uncertainty with methods that don't rely on large samples.

---

## 29.2 Switchback experiments: randomising time (70 min)

**Concept.** In a switchback, the whole marketplace switches between treatment and control on a random schedule (by day, hour, or block of days). Each period is one unit. Analysis compares treated periods with control periods, controlling for time patterns (day of week, trend).

Create `day29/switchback.py`. It's a toy marketplace (not the customer lab): 112 days of orders with weekly seasonality, an upward trend, autocorrelated noise and a true effect of +40 orders/day. Half of the effect lingers into the day after a switch-off.

```python
# day29/switchback.py   (python -m day29.switchback)
"""
A change that hits EVERY customer at once: a new delivery-promise algorithm on the website.
Customer-level randomisation is impossible, so randomise TIME instead (switchback).
Toy marketplace: 112 days of daily orders with weekly seasonality, a slow upward trend,
autocorrelated noise, and a true effect of +40 orders/day. Part of the effect carries over
into the day after the algorithm is switched off (customers' expectations adjust slowly).
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

DAYS, EFFECT, CARRYOVER = 112, 40.0, 0.5
dow = np.arange(DAYS) % 7
season = np.array([0, -20, -10, 0, 30, 80, 60])[dow]


def simulate(treated: np.ndarray, rng) -> np.ndarray:
    noise = np.zeros(DAYS)
    for d in range(1, DAYS):
        noise[d] = 0.6 * noise[d - 1] + rng.normal(0, 25)
    carry = np.r_[0, treated[:-1]] * (1 - treated) * CARRYOVER * EFFECT    # lingering effect after switch-off
    return 1000 + 1.5 * np.arange(DAYS) + season + EFFECT * treated + carry + noise


def estimate(y: np.ndarray, t: np.ndarray, drop_after_switch: bool = False) -> float:
    df = pl.DataFrame({"y": y, "t": t, "dow": dow, "day": np.arange(DAYS)})
    if drop_after_switch:                                  # wash-out: ignore the first day of each period
        df = df.filter(pl.Series(np.r_[False, t[1:] != t[:-1]]).not_())
    return smf.ols("y ~ t + C(dow) + day", df.to_pandas()).fit().params["t"]


rng = np.random.default_rng(29)
designs = {"before/after (56 days each)": [], "switchback, daily": [],
           "switchback, weekly blocks": [], "switchback, weekly + wash-out": []}
for r in range(1000):
    before_after = (np.arange(DAYS) >= 56).astype(float)
    daily = rng.integers(0, 2, DAYS).astype(float)
    weekly = np.repeat(rng.integers(0, 2, DAYS // 7), 7).astype(float)
    ba_y = simulate(before_after, rng)
    designs["before/after (56 days each)"].append(ba_y[56:].mean() - ba_y[:56].mean())
    designs["switchback, daily"].append(estimate(simulate(daily, rng), daily))
    y_w = simulate(weekly, rng)
    designs["switchback, weekly blocks"].append(estimate(y_w, weekly))
    designs["switchback, weekly + wash-out"].append(estimate(y_w, weekly, drop_after_switch=True))

with pl.Config(fmt_str_lengths=40, tbl_width_chars=100):
    print(pl.DataFrame([{"design": k, "mean estimate": round(np.mean(v), 1), "SD": round(np.std(v), 1)}
                        for k, v in designs.items()]))
print(f"True effect: +{EFFECT:.0f} orders per day")
```

```bash
python -m day29.switchback
```

```
│ design                        ┆ mean estimate ┆ SD   │
│ before/after (56 days each)   ┆ 124.1         ┆ 11.4 │
│ switchback, daily             ┆ 29.7          ┆ 6.3  │
│ switchback, weekly blocks     ┆ 38.7          ┆ 10.9 │
│ switchback, weekly + wash-out ┆ 40.3          ┆ 11.5 │
True effect: +40 orders per day
```

- **Before/after** ("launch it and compare with last month") reports +124: three times the truth, because the trend adds 1.5 orders a day. No standard error fixes that.
- **Daily switchback** is the most precise (SD 6.3) but biased low (29.7): every switch-off day still carries half the effect, contaminating the control days.

---

## 29.3 Carryover and wash-out periods (45 min)

**Carryover** is interference across time: today's treatment affects tomorrow's outcome (customers' expectations, inventory positions, drivers' locations). It biases switchbacks towards zero, more so the more often you switch.

Two fixes, used together:

1. **Longer periods** (weekly blocks): fewer switches, so less contamination. Bias almost disappears (38.7), but precision drops (fewer, longer periods).
2. **Wash-out**: drop the first day after each switch from the analysis. With weekly blocks, the estimate is unbiased (40.3).

Choosing the period length is a trade-off between bias (carryover) and variance (number of periods). Run an A/A switchback first to learn the noise, and estimate carryover from past launches.

**Try it.** Set `CARRYOVER = 0` and rerun. Which design is best now?

<details>
<summary>Solution</summary>

```
│ before/after (56 days each)   ┆ 124.6         ┆ 11.5 │
│ switchback, daily             ┆ 39.8          ┆ 6.0  │
│ switchback, weekly blocks     ┆ 41.0          ┆ 10.6 │
│ switchback, weekly + wash-out ┆ 41.3          ┆ 11.4 │
```

(400 replays instead of 1,000, to save time.) Without carryover, the daily switchback is unbiased and still the most precise, so it wins clearly. Before/after stays biased by the trend whatever the carryover. Short periods are only a problem when the effect lingers, so the period length should follow how long effects last in your business.

</details>

---

## 29.4 Geo experiments: randomising places (40 min)

**Concept.** In a geo experiment, regions or cities are the units. Typical numbers are 10–200 units of very different sizes, with strong common shocks. Three habits make them work:

1. **Use the pre-period.** Compare each region's growth with its own history (the CUPED idea, applied to regions), not raw levels.
2. **Pair or stratify** similar regions before randomising (Day 23), when the pre-period doesn't already absorb the differences.
3. **Randomisation inference** (Day 13): with 20 units, normal-approximation p-values are shaky. Re-draw the assignment thousands of times under "no effect" and see where the real result falls.

---

## 29.5 Day 29 mini-project: a 20-city radio campaign (70 min)

**Goal.** NorthMart buys local radio in 10 of 20 cities. Cities differ in size by a factor of 20. True effect: +4% of a city's sales. Compare two analyses over 2,000 simulated campaigns, then analyse one campaign with randomisation inference.

Create `day29/geo_experiment.py`:

```python
# day29/geo_experiment.py   (python -m day29.geo_experiment)
"""
Mini-project: a geo experiment. A local radio campaign can only be bought by CITY.
20 cities of very different sizes; 10 get the campaign. With only 20 units, the analysis matters.
Toy data: weekly sales for 8 pre-period weeks and 4 test weeks; true effect +4% of a city's sales.
"""
import numpy as np
import polars as pl

rng = np.random.default_rng(29)
N, PRE, TEST, LIFT = 20, 8, 4, 0.04
size = rng.lognormal(np.log(50_000), 0.8, N)                    # typical weekly sales per city ($)


def sales(treated: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    drift = r.normal(0, 0.02, N)                                 # cities drift a little
    pre = size[:, None] * (1 + r.normal(0, 0.03, (N, PRE)))
    test = size[:, None] * (1 + drift[:, None] + LIFT * treated[:, None] + r.normal(0, 0.03, (N, TEST)))
    return pre.sum(1), test.sum(1)


def raw_lift(pre, test, t) -> float:
    """Difference in average test-period sales, as a share of average sales (cities differ hugely in size)."""
    return (test[t == 1].mean() - test[t == 0].mean()) / test.mean()


def growth_lift(pre, test, t) -> float:
    """Each city compared with ITSELF first: growth vs its own pre-period, then treated minus control."""
    growth = test / (pre / PRE * TEST) - 1
    return growth[t == 1].mean() - growth[t == 0].mean()


est = {"raw sales": [], "growth vs own pre-period": []}
for r in range(2000):
    t = np.zeros(N); t[rng.choice(N, 10, replace=False)] = 1
    pre, test = sales(t, r)
    est["raw sales"].append(raw_lift(pre, test, t))
    est["growth vs own pre-period"].append(growth_lift(pre, test, t))
print(pl.DataFrame([{"analysis": k, "mean": f"{np.mean(v):+.1%}", "SD": f"{np.std(v):.1%}"} for k, v in est.items()]))
print(f"True effect: +{LIFT:.0%}")

# One real geo test, analysed with randomisation inference: re-draw the assignment 20,000 times
t = np.zeros(N); t[rng.choice(N, 10, replace=False)] = 1
pre, test = sales(t, 424_242)
observed = growth_lift(pre, test, t)
null = []
for _ in range(20_000):
    fake = np.zeros(N); fake[rng.choice(N, 10, replace=False)] = 1
    null.append(growth_lift(pre, test, fake))                    # sharp null: the campaign did nothing anywhere
p = np.mean(np.abs(null) >= abs(observed))
print(f"\nOne geo test: estimated lift {observed:+.1%}, randomisation-inference p-value {p:.4f}")
```

```bash
python -m day29.geo_experiment
```

```
│ analysis                 ┆ mean  ┆ SD    │
│ raw sales                ┆ +4.0% ┆ 20.4% │
│ growth vs own pre-period ┆ +4.0% ┆ 1.2%  │
True effect: +4%

One geo test: estimated lift +4.9%, randomisation-inference p-value 0.0022
```

- Comparing raw sales is unbiased but useless: an SD of 20 points against a 4-point effect. Which cities happen to be treated matters far more than the campaign.
- Comparing **growth** against each city's own pre-period cuts the noise 17-fold. Same data, same 20 cities.
- In the single test, randomisation inference re-draws 20,000 assignments of 10 cities: only 0.22% give a lift as large as the one observed. No distributional assumptions needed.

Write a one-paragraph **geo test plan** in `docs/` for a real campaign: number of cities, how you'd pair them, pre-period length, the analysis, and the MDE implied by the 1.2% standard deviation.

---

## 29.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 29: switchback designs, carryover, geo experiment with randomisation inference"
git push
```

### Day 29 checklist

- [ ] I can tell when customer-level randomisation is impossible, and choose a switchback or geo design
- [ ] I can explain why before/after comparisons fail, even with long periods
- [ ] I can handle carryover with block length and wash-out periods
- [ ] I can analyse a geo test with pre-period growth and randomisation inference

---

# DAY 30 — Capstone: The Free-Shipping Experiment

| Time | Block |
|---|---|
| 0:00 – 0:30 | 30.1 The brief and how it is assessed |
| 0:30 – 1:15 | 30.2 The plan, before any data |
| 1:15 – 1:30 | Break |
| 1:30 – 2:00 | 30.3 Assign, audit, run |
| 2:00 – 3:00 | 30.4 The readout: primary metric and guardrail |
| 3:00 – 3:15 | Break |
| 3:15 – 3:50 | 30.5 Pre-registered segments |
| 3:50 – 4:40 | 30.6 The decision memo |
| 4:40 – 5:00 | 30.7 Answer key, final commit and Module 3 retrospective |

---

## 30.1 The brief and how it is assessed (30 min)

**The brief.** NorthMart's e-commerce director writes:

> *"Competitors offer free shipping from $35. We charge $7 below $50. Marketing is sure lowering our threshold to $35 will raise sales; Finance is worried we'll give away margin on every order between $35 and $50. I want a proper test before deciding. I need to know three things: does it raise revenue, does it hurt margin by more than we can accept (Finance says a loss of up to $1 per customer per month is fine if revenue grows), and should we launch it for everyone or only for some customers? Plan it before you run it."*

**Deliverables.**

| # | Deliverable | Where |
|---|---|---|
| D1 | Pre-registered plan, written before running | `docs/FREE_SHIPPING_PLAN.md` |
| D2 | Power and MDE for revenue and the margin guardrail, overall and by segment | `day30/plan.py` |
| D3 | Assignment audit (SRM, balance) | `day30/run_test.py` |
| D4 | Readout: CUPED-adjusted effects, non-inferiority guardrail, corrected segment tests | `day30/readout.py` |
| D5 | One-page decision memo | `docs/FREE_SHIPPING_DECISION.md` |

**Assessment rubric.**

| Criterion | Weight | Excellent |
|---|---|---|
| Pre-registration | 20% | Metrics, guardrail margin, segments, analysis and decision rule fixed before running; committed to Git first |
| Design | 20% | Power justified; stratified assignment; audit clean |
| Analysis | 25% | CUPED, robust SEs, non-inferiority test done correctly, segments corrected |
| Decision | 20% | Clear recommendation that follows from the pre-registered rule and states its uncertainty |
| Communication | 15% | A director can read the memo in five minutes and act |

---

## 30.2 The plan, before any data (45 min)

Write `docs/FREE_SHIPPING_PLAN.md` using the Day 21 template. The brief fixes most of it:

| Element | Choice |
|---|---|
| Hypothesis | Lowering the free-shipping threshold to $35 raises revenue; margin loss is at most $1/customer |
| Unit, population | customer; all 40,000 |
| Duration | 4 weeks |
| Primary metric | revenue per customer (two-sided test, α = 0.05) |
| Secondary | orders, conversion (Holm-corrected) |
| **Guardrail** | margin per customer: **non-inferiority** with margin −$1.00 (one-sided, α = 0.05): pass if the lower bound of the one-sided 95% interval is above −$1.00 |
| Segments | loyalty members vs non-members, **pre-registered**, Holm correction across segment tests (the director asked "for whom?") |
| Analysis | regression with CUPED covariates (pre_revenue, pre_orders) and stratum dummies; robust SEs |
| Assignment | stratified hash assignment, salt `free_shipping_2026` |
| Decision rule | launch for a population if its revenue effect is positive and significant **and** it passes the margin guardrail |

**Non-inferiority, in one sentence:** instead of asking "did margin fall?" (which needs 81,000 customers to answer, Day 25), we ask "can we rule out a fall worse than $1?", which is the question Finance actually asked.

Now check feasibility. Create `day30/plan.py`:

```python
# day30/plan.py   (python -m day30.plan)
"""
Capstone step 1 (before running anything): can 40,000 customers answer the questions?
Primary metric: revenue (superiority).  Guardrail: margin (non-inferiority: no loss worse than $1.00).
Pre-registered segments: loyalty members and non-members.
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import norm

from causalkit.lab import population, run

NI_MARGIN = -1.00                       # the largest margin loss per customer we would accept
pop = population()
last_month = run(pop, np.zeros(pop.height), "aa_test", seed=99).join(
    pop.select("customer_id", "pre_revenue", "pre_orders", "loyalty"), on="customer_id").to_pandas()

Z_TWO = norm.ppf(0.975) + norm.ppf(0.80)     # two-sided superiority test, 80% power
Z_ONE = norm.ppf(0.95) + norm.ppf(0.80)      # one-sided non-inferiority test, 80% power

rows = []
for label, sel in [("everyone", np.ones(len(last_month), bool)),
                   ("non-members", last_month.loyalty.to_numpy() == 0),
                   ("loyalty members", last_month.loyalty.to_numpy() == 1)]:
    d = last_month[sel]
    n_group = sel.sum() / 2
    for metric in ["revenue", "margin"]:
        r2 = smf.ols(f"{metric} ~ pre_revenue + pre_orders", d).fit().rsquared
        sd = d[metric].std() * np.sqrt(1 - r2)                          # CUPED-adjusted SD
        mde = Z_TWO * sd * np.sqrt(2 / n_group)
        row = {"population": label, "metric": metric, "per group": int(n_group),
               "CUPED SD": round(sd, 2), "MDE (two-sided)": round(mde, 2)}
        if metric == "margin":   # power to show 'no worse than -$1' if the true effect is 0
            se = sd * np.sqrt(2 / n_group)
            row["NI power if truly 0"] = f"{norm.cdf((0 - NI_MARGIN) / se - norm.ppf(0.95)):.0%}"
        rows.append(row)
with pl.Config(tbl_width_chars=120):
    print(pl.DataFrame(rows))
```

```bash
python -m day30.plan
```

```
│ population      ┆ metric  ┆ per group ┆ CUPED SD ┆ MDE (two-sided) ┆ NI power if truly 0 │
│ everyone        ┆ revenue ┆ 20000     ┆ 85.22    ┆ 2.39            ┆ null                │
│ everyone        ┆ margin  ┆ 20000     ┆ 25.57    ┆ 0.72            ┆ 99%                 │
│ non-members     ┆ revenue ┆ 14003     ┆ 76.94    ┆ 2.58            ┆ null                │
│ non-members     ┆ margin  ┆ 14003     ┆ 23.08    ┆ 0.77            ┆ 98%                 │
│ loyalty members ┆ revenue ┆ 5996      ┆ 100.46   ┆ 5.14            ┆ null                │
│ loyalty members ┆ margin  ┆ 5996      ┆ 30.14    ┆ 1.54            ┆ 57%                 │
```

Record in the plan: overall and for non-members, the test can detect a revenue effect of about $2.40–$2.60 and has near-certain power to pass the guardrail if margin is truly unchanged. For loyalty members (only 30% of customers), the revenue MDE is $5 and guardrail power is 57%: a "fail" there will be hard to distinguish from "uncertain". Write that down **now**, so nobody can reinterpret it later.

**Commit the plan before running anything:**

```bash
git add docs/FREE_SHIPPING_PLAN.md day30/plan.py
git commit -m "Day 30: pre-registered free-shipping plan"
```

The commit timestamp is your proof of pre-registration.

---

## 30.3 Assign, audit, run (30 min)

Create `day30/run_test.py`:

```python
# day30/run_test.py   (python -m day30.run_test)
"""Capstone step 2: assign (stratified), audit, run the 4-week test, store the results."""
from pathlib import Path

from causalkit.assignment import add_strata, assign_stratified, balance_table, srm_pvalue
from causalkit.lab import population, run

pop = population()
z = assign_stratified(pop, "free_shipping_2026")
print(f"Treated {z.sum():,} of {len(z):,}   SRM p = {srm_pvalue(z):.3f}")
print(balance_table(pop, z))

out = run(pop, z, "free_shipping", seed=30)          # the four weeks happen here
results = out.join(add_strata(pop).select("customer_id", "loyalty", "pre_orders", "pre_revenue", "stratum"),
                   on="customer_id")
Path("data/experiments").mkdir(parents=True, exist_ok=True)
results.write_parquet("data/experiments/free_shipping.parquet")
print(f"Saved {results.height:,} rows to data/experiments/free_shipping.parquet")
```

```bash
python -m day30.run_test
```

```
Treated 19,997 of 40,000   SRM p = 0.976
│ covariate   ┆ treated ┆ control ┆ smd     ┆ flag  │
│ pre_orders  ┆ 2.263   ┆ 2.274   ┆ -0.0051 ┆ false │
│ pre_revenue ┆ 137.758 ┆ 138.956 ┆ -0.0072 ┆ false │
│ loyalty     ┆ 0.3     ┆ 0.3     ┆ -0.0001 ┆ false │
│ mobile      ┆ 0.62    ┆ 0.621   ┆ -0.0027 ┆ false │
Saved 40,000 rows to data/experiments/free_shipping.parquet
```

Clean audit: no SRM, every |SMD| below 0.01. Only now may you look at outcomes.

---

## 30.4 The readout: primary metric and guardrail (60 min)

Create `day30/readout.py`:

```python
# day30/readout.py   (python -m day30.readout)
"""Capstone step 3: the pre-registered readout. CUPED throughout; margin as a non-inferiority guardrail."""
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from causalkit.analysis import readout

NI_MARGIN = -1.00
df = pl.read_parquet("data/experiments/free_shipping.parquet")
COVS = ["pre_revenue", "pre_orders", "C(stratum)"]

print("1. All customers (CUPED-adjusted)")
with pl.Config(tbl_width_chars=120):
    print(readout(df, ["revenue", "orders", "converted", "margin"], covariates=COVS))


def guardrail(d: pl.DataFrame, label: str) -> dict:
    fit = smf.ols("margin ~ treated + " + " + ".join(COVS), d.to_pandas()).fit(cov_type="HC2")
    eff, se = fit.params["treated"], fit.bse["treated"]
    lower = eff - norm.ppf(0.95) * se                       # one-sided 95% lower bound
    p_ni = norm.sf((eff - NI_MARGIN) / se)                  # H0: effect <= -1
    return {"population": label, "margin_effect": round(eff, 3), "lower_bound_95": round(lower, 3),
            "p_noninferiority": fmt(p_ni), "passes": bool(lower > NI_MARGIN)}


def fmt(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


print(f"\n2. Margin guardrail: non-inferiority, margin {NI_MARGIN:+.2f} per customer")
print(pl.DataFrame([guardrail(df, "everyone")]))

print("\n3. Pre-registered segments (Holm correction across the 4 segment tests)")
rows = []
for flag, label in [(0, "non-members"), (1, "loyalty members")]:
    seg = df.filter(pl.col("loyalty") == flag)
    for metric in ["revenue", "margin"]:
        fit = smf.ols(f"{metric} ~ treated + pre_revenue + pre_orders + C(stratum)",
                      seg.to_pandas()).fit(cov_type="HC2")
        rows.append({"segment": label, "metric": metric, "effect": round(fit.params["treated"], 3),
                     "se": round(fit.bse["treated"], 3), "p": fit.pvalues["treated"]})
seg_table = pl.DataFrame(rows)
reject, p_holm, _, _ = multipletests(seg_table["p"].to_numpy(), method="holm")
seg_table = seg_table.with_columns(p=pl.Series([fmt(v) for v in seg_table["p"]]),
                                   p_holm=pl.Series([fmt(v) for v in p_holm]), significant=pl.Series(reject))
print(seg_table)
print(pl.DataFrame([guardrail(df.filter(pl.col("loyalty") == f), lab)
                    for f, lab in [(0, "non-members"), (1, "loyalty members")]]))
```

```bash
python -m day30.readout
```

**Part 1 and 2:**

```
1. All customers (CUPED-adjusted)
│ metric    ┆ control_mean ┆ effect ┆ se    ┆ ci_low ┆ ci_high ┆ lift   ┆ p_value │
│ revenue   ┆ 69.967       ┆ 5.365  ┆ 0.867 ┆ 3.665  ┆ 7.065   ┆ +7.67% ┆ <0.0001 │
│ orders    ┆ 1.151        ┆ 0.079  ┆ 0.012 ┆ 0.055  ┆ 0.103   ┆ +6.90% ┆ <0.0001 │
│ converted ┆ 0.603        ┆ 0.028  ┆ 0.005 ┆ 0.019  ┆ 0.037   ┆ +4.63% ┆ <0.0001 │
│ margin    ┆ 20.99        ┆ -0.395 ┆ 0.256 ┆ -0.897 ┆ 0.107   ┆ -1.88% ┆ 0.1228  │

2. Margin guardrail: non-inferiority, margin -1.00 per customer
│ population ┆ margin_effect ┆ lower_bound_95 ┆ p_noninferiority ┆ passes │
│ everyone   ┆ -0.395        ┆ -0.816         ┆ 0.0091           ┆ true   │
```

For all customers: revenue up **$5.37 (+7.7%)**, clearly significant. Margin down $0.40, not significantly different from zero, and the guardrail **passes**: we can rule out a loss worse than $0.82, inside Finance's $1 limit. Following the overall rule, we'd launch for everyone.

But the director asked a third question.

---

## 30.5 Pre-registered segments (35 min)

**Part 3:**

```
3. Pre-registered segments (Holm correction across the 4 segment tests)
│ segment         ┆ metric  ┆ effect ┆ se    ┆ p       ┆ p_holm  ┆ significant │
│ non-members     ┆ revenue ┆ 6.975  ┆ 0.95  ┆ <0.0001 ┆ <0.0001 ┆ true        │
│ non-members     ┆ margin  ┆ 0.316  ┆ 0.28  ┆ 0.2605  ┆ 0.5210  ┆ false       │
│ loyalty members ┆ revenue ┆ 1.615  ┆ 1.856 ┆ 0.3840  ┆ 0.5210  ┆ false       │
│ loyalty members ┆ margin  ┆ -2.052 ┆ 0.547 ┆ 0.0002  ┆ 0.0005  ┆ true        │
│ population      ┆ margin_effect ┆ lower_bound_95 ┆ p_noninferiority ┆ passes │
│ non-members     ┆ 0.316         ┆ -0.146         ┆ <0.0001          ┆ true   │
│ loyalty members ┆ -2.052        ┆ -2.952         ┆ 0.9728           ┆ false  │
```

The overall result averages two very different stories:

- **Non-members:** revenue +$6.98, significant; margin unchanged (+$0.32, guardrail passes easily). The lower threshold brings in orders that wouldn't have happened, enough to pay for the shipping.
- **Loyalty members:** revenue +$1.62, not significant; margin **−$2.05**, significantly negative even after correction, and the guardrail **fails**. Members already buy regularly; free shipping mostly subsidises orders they'd have placed anyway.

This is a legitimate segment finding, unlike Day 26's "best region": the segments were **pre-registered**, there are only two, the tests are Holm-corrected, and the planned power analysis said members would be the hard case. The margin loss for members is large enough to be detected even so.

---

## 30.6 The decision memo (50 min)

Write `docs/FREE_SHIPPING_DECISION.md`: one page, for the director. Headings:

1. **Recommendation** (two sentences).
2. **What we tested** (one paragraph: design, size, duration, pre-registration).
3. **Results** (a small table: revenue and margin, overall and by segment, with intervals).
4. **Why not launch for everyone?** The overall guardrail passed. Explain why the segment result should still change the decision, and why it isn't fishing.
5. **Risks and next steps** (for example: the effect after four weeks may fade or grow; members excluded from free shipping may notice; re-test members with a different perk).

<details>
<summary>Model answer for sections 1 and 4</summary>

> **Recommendation.** Lower the free-shipping threshold to $35 **for customers who are not loyalty members**: it raised their revenue by about $7 per customer per month with no detectable loss of margin. Keep the $50 threshold for loyalty members, for whom it cost about $2 of margin per customer with no significant revenue gain.
>
> **Why not launch for everyone?** Across all customers, the margin loss ($0.40) was within Finance's $1 limit, so the overall rule would say "launch". But that average hides a clear split we planned to check: members lose $2 of margin each, twice the acceptable limit. Launching for non-members only keeps nearly all of the revenue gain ($6.98 × 70% of customers ≈ $4.88 per customer overall, against $5.37 for everyone) and turns a margin loss into a small gain. Because this split was written into the plan before the test, tested with a correction for multiple comparisons, and is large relative to its uncertainty, it's a finding, not a fishing expedition.

</details>

---

## 30.7 Answer key, final commit and Module 3 retrospective (20 min)

After the memo is committed, compare with the truth. Create `day30/answer_key.py`:

```python
# day30/answer_key.py   (python -m day30.answer_key)
"""Instructor check, after the memo is written: the true effects, overall and by segment."""
import numpy as np
import polars as pl
from scipy.stats import norm

from causalkit import lab

pop = lab._population(2030)
rate, basket = pop["_rate"].to_numpy(), pop["_basket"].to_numpy()
loyal = pop["loyalty"].to_numpy().astype(bool)
lift = np.where(loyal, *lab.EXPERIMENTS["free_shipping"]["lift"][::-1])

mu0 = rate * lab.TEST_WEEKS
mu1 = mu0 * (1 + lift)
share_35_50 = (norm.cdf((np.log(lab.OLD_THRESHOLD / basket) + 0.045) / 0.3)
               - norm.cdf((np.log(lab.NEW_THRESHOLD / basket) + 0.045) / 0.3))
d_revenue = (mu1 - mu0) * basket
d_margin = lab.GROSS_MARGIN * d_revenue - lab.SHIPPING_FEE * share_35_50 * mu1

rows = []
for label, sel in [("everyone", np.ones_like(loyal)), ("non-members", ~loyal), ("loyalty members", loyal)]:
    rows.append({"population": label, "true revenue effect": round(d_revenue[sel].mean(), 2),
                 "true margin effect": round(d_margin[sel].mean(), 2),
                 "true order lift": f"{lift[sel].mean():.0%}"})
print(pl.DataFrame(rows))
```

```bash
python -m day30.answer_key
```

```
│ population      ┆ true revenue effect ┆ true margin effect ┆ true order lift │
│ everyone        ┆ 4.98                ┆ -0.52              ┆ 8%              │
│ non-members     ┆ 5.9                 ┆ 0.02               ┆ 10%             │
│ loyalty members ┆ 2.83                ┆ -1.79              ┆ 3%              │
```

Every estimate's confidence interval contains its true value, and the segment decision is the right one: free shipping is break-even on margin for non-members (+$0.02) and loses $1.79 per member.

```bash
git add .
git commit -m "Day 30: free-shipping capstone: readout and decision"
git push
```

**What you learned in Module 3.**

| Day | Idea | The lab lesson |
|---|---|---|
| 21 | Plans, hashing, A/A tests, SRM | A 10% logging bug inflated the lift by 65%; SRM caught it |
| 22 | Power | 5% revenue lift needs ~25,000 customers; a $0.50 margin loss needs 110,000 |
| 23 | Stratification | 11% less noise for free |
| 24 | Regression readout | Covariates cut noise 13%; the estimate's target doesn't move |
| 25 | CUPED, capping | CUPED is worth 32% more customers; capping changes the estimand |
| 26 | Multiple tests, peeking | Daily peeking: 29% false positives; the "best region" overstates by 74% |
| 27 | Sequential tests | O'Brien–Fleming saves a third of test time at 5% error |
| 28 | Interference | Household spillover hid 35% of the effect; cluster randomisation fixes it |
| 29 | Switchbacks, geo tests | Before/after was 3× the truth; pre-period growth cut geo noise 17-fold |
| 30 | Capstone | Overall "launch" would have cost $2 per loyalty member; the pre-registered segments caught it |

**Coming next: Module 4, Selection on Observables & Instrumental Variables.** Experiments are the benchmark, but most business decisions were never randomised. Module 4 returns to observational data with propensity scores, matching, weighting, doubly robust estimators and instrumental variables, and grades each one against the truth.

### Day 30 checklist

- [ ] I pre-registered a complete plan and committed it before running the test
- [ ] I justified the design with power analysis, overall and by segment
- [ ] I tested a guardrail with non-inferiority instead of a significance test
- [ ] I analysed pre-registered segments with a multiple-testing correction
- [ ] I wrote a decision memo whose recommendation follows from the plan
