# Module 8 · Decision Intelligence Delivery
## Days 77–80: Monitoring, Re-Running Experiments and the Final Capstone

**Goal by end of Day 80:** Students can detect when causal estimates have gone stale, put a dollar value on re-estimating or re-testing, design the re-test, and carry a full decision cycle (monitor, rebuild, simulate, recommend, defend) on new data.

---

# DAY 77 — Monitoring Estimates over Time

| Time | Block |
|---|---|
| 0:00 – 0:40 | 77.1 Causal estimates go stale |
| 0:40 – 1:30 | 77.2 A rolling re-estimate |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 77.3 Alert rules: sensitivity vs false alarms |
| 2:30 – 3:10 | 77.4 What a monitor can't see |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 77.5 Day 77 mini-project: CUSUM |
| 4:40 – 5:00 | 77.6 Save your work with Git |

---

## 77.1 Causal estimates go stale (40 min)

An elasticity is a fact about customers at a time. Competitors open stores, incomes change, a segment's mix shifts. Prediction models are monitored for **data drift** (inputs change) and **performance drift** (accuracy falls). Causal parameters need a third kind: **effect drift**. The same price change now does something different, and nothing in the inputs warns you.

You can't monitor an effect by comparing predictions with outcomes week by week: that confuses effects with everything else that moves sales. You monitor it the way you estimated it: **re-run the identification strategy on recent data**, and compare.

The weekly feed after the simulator went live. Create `data/make_monitoring_data.py`:

```python
"""
Module 8 monitoring data: the weekly feed after the simulator went live. Provided.

    python data/make_monitoring_data.py

Writes data/scenario/weekly_feed.csv: 130 weeks of store-category sales. Something changes partway through;
the monitoring on Day 77 should find what and when, without being told.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.demand import demand, public  # noqa: E402

root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
public(demand(seed=77, weeks=130, drift_week=105, drift_segment="student", drift_factor=1.3)).write_csv(
    root / "scenario" / "weekly_feed.csv")
print("Wrote data/scenario/weekly_feed.csv (130 weeks)")
```

```bash
python data/make_monitoring_data.py
```

```
Wrote data/scenario/weekly_feed.csv (130 weeks)
```

Something changes partway through. Today's job is to find what and when, without looking at the generator's arguments.

---

## 77.2 A rolling re-estimate (50 min)

Every two weeks, re-estimate each segment's elasticity on the last 13 weeks, with Module 6's 2SLS (supplier cost as the instrument). 2SLS rather than DML: it's fast, it has an honest standard error, and in a short window flexible ML has too little data. Create `day77/monitor.py`:

```python
# day77/monitor.py   (python -m day77.monitor)
"""
Monitoring causal estimates: re-estimate each segment's elasticity on a rolling 13-week window of the
weekly feed (2SLS with the supplier-cost instrument, Module 6), and compare it with what the simulator
is using. Alert when the estimate falls outside the reference value +/- 3 standard errors in three windows
in a row (six weeks): single excursions happen, because neighbouring windows share most of their data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

from engine import load_parameters

WINDOW, STEP = 13, 2


def rolling_estimates(feed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for end in range(WINDOW, feed.week.max() + 1, STEP):
        win = feed[(feed.week > end - WINDOW) & (feed.week <= end)]
        for seg, s in win.groupby("segment"):
            fit = pf.feols("log_units ~ promo | store_id^category + week | log_price ~ cost_index", s,
                           vcov={"CRV1": "store_id"})
            rows.append({"segment": seg, "end_week": end, "estimate": fit.coef()["log_price"],
                         "se": fit.se()["log_price"]})
    return pd.DataFrame(rows)


def reference(params: dict) -> dict:
    """The simulator's view of each segment: the average of its cells' elasticities."""
    cells = pd.DataFrame(params["pricing"])
    return cells.groupby("segment").elasticity.mean().to_dict()


def alerts(est: pd.DataFrame, ref: dict) -> pd.DataFrame:
    est = est.assign(reference=est.segment.map(ref))
    est["outside"] = (est.estimate - est.reference).abs() > 3 * est.se
    est["alert"] = est.groupby("segment").outside.transform(
        lambda s: s & s.shift(1, fill_value=False) & s.shift(2, fill_value=False))
    return est


if __name__ == "__main__":
    feed = pd.read_csv("data/scenario/weekly_feed.csv")
    est = alerts(rolling_estimates(feed), reference(load_parameters()))
    est.to_csv("data/day77_rolling.csv", index=False)
    first = est[est.alert].groupby("segment").end_week.min()
    print("Typical standard error of a 13-week segment estimate:", f"{est.se.median():.2f}")
    summary = est.groupby("segment").agg(reference=("reference", "first"), windows=("outside", "size"),
                                         windows_outside=("outside", "sum")).reset_index()
    summary["first_alert_week"] = summary.segment.map(first).astype("Int64")
    print(pl.from_pandas(summary).with_columns(pl.col("reference").round(2)))
    late = est[est.end_week >= 120].groupby("segment").estimate.mean().round(2)
    print("Average estimate in windows ending week 120 or later:", late.to_dict())

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, (seg, s) in zip(axes.ravel(), est.groupby("segment")):
        ax.fill_between(s.end_week, s.reference - 3 * s.se, s.reference + 3 * s.se, color="#cbd5e1", alpha=0.6)
        ax.plot(s.end_week, s.estimate, color="#2563eb", marker="o", ms=3)
        ax.plot(s.end_week[s.alert], s.estimate[s.alert], "o", color="#dc2626", ms=6)
        ax.axhline(s.reference.iloc[0], color="#334155", lw=1)
        ax.set_title(seg)
    for ax in axes[1]:
        ax.set_xlabel("last week of the 13-week window")
    fig.tight_layout()
    fig.savefig("figures/day77_monitor.png", dpi=120)
    print("Saved figures/day77_monitor.png")
```

```bash
python -m day77.monitor
```

```
Typical standard error of a 13-week segment estimate: 0.21
│ segment         ┆ reference ┆ windows ┆ windows_outside ┆ first_alert_week │
│ rural value     ┆ -3.15     ┆ 59      ┆ 2               ┆ null             │
│ student         ┆ -3.53     ┆ 59      ┆ 7               ┆ 121              │
│ suburban family ┆ -2.54     ┆ 59      ┆ 3               ┆ null             │
│ urban premium   ┆ -1.38     ┆ 59      ┆ 2               ┆ null             │
Average estimate in windows ending week 120 or later: {'rural value': -3.78, 'student': -5.19, 'suburban family': -2.8, 'urban premium': -1.31}
Saved figures/day77_monitor.png
```

Open the figure. Student stores drift below the band from about week 110 and stay there: their elasticity went from about −3.5 to about −5.2, roughly 1.3 to 1.5 times as price-sensitive. The first alert comes at week 121. (The simulator's truth: ×1.3 from week 105. The monitor needed about 16 weeks to be sure.)

---

## 77.3 Alert rules: sensitivity vs false alarms (45 min)

Every alert rule trades detection speed for false alarms:

| Rule | Detects drift | False alarms |
|---|---|---|
| one window outside ±2 SE | fast | often: 59 windows × 4 segments × 5% |
| one window outside ±3 SE | slower | still several; rural and suburban both had windows outside |
| **three consecutive windows outside ±3 SE** | slower still | rare: neighbouring windows share data, so one noisy stretch isn't enough |

Rural value's late average (−3.78 vs −3.15) looks alarming in the table, but only 2 of its 59 windows were outside the band and never three in a row: it's noise. The alert rule is the thing to agree with the business **before** the first alert, so nobody argues about it under pressure.

**Try it.** Change the window to 8 and to 26 weeks (set `day77.monitor.WINDOW` before calling the functions). What happens to the standard errors and the alerts? Create `day77/try_window.py`.

<details>
<summary>Solution</summary>

```python
# day77/try_window.py   (python -m day77.try_window)
import pandas as pd

import day77.monitor as mon
from engine import load_parameters

feed = pd.read_csv("data/scenario/weekly_feed.csv")
for w in [8, 13, 26]:
    mon.WINDOW = w
    est = mon.alerts(mon.rolling_estimates(feed), mon.reference(load_parameters()))
    first = est[est.alert].groupby("segment").end_week.min().to_dict()
    print(f"window {w:2} weeks: median SE {est.se.median():.2f}, first alerts {first}")
```

```
window  8 weeks: median SE 0.27, first alerts {'student': 120}
window 13 weeks: median SE 0.21, first alerts {'student': 121}
window 26 weeks: median SE 0.15, first alerts {'rural value': 40, 'student': 54, 'suburban family': 40, 'urban premium': 38}
```

A longer window means smaller standard errors, and suddenly every segment alerts, from week 38. That isn't drift: the reference values were estimated by **DML on a different year**, and differ from 2SLS on this feed by a few hundredths to a tenth. With SE 0.15 a gap of 0.45 is "significant". **A monitor compares two estimates, and both have error.** Fixes: compare 2SLS with 2SLS on the same feed's first year as the reference, or add the reference's own SE to the band (√(SE₁² + SE₂²)). The 13-week window works here partly because its noise hides the method gap: don't rely on that.

</details>

---

## 77.4 What a monitor can't see (40 min)

- **Drift in the instrument.** If supplier costs stop moving prices (a long-term contract), 2SLS loses its first stage and the monitor goes blind. Track the first-stage F alongside the estimate.
- **Drift inside a segment.** A segment average can stay put while two categories move in opposite directions. Monitor cells when you have the data for it; here cells are too noisy at 13 weeks.
- **Drift in coupons.** The coupon curve came from one experiment. Nothing in weekly sales data re-identifies it; only a holdout (Module 7, Day 68) does. Keep a 5% holdout and monitor its uplift instead.

---

## 77.5 Day 77 mini-project: CUSUM (75 min)

Rolling windows overlap, so their alerts are correlated and hard to calibrate. Page's CUSUM (1954) uses non-overlapping blocks and accumulates evidence: S = max(0, S + z − k), alarm when S > h. A persistent shift of size k or more builds up; noise drains away. Create `day77/cusum.py`:

```python
# day77/cusum.py   (python -m day77.cusum)
"""
Mini-project: a CUSUM chart (Page, 1954) for elasticities. Estimate each segment's elasticity on
NON-overlapping 4-week blocks (independent, but noisy), standardise the gap from the reference, and add
it up: S = max(0, S_previous + z - k). A small persistent shift accumulates; noise doesn't. Alert at S > h.
Here z measures 'more price-sensitive than the reference' (the direction a competitor would push).
"""
import pandas as pd
import polars as pl
import pyfixest as pf

from day77.monitor import reference
from engine import load_parameters

BLOCK, K, H = 4, 0.5, 5.0
feed = pd.read_csv("data/scenario/weekly_feed.csv")
ref = reference(load_parameters())
rows = []
for seg, s in feed.groupby("segment"):
    cusum, alarm = 0.0, None
    for start in range(1, feed.week.max() + 1, BLOCK):
        block = s[(s.week >= start) & (s.week < start + BLOCK)]
        fit = pf.feols("log_units ~ promo | store_id^category + week | log_price ~ cost_index", block,
                       vcov={"CRV1": "store_id"})
        z = (ref[seg] - fit.coef()["log_price"]) / fit.se()["log_price"]       # > 0: more elastic than reference
        cusum = max(0.0, cusum + z - K)
        if alarm is None and cusum > H:
            alarm = start + BLOCK - 1
    rows.append({"segment": seg, "alarm at week": alarm, "final CUSUM": round(cusum, 1)})
print(pl.DataFrame(rows))
print(f"Blocks of {BLOCK} weeks, k = {K}, h = {H}.")
```

```bash
python -m day77.cusum
```

```
│ segment         ┆ alarm at week ┆ final CUSUM │
│ rural value     ┆ null          ┆ 3.5         │
│ student         ┆ 116           ┆ 13.8        │
│ suburban family ┆ null          ┆ 0.5         │
│ urban premium   ┆ null          ┆ 0.0         │
Blocks of 4 weeks, k = 0.5, h = 5.0.
```

Student alarms at week 116, five weeks before the rolling monitor, with no false alarms elsewhere. Rural's 3.5 is worth watching. CUSUM is designed for exactly this: small, persistent, one-directional shifts. Use it as the primary alarm and the rolling chart as the picture you show.

---

## 77.6 Save your work with Git (20 min)

```bash
git add data/make_monitoring_data.py day77
git commit -m "Day 77: rolling 2SLS monitor and CUSUM for elasticity drift"
git push
```

### Day 77 checklist

- [ ] I monitor effects by re-running the identification strategy, not by tracking forecasts
- [ ] I can explain my alert rule's trade-off between speed and false alarms
- [ ] I know that a monitor compares two noisy estimates
- [ ] CUSUM found the drift earlier than the rolling window, without false alarms


---

# DAY 78 — When to Re-Run the Experiment

| Time | Block |
|---|---|
| 0:00 – 0:40 | 78.1 Rebuild, re-test, or carry on |
| 0:40 – 1:30 | 78.2 What stale estimates cost |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 78.3 Designing the re-test |
| 2:30 – 3:10 | 78.4 Why size beats length |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 78.5 Day 78 mini-project: value of information and re-run rules |
| 4:40 – 5:00 | 78.6 Save your work with Git |

---

## 78.1 Rebuild, re-test, or carry on (40 min)

When the monitor fires, there are three responses, in increasing cost:

| Response | What it is | Cost | When it's enough |
|---|---|---|---|
| **Carry on** | keep the parameters, note the alert | nothing | the drift doesn't change any decision |
| **Rebuild** | re-run `engine.build` on recent data | an afternoon | the identification strategy still works on recent data |
| **Re-test** | a new randomised price test | weeks of deliberately "wrong" prices | observational estimates are too noisy, or their assumptions are in doubt |

Module 7's value-of-information logic (Day 68) decides between them: do it if the expected improvement in decisions is worth more than it costs.

The `demand()` simulator can now run a price test: `price_test={"segment": "student", "start": 105, "weeks": 8, "change": 0.10}` raises prices by 10% in a random half of student stores for 8 weeks, and adds a `price_test` column.

---

## 78.2 What stale estimates cost (50 min)

Suppose every student elasticity is now 1.3 times what the parameters file says (Day 77's finding). For each student cell, compare a +5% decision under the stale and the updated values. Create `day78/stale_cost.py`:

```python
# day78/stale_cost.py   (python -m day78.stale_cost)
"""
What does acting on stale estimates cost? The monitoring says students became more price-sensitive.
Suppose (as the simulator knows) every student elasticity is now 1.3 times what the parameters file says.
For each student cell, compare the +5% price decision under the stale and the updated elasticities.
"""
import copy

import polars as pl

from engine import Scenario, load_parameters, simulate

stale = load_parameters()
updated = copy.deepcopy(stale)
for c in updated["pricing"]:
    if c["segment"] == "student":
        c["elasticity"] = round(c["elasticity"] * 1.3, 3)

rows = []
for c in [c for c in stale["pricing"] if c["segment"] == "student"]:
    key = f"student/{c['category']}"
    sc = Scenario(price_change={key: 0.05}, seed=1)
    old = next(x for x in simulate(stale, sc)["cells"] if (x["segment"], x["category"]) == ("student", c["category"]))
    new = next(x for x in simulate(updated, sc)["cells"] if (x["segment"], x["category"]) == ("student", c["category"]))
    rows.append({"cell": key, "stale elasticity": c["elasticity"], "updated": round(c["elasticity"] * 1.3, 2),
                 "+5% profit, stale view ($k/yr)": round(old["profit_change_median"] / 1e3, 1),
                 "+5% profit, updated ($k/yr)": round(new["profit_change_median"] / 1e3, 1),
                 "decision flips?": "YES" if (old["profit_change_median"] > 0) != (new["profit_change_median"] > 0) else ""})
with pl.Config(tbl_width_chars=140, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
plan = Scenario(price_change={"student/pantry": 0.05})
print(f"Day 60's plan raises student pantry prices. Stale view: "
      f"{simulate(stale, plan)['change']['gross_profit']['median'] / 1e3:+.1f} $k/yr; "
      f"updated: {simulate(updated, plan)['change']['gross_profit']['median'] / 1e3:+.1f} $k/yr")
```

```bash
python -m day78.stale_cost
```

```
│ cell                  ┆ stale elasticity ┆ updated ┆ +5% profit, stale view ($k/yr) ┆ +5% profit, updated ($k/yr) ┆ decision flips? │
│ student/fresh         ┆ -3.522           ┆ -4.58   ┆ -3.5                           ┆ -26.1                       ┆                 │
│ student/household     ┆ -3.811           ┆ -4.95   ┆ -36.4                          ┆ -97.4                       ┆                 │
│ student/pantry        ┆ -3.114           ┆ -4.05   ┆ 18.0                           ┆ 2.5                         ┆                 │
│ student/personal care ┆ -2.828           ┆ -3.68   ┆ -5.5                           ┆ -57.2                       ┆                 │
│ student/snacks        ┆ -4.37            ┆ -5.68   ┆ -23.3                          ┆ -47.1                       ┆                 │
Day 60's plan raises student pantry prices. Stale view: +18.0 $k/yr; updated: +2.4 $k/yr
```

No decision flips: pantry still gains, the others still lose, just by more. The plan's value falls by 87%, but its sign holds. **Drift matters for decisions only when it moves a cell across break-even.** That's why "the monitor fired" isn't by itself a reason for an expensive re-test.

---

## 78.3 Designing the re-test (45 min)

**Concept.** A randomised price test identifies the elasticity with no assumptions about the pricing team: the random assignment is the instrument (Module 4). The design choices are the size of the change, the length and the stores. Simulate before you run it. Create `day78/test_design.py`:

```python
# day78/test_design.py   (python -m day78.test_design)
"""
Designing the re-test: raise prices in a random half of student stores for k weeks, and estimate the
student elasticity from the test alone (the random price change is the instrument: no assumptions about
the pricing team needed). The window includes the 8 weeks before the test, so each store is compared with
itself; cost_index soaks up supplier-cost noise in prices. Which design is precise enough?
Simulated 12 times per design; every design sees the same 12 markets (common random numbers),
so the differences between rows come from the design, not from luck.
"""
import numpy as np
import polars as pl
import pyfixest as pf

from causalkit.demand import demand

rows = []
for change in [0.05, 0.10]:
    for weeks in [4, 8, 13]:
        est, ses, fs_t, truth = [], [], [], []
        for s in range(12):
            df = demand(seed=7800 + s, weeks=117, drift_week=95, drift_segment="student", drift_factor=1.3,
                        price_test={"segment": "student", "start": 105, "weeks": weeks, "change": change})
            win = df.filter((pl.col("segment") == "student") & pl.col("week").is_between(97, 104 + weeks)).to_pandas()
            fe = "store_id^category + category^week"
            fs = pf.feols(f"log_price ~ price_test + cost_index + promo | {fe}", win, vcov={"CRV1": "store_id"})
            fit = pf.feols(f"log_units ~ promo + cost_index | {fe} | log_price ~ price_test", win,
                           vcov={"CRV1": "store_id"})
            fs_t.append(fs.tstat()["price_test"])
            est.append(fit.coef()["log_price"])
            ses.append(fit.se()["log_price"])
            truth.append(win._elasticity.mean())
        err = np.array(est) - np.array(truth)
        rows.append({"price change": f"{change:.0%}", "weeks": weeks, "first-stage t": round(np.median(fs_t), 1),
                     "average error": round(err.mean(), 2), "SD of estimates": round(err.std(), 2),
                     "median SE": round(np.median(ses), 2)})
print(pl.DataFrame(rows))
print("true student elasticity (average):", round(np.mean(truth), 2))
```

```bash
python -m day78.test_design
```

```
│ price change ┆ weeks ┆ first-stage t ┆ average error ┆ SD of estimates ┆ median SE │
│ 5%           ┆ 4     ┆ 2.0           ┆ -0.26         ┆ 0.93            ┆ 0.7       │
│ 5%           ┆ 8     ┆ 3.4           ┆ 0.05          ┆ 0.27            ┆ 0.53      │
│ 5%           ┆ 13    ┆ 2.8           ┆ -0.7          ┆ 1.11            ┆ 0.76      │
│ 10%          ┆ 4     ┆ 4.5           ┆ 0.03          ┆ 0.3             ┆ 0.34      │
│ 10%          ┆ 8     ┆ 4.9           ┆ -0.02         ┆ 0.32            ┆ 0.36      │
│ 10%          ┆ 13    ┆ 5.4           ┆ -0.14         ┆ 0.35            ┆ 0.31      │
true student elasticity (average): -4.73
```

(The analysis window starts 8 weeks before the test so each store is compared with itself. A first version that used only the test weeks failed with a singular matrix: with store × category and category × week fixed effects, a treatment that never varies within a store in the window is absorbed completely.)

---

## 78.4 Why size beats length (40 min)

Read the table by the first-stage t-statistic (Module 4's weak-instrument diagnostic):

- At 5%, the test barely moves prices relative to their normal week-to-week noise: t of 2 to 3.4, a weak instrument. Estimates are erratic (SD up to 1.1) and **longer doesn't help**: 13 weeks is worse than 8 in these 12 markets.
- At 10%, t is 4.5 or more at every length and the estimates are tight (SD about 0.3) from 4 weeks on.

Length adds weeks from the **same 20 stores**, and with errors clustered by store, the effective sample is the number of stores, not store-weeks. Doubling the price change doubles the signal in every one of them. Recommend: 10% for 4–8 weeks. (Cost: Day 78.5.)

---

## 78.5 Day 78 mini-project: value of information and re-run rules (75 min)

Is the test worth running? Our belief after Day 77: student elasticities are k times the file, with k ~ Normal(1.3, 0.15). Compare acting now with acting after perfect information (EVPI, an upper bound) and after the realistic test (SE 0.35 from 78.3), against what the test costs. Create `day78/rerun_rules.py`:

```python
# day78/rerun_rules.py   (python -m day78.rerun_rules)   — Day 78 mini-project
"""
Is a new student price test worth running? Value of information, in dollars.

Decision: raise each student cell's price 5%, or leave it. Our belief about the drift (from Day 77's
monitor): student elasticities are now k times the parameters file, with k ~ Normal(1.3, 0.15).
  acting now     = pick, per cell, the action with the best EXPECTED profit under that belief
  perfect info   = learn k first, then pick the best action for that k
  EVPI           = E[perfect info] - acting now   (an upper bound on what any test is worth)
A real 8-week, 10% test does not reveal k exactly: Day 78's design gives SE about 0.35 on the elasticity,
so we also compute the value of THAT (imperfect) information. Then compare with what the test costs.
"""
import numpy as np

from engine import load_parameters

params = load_parameters()
cells = [c for c in params["pricing"] if c["segment"] == "student"]
HORIZON_YEARS = 1.0


def profit_change(c, d, k):
    """Annual gross-profit change of price change d in cell c if the elasticity is k * estimate."""
    e = c["elasticity"] * k
    base = c["margin"] * c["price"] * c["weekly_units"] * 52
    return (c["margin"] + d) * c["price"] * c["weekly_units"] * (1 + d) ** e * 52 - base


if __name__ == "__main__":
    rng = np.random.default_rng(78)
    k = rng.normal(1.3, 0.15, 20_000)             # belief about the drift factor
    now = perfect = sample = 0.0
    for c in cells:
        gain = profit_change(c, 0.05, k)              # profit of raising vs leaving, one value per belief draw
        now += max(gain.mean(), 0)                    # act on the expected value
        perfect += np.maximum(gain, 0).mean()         # act knowing k
        # imperfect test: observe the cell elasticity with noise SE 0.35, update by Bayes on a grid of k
        e_hat = c["elasticity"] * k + rng.normal(0, 0.35, k.size)
        grid = np.linspace(0.7, 1.9, 241)
        prior = np.exp(-0.5 * ((grid - 1.3) / 0.15) ** 2)
        like = np.exp(-0.5 * ((e_hat[:, None] - c["elasticity"] * grid[None, :]) / 0.35) ** 2)
        post = like * prior
        post /= post.sum(axis=1, keepdims=True)
        exp_gain = post @ profit_change(c, 0.05, grid)    # expected gain after seeing the test
        sample += np.where(exp_gain > 0, gain, 0).mean()

    test_cost = 15_000 + sum(                         # fixed cost + profit lost while half the student stores pay +10%
        max(-profit_change(c, 0.10, 1.3), 0) / 52 * 8 / 2 for c in cells)
    print(f"Expected profit change, acting now:        {now / 1e3:8.1f} $k/yr")
    print(f"With perfect information:                  {perfect / 1e3:8.1f} $k/yr   -> EVPI {(perfect - now) / 1e3:.1f} $k/yr")
    print(f"With an 8-week 10% test (SE 0.35):         {sample / 1e3:8.1f} $k/yr   -> value {(sample - now) / 1e3:.1f} $k/yr")
    print(f"Cost of the test:                          {test_cost / 1e3:8.1f} $k")
    print("Run the test" if (sample - now) * HORIZON_YEARS > test_cost else "Do not run the test: act on current beliefs")

    print("""
    Re-run rules for NorthMart (agreed with the pricing team):
      1. The monitor alerts on a segment (3 consecutive windows outside +-3 SE) or CUSUM alarms.
      2. A decision recommended by the app would flip under the monitor's latest estimate.
      3. The value of a test (as above) exceeds its cost over the decision horizon.
      4. Anyway, every 12 months: assumptions age even when no alarm fires.
      Rule 1 or 4 -> rebuild models/causal_parameters.json on recent data; rule 2 + 3 -> run the experiment.""")
```

```bash
python -m day78.rerun_rules
```

```
Expected profit change, acting now:             2.5 $k/yr
With perfect information:                       4.6 $k/yr   -> EVPI 2.2 $k/yr
With an 8-week 10% test (SE 0.35):              3.9 $k/yr   -> value 1.4 $k/yr
Cost of the test:                              50.9 $k
Do not run the test: act on current beliefs
```

The test would cost about $51k (mostly profit lost while half the student stores charge 10% more) to improve a decision worth $2.5k a year. Even perfect information is worth $2.2k a year. **Don't test.** Rebuild on recent data instead (free), and raise student pantry only.

The script also prints NorthMart's four re-run rules. Adopt them in writing: when the monitor alerts or a year has passed, rebuild; when a decision would flip **and** the test is worth more than it costs, re-test.

**Try it.** How does EVPI depend on how unsure we are about the drift? Recompute it for belief SDs of 0.05, 0.15 and 0.30. Create `day78/try_voi.py` (reuse `cells` and `profit_change`).

<details>
<summary>Solution</summary>

```python
# day78/try_voi.py   (python -m day78.try_voi)
"""How does the value of perfect information change with how unsure we are about the drift?"""
import numpy as np

from day78.rerun_rules import cells, profit_change

rng = np.random.default_rng(1)
for sd in [0.05, 0.15, 0.30]:
    k = rng.normal(1.3, sd, 20_000)
    now = sum(max(profit_change(c, 0.05, k).mean(), 0) for c in cells)
    perfect = sum(np.maximum(profit_change(c, 0.05, k), 0).mean() for c in cells)
    print(f"belief SD {sd:.2f}: EVPI {(perfect - now) / 1e3:5.1f} $k/yr")
```

```
belief SD 0.05: EVPI   0.2 $k/yr
belief SD 0.15: EVPI   2.1 $k/yr
belief SD 0.30: EVPI  11.9 $k/yr
```

Doubling the uncertainty (0.15 to 0.30) multiplies EVPI by almost 6, because more of the belief now lies on the other side of break-even for student pantry. Information is worth most when you're both unsure **and** close to a decision boundary. Still not enough to pay for a $51k test on a $2.5k decision: the test would pay only for a bigger decision, such as a price change across all student stores and categories.

</details>

---

## 78.6 Save your work with Git (20 min)

```bash
git add day78 causalkit/demand.py
git commit -m "Day 78: cost of stale estimates, price-test design, value of information and re-run rules"
git push
```

### Day 78 checklist

- [ ] I distinguish carrying on, rebuilding and re-testing, and know what each costs
- [ ] I checked whether the drift flips any decision before recommending action
- [ ] I simulated the re-test's precision before proposing it, and read its first stage
- [ ] I compared the value of information with the test's cost
- [ ] NorthMart has written re-run rules


---

# DAY 79 — Final Capstone Workshop

| Time | Block |
|---|---|
| 0:00 – 0:30 | 79.1 The brief |
| 0:30 – 1:20 | 79.2 Step 1: what does the monitor say? |
| 1:20 – 1:35 | Break |
| 1:35 – 2:40 | 79.3 Step 2: rebuild the parameters |
| 2:40 – 3:30 | 79.4 Step 3: the scenario pack |
| 3:30 – 3:45 | Break |
| 3:45 – 4:40 | 79.5 Step 4: the recommendation and the defence deck |
| 4:40 – 5:00 | 79.6 Save your work with Git |

---

## 79.1 The brief (30 min)

> It's NorthMart's annual pricing review. The simulator has been live for 130 weeks on the parameters in `models/causal_parameters.json`. The board wants: (1) are the estimates still right? (2) which of the standard scenarios should we adopt this year, and what will it earn? (3) what should we do differently in how we run the simulator? You present tomorrow: 10 minutes, then questions.

You work from new data only. Generate it:

```python
"""
Final capstone data (Days 79-80). Provided: run it, then work from data/final/ only.

    python data/make_final_capstone.py

data/final/weekly_feed.csv        130 weeks of NorthMart demand, as the monitoring feed delivers it
data/final/coupon_experiment.csv  this year's randomised coupon test (40,000 customers)
data/truth/final_truth.json       the simulator's truth, for the instructor's answer key. Do not open it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.coupons import coupons  # noqa: E402
from causalkit.coupons import public as coupons_public  # noqa: E402
from causalkit.demand import CATEGORIES, SEGMENTS, demand, elasticity, public  # noqa: E402

DRIFT = {"segment": "suburban family", "week": 105, "factor": 1.25}
root = Path(__file__).parent
(root / "final").mkdir(exist_ok=True)
(root / "truth").mkdir(exist_ok=True)
df = demand(seed=80, weeks=130, drift_week=DRIFT["week"], drift_segment=DRIFT["segment"], drift_factor=DRIFT["factor"])
public(df).write_csv(root / "final" / "weekly_feed.csv")
coupons_public(coupons(n=40_000, seed=810, randomized=True)).write_csv(root / "final" / "coupon_experiment.csv")
truth = {"drift": DRIFT, "elasticity_now": {f"{s}/{c}": round(elasticity(s, c) * (DRIFT["factor"] if s == DRIFT["segment"] else 1), 3)
                                             for s in SEGMENTS for c in CATEGORIES}}
(root / "truth" / "final_truth.json").write_text(json.dumps(truth, indent=2))
print("Wrote data/final/weekly_feed.csv, data/final/coupon_experiment.csv and data/truth/final_truth.json")
```

```bash
python data/make_final_capstone.py
```

**Do not open `data/truth/`.** The instructor uses it tomorrow.

Rules, as for every capstone: every number in your deck comes from code in `day79/`, committed; you state each assumption you rely on; and you say what you didn't check.

---

## 79.2 Step 1: what does the monitor say? (50 min)

Run Day 77's monitor on this year's feed. Create `day79/check_feed.py`:

```python
# day79/check_feed.py   (python -m day79.check_feed)   — capstone step 1
"""Run Day 77's monitor on this year's feed against the parameters the app is serving. Which segment moved, and when?"""
import pandas as pd
import polars as pl

from day77.monitor import alerts, reference, rolling_estimates
from engine import load_parameters

feed = pd.read_csv("data/final/weekly_feed.csv")
est = alerts(rolling_estimates(feed), reference(load_parameters()))
summary = est.groupby("segment").agg(reference=("reference", "first"), windows_outside=("outside", "sum"),
                                     first_alert=("end_week", lambda w: w[est.loc[w.index, "alert"]].min())).reset_index()
late = est[est.end_week >= 120].groupby("segment").estimate.mean()
summary["late_estimate"] = summary.segment.map(late)
print(pl.from_pandas(summary).with_columns(pl.col("reference", "late_estimate").round(2)))
```

```bash
python -m day79.check_feed
```

```
│ segment         ┆ reference ┆ windows_outside ┆ first_alert ┆ late_estimate │
│ rural value     ┆ -3.15     ┆ 1               ┆ null        ┆ -3.03         │
│ student         ┆ -3.53     ┆ 3               ┆ null        ┆ -4.0          │
│ suburban family ┆ -2.54     ┆ 7               ┆ 119.0       ┆ -3.21         │
│ urban premium   ┆ -1.38     ┆ 9               ┆ 79.0        ┆ -1.51         │
```

Two segments alert. Decide for each whether it's material:

- **Suburban family**: from −2.54 to about −3.2, a quarter more price-sensitive, alerting at week 119. Material: several suburban cells were in the "raise" list.
- **Urban premium**: alerted at week 79, before anything else, but −1.38 against −1.51 is a 9% difference in the least elastic segment, and it was there from early on. Day 77's Try-it explains it: a tight standard error turns a small method gap into an "alert". Statistically real, economically small.

Run `day77.cusum`'s logic on this feed too, if you have time, and say whether it agrees.

---

## 79.3 Step 2: rebuild the parameters (65 min)

Rebuild on the weeks after the change, but think about what you lose: 26 weeks instead of 104 means larger standard errors everywhere. The sensible compromise is a **hybrid**: recent estimates for the segment that moved, long-sample estimates elsewhere. Create `day79/rebuild.py`:

```python
# day79/rebuild.py   (python -m day79.rebuild)   — capstone step 2
"""
Rebuild the simulator's parameters from this year's data.
1. Re-estimate every cell on weeks 105-130 only (the drift started around week 105: older weeks describe
   customers who no longer exist) -> models/final_recent_only.json. Prices and volumes are this year's.
2. 26 weeks is a short sample. For segments where the monitor found no material change, keep the
   long-sample elasticities (smaller SE); take the recent ones only for the segment that moved
   -> models/final_parameters.json, the file the app will serve.
"""
import copy

import pandas as pd

from engine import load_parameters, save_parameters
from engine.build import build

MOVED = ["suburban family"]     # from day79.check_feed: urban premium's alert is real but tiny (-1.38 -> -1.51)

feed = pd.read_csv("data/final/weekly_feed.csv")
feed[feed.week >= 105].to_csv("data/final/recent_demand.csv", index=False)
recent = build("data/final/recent_demand.csv", "data/final/coupon_experiment.csv", out="models/final_recent_only.json")
old = {(c["segment"], c["category"]): c for c in load_parameters()["pricing"]}

final = copy.deepcopy(recent)
final["version"] += "-hybrid"
final["sources"]["pricing_method"] += f"; recent weeks for {', '.join(MOVED)}, long sample elsewhere"
for c in final["pricing"]:
    if c["segment"] not in MOVED:
        c["elasticity"], c["se"] = old[(c["segment"], c["category"])]["elasticity"], old[(c["segment"], c["category"])]["se"]
save_parameters(final, "models/final_parameters.json")

print("version", final["version"], "| weeks", recent["sources"]["demand_weeks"])
print(f"{'cell':32}{'old':>8}{'recent':>8}{'SE':>7}{'final':>8}")
for c, r in zip(final["pricing"], recent["pricing"]):
    o = old[(c["segment"], c["category"])]
    print(f"{c['segment'] + '/' + c['category']:32}{o['elasticity']:8.2f}{r['elasticity']:8.2f}{r['se']:7.2f}{c['elasticity']:8.2f}")
print("coupon uplift at 50%:", final["coupons"]["uplift"][5], "(old", load_parameters()["coupons"]["uplift"][5], ")")
```

```bash
python -m day79.rebuild            # about 4 minutes: 20 DML fits and the coupon curve
```

```
version 2026-11-05-082c2977-hybrid | weeks [105, 130]
cell                                 old  recent     SE   final
rural value/fresh                  -3.05   -3.03   0.18   -3.05
...
suburban family/fresh              -2.45   -3.19   0.21   -3.19
suburban family/household          -2.78   -3.38   0.16   -3.38
suburban family/pantry             -2.26   -2.75   0.49   -2.75
suburban family/personal care      -2.03   -2.52   0.32   -2.52
suburban family/snacks             -3.15   -3.92   0.39   -3.92
...
urban premium/snacks               -1.69   -1.40   0.17   -1.69
coupon uplift at 50%: 14.019 (old 14.388 )
```

Every suburban cell moved by about a quarter. Elsewhere the recent estimates are within noise of the old ones, but with SEs two to four times larger, so keeping the long-sample values loses nothing and gains precision. The coupon curve from this year's experiment is close to last year's.

---

## 79.4 Step 3: the scenario pack (50 min)

Run the standard scenarios through the **API**, old and rebuilt parameters side by side, the way the board's planning team will. Create `day79/scenario_pack.py`:

```python
# day79/scenario_pack.py   (python -m day79.scenario_pack)   — capstone step 3
"""Run the preset menu through the API twice: old parameters and rebuilt ones. The pack for the memo."""
import os

import polars as pl
from fastapi.testclient import TestClient

import service.app as api
from engine.presets import PRESETS

body = [{"name": n, "scenario": s.model_dump()} for n, s in PRESETS.items() if n != "status quo"]
rows = {}
for label, path in [("old", "models/causal_parameters.json"), ("rebuilt", "models/final_parameters.json")]:
    os.environ["PARAMS_PATH"] = path
    api.params.cache_clear()
    with TestClient(api.app) as client:
        out = client.post("/compare", json=body).json()
        print(label, "parameters", out["parameters_version"])
        for r in out["rows"]:
            row = rows.setdefault(r["name"], {"scenario": r["name"]})
            row[f"{label} median $k"] = round(r["profit_change_median"] / 1e3)
            row[f"{label} 90% range"] = f"{r['p5'] / 1e3:,.0f} to {r['p95'] / 1e3:,.0f}"
            if label == "rebuilt":
                row["P(profit up)"] = f"{r['prob_profit_up']:.0%}"
                row["warnings"] = r["warnings"]
with pl.Config(tbl_width_chars=160, fmt_str_lengths=30):
    print(pl.DataFrame(list(rows.values())))
```

```bash
python -m day79.scenario_pack
```

```
old parameters 2026-11-05-e117a0ef
rebuilt parameters 2026-11-05-082c2977-hybrid
│ scenario                ┆ old median $k ┆ old 90% range    ┆ rebuilt median $k ┆ rebuilt 90% range ┆ P(profit up) ┆ warnings │
│ 5% everywhere           ┆ 459           ┆ 419 to 502       ┆ 431               ┆ 359 to 507        ┆ 100%         ┆ 0        │
│ 5% in the 'raise' cells ┆ 517           ┆ 489 to 544       ┆ 576               ┆ 518 to 632        ┆ 100%         ┆ 0        │
│ coupons to the top 50%  ┆ 5625          ┆ 4,824 to 6,361   ┆ 5171              ┆ 4,477 to 5,809    ┆ 100%         ┆ 0        │
│ 'raise' cells + coupons ┆ 6142          ┆ 5,334 to 6,877   ┆ 5745              ┆ 5,041 to 6,394    ┆ 100%         ┆ 0        │
│ coupons to everyone     ┆ -2827         ┆ -3,902 to -1,840 ┆ -3812             ┆ -4,828 to -2,879  ┆ 0%           ┆ 0        │
```

(The rebuilt figures also use this year's prices and volumes, which is why some move even though their elasticities didn't.) Then explore beyond the menu in the Streamlit app, pointed at the new file. The app reads the default path, so copy it over first and restore it afterwards, or set `PARAMS_PATH` and run with `API_URL`:

```bash
PARAMS_PATH=models/final_parameters.json uvicorn service.app:app --port 8001 &
API_URL=http://127.0.0.1:8001 streamlit run app/streamlit_app.py
```

Questions worth answering with the sliders: should the suburban cells stay in the "raise" list? What does the board lose if suburban prices are left alone? How much of the plan's value depends on the error-correlation slider?

---

## 79.5 Step 4: the recommendation and the defence deck (55 min)

Write `day79/RECOMMENDATION.md` (one page) and a 6–8 slide deck. Structure:

1. **The answer**: the scenario you recommend, its median and 90% range (at ρ = 0 and ρ = 0.5), and the probability it raises profit.
2. **What changed**: the suburban drift, when it was detected, and how you rebuilt.
3. **What didn't**: urban premium, and why you didn't act on its alert.
4. **What it rests on**: constant elasticity inside the supported range, no price-coupon interaction, 15% pull-forward.
5. **What we'll do differently**: the re-run rules (Day 78), monitoring with CUSUM, a 5% coupon holdout.
6. **The test we're not running, and why**: the value-of-information argument.

<details>
<summary>A model recommendation (read after writing yours)</summary>

> **Adopt the 'raise' cells with coupons to the top 50%: about +$5.7M gross profit a year (90% range $5.0M–$6.4M, almost unchanged at the cautious correlation setting, because the coupon programme dominates). Profit going up is virtually certain.** Suburban family shoppers became about 25% more price-sensitive from around week 105; the monitor flagged it at week 119 and we rebuilt their five elasticities on the 26 weeks since. Urban premium also triggered an alert, but the shift is 9% in our least price-sensitive segment and changes no decision; we kept its long-run estimates. With the new suburban values the three suburban cells in the plan still gain, by less ($7k, $29k and $17k a year); dropping them would cost only about $51k, so if the board prefers caution with a segment that is still changing, that is a cheap option. These numbers assume constant elasticity within ±10% price moves, no interaction between price and coupons, and that 15% of coupon spend is pulled forward. From this year we monitor with CUSUM, rebuild annually or on alert, and keep a 5% coupon holdout. A new price test isn't worth its cost for this decision.

</details>

---

## 79.6 Save your work with Git (20 min)

```bash
git add data/make_final_capstone.py day79 models/final_parameters.json models/final_recent_only.json
git commit -m "Day 79: final capstone: monitor, hybrid rebuild, scenario pack, recommendation"
git push
```

### Day 79 checklist

- [ ] I ran the monitor on new data and judged each alert for materiality, not just significance
- [ ] I rebuilt the parameters with a reasoned choice of window, and committed the new file
- [ ] My scenario pack compares old and rebuilt parameters through the API
- [ ] My recommendation leads with the decision, the money and the probability
- [ ] Every number in the deck comes from committed code


---

# DAY 80 — Final Capstone Defence

| Time | Block |
|---|---|
| 0:00 – 0:30 | 80.1 Format and rubric |
| 0:30 – 2:30 | 80.2 Defences (10 minutes + 10 minutes of questions each; breaks between) |
| 2:30 – 3:30 | 80.3 The answer key |
| 3:30 – 3:45 | Break |
| 3:45 – 4:30 | 80.4 Module 8 and course retrospective |
| 4:30 – 5:00 | 80.5 Final commit and what's next |

---

## 80.1 Format and rubric (30 min)

Each student (or pair) presents the Day 79 deck to a "board" of instructor and peers, then takes questions. The board may ask for any number to be reproduced live from the repository.

| Criterion | Weight | Excellent looks like |
|---|---|---|
| Decision first | 15% | first slide: what to do, how much it earns, how sure |
| Diagnosis | 20% | found the suburban drift; judged the urban alert as immaterial, with a reason |
| Re-estimation | 20% | rebuilt on post-change weeks; reasoned about precision lost; committed and versioned |
| Honest uncertainty | 20% | ranges in words; the correlated-error range; what the range leaves out |
| Engineering | 15% | numbers from committed code; tests pass; API and app use the same parameters file |
| Answers under questioning | 10% | concise, no new claims without evidence, "I don't know, here's how I'd find out" |

---

## 80.2 Defences (120 min)

Questions the board will ask (prepare for all of them):

1. Your range for "5% everywhere" is $359k to $507k. What would have to be true for the real result to be outside it?
2. Urban premium alerted first. Why didn't you act on it? What would make you change your mind?
3. You used 26 weeks for suburban family and 104 for everyone else. What if the other segments changed too, just less?
4. The coupon curve comes from one experiment. How do you know it still holds next year?
5. A board member says: "just set the price that maximises profit". What do you say?
6. If we deploy your recommendation, how will we know in six months whether it worked?
7. What's the single assumption that, if wrong, would change your recommendation?

Good answers to 6 name a design, not a hope: a holdout of stores at the old prices, or a staggered rollout analysed as a difference-in-differences (Module 5).

---

## 80.3 The answer key (60 min)

After the defences, compare every projection with the simulator's truth. The key applies the true post-drift elasticities to this year's prices and volumes, and puts them next to three projections: the old file, the recent-only rebuild and the hybrid. Create `day80/answer_key.py`:

```python
# day80/answer_key.py   (python -m day80.answer_key)   — instructor only
"""
After the defence: each price scenario's TRUE annual gross-profit change (simulator elasticities after the
drift, applied to this year's prices and volumes), next to what the old and the rebuilt parameters projected.
Coupons are left out: their truth did not drift, and Module 7's key already covers them.
"""
import json

import polars as pl

from engine import Scenario, load_parameters, simulate
from engine.presets import PRESETS

truth = json.load(open("data/truth/final_truth.json"))
old, new = load_parameters(), load_parameters("models/final_parameters.json")
recent = load_parameters("models/final_recent_only.json")
true_params = json.loads(json.dumps(new))
for c in true_params["pricing"]:
    c["elasticity"], c["se"] = truth["elasticity_now"][f"{c['segment']}/{c['category']}"], 0.0

scenarios = {n: s for n, s in PRESETS.items() if s.price_change and not s.coupon_share}
scenarios["5% outside suburban family"] = Scenario(price_change={"all": 0.05, "suburban family": 0.0})
scenarios["5% urban premium only"] = Scenario(price_change={"urban premium": 0.05})
def band(p, sc):
    g = simulate(p, sc)["change"]["gross_profit"]
    return g["p5"] / 1e3, g["p95"] / 1e3


rows = []
for name, sc in scenarios.items():
    sc = sc.model_copy(update={"coupon_share": 0.0})
    med = lambda p: simulate(p, sc)["change"]["gross_profit"]["median"] / 1e3   # noqa: E731
    rows.append({"scenario": name, "old projection $k": round(med(old)), "recent-only $k": round(med(recent)),
                 "final (hybrid) $k": round(med(new)), "final 90% range": "{:.0f} to {:.0f}".format(*band(new, sc)),
                 "TRUE $k": round(med(true_params))})
print(f"Drift: {truth['drift']}")
with pl.Config(tbl_width_chars=160, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
print("""Grade the defence on: noticing the suburban-family drift (and that the urban-premium alert is
statistically real but economically small), rebuilding on post-drift weeks only, keeping projections
inside valid_change, and stating the uncertainty in words the board can use. Note that TRUE can fall outside
the 90% range: the bands cover sampling noise, not the small attenuation bias every estimate shares.""")
```

```bash
python -m day80.answer_key
```

```
Drift: {'segment': 'suburban family', 'week': 105, 'factor': 1.25}
│ scenario                   ┆ old projection $k ┆ recent-only $k ┆ final (hybrid) $k ┆ final 90% range ┆ TRUE $k │
│ 5% everywhere              ┆ 459               ┆ 508            ┆ 431               ┆ 359 to 507      ┆ 341     │
│ 5% in the 'raise' cells    ┆ 517               ┆ 603            ┆ 576               ┆ 518 to 632      ┆ 546     │
│ 5% outside suburban family ┆ 324               ┆ 486            ┆ 411               ┆ 363 to 461      ┆ 356     │
│ 5% urban premium only      ┆ 356               ┆ 489            ┆ 467               ┆ 440 to 493      ┆ 459     │
```

What the key says:

- **The hybrid beats the recent-only rebuild everywhere.** Re-estimating every cell on 26 weeks overstated every price scenario, by $30k to $167k. Short samples didn't just add noise: they made every elasticity a little less negative, and in a near-break-even calculation that is worth a lot.
- **The recommended scenario holds.** The 'raise' cells earn $546k in truth; the hybrid said $576k with a range of $518k–$632k. The decision was right, and the number was close.
- **"5% everywhere" is the warning.** True $341k lies below the hybrid's range. Every estimate in the file shares a small attenuation bias (Day 73's correlated errors, made real). The range covers sampling noise, not shared bias, and a scenario that sums 20 cells near break-even magnifies it. Students who quoted the ρ = 0.5 range were closer to honest.
- **Coupons are not in the key**: their truth didn't drift, and Module 7's key covers them. The coupon programme is most of the plan's value, so the holdout matters more than any pricing refinement.

---

## 80.4 Module 8 and course retrospective (45 min)

**What you learned in Module 8.**

| Day | Idea | The simulator lesson |
|---|---|---|
| 71 | Serving causal models | Estimation once (minutes), simulation per request (10 ms); a versioned parameters file in between |
| 72 | Scenario design | Revenue fell and profit rose; the optimiser left the data in 13 of 20 cells |
| 73 | Communicating uncertainty | Rounded medians, ranges in words, calibrated probabilities; correlated errors made the range 4× wider |
| 74 | The front end | Inputs are decisions; the first screen is the counterfactual; scenarios are links |
| 75 | Counterfactual charts | Zero-based axes, grey counterfactual, ranges always; decompositions that add up |
| 76 | Testing and deployment | Tests of economic properties; CI; a container per parameters version; a smoke test |
| 77 | Monitoring | Rolling 2SLS found the student drift in 16 weeks, CUSUM in 11; a tight SE turns method gaps into alarms |
| 78 | Re-run decisions | The drift flipped no decision; a 10% test beats a 5% one; the test wasn't worth its $51k |
| 79–80 | Capstone | The hybrid rebuild got the decision right; shared bias is what ranges miss |

**The whole course, in one table.**

| Module | Question it answers | The habit it leaves |
|---|---|---|
| 1 Causal-ready data | Can I trust the data's history? | contracts, lineage, reproducible pipelines |
| 2 Causal reasoning | What exactly am I estimating, and what must I assume? | estimands first, DAGs before regressions |
| 3 Experiments | How do I learn by intervening? | power before launch, pre-registration, variance reduction |
| 4 Observational methods | What if I can't randomise? | overlap, doubly robust estimators, instruments with honest first stages |
| 5 Quasi-experiments | What if the world randomised for me? | parallel trends, synthetic controls, discontinuities, placebos |
| 6 Causal ML | What if the confounding is complicated, and effects differ? | cross-fitting, heterogeneity that's tested, not mined |
| 7 Decisions | What should we do, given what we don't know? | sensitivity, policy value, value of information |
| 8 Delivery | How do people use it, and when is it out of date? | versioned parameters, honest interfaces, monitoring and re-run rules |

The through-line: **every number is a comparison with a counterfactual, every counterfactual rests on an assumption, and every assumption has a price.** Your job is to make that chain visible to the people who decide.

---

## 80.5 Final commit and what's next (30 min)

```bash
python -m pytest -q
git add .
git commit -m "Day 80: final capstone defence and answer key. Course complete."
git tag v1.0
git push --tags
```

Your repository now contains a reusable causal toolkit (`causalkit`), a parameters builder, a tested scenario API and app, monitoring scripts, and 80 days of worked analyses. It's a portfolio: pin it on your GitHub profile, and write a short README that links the capstones of Modules 3, 5, 7 and 8.

Where to go next:

- **Read:** Cunningham, *Causal Inference: The Mixtape*; Huntington-Klein, *The Effect*; Chernozhukov et al., *Applied Causal Inference Powered by ML and AI*; Hernán & Robins, *Causal Inference: What If*.
- **Practise:** reproduce a published study's main table from its replication files, then run its sensitivity analysis.
- **Build:** add one estimator you met only in passing (synthetic DiD, Bayesian structural time series, off-policy evaluation for bandits) to `causalkit`, with tests.

### Day 80 checklist

- [ ] I defended a recommendation under questioning, reproducing numbers live
- [ ] I compared my projections with the truth and can explain every gap
- [ ] I know which uncertainty my ranges covered and which they didn't
- [ ] My repository is tagged v1.0 and its tests pass
- [ ] Course complete

