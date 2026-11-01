# Module 5 · Quasi-Experiments: DiD, Synthetic Control, RDD
## Days 45–47: Synthetic Control, Synthetic DiD and Bayesian Time Series

**Goal by end of Day 47:** Students can build a synthetic control and judge it by its pre-period fit and robustness to dropped donors, explain when synthetic control fails and how an intercept or synthetic DiD fixes it, and estimate a campaign's lift with a Bayesian time-series model in CausalPy, turning the posterior into a probability that the campaign paid off.

---

# DAY 45 — Synthetic Control

| Time | Block |
|---|---|
| 0:00 – 0:40 | 45.1 One treated region |
| 0:40 – 1:30 | 45.2 Why DiD fails here |
| 1:30 – 1:45 | Break |
| 1:45 – 2:50 | 45.3 Building a synthetic region |
| 2:50 – 3:20 | 45.4 How long a pre-period? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 45.5 Day 45 mini-project: leave-one-out |
| 4:40 – 5:00 | 45.6 Save your work with Git |

---

## 45.1 One treated region (40 min)

Region R07 ran an 8-week TV campaign starting in week 80. Its sales lift was about $30k a week during the campaign, then faded over the following weeks. That's a typical marketing question: **one** treated unit, many untreated ones, a long history before.

Create `causalkit/regions.py` (provided; read only the docstrings):

<details>
<summary>causalkit/regions.py (provided; click to expand)</summary>

```python
"""
NorthMart regional sales (simulated). Provided for Module 5: read the docstrings, not the code,
until the Day 50 capstone is done.

    from causalkit.regions import regions, truth, wide

    df = regions()          # 30 regions x 104 weeks; region R07 runs an 8-week TV campaign from week 80
    truth(df)               # true lift: average per post week, cumulative, during the campaign (for checking only)
    Y = wide(df)            # weeks x regions matrix of sales, for synthetic control

Columns an analyst has:
    region, week, sales ($ thousands per week), treated (1 = the campaign region), post (1 = week >= start)
Columns starting with '_' are the simulator's hidden truth: _y0 (sales without the campaign), _tau (lift).

Regions differ in size, seasonality and exposure to two national trends; one of them, urban growth,
accelerates over the two years, and some regions ride it much more than others.
"""
import numpy as np
import polars as pl


def regions(n_regions: int = 30, weeks: int = 104, start: int = 80, seed: int = 45, treated: str = "R07",
            campaign_weeks: int = 8, lift: float = 30.0, decay: float = 0.8, treated_urban: float = 1.6,
            treated_size: float = 0.0,
            noise: float = 4.0) -> pl.DataFrame:
    """treated_size: extra weekly sales of the campaign region ($ thousands) on top of the median region.
    lift:   true weekly sales lift during the campaign ($ thousands); after it ends the lift decays
               by `decay` per week (carry-over).
    treated_urban: the campaign region's exposure to urban growth (other regions: uniform 0 to 2)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, weeks + 1)
    names = [f"R{j:02d}" for j in range(1, n_regions + 1)]
    level = rng.normal(500, 60, n_regions)
    season_amp = rng.uniform(10, 30, n_regions)
    national = 0.4 * t + np.cumsum(rng.normal(0, 1.5, weeks))
    urban = 30 * (t / weeks) ** 2 + np.cumsum(rng.normal(0, 0.8, weeks))
    load_nat = rng.uniform(0.6, 1.4, n_regions)
    load_urb = rng.uniform(0, 2, n_regions)
    j = names.index(treated)
    load_urb[j] = treated_urban
    level[j] = np.median(level) + treated_size
    season = np.sin(2 * np.pi * (t - 10) / 52)
    y0 = (level[:, None] + season_amp[:, None] * season[None, :] + load_nat[:, None] * national[None, :]
          + load_urb[:, None] * urban[None, :] + rng.normal(0, noise, (n_regions, weeks)))
    k = t - start
    tau_t = np.where(k < 0, 0.0, np.where(k < campaign_weeks, lift, lift * decay ** (k - campaign_weeks + 1)))
    tau = np.zeros_like(y0)
    tau[j] = tau_t
    return pl.DataFrame({
        "region": np.repeat(names, weeks),
        "week": np.tile(t, n_regions).astype(np.int64),
        "sales": np.round((y0 + tau).ravel(), 2),
        "treated": np.repeat((np.arange(n_regions) == j).astype(np.int64), weeks),
        "post": np.tile((t >= start).astype(np.int64), n_regions),
        "_y0": y0.ravel(),
        "_tau": tau.ravel(),
    })


def truth(df: pl.DataFrame, campaign_weeks: int = 8) -> dict:
    tr = df.filter((pl.col("treated") == 1) & (pl.col("post") == 1)).sort("week")
    tau = tr["_tau"].to_numpy()
    return {"avg_post_lift": round(float(tau.mean()), 2), "cumulative_lift": round(float(tau.sum()), 1),
            "campaign_weeks_lift": round(float(tau[:campaign_weeks].sum()), 1)}


def wide(df: pl.DataFrame, col: str = "sales") -> pl.DataFrame:
    """Weeks in rows, one column per region."""
    return df.pivot(on="region", index="week", values=col).sort("week")


def public(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("_")])
```

</details>

The docstring gives a hint: regions ride a national trend and an **urban growth** trend, to different degrees, and urban growth accelerates.

---

## 45.2 Why DiD fails here (50 min)

Create `day45/did_fails.py`:

```python
# day45/did_fails.py   (python -m day45.did_fails)
"""
Region R07 ran an 8-week TV campaign from week 80. Only ONE region was treated, and it is an urban
region that was already growing faster than most. DiD against the average region, and why it fails.
"""
import matplotlib.pyplot as plt
import polars as pl

from causalkit.regions import public, regions, truth
from causalkit.synth import did

full = regions()
df = public(full)
t = truth(full)
print(f"{df['region'].n_unique()} regions x {df['week'].n_unique()} weeks; campaign region R07 from week 80")
print(f"True lift: {t['avg_post_lift']:.1f} $k per post week on average, "
      f"{t['cumulative_lift']:.0f} $k in total ({t['campaign_weeks_lift']:.0f} during the 8 campaign weeks)")

d = did(df, "R07", 80)
print(f"\nDiD vs the average of the other 29 regions: {d['effect']:.1f} $k per week")

avg = df.filter(pl.col("region") != "R07").group_by("week").agg(pl.col("sales").mean()).sort("week")
r07 = df.filter(pl.col("region") == "R07").sort("week")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(r07["week"], r07["sales"], label="R07", color="#dc2626")
axes[0].plot(avg["week"], avg["sales"], label="average of other regions", color="#64748b")
axes[0].set_title("Sales ($k per week)")
axes[1].plot(d["weeks"], d["gap"], color="#2563eb")
axes[1].set_title("R07 minus average, centred on the pre-period")
for ax in axes:
    ax.axvline(80, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("week")
axes[0].legend()
fig.tight_layout()
fig.savefig("figures/day45_did_fails.png", dpi=120)
print("Saved figures/day45_did_fails.png")
```

```bash
python -m day45.did_fails
```

```
30 regions x 104 weeks; campaign region R07 from week 80
True lift: 14.3 $k per post week on average, 357 $k in total (240 during the 8 campaign weeks)

DiD vs the average of the other 29 regions: 32.1 $k per week
Saved figures/day45_did_fails.png
```

DiD more than doubles the lift. Open the figure: the gap between R07 and the average region drifts upward long before week 80. R07 is an urban region, more exposed to urban growth than the average region, so parallel trends fail. With 240 stores (Day 41) the average of many controls was a fine comparison; with one urban region, the average region is not.

---

## 45.3 Building a synthetic region (65 min)

**Concept.** Synthetic control (Abadie & Gardeazabal 2003; Abadie, Diamond & Hainmueller 2010) builds a **synthetic R07**: a weighted average of other regions, with weights that are non-negative and sum to 1, chosen so that synthetic R07's weekly sales match real R07's **before** week 80 as closely as possible. If the match is close for 79 weeks, it's credible that it would have kept matching without the campaign. After week 80, the gap is the effect.

Why non-negative weights that sum to 1? The synthetic region is then an **interpolation** of real regions, never an extrapolation, and you can read the weights: "synthetic R07 is 23% R18, 22% R14, ...".

Create `causalkit/synth.py`. Today you need `simplex_ls`, `_split`, `did` and `synthetic_control`; `sdid` and `placebo_in_space` follow on Days 46 and 49.

<details>
<summary>causalkit/synth.py (complete, built Days 45–49; click to expand)</summary>

```python
"""Synthetic control, synthetic DiD and their placebo tests (Days 45-49)."""
import numpy as np
import polars as pl
from scipy.optimize import minimize

from causalkit.regions import wide


def simplex_ls(X: np.ndarray, y: np.ndarray, intercept: bool = False, ridge: float = 0.0):
    """Weights w >= 0 summing to 1 that minimise ||y - (a) - X w||^2 + ridge * ||w||^2.
    With intercept=True, a free constant a absorbs any level difference. Returns (w, a)."""
    k = X.shape[1]
    if intercept:                           # the optimal constant is the mean gap: demean, then solve
        Xc, yc = X - X.mean(0), y - y.mean()
    else:
        Xc, yc = X, y

    def loss(w):
        r = yc - Xc @ w
        return r @ r + ridge * w @ w

    def grad(w):
        return -2 * Xc.T @ (yc - Xc @ w) + 2 * ridge * w

    res = minimize(loss, np.full(k, 1 / k), jac=grad, method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w /= w.sum()
    a = float((y - X @ w).mean()) if intercept else 0.0
    return w, a


def _split(df: pl.DataFrame, treated: str, start: int):
    Y = wide(df)
    weeks = Y["week"].to_numpy()
    donors = [c for c in Y.columns if c not in ("week", treated)]
    y1 = Y[treated].to_numpy()
    Y0 = Y.select(donors).to_numpy()
    pre = weeks < start
    return weeks, donors, y1, Y0, pre


def did(df: pl.DataFrame, treated: str, start: int) -> dict:
    """Plain DiD: treated region vs the average of all other regions, post minus pre."""
    weeks, donors, y1, Y0, pre = _split(df, treated, start)
    gap = y1 - Y0.mean(1)
    return {"effect": float(gap[~pre].mean() - gap[pre].mean()), "gap": gap - gap[pre].mean(), "weeks": weeks}


def synthetic_control(df: pl.DataFrame, treated: str, start: int, intercept: bool = False) -> dict:
    """Abadie-style synthetic control: donor weights fitted on pre-period sales only.
    Returns the average post-period gap (effect), the weekly gap, weights and fit statistics."""
    weeks, donors, y1, Y0, pre = _split(df, treated, start)
    w, a = simplex_ls(Y0[pre], y1[pre], intercept=intercept)
    synth = a + Y0 @ w
    gap = y1 - synth
    rmspe_pre = float(np.sqrt((gap[pre] ** 2).mean()))
    rmspe_post = float(np.sqrt((gap[~pre] ** 2).mean()))
    return {"effect": float(gap[~pre].mean()), "gap": gap, "synthetic": synth, "actual": y1, "weeks": weeks,
            "weights": dict(sorted(((d, round(float(x), 3)) for d, x in zip(donors, w) if x > 0.005),
                                   key=lambda kv: -kv[1])),
            "rmspe_pre": rmspe_pre, "rmspe_post": rmspe_post, "ratio": rmspe_post / rmspe_pre}


def sdid(df: pl.DataFrame, treated: str, start: int) -> dict:
    """Synthetic difference-in-differences (Arkhangelsky et al., 2021), one treated unit.
    Unit weights: like SC but with an intercept and a ridge penalty.
    Time weights: pre-period weeks that best predict the donors' post-period average."""
    weeks, donors, y1, Y0, pre = _split(df, treated, start)
    T0, T1 = pre.sum(), (~pre).sum()
    zeta = (1 * T1) ** 0.25 * np.diff(Y0[pre], axis=0).std(ddof=1)
    omega, _ = simplex_ls(Y0[pre], y1[pre], intercept=True, ridge=zeta**2 * T0)
    lam, _ = simplex_ls(Y0[pre].T, Y0[~pre].mean(0), intercept=True, ridge=1e-6 * T0)
    treated_change = y1[~pre].mean() - lam @ y1[pre]
    donor_change = (Y0[~pre].mean(0) - lam @ Y0[pre]) @ omega
    return {"effect": float(treated_change - donor_change),
            "unit_weights": dict(sorted(((d, round(float(x), 3)) for d, x in zip(donors, omega) if x > 0.005),
                                        key=lambda kv: -kv[1])),
            "time_weights": {int(w): round(float(x), 3) for w, x in zip(weeks[pre], lam) if x > 0.005}}


def placebo_in_space(df: pl.DataFrame, treated: str, start: int, intercept: bool = False) -> pl.DataFrame:
    """Re-run synthetic control pretending each region was treated (the real one is dropped from donor pools).
    Inference: where does the real region's post/pre RMSPE ratio rank?"""
    rows = []
    for r in df["region"].unique().sort().to_list():
        sub = df if r == treated else df.filter(pl.col("region") != treated)
        sc = synthetic_control(sub, r, start, intercept=intercept)
        rows.append({"region": r, "is_treated": r == treated, "effect": sc["effect"],
                     "rmspe_pre": sc["rmspe_pre"], "ratio": sc["ratio"]})
    return pl.DataFrame(rows).sort("ratio", descending=True)
```

</details>

Create `day45/synthetic_control.py`:

```python
# day45/synthetic_control.py   (python -m day45.synthetic_control)
"""
Synthetic control (Abadie, Diamond & Hainmueller, 2010): build a 'synthetic R07' as a weighted average
of other regions, with weights >= 0 that sum to 1, chosen to match R07's sales BEFORE the campaign.
After week 80, the synthetic region shows what R07 would have sold without the campaign.
"""
import matplotlib.pyplot as plt

from causalkit.regions import public, regions, truth
from causalkit.synth import synthetic_control

full = regions()
sc = synthetic_control(public(full), "R07", 80)
t = truth(full)

print("Donor weights:", sc["weights"])
print(f"Pre-period fit: RMSPE {sc['rmspe_pre']:.1f} $k (R07 sells about {sc['actual'][:79].mean():.0f} $k a week)")
print(f"Effect: {sc['effect']:.1f} $k per post week (truth {t['avg_post_lift']:.1f}); "
      f"cumulative {sc['gap'][79:].sum():.0f} $k (truth {t['cumulative_lift']:.0f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(sc["weeks"], sc["actual"], color="#dc2626", label="R07")
axes[0].plot(sc["weeks"], sc["synthetic"], color="#2563eb", ls="--", label="synthetic R07")
axes[0].legend()
axes[0].set_title("Sales ($k per week)")
axes[1].plot(sc["weeks"], sc["gap"], color="#2563eb", label="estimated")
axes[1].plot(sc["weeks"], full.filter(full["region"] == "R07")["_tau"], color="k", lw=1, label="true lift")
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].legend()
axes[1].set_title("Gap: actual minus synthetic")
for ax in axes:
    ax.axvline(80, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("week")
fig.tight_layout()
fig.savefig("figures/day45_synthetic_control.png", dpi=120)
print("Saved figures/day45_synthetic_control.png")
```

```bash
python -m day45.synthetic_control
```

```
Donor weights: {'R18': 0.225, 'R14': 0.216, 'R13': 0.209, 'R05': 0.174, 'R22': 0.161, 'R03': 0.01}
Pre-period fit: RMSPE 3.9 $k (R07 sells about 514 $k a week)
Effect: 17.3 $k per post week (truth 14.3); cumulative 433 $k (truth 357)
Saved figures/day45_synthetic_control.png
```

Six regions get weight; the other 23 get zero. The pre-period fit is within $3.9k a week (under 1% of sales). The estimate is $17.3k against a true $14.3k: much closer than DiD's $32k. Open the figure: the estimated gap tracks the true lift's shape, the jump at week 80 and the decay after week 88.

---

## 45.4 How long a pre-period? (30 min)

**Concept.** A close pre-period fit is only convincing if it's long. With a few weeks, any mix of regions can match by luck, including a mix that doesn't share R07's trends.

**Try it.** Fit synthetic control using only weeks 60 onwards (20 pre-period weeks instead of 79). Create `day45/short_pre_period.py`.

<details>
<summary>Solution</summary>

```python
# day45/short_pre_period.py   (python -m day45.short_pre_period)
import polars as pl

from causalkit.regions import public, regions, truth
from causalkit.synth import synthetic_control

full = regions()
df = public(full)
for first_week in [1, 60]:
    sc = synthetic_control(df.filter(pl.col("week") >= first_week), "R07", 80)
    print(f"pre-period from week {first_week:>2}: effect {sc['effect']:.1f}, pre RMSPE {sc['rmspe_pre']:.1f}, "
          f"{len(sc['weights'])} donors")
print(f"truth {truth(full)['avg_post_lift']:.1f}")
```

```
pre-period from week  1: effect 17.3, pre RMSPE 3.9, 6 donors
pre-period from week 60: effect 27.4, pre RMSPE 2.9, 6 donors
truth 14.3
```

The short pre-period gives a **better** fit (RMSPE 2.9) and a **worse** answer ($27.4k). The weights fitted the noise of 20 weeks, not R07's long-run exposure to urban growth. A good fit over a short window is weak evidence; Abadie (2021) recommends a pre-period much longer than the number of donors that end up with weight.

</details>

---

## 45.5 Day 45 mini-project: leave-one-out (65 min)

Is the answer driven by one donor? Drop each donor with a positive weight in turn and re-estimate. Create `day45/leave_one_out.py`:

```python
# day45/leave_one_out.py   (python -m day45.leave_one_out)
"""
Mini-project: is the answer driven by one donor? Drop each donor with a positive weight in turn and
re-estimate. A robust synthetic control gives similar answers every time.
"""
import polars as pl

from causalkit.regions import public, regions
from causalkit.synth import synthetic_control

df = public(regions())
base = synthetic_control(df, "R07", 80)
rows = [{"dropped": "(none)", "effect": round(base["effect"], 1), "pre RMSPE": round(base["rmspe_pre"], 1),
         "largest weights": str(dict(list(base["weights"].items())[:3]))}]
for donor in base["weights"]:
    sc = synthetic_control(df.filter(pl.col("region") != donor), "R07", 80)
    rows.append({"dropped": donor, "effect": round(sc["effect"], 1), "pre RMSPE": round(sc["rmspe_pre"], 1),
                 "largest weights": str(dict(list(sc["weights"].items())[:3]))})
with pl.Config(tbl_width_chars=140, fmt_str_lengths=60):
    print(pl.DataFrame(rows))
```

```bash
python -m day45.leave_one_out
```

```
│ dropped ┆ effect ┆ pre RMSPE ┆ largest weights                            │
│ (none)  ┆ 17.3   ┆ 3.9       ┆ {'R18': 0.225, 'R14': 0.216, 'R13': 0.209} │
│ R18     ┆ 20.6   ┆ 4.2       ┆ {'R14': 0.286, 'R22': 0.167, 'R04': 0.163} │
│ R14     ┆ 20.0   ┆ 4.1       ┆ {'R13': 0.31, 'R18': 0.293, 'R05': 0.205}  │
│ R13     ┆ 17.8   ┆ 4.0       ┆ {'R14': 0.314, 'R18': 0.209, 'R22': 0.178} │
│ R05     ┆ 17.3   ┆ 3.9       ┆ {'R18': 0.274, 'R14': 0.244, 'R13': 0.202} │
│ R22     ┆ 17.4   ┆ 3.9       ┆ {'R18': 0.296, 'R13': 0.212, 'R14': 0.207} │
│ R03     ┆ 17.3   ┆ 3.9       ┆ {'R18': 0.229, 'R14': 0.212, 'R13': 0.209} │
```

The estimate stays between $17.3k and $20.6k whatever is dropped, and the fit hardly changes. That's what a robust synthetic control looks like. If dropping one donor had moved the estimate by half, the result would be that donor's story, not R07's.

Report the leave-one-out range alongside the estimate: "$17k a week (between $17k and $21k when any single donor region is dropped)".

---

## 45.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 45: regions simulator, synthetic control, pre-period length, leave-one-out"
git push
```

### Day 45 checklist

- [ ] I can explain why DiD fails with one treated unit on a different trend
- [ ] I can build a synthetic control, read its weights and judge its pre-period fit
- [ ] I know that a close fit over a short pre-period is weak evidence
- [ ] I check robustness by dropping each donor in turn

---

# DAY 46 — Synthetic Difference-in-Differences

| Time | Block |
|---|---|
| 0:00 – 0:40 | 46.1 When synthetic control can't fit |
| 0:40 – 1:30 | 46.2 The biggest region |
| 1:30 – 1:45 | Break |
| 1:45 – 2:50 | 46.3 Synthetic DiD |
| 2:50 – 3:20 | 46.4 What do the time weights do? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 46.5 Day 46 mini-project: four estimators, two worlds |
| 4:40 – 5:00 | 46.6 Save your work with Git |

---

## 46.1 When synthetic control can't fit (40 min)

Synthetic control's weights are non-negative and sum to 1, so synthetic R07 always lies **inside** the range of the donors. If R07 is the biggest region (or the fastest-growing, or the most seasonal), no mix of smaller regions can match it. Toronto, a flagship store, a market leader: the units companies most often run campaigns in are the hardest to match.

Two fixes:

| Fix | Idea |
|---|---|
| **Synthetic control with an intercept** (Doudchenko & Imbens 2016; Ferman & Pinto 2021) | match R07's *changes*, allowing a constant level difference, like DiD does |
| **Synthetic DiD** (Arkhangelsky et al., 2021) | intercept + spread-out unit weights + weights on pre-period weeks, then a weighted DiD |

---

## 46.2 The biggest region (50 min)

`regions(treated_size=250)` is the same world, except that R07 sells $250k a week more than the median region. Create `day46/big_region.py`:

```python
# day46/big_region.py   (python -m day46.big_region)
"""
Synthetic control's weakness: weights are >= 0 and sum to 1, so the synthetic region can never be
BIGGER than the biggest donor. Same world, but now the campaign ran in a region that sells 250 $k a week
more than the median region. Adding an intercept (matching changes, not levels) fixes it.
"""
import matplotlib.pyplot as plt
import polars as pl

from causalkit.regions import public, regions, truth
from causalkit.synth import synthetic_control

full = regions(treated_size=250)
df = public(full)
t = truth(full)
plain = synthetic_control(df, "R07", 80)
demeaned = synthetic_control(df, "R07", 80, intercept=True)
print(f"R07 sells {plain['actual'][:79].mean():.0f} $k a week; the biggest donor sells "
      f"{df.filter(pl.col('region') != 'R07').group_by('region').agg(pl.col('sales').mean())['sales'].max():.0f}")
for name, sc in [("synthetic control", plain), ("synthetic control + intercept", demeaned)]:
    print(f"{name:<31} effect {sc['effect']:6.1f}   pre RMSPE {sc['rmspe_pre']:5.1f}   weights {sc['weights']}")
print(f"Truth: {t['avg_post_lift']:.1f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(plain["weeks"], plain["actual"], color="#dc2626", label="R07")
ax.plot(plain["weeks"], plain["synthetic"], color="#64748b", ls="--", label="synthetic control")
ax.plot(demeaned["weeks"], demeaned["synthetic"], color="#2563eb", ls="--", label="with intercept")
ax.axvline(80, color="grey", ls="--", lw=0.8)
ax.legend()
ax.set_xlabel("week")
ax.set_ylabel("sales ($k per week)")
fig.tight_layout()
fig.savefig("figures/day46_big_region.png", dpi=120)
print("Saved figures/day46_big_region.png")
```

```bash
python -m day46.big_region
```

```
R07 sells 764 $k a week; the biggest donor sells 604
synthetic control               effect  214.8   pre RMSPE 164.9   weights {'R25': 0.987}
synthetic control + intercept   effect   17.3   pre RMSPE   3.9   weights {'R14': 0.211, 'R18': 0.21, 'R13': 0.209, 'R05': 0.171, 'R22': 0.169, 'R16': 0.015, 'R23': 0.012}
Truth: 14.3
```

Plain synthetic control puts all its weight on the biggest donor and still misses by $165k a week before the campaign. Its "effect" of $215k is just the size gap. The warning is right there in the pre-period RMSPE: **never report a synthetic control without its pre-period fit**.

With an intercept, the weights and the estimate are exactly Day 45's: the level difference is absorbed, and the donors are chosen on how R07 *moves*.

---

## 46.3 Synthetic DiD (65 min)

**Concept.** Synthetic DiD (SDID) combines three ideas:

1. **Unit weights**, like synthetic control, but with an intercept (levels don't need to match) and a ridge penalty that spreads weight over more donors (less dependence on any one).
2. **Time weights**: choose pre-period weeks whose donor sales best predict the donors' post-period average. Weeks that look like the post-period count more.
3. A **weighted DiD**: R07's change minus the weighted donors' change, each from the weighted pre-period to the post-period.

It's DiD made robust to non-parallel trends, or synthetic control made robust to level differences, depending on where you start. Type `sdid` into `synth.py`. Create `day46/sdid.py`:

```python
# day46/sdid.py   (python -m day46.sdid)
"""
Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens & Wager, 2021):
  unit weights: like synthetic control, but with an intercept and a penalty that spreads weight;
  time weights: the pre-period weeks that best resemble the post-period (for the donors);
  then a weighted DiD. It matches trends, not levels, and borrows strength from more donors.
"""
from causalkit.regions import public, regions, truth
from causalkit.synth import did, sdid, synthetic_control

for title, size in [("R07 a typical-size region", 0), ("R07 a very large region", 250)]:
    full = regions(treated_size=size)
    df = public(full)
    s = sdid(df, "R07", 80)
    print(f"{title}:   truth {truth(full)['avg_post_lift']:.1f}")
    print(f"   DiD {did(df, 'R07', 80)['effect']:6.1f}   SC {synthetic_control(df, 'R07', 80)['effect']:6.1f}   "
          f"SDID {s['effect']:6.1f}")
print(f"\nSDID unit weights ({len(s['unit_weights'])} donors):", dict(list(s["unit_weights"].items())[:6]), "...")
print("SDID time weights:", s["time_weights"])
```

```bash
python -m day46.sdid
```

```
R07 a typical-size region:   truth 14.3
   DiD   32.1   SC   17.3   SDID   20.9
R07 a very large region:   truth 14.3
   DiD   32.1   SC  214.8   SDID   20.9

SDID unit weights (22 donors): {'R14': 0.094, 'R13': 0.089, 'R18': 0.089, 'R22': 0.084, 'R16': 0.062, 'R05': 0.061} ...
SDID time weights: {49: 0.172, 51: 0.01, 52: 0.056, 55: 0.106, 64: 0.014, 66: 0.059, 74: 0.155, 78: 0.11, 79: 0.319}
```

SDID is unaffected by R07's size. It spreads weight over 22 donors (synthetic control used 6), and it puts most time weight on recent weeks (78, 79) and on weeks 49–55, a year before the post-period, so the same season.

On this dataset SDID ($20.9k) is further from the truth than synthetic control ($17.3k). One dataset can't tell you which estimator is better: that's the mini-project.

---

## 46.4 What do the time weights do? (30 min)

**Try it.** Rerun SDID over the 20 datasets of the mini-project (`regions(seed=100 + s)`) with and without time weights (without: compare the post-period with the plain pre-period average). Create `day46/no_time_weights.py`. You need `_split` and `simplex_ls` from `synth.py`.

<details>
<summary>Solution</summary>

```python
# day46/no_time_weights.py   (python -m day46.no_time_weights)
import numpy as np

from causalkit.regions import public, regions, truth
from causalkit.synth import _split, sdid, simplex_ls

errs = {"SDID": [], "SDID without time weights": []}
for s in range(20):
    full = regions(seed=100 + s)
    df, tr = public(full), truth(full)["avg_post_lift"]
    errs["SDID"].append(sdid(df, "R07", 80)["effect"] - tr)
    weeks, donors, y1, Y0, pre = _split(df, "R07", 80)
    zeta = (~pre).sum() ** 0.25 * np.diff(Y0[pre], axis=0).std(ddof=1)
    omega, _ = simplex_ls(Y0[pre], y1[pre], intercept=True, ridge=zeta**2 * pre.sum())
    flat = (y1[~pre].mean() - y1[pre].mean()) - (Y0[~pre].mean(0) - Y0[pre].mean(0)) @ omega
    errs["SDID without time weights"].append(flat - tr)
for k, e in errs.items():
    print(f"{k:<28} average error {np.mean(e):+.1f}   SD {np.std(e):.1f}")
```

```
SDID                         average error +2.2   SD 4.7
SDID without time weights    average error +4.3   SD 3.8
```

Without time weights, the bias doubles. Urban growth accelerates, so the early pre-period weeks are a poor baseline for the post-period; the time weights shift the baseline towards recent and same-season weeks. The price is a little more variance: the estimate rests on fewer weeks.

</details>

---

## 46.5 Day 46 mini-project: four estimators, two worlds (65 min)

Create `day46/monte_carlo.py`:

```python
# day46/monte_carlo.py   (python -m day46.monte_carlo)
"""
Mini-project: four estimators, 20 simulated datasets, two worlds. Average error and its spread.
"""
import numpy as np
import polars as pl

from causalkit.regions import public, regions, truth
from causalkit.synth import did, sdid, synthetic_control

estimators = {
    "DiD": lambda d: did(d, "R07", 80)["effect"],
    "synthetic control": lambda d: synthetic_control(d, "R07", 80)["effect"],
    "SC + intercept": lambda d: synthetic_control(d, "R07", 80, intercept=True)["effect"],
    "SDID": lambda d: sdid(d, "R07", 80)["effect"],
}
rows = []
for world, size in [("typical-size region", 0), ("very large region", 250)]:
    errors = {k: [] for k in estimators}
    for s in range(20):
        full = regions(seed=100 + s, treated_size=size)
        df, tr = public(full), truth(full)["avg_post_lift"]
        for k, f in estimators.items():
            errors[k].append(f(df) - tr)
    for k, e in errors.items():
        rows.append({"world": world, "estimator": k, "average error": round(np.mean(e), 1),
                     "SD of error": round(np.std(e), 1)})
with pl.Config(tbl_rows=10, tbl_width_chars=100):
    print(pl.DataFrame(rows))
```

```bash
python -m day46.monte_carlo
```

```
│ world               ┆ estimator         ┆ average error ┆ SD of error │
│ typical-size region ┆ DiD               ┆ 7.9           ┆ 6.1         │
│ typical-size region ┆ synthetic control ┆ 1.0           ┆ 3.8         │
│ typical-size region ┆ SC + intercept    ┆ 2.5           ┆ 3.0         │
│ typical-size region ┆ SDID              ┆ 2.2           ┆ 4.7         │
│ very large region   ┆ DiD               ┆ 7.9           ┆ 6.1         │
│ very large region   ┆ synthetic control ┆ 143.3         ┆ 28.8        │
│ very large region   ┆ SC + intercept    ┆ 2.5           ┆ 3.0         │
│ very large region   ┆ SDID              ┆ 2.2           ┆ 4.7         │
```

- **DiD** is biased in both worlds: parallel trends fail.
- **Plain synthetic control** is the best estimator when R07 is a typical region, and useless when it's the biggest.
- **SC with an intercept and SDID** are unaffected by size, with a small bias (+$2–2.5k on a $14k effect) and similar spread.

No estimator wins everywhere. The rule that does: **check the pre-period fit, then use the estimator whose assumptions the fit supports.** If plain synthetic control fits well, it's hard to beat; if it can't fit, use an intercept or SDID, and report both.

---

## 46.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 46: big-region failure, SC with intercept, synthetic DiD, Monte Carlo"
git push
```

### Day 46 checklist

- [ ] I can explain why synthetic control fails when the treated unit is outside the donors' range
- [ ] I can fix it with an intercept, and explain what that changes
- [ ] I can explain SDID's unit weights, time weights and weighted DiD
- [ ] I choose among DiD, SC and SDID using the pre-period fit, and report more than one

---

# DAY 47 — Bayesian Structural Time Series with CausalPy

| Time | Block |
|---|---|
| 0:00 – 0:40 | 47.1 Forecasting the counterfactual |
| 0:40 – 1:20 | 47.2 A time series without controls |
| 1:20 – 1:35 | Break |
| 1:35 – 2:40 | 47.3 CausalImpact-style: control regions as predictors |
| 2:40 – 3:20 | 47.4 How many controls? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 47.5 Day 47 mini-project: did the campaign pay off? |
| 4:40 – 5:00 | 47.6 Save your work with Git |

---

## 47.1 Forecasting the counterfactual (40 min)

**Concept.** Synthetic control is one way to predict what R07 would have sold without the campaign. A time-series model is another: learn the relationship between R07's sales and things the campaign couldn't have affected (other regions' sales, trends, seasons) from the pre-period, then **forecast** the post-period. Google's **CausalImpact** (Brodersen et al., 2015) made this the standard tool in marketing analytics: a Bayesian structural time-series (BSTS) model with control series as predictors.

Being Bayesian gives two things synthetic control doesn't:

- a full **posterior** for the lift in every week, the cumulative lift, and anything computed from it, like profit;
- direct probability statements: "there's an 86% chance the campaign paid for itself".

We use **CausalPy** (by PyMC Labs), which runs these models on PyMC. Its `InterruptedTimeSeries` fits a Bayesian regression on the pre-period and forecasts the post-period. CausalPy also offers a full state-space version (`StateSpaceTimeSeries`, with a local level and trend that evolve over time); the regression version is faster and, with good control series, gives the same answers here.

PyMC samples with MCMC. The first run compiles the model, which takes a minute; later runs are faster. You'll see sampling messages: they're normal.

---

## 47.2 A time series without controls (40 min)

Create `day47/causal_impact.py` first: `day47/its_no_controls.py` imports its sampling settings.

```python
# day47/causal_impact.py   (python -m day47.causal_impact)
"""
A Bayesian time-series model in the spirit of Google's CausalImpact (Brodersen et al., 2015), with CausalPy:
learn how R07's weekly sales move with other regions' sales BEFORE the campaign, forecast the post-period
from the other regions, and read the lift as actual minus forecast, with a posterior for every week.
"""
import causalpy as cp
import matplotlib.pyplot as plt
import polars as pl

from causalkit.regions import public, regions, truth, wide

SAMPLE = {"random_seed": 1, "progressbar": False, "draws": 1000, "tune": 1000, "chains": 2}


def causal_impact(df: pl.DataFrame, treated: str, start: int, controls: list[str]):
    """Fit CausalPy's InterruptedTimeSeries with control regions as regressors.
    Sales are centred on each region's pre-period mean and scaled to tens of $k, which makes sampling
    much faster; results are converted back to $k. Returns (result, scale)."""
    W = wide(df).to_pandas().set_index("week")
    Z = (W - W[W.index < start].mean()) / 10
    result = cp.InterruptedTimeSeries(Z, start, formula=f"{treated} ~ 1 + " + " + ".join(controls),
                                      model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE))
    return result, 10.0


def summary(result, scale: float):
    tab = result.effect_summary().table
    return {k: {"mean": tab.loc[k, "mean"] * scale, "low": tab.loc[k, "hdi_lower"] * scale,
                "high": tab.loc[k, "hdi_upper"] * scale} for k in ["average", "cumulative"]}


if __name__ == "__main__":
    full = regions()
    df = public(full)
    t = truth(full)
    controls = ["R13", "R14", "R18", "R05", "R22"]          # Day 45's synthetic-control donors
    result, scale = causal_impact(df, "R07", 80, controls)
    s = summary(result, scale)
    print(f"Average lift per post week: {s['average']['mean']:.1f} $k  "
          f"[94% HDI {s['average']['low']:.1f} to {s['average']['high']:.1f}]   truth {t['avg_post_lift']:.1f}")
    print(f"Cumulative lift:            {s['cumulative']['mean']:.0f} $k  "
          f"[94% HDI {s['cumulative']['low']:.0f} to {s['cumulative']['high']:.0f}]   truth {t['cumulative_lift']:.0f}")
    fig, ax = result.plot()
    fig.savefig("figures/day47_causal_impact.png", dpi=110)
    plt.close(fig)
    print("Saved figures/day47_causal_impact.png")
```

Then create `day47/its_no_controls.py`:

```python
# day47/its_no_controls.py   (python -m day47.its_no_controls)
"""
An interrupted time series with NO control regions: fit trend + seasonality to R07 before week 80 and
extend it. The model can't know that urban growth accelerated, so it credits the acceleration to the campaign.
"""
import causalpy as cp
import numpy as np

from causalkit.regions import public, regions, truth, wide
from day47.causal_impact import SAMPLE

full = regions()
W = wide(public(full)).to_pandas().set_index("week")
Z = (W - W[W.index < 80].mean()) / 10
Z["t"] = np.arange(len(Z)) / 52
Z["s1"], Z["c1"] = np.sin(2 * np.pi * Z.index / 52), np.cos(2 * np.pi * Z.index / 52)
result = cp.InterruptedTimeSeries(Z, 80, formula="R07 ~ 1 + t + s1 + c1",
                                  model=cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE))
tab = result.effect_summary().table
print(f"Average lift per post week: {tab.loc['average', 'mean'] * 10:.1f} $k  "
      f"[94% HDI {tab.loc['average', 'hdi_lower'] * 10:.1f} to {tab.loc['average', 'hdi_upper'] * 10:.1f}]"
      f"   truth {truth(full)['avg_post_lift']:.1f}")
```

```bash
python -m day47.its_no_controls
```

```
Average lift per post week: 37.4 $k  [94% HDI 34.2 to 40.8]   truth 14.3
```

A confident and wrong answer: $37k a week, with a tight credible interval that excludes the truth. A trend fitted to R07's own past can't anticipate that urban growth **accelerated** after week 80, so the model credits the acceleration to the campaign. The credible interval only reflects uncertainty *within* the model; it can't warn you that the model is wrong. Without control series, an interrupted time series is a before-after comparison with a forecast attached.

(CausalPy reports a 94% highest-density interval, HDI, by default: the narrowest range containing 94% of the posterior.)

---

## 47.3 CausalImpact-style: control regions as predictors (65 min)

Now the control regions do the work: R07's sales are predicted from the five regions synthetic control chose on Day 45. Unlike synthetic control, the regression coefficients are unconstrained (any sign, any sum), and an intercept absorbs level differences.

One practical detail in `causal_impact`: each region's sales are centred on its pre-period mean and divided by 10 before fitting. On the original scale ($500k a week) the sampler needs many small steps and the run takes five minutes; centred and scaled, it takes twenty seconds. Results are converted back to $k.

```bash
python -m day47.causal_impact
```

```
Average lift per post week: 15.0 $k  [94% HDI 12.1 to 18.2]   truth 14.3
Cumulative lift:            375 $k  [94% HDI 302 to 454]   truth 357
Saved figures/day47_causal_impact.png
```

Within $1k a week of the truth, and the interval contains it. Open the figure: CausalPy's standard three panels (actual vs counterfactual, weekly impact, cumulative impact), in the scaled units (tens of $k, centred).

---

## 47.4 How many controls? (30 min)

**Try it.** Use all 29 other regions as predictors. Create `day47/all_controls.py`. Better or worse?

<details>
<summary>Solution</summary>

```python
# day47/all_controls.py   (python -m day47.all_controls)
from causalkit.regions import public, regions, truth
from day47.causal_impact import causal_impact, summary

full = regions()
df = public(full)
controls = [r for r in df["region"].unique().sort().to_list() if r != "R07"]
s = summary(*causal_impact(df, "R07", 80, controls))
print(f"29 controls: {s['average']['mean']:.1f} $k/week [94% HDI {s['average']['low']:.1f}, {s['average']['high']:.1f}]"
      f"   truth {truth(full)['avg_post_lift']:.1f}")
```

```
29 controls: 13.1 $k/week [94% HDI 8.4, 18.3]   truth 14.3
```

Still close, with a wider interval: 29 coefficients estimated from 79 weeks is a lot, and the posterior honestly reflects that. With many candidate controls, CausalImpact uses a **spike-and-slab** prior that lets most coefficients be exactly zero; CausalPy's `VariableSelectionPrior` does the same. Choosing controls with synthetic control first, as we did, is a simple alternative. Either way, **choose controls the campaign couldn't have affected**: a neighbouring region that saw the same TV ads is not a control.

</details>

---

## 47.5 Day 47 mini-project: did the campaign pay off? (65 min)

The campaign cost $100k. NorthMart keeps 30% of extra sales as margin. Every posterior draw of the cumulative lift gives a draw of profit = 0.30 × lift − 100. Create `day47/posterior_roi.py`:

```python
# day47/posterior_roi.py   (python -m day47.posterior_roi)
"""
Mini-project: a posterior for the decision, not just for the lift.
The campaign cost $100k; NorthMart keeps 30% of extra sales as margin. Profit = 0.30 x cumulative lift - 100.
Every posterior draw of the lift gives a draw of profit, so we can state P(the campaign paid off).
"""
import matplotlib.pyplot as plt
import numpy as np

from causalkit.regions import public, regions, truth
from day47.causal_impact import causal_impact

MARGIN, COST = 0.30, 100.0
full = regions()
result, scale = causal_impact(public(full), "R07", 80, ["R13", "R14", "R18", "R05", "R22"])
lift = (result.post_impact.sum("obs_ind") * scale).values.ravel()        # cumulative lift, one per draw
profit = MARGIN * lift - COST
lo, hi = np.percentile(profit, [3, 97])
print(f"Profit: mean {profit.mean():+.1f} $k, 94% interval {lo:+.1f} to {hi:+.1f}")
print(f"P(profit > 0) = {(profit > 0).mean():.2f}")
print(f"P(return on spend above 20%) = {(profit > 0.2 * COST).mean():.2f}")
print(f"True profit: {MARGIN * truth(full)['cumulative_lift'] - COST:+.1f} $k")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(profit, bins=50, color="#2563eb", alpha=0.8)
ax.axvline(0, color="#dc2626", lw=1)
ax.set_xlabel("campaign profit ($k)")
ax.set_ylabel("posterior draws")
fig.tight_layout()
fig.savefig("figures/day47_profit_posterior.png", dpi=120)
print("Saved figures/day47_profit_posterior.png")
```

```bash
python -m day47.posterior_roi
```

```
Profit: mean +12.6 $k, 94% interval -9.5 to +34.5
P(profit > 0) = 0.86
P(return on spend above 20%) = 0.26
True profit: +7.2 $k
Saved figures/day47_profit_posterior.png
```

This is the output a decision-maker can use: "the campaign most likely made a small profit (86% probability), but it's unlikely to have returned 20% or more (26%)". Compare that with "the lift was significant", which says nothing about whether the money was well spent.

Write in `day47/NOTES.md`: if the marketing director asks whether to repeat the campaign at the same cost, what would you say, using these three numbers?

---

## 47.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 47: CausalPy interrupted time series, control regions, profit posterior"
git push
```

### Day 47 checklist

- [ ] I can explain the forecasting approach to a counterfactual, and what CausalImpact adds
- [ ] I know that a time series without controls is a before-after comparison, whatever its interval says
- [ ] I can fit a CausalPy model with control series and read its HDIs
- [ ] I choose controls the treatment couldn't have affected
- [ ] I turn a posterior for the lift into a posterior for profit and probability statements

