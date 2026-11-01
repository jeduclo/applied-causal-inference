# Module 4 · Selection on Observables & Instrumental Variables
## Days 34–36: Weighting, Overlap and Doubly Robust Estimation

**Goal by end of Day 36:** Students can estimate effects by inverse probability weighting, diagnose extreme weights with the effective sample size, handle poor overlap by trimming or overlap weights while naming the estimand that results, and estimate ATE and ATT with AIPW: doubly robust, cross-fitted, with influence-function standard errors.

---

# DAY 34 — Inverse Probability Weighting

| Time | Block |
|---|---|
| 0:00 – 0:45 | 34.1 Weighting creates a pseudo-population |
| 0:45 – 1:35 | 34.2 Horvitz–Thompson and Hajek estimators |
| 1:35 – 1:50 | Break |
| 1:50 – 2:40 | 34.3 Weighted balance and the effective sample size |
| 2:40 – 3:20 | 34.4 Is IPW biased, or just noisy? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 34.5 Day 34 mini-project: Horvitz–Thompson vs Hajek |
| 4:40 – 5:00 | 34.6 Save your work with Git |

---

## 34.1 Weighting creates a pseudo-population (45 min)

**Concept.** A customer with propensity 0.2 who joined is **unusual**: four in five customers like them didn't join. In the data they stand in for the missing joiners like them, so give them weight 1/0.2 = 5. A non-joiner with propensity 0.2 is typical: weight 1/0.8 = 1.25. After weighting, each group looks like the whole population, as if joining had been randomised. This is **inverse probability weighting (IPW)**, the survey statistician's idea from Day 12 applied to treatment.

| Estimand | Joiners' weight | Non-joiners' weight | Idea |
|---|---|---|---|
| ATE | 1 / e(X) | 1 / (1 − e(X)) | both groups made to look like everyone |
| ATT | 1 | e(X) / (1 − e(X)) | non-joiners made to look like the joiners |

Type `ipw_weights`, `ipw` and `effective_sample_size` into `estimators.py`.

---

## 34.2 Horvitz–Thompson and Hajek estimators (50 min)

There are two ways to turn weights into an average:

- **Horvitz–Thompson (HT):** weighted sum divided by the **number of customers**. Unbiased in theory, but if the weights happen to sum to more than n, the whole estimate is inflated.
- **Hajek:** weighted sum divided by the **sum of the weights**: an ordinary weighted mean. Slightly biased in small samples, usually much less variable. This is the default in practice.

Create `day34/ipw.py`:

```python
# day34/ipw.py   (python -m day34.ipw)
"""
Inverse probability weighting: a joiner with a 25% chance of joining stands in for four customers
like them. Re-weighted, each group looks like the population the estimand is about.
"""
import numpy as np

from causalkit.estimators import effective_sample_size, ipw, ipw_weights, propensity
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)
ps = propensity(df)
d = df.joined.to_numpy()

for name, normalised in [("Horvitz-Thompson", False), ("Hajek (normalised)", True)]:
    e = ipw(df, ps, normalised=normalised)
    print(f"{name:<20} ATE {e['ATE']:7.1f}   ATT {e['ATT']:7.1f}")
print(f"{'Truth':<20} ATE {t['ATE']:7.1f}   ATT {t['ATT']:7.1f}")

w = ipw_weights(d, ps, "ATE")
print(f"\nATE weights: min {w.min():.2f}, median {np.median(w):.2f}, max {w.max():.1f}")
print(f"Effective sample size: {effective_sample_size(w[d == 1]):,.0f} of {(d == 1).sum():,} joiners, "
      f"{effective_sample_size(w[d == 0]):,.0f} of {(d == 0).sum():,} non-joiners")
```

```bash
python -m day34.ipw
```

```
Horvitz-Thompson     ATE   180.8   ATT   229.4
Hajek (normalised)   ATE   176.1   ATT   223.5
Truth                ATE   158.4   ATT   186.0

ATE weights: min 1.01, median 1.61, max 23.4
Effective sample size: 6,760 of 9,630 joiners, 7,879 of 10,370 non-joiners
```

Both are well off: +$18 on the ATE, +$37 on the ATT. G-computation got within $2 on the same data. Before blaming the method, look at the weights.

---

## 34.3 Weighted balance and the effective sample size (50 min)

**Concept.** The **effective sample size** (Kish): ESS = (Σw)² / Σw². With equal weights it equals n. When a few customers carry large weights, the estimate rests mostly on them, and the ESS falls. Here 9,630 joiners are "worth" 6,760 equally weighted ones: about 30% of the information is lost to weighting.

Create `day34/weighted_balance.py`:

```python
# day34/weighted_balance.py   (python -m day34.weighted_balance)
"""Do the weights balance the covariates? Weighted standardised mean differences, ATE and ATT weights."""
import polars as pl

from causalkit.estimators import COVARIATES, ipw_weights, propensity, smd
from causalkit.loyalty import public, world

df = public(world()).to_pandas()
ps = propensity(df)
d = df.joined.to_numpy()
t, c = d == 1, d == 0

rows = []
for col in COVARIATES:
    x = df[col]
    w_ate, w_att = ipw_weights(d, ps, "ATE"), ipw_weights(d, ps, "ATT")
    rows.append({"covariate": col,
                 "unweighted": round(smd(x[t], x[c]), 3),
                 "ATE weights": round(smd(x[t], x[c], w_ate[t], w_ate[c]), 3),
                 "ATT weights": round(smd(x[t], x[c], w_att[t], w_att[c]), 3)})
print(pl.DataFrame(rows))
```

```bash
python -m day34.weighted_balance
```

```
│ covariate    ┆ unweighted ┆ ATE weights ┆ ATT weights │
│ pre_spend    ┆ 0.649      ┆ 0.024       ┆ 0.051       │
│ pre_orders   ┆ 0.595      ┆ 0.013       ┆ 0.03        │
│ engaged      ┆ 0.465      ┆ -0.003      ┆ 0.001       │
│ tenure_years ┆ 0.553      ┆ 0.003       ┆ 0.0         │
│ age          ┆ -0.097     ┆ -0.003      ┆ -0.007      │
│ mobile       ┆ 0.017      ┆ 0.006       ┆ 0.011       │
```

Weighted balance is excellent. So the propensity model is fine, and the estimate is still $18 off. Is IPW biased here, or did this dataset just get unlucky?

---

## 34.4 Is IPW biased, or just noisy? (40 min)

One dataset can't tell you. The simulator can: draw 100 datasets from the same world and look at the **distribution** of errors. Create `day34/ipw_noise.py`:

```python
# day34/ipw_noise.py   (python -m day34.ipw_noise)
"""
Mini-project: is IPW wrong, or just noisy? Simulate 100 loyalty datasets (new customers each time)
and compare the error of IPW and g-computation against each dataset's own truth.
"""
import numpy as np
import polars as pl

from causalkit.estimators import g_computation, ipw
from causalkit.loyalty import public, truth, world

errors = {"IPW (Hajek)": [], "g-computation": []}
for s in range(100):
    full = world(seed=1000 + s)
    df = public(full).to_pandas()
    ate = truth(full)["ATE"]
    errors["IPW (Hajek)"].append(ipw(df)["ATE"] - ate)
    errors["g-computation"].append(g_computation(df)["ATE"] - ate)

print(pl.DataFrame([{"estimator": k, "average error (bias)": round(np.mean(v), 2),
                     "SD of error": round(np.std(v), 2), "worst error": round(max(v, key=abs), 1)}
                    for k, v in errors.items()]))
```

```bash
python -m day34.ipw_noise
```

```
│ estimator     ┆ average error (bias) ┆ SD of error ┆ worst error │
│ IPW (Hajek)   ┆ 1.17                 ┆ 17.37       ┆ -132.2      │
│ g-computation ┆ 0.08                 ┆ 1.98        ┆ -5.0        │
```

IPW is essentially **unbiased** (+$1 on average), but its standard deviation is **nine times** g-computation's, and one dataset in a hundred was $132 off. Our dataset's +$18 error was one ordinary draw from that wide distribution.

Why so noisy? Spending is very spread out (some customers spend $5,000), and IPW multiplies each customer's spending by their weight. One big spender with a weight of 20 moves the average by a lot. G-computation uses a model to smooth over those customers; IPW uses their raw spending.

**The lesson is about method, not just IPW:** when one estimate surprises you, simulate before you conclude. A single draw can't distinguish bias from noise.

---

## 34.5 Day 34 mini-project: Horvitz–Thompson vs Hajek (65 min)

Repeat 34.4 for both the HT and Hajek ATE estimators, using the same 100 seeds (`world(seed=1000 + s)`). Report the average error and SD of each. Create `day34/ht_vs_hajek.py`.

<details>
<summary>Solution</summary>

```python
# day34/ht_vs_hajek.py   (python -m day34.ht_vs_hajek)
import numpy as np

from causalkit.estimators import ipw, propensity
from causalkit.loyalty import public, truth, world

errs = {"Horvitz-Thompson": [], "Hajek": []}
for s in range(100):
    full = world(seed=1000 + s)
    df = public(full).to_pandas()
    ps, ate = propensity(df), truth(full)["ATE"]
    errs["Horvitz-Thompson"].append(ipw(df, ps, normalised=False)["ATE"] - ate)
    errs["Hajek"].append(ipw(df, ps, normalised=True)["ATE"] - ate)
for k, v in errs.items():
    print(f"{k:<17} average error {np.mean(v):+6.1f}   SD {np.std(v):5.1f}")
```

```
Horvitz-Thompson  average error   +0.2   SD  18.7
Hajek             average error   +1.2   SD  17.4
```

HT is less biased, Hajek less variable. Here the difference is small, because the propensity scores are moderate. When some scores are close to 0 or 1, HT's weights can sum to far more or less than n, and its variance explodes. That's why Hajek is the default. Neither fixes the real problem, the big weights themselves, which is tomorrow's topic.

</details>

---

## 34.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 34: IPW (HT and Hajek), weighted balance, ESS, Monte Carlo check"
git push
```

### Day 34 checklist

- [ ] I can write IPW weights for the ATE and the ATT and explain the pseudo-population
- [ ] I can explain the difference between Horvitz–Thompson and Hajek estimators
- [ ] I report weighted balance and the effective sample size with every IPW estimate
- [ ] I use simulation to tell bias from noise before trusting a surprising estimate

---

# DAY 35 — Overlap and Trimming

| Time | Block |
|---|---|
| 0:00 – 0:40 | 35.1 Positivity: the assumption you can see |
| 0:40 – 1:40 | 35.2 A world with poor overlap |
| 1:40 – 1:55 | Break |
| 1:55 – 2:45 | 35.3 Three remedies, three estimands |
| 2:45 – 3:20 | 35.4 Who gets trimmed? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 35.5 Day 35 mini-project: trimming rules compared |
| 4:40 – 5:00 | 35.6 Save your work with Git |

---

## 35.1 Positivity: the assumption you can see (40 min)

**Concept.** Overlap (positivity) requires that every kind of customer had some chance of both joining and not joining. If customers who spend over $3,000 **always** join, there's nobody to compare them with. Every method fails there, in its own way:

| Method | What happens without overlap |
|---|---|
| g-computation | silently **extrapolates** the outcome model into empty regions |
| matching | reuses the same few distant matches, or drops joiners via the caliper |
| IPW | weights explode: 1 / 0.002 = 500 |

Unlike unconfoundedness, overlap can be **checked**: plot the propensity scores (Day 32's figure) and count customers near 0 or 1.

---

## 35.2 A world with poor overlap (60 min)

`world(overlap=2.5)` makes selection 2.5 times sharper: the same customers, but joining is much more predictable. Create `day35/poor_overlap.py`:

```python
# day35/poor_overlap.py   (python -m day35.poor_overlap)
"""
Poor overlap: when the covariates almost DETERMINE who joins, some customers have no comparable
counterpart in the other group. Same customers, but joining is 2.5x more predictable.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causalkit.estimators import effective_sample_size, ipw, ipw_weights, propensity
from causalkit.loyalty import public, truth, world

for label, k in [("normal overlap", 1.0), ("poor overlap", 2.5)]:
    full = world(overlap=k)
    df = public(full).to_pandas()
    ps = propensity(df)
    d = df.joined.to_numpy()
    w = ipw_weights(d, ps, "ATE")
    print(f"{label}: propensity range [{ps.min():.4f}, {ps.max():.4f}], "
          f"share below 0.05 or above 0.95: {np.mean((ps < 0.05) | (ps > 0.95)):.1%}")
    print(f"   largest ATE weight {w.max():,.0f};  effective sample size {effective_sample_size(w):,.0f} of {len(w):,}")
    print(f"   IPW ATE {ipw(df, ps)['ATE']:.1f}   truth {truth(full)['ATE']:.1f}")

full = world(overlap=2.5)
df = public(full).to_pandas()
ps = propensity(df)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ps[df.joined == 0], bins=60, alpha=0.6, label="did not join", color="#d95926")
ax.hist(ps[df.joined == 1], bins=60, alpha=0.6, label="joined", color="#3987e5")
ax.set_xlabel("Estimated propensity to join"); ax.set_ylabel("Customers"); ax.legend()
ax.set_title("Poor overlap: piles near 0 and 1 with few comparisons")
fig.tight_layout(); fig.savefig("figures/day35_poor_overlap.png", dpi=150)
print("Saved figures/day35_poor_overlap.png")
```

```bash
python -m day35.poor_overlap
```

```
normal overlap: propensity range [0.0127, 0.9724], share below 0.05 or above 0.95: 0.4%
   largest ATE weight 23;  effective sample size 14,549 of 20,000
   IPW ATE 176.1   truth 158.4
poor overlap: propensity range [0.0000, 0.9996], share below 0.05 or above 0.95: 30.6%
   largest ATE weight 404;  effective sample size 2,273 of 20,000
   IPW ATE 205.9   truth 158.4
Saved figures/day35_poor_overlap.png
```

With poor overlap, almost a third of customers have scores below 0.05 or above 0.95, one customer carries a weight of 404, and 20,000 customers are worth 2,273. Open the figure: the two histograms barely touch.

**Rule of thumb.** If more than a few percent of scores are outside [0.05, 0.95], or the largest weight is a large share of the total, the ATE for everyone is **not identified well by these data**. Change the question before you change the estimator.

---

## 35.3 Three remedies, three estimands (50 min)

| Remedy | What it does | What it estimates |
|---|---|---|
| **Trimming** (Crump et al., 2009) | drop customers with e(X) outside [α, 1 − α], usually α = 0.1 | ATE for the **trimmed population** |
| **Overlap weights** (Li, Morgan & Zaslavsky, 2018) | weight joiners by 1 − e(X), non-joiners by e(X) | **ATO**: effect for customers who could plausibly go either way |
| **Change the estimand** | estimate the ATT instead of the ATE, if few joiners are unmatched | ATT |

All three buy precision by answering a **different question**. That's fine, as long as the report says so. "The programme raised spending by $160 among customers whose decision to join wasn't a foregone conclusion" is honest. Calling the same number "the effect" is not.

---

## 35.4 Who gets trimmed? (35 min)

**Try it.** In the poor-overlap world, trim to [0.05, 0.95]. Compare the true ATE for everyone with the true ATE for the customers you kept (use the hidden `_tau` column: we're grading, not analysing). Then compare the kept and dropped customers' average `pre_spend` and share `engaged`. Create `day35/who_is_trimmed.py`.

<details>
<summary>Solution</summary>

```python
# day35/who_is_trimmed.py   (python -m day35.who_is_trimmed)
import numpy as np

from causalkit.estimators import propensity
from causalkit.loyalty import public, world

full = world(overlap=2.5)
df = public(full).to_pandas()
ps = propensity(df)
keep = (ps >= 0.05) & (ps <= 0.95)
tau = full["_tau"].to_numpy()
print(f"True ATE, everyone:            {tau.mean():.1f}")
print(f"True ATE, kept after trimming: {tau[keep].mean():.1f}")
print(f"Average pre_spend, kept {df.pre_spend[keep].mean():.0f} vs dropped {df.pre_spend[~keep].mean():.0f}")
print(f"Share engaged, kept {df.engaged[keep].mean():.2f} vs dropped {df.engaged[~keep].mean():.2f}")
```

```
True ATE, everyone:            158.4
True ATE, kept after trimming: 161.9
Average pre_spend, kept 718 vs dropped 707
Share engaged, kept 0.64 vs dropped 0.39
```

The trimmed population's true effect differs by $3.5, and it is a different group of customers: engaged customers are kept more often (64% vs 39%). The dropped customers are the "sure things" at both ends: high-scoring customers who almost always join and low-scoring customers who almost never do. If the decision is about converting the hesitant middle, the trimmed estimand may actually be the better question.

</details>

---

## 35.5 Day 35 mini-project: trimming rules compared (65 min)

Create `day35/trimming.py`. Over 50 poor-overlap datasets, each estimate is compared with the true effect **for the population it describes**:

```python
# day35/trimming.py   (python -m day35.trimming)
"""
Mini-project: trimming (dropping customers with propensity outside [a, 1 - a]) and overlap weights.
Both trade the estimand for precision: each estimate is compared with the truth for ITS population.
50 simulated datasets with poor overlap.
"""
import numpy as np
import polars as pl

from causalkit.estimators import ipw, propensity
from causalkit.loyalty import public, world

RULES = [0.0, 0.01, 0.05, 0.10]
res = {a: {"err": [], "kept": []} for a in RULES}
ato = {"err": []}
for s in range(50):
    full = world(seed=2000 + s, overlap=2.5)
    df = public(full).to_pandas()
    ps = propensity(df)
    tau = full["_tau"].to_numpy()
    for a in RULES:
        keep = (ps >= a) & (ps <= 1 - a)
        est = ipw(df[keep].reset_index(drop=True), ps[keep])["ATE"]
        res[a]["err"].append(est - tau[keep].mean())          # truth for the trimmed population
        res[a]["kept"].append(keep.mean())
    # Overlap weights: joiners weighted by 1 - ps, non-joiners by ps (estimand: the 'overlap population')
    d, y = df.joined.to_numpy(), df.spend.to_numpy()
    est = np.average(y[d == 1], weights=1 - ps[d == 1]) - np.average(y[d == 0], weights=ps[d == 0])
    ato["err"].append(est - np.average(tau, weights=ps * (1 - ps)))

rows = [{"rule": "no trimming" if a == 0 else f"trim to [{a}, {1 - a}]",
         "customers kept": f"{np.mean(r['kept']):.0%}", "average error": round(np.mean(r["err"]), 1),
         "SD of error": round(np.std(r["err"]), 1)} for a, r in res.items()]
rows.append({"rule": "overlap weights (ATO)", "customers kept": "100%",
             "average error": round(np.mean(ato["err"]), 1), "SD of error": round(np.std(ato["err"]), 1)})
with pl.Config(fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day35.trimming
```

```
│ rule                  ┆ customers kept ┆ average error ┆ SD of error │
│ no trimming           ┆ 100%           ┆ 2.4           ┆ 116.1       │
│ trim to [0.01, 0.99]  ┆ 90%            ┆ 5.2           ┆ 16.6        │
│ trim to [0.05, 0.95]  ┆ 70%            ┆ 0.3           ┆ 8.7         │
│ trim to [0.1, 0.9]    ┆ 56%            ┆ -1.2          ┆ 7.0         │
│ overlap weights (ATO) ┆ 100%           ┆ 0.6           ┆ 3.5         │
```

- Untrimmed IPW is unbiased **on average** and useless in practice: SD $116.
- Trimming at 0.05 or 0.1 cuts the SD by more than 90%, while dropping 30–44% of customers.
- **Overlap weights** have the smallest error of all and drop nobody. Their weights are bounded between 0 and 1, so they can't explode. The price is an estimand (ATO) that emphasises customers near e(X) = 0.5.

Write in `day35/NOTES.md`: if the director asks "what's the effect of the programme?" in the poor-overlap world, what would you report, and how would you word the population it applies to?

---

## 35.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 35: overlap diagnostics, trimming and overlap weights"
git push
```

### Day 35 checklist

- [ ] I check overlap before estimating: propensity ranges, extreme shares, largest weight, ESS
- [ ] I can explain how g-computation, matching and IPW each fail without overlap
- [ ] I can trim or use overlap weights, and name the estimand each one targets
- [ ] I can describe who was trimmed, and whether that matters for the decision

---

# DAY 36 — Doubly Robust Estimation (AIPW)

| Time | Block |
|---|---|
| 0:00 – 0:45 | 36.1 Two models, one estimator |
| 0:45 – 1:35 | 36.2 AIPW and its standard errors |
| 1:35 – 1:50 | Break |
| 1:50 – 2:50 | 36.3 Double robustness, demonstrated |
| 2:50 – 3:20 | 36.4 Do the confidence intervals work? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 36.5 Day 36 mini-project: machine learning and cross-fitting |
| 4:40 – 5:00 | 36.6 Save your work with Git |

---

## 36.1 Two models, one estimator (45 min)

**Concept.** G-computation relies on the outcome model; IPW relies on the propensity model. **Augmented IPW (AIPW)** uses both. For each customer:

```
ψ = μ₁(X) − μ₀(X)                       ← g-computation's prediction of their effect
    + D · (Y − μ₁(X)) / e(X)             ← IPW-weighted correction: joiners' prediction errors
    − (1 − D) · (Y − μ₀(X)) / (1 − e(X)) ← IPW-weighted correction: non-joiners' prediction errors
```

The ATE estimate is the average of ψ. Read it as "g-computation, plus a weighted average of g-computation's mistakes".

- If the **outcome model is right**, its prediction errors average to zero, so the correction does nothing harmful, even with a wrong propensity score.
- If the **propensity model is right**, the weighted correction removes whatever bias the outcome model has.

Either one being right is enough: that's **double robustness**. A bonus: the variance of ψ gives the standard error directly, SE = sd(ψ) / √n. No bootstrap needed. (ψ is the **efficient influence function**; you'll meet it again in Module 6.)

---

## 36.2 AIPW and its standard errors (50 min)

Type `aipw` and its two helpers into `estimators.py`. Two details:

- **Cross-fitting:** customers are split into 5 folds, and each customer's predictions come from models fitted on the **other** four. With simple regressions it hardly matters; with machine learning it matters a lot (36.5).
- **Clipping:** propensity scores are clipped to [0.01, 0.99] so no weight exceeds 100.

Create `day36/aipw.py`:

```python
# day36/aipw.py   (python -m day36.aipw)
"""
AIPW = g-computation + an IPW correction of its mistakes. Consistent if EITHER the outcome model
OR the propensity model is right ('doubly robust'), with standard errors from a simple formula.
"""
from causalkit.estimators import aipw, g_computation, ipw
from causalkit.loyalty import public, truth, world

full = world()
df = public(full).to_pandas()
t = truth(full)

g, w, a = g_computation(df), ipw(df), aipw(df)
print(f"{'':<16}{'ATE':>8}{'ATT':>8}")
print(f"{'g-computation':<16}{g['ATE']:8.1f}{g['ATT']:8.1f}")
print(f"{'IPW':<16}{w['ATE']:8.1f}{w['ATT']:8.1f}")
print(f"{'AIPW':<16}{a['ATE']:8.1f}{a['ATT']:8.1f}   SEs {a['ATE_se']:.1f} / {a['ATT_se']:.1f}")
print(f"{'Truth':<16}{t['ATE']:8.1f}{t['ATT']:8.1f}")
```

```bash
python -m day36.aipw
```

```
                     ATE     ATT
g-computation      156.8   183.9
IPW                176.1   223.5
AIPW               157.8   184.6   SEs 2.1 / 2.7
Truth              158.4   186.0
```

AIPW sits with g-computation, close to the truth, not with IPW. When the outcome model is good, AIPW inherits its precision; the IPW part only corrects what the model gets wrong.

---

## 36.3 Double robustness, demonstrated (60 min)

Break each model on purpose, over 40 datasets. The "wrong" outcome model is Day 31's linear one; the "wrong" propensity model forgets `engaged`. Create `day36/double_robustness.py`:

```python
# day36/double_robustness.py   (python -m day36.double_robustness)
"""
Double robustness, demonstrated: break the outcome model, the propensity model, or both,
across 40 simulated datasets. Wrong outcome model = the plain linear model of Day 31 (it misses
the curvature in pre_spend and tenure); wrong propensity model = forgets 'engaged' (a real driver of joining).
"""
import numpy as np
import polars as pl

from causalkit.estimators import COVARIATES, aipw, g_computation, ipw, propensity
from causalkit.loyalty import public, truth, world

WRONG_OUTCOME = "spend ~ " + " + ".join(COVARIATES)
WRONG_PS = "joined ~ np.log(pre_spend) + pre_orders + tenure_years + age + mobile"
NO_ENGAGED_OUTCOME = "spend ~ " + " + ".join(c for c in COVARIATES if c != "engaged")

cases = {
    "g-computation, wrong outcome model": lambda df: g_computation(df, WRONG_OUTCOME)["ATE"],
    "IPW, wrong propensity model": lambda df: ipw(df, propensity(df, WRONG_PS))["ATE"],
    "AIPW, both right": lambda df: aipw(df)["ATE"],
    "AIPW, outcome wrong, propensity right": lambda df: aipw(df, outcome_formula=WRONG_OUTCOME)["ATE"],
    "AIPW, outcome right, propensity wrong": lambda df: aipw(df, ps_formula=WRONG_PS)["ATE"],
    "AIPW, both models miss 'engaged'": lambda df: aipw(df, outcome_formula=NO_ENGAGED_OUTCOME,
                                                         ps_formula=WRONG_PS)["ATE"],
}
errors = {k: [] for k in cases}
for s in range(40):
    full = world(seed=3000 + s)
    df = public(full).to_pandas()
    ate = truth(full)["ATE"]
    for k, f in cases.items():
        errors[k].append(f(df) - ate)

with pl.Config(fmt_str_lengths=45, tbl_width_chars=100):
    print(pl.DataFrame([{"estimator": k, "average error": round(np.mean(v), 1), "SD": round(np.std(v), 1)}
                        for k, v in errors.items()]))
```

```bash
python -m day36.double_robustness
```

```
│ estimator                             ┆ average error ┆ SD   │
│ g-computation, wrong outcome model    ┆ -24.2         ┆ 4.9  │
│ IPW, wrong propensity model           ┆ 25.1          ┆ 14.0 │
│ AIPW, both right                      ┆ 0.6           ┆ 1.9  │
│ AIPW, outcome wrong, propensity right ┆ -0.5          ┆ 7.1  │
│ AIPW, outcome right, propensity wrong ┆ 0.5           ┆ 1.9  │
│ AIPW, both models miss 'engaged'      ┆ 25.8          ┆ 7.1  │
```

Read it row by row:

1. Each single-model method is biased by about $25 when its model is wrong.
2. AIPW is unbiased if **either** model is right. When the outcome model is wrong, AIPW pays in variance (SD 7.1 instead of 1.9), because the IPW correction now does real work.
3. The last row is the warning. Double robustness protects against a **wrong functional form**, not against a **missing confounder**. If `engaged` isn't in the data, no model can use it, and AIPW is as biased as everything else.

---

## 36.4 Do the confidence intervals work? (30 min)

**Try it.** Using the same 40 seeds (`world(seed=3000 + s)`), count how often AIPW's 95% interval (ATE ± 1.96 × SE) contains the true ATE. Create `day36/coverage.py`.

<details>
<summary>Solution</summary>

```python
# day36/coverage.py   (python -m day36.coverage)
import numpy as np

from causalkit.estimators import aipw
from causalkit.loyalty import public, truth, world

hits, n = 0, 40
for s in range(n):
    full = world(seed=3000 + s)
    a = aipw(public(full).to_pandas())
    hits += abs(a["ATE"] - truth(full)["ATE"]) <= 1.96 * a["ATE_se"]
print(f"95% CIs that contain the true ATE: {hits} of {n}")
```

```
95% CIs that contain the true ATE: 38 of 40
```

95%, as promised. Coverage checks like this one are the best test of any standard-error formula: Day 21's A/A tests did the same job for experiments.

</details>

---

## 36.5 Day 36 mini-project: machine learning and cross-fitting (65 min)

On Day 31, gradient boosting in g-computation overshot the ATT by $24. Does AIPW with boosted models do better, and does cross-fitting matter? Create `day36/aipw_ml.py`:

```python
# day36/aipw_ml.py   (python -m day36.aipw_ml)
"""
Mini-project: machine learning for the nuisance models, with no formulas to get wrong.
Day 31 showed that plugging gradient boosting into g-computation overshoots. AIPW with cross-fitting
corrects it, and the influence-function SE gives a confidence interval. (This is the core of
Double Machine Learning, Module 6.)
"""
import polars as pl

from causalkit.estimators import aipw
from causalkit.loyalty import public, truth, world
from day31.specification_table import g_comp_boosting

full = world()
df = public(full).to_pandas()
t = truth(full)

rows = []
plain = g_comp_boosting(df)
rows.append({"estimator": "boosting g-computation (Day 31)", "ATE": round(plain["ATE"], 1),
             "ATT": round(plain["ATT"], 1), "ATT 95% CI": "none"})
for label, folds in [("AIPW + boosting, no cross-fitting", 1), ("AIPW + boosting, 5-fold cross-fitting", 5)]:
    a = aipw(df, outcome="boosting", ps_model="boosting", folds=folds)
    rows.append({"estimator": label, "ATE": round(a["ATE"], 1), "ATT": round(a["ATT"], 1),
                 "ATT 95% CI": f"[{a['ATT'] - 1.96 * a['ATT_se']:.1f}, {a['ATT'] + 1.96 * a['ATT_se']:.1f}]"})
rows.append({"estimator": "TRUTH", "ATE": round(t["ATE"], 1), "ATT": round(t["ATT"], 1), "ATT 95% CI": ""})
with pl.Config(fmt_str_lengths=45, tbl_width_chars=110):
    print(pl.DataFrame(rows))
```

```bash
python -m day36.aipw_ml
```

```
│ estimator                             ┆ ATE   ┆ ATT   ┆ ATT 95% CI     │
│ boosting g-computation (Day 31)       ┆ 170.0 ┆ 209.7 ┆ none           │
│ AIPW + boosting, no cross-fitting     ┆ 167.5 ┆ 205.2 ┆ [196.1, 214.2] │
│ AIPW + boosting, 5-fold cross-fitting ┆ 164.2 ┆ 198.9 ┆ [180.4, 217.5] │
│ TRUTH                                 ┆ 158.4 ┆ 186.0 ┆                │
```

- Without cross-fitting, each customer's prediction error comes from a model that has **seen** that customer, so errors look smaller than they are. The correction is too small, and the CI is too narrow: it misses the truth.
- With cross-fitting, the estimate moves towards the truth, and the CI is honest (and wider): it contains $186.
- With no formulas to write, boosting still leaves some bias at n = 20,000. With hand-crafted features (the flexible model), AIPW was better. Machine learning is a tool for when you **don't** know the functional form, not a replacement for thinking about it.

This combination, ML models + an orthogonal score like ψ + cross-fitting, is **Double/Debiased Machine Learning**, the core of Module 6.

---

## 36.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 36: AIPW, double robustness, coverage, cross-fitting with ML"
git push
```

**Coming next (Days 37–40).** Everything so far assumed that every confounder is measured. What if loyal fans, people who simply like NorthMart, both join and spend more, and nobody measures "liking NorthMart"? Days 37–38 use the random $20 invitation as an **instrument**; Day 39 races every estimator in both worlds; Day 40 is the capstone.

### Day 36 checklist

- [ ] I can write the AIPW score and explain each of its three terms
- [ ] I can show by simulation that AIPW is consistent if either model is right
- [ ] I can explain why double robustness doesn't protect against a missing confounder
- [ ] I use cross-fitting whenever the nuisance models are machine-learning models
- [ ] I check a standard-error formula by its coverage in simulation

