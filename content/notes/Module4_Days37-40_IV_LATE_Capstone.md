# Module 4 · Selection on Observables & Instrumental Variables
## Days 37–40: Instrumental Variables, LATE, the Horse Race and the Capstone

**Goal by end of Day 40:** Students can estimate a causal effect with an instrument (Wald and 2SLS), check instrument strength, argue for the exclusion restriction and show what happens when it fails, describe the compliers a LATE applies to, choose between selection-on-observables and IV for a real decision, and turn the estimate into a recommendation with money attached.

---

# DAY 37 — Instrumental Variables: Intuition and 2SLS

| Time | Block |
|---|---|
| 0:00 – 0:40 | 37.1 When selection on observables fails |
| 0:40 – 1:30 | 37.2 An instrument: the random invitation |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 37.3 The Wald estimator and the LATE |
| 2:30 – 3:15 | 37.4 Two-stage least squares |
| 3:15 – 3:30 | Break |
| 3:30 – 4:40 | 37.5 Day 37 mini-project: IV with covariates |
| 4:40 – 5:00 | 37.6 Save your work with Git |

---

## 37.1 When selection on observables fails (40 min)

The brand team has a worry: some customers simply **like NorthMart**. They join the programme and they spend more, and the data has no column for "likes NorthMart". In DAG terms (Day 14), *brand affinity* is an unmeasured confounder: an open backdoor path that no adjustment can close.

World B (`hidden=1.0`) builds that in. Create `day37/selection_fails.py`:

```python
# day37/selection_fails.py   (python -m day37.selection_fails)
"""
Same customers, one change: an UNOBSERVED trait (brand affinity) now makes customers both more
likely to join and bigger spenders. Everything from Days 31-36 assumes no such trait exists.
"""
from causalkit.estimators import aipw, g_computation
from causalkit.loyalty import public, truth, world

for label, h in [("selection on observables holds", 0.0), ("hidden brand affinity", 1.0)]:
    full = world(hidden=h)
    df = public(full).to_pandas()
    t, g, a = truth(full), g_computation(df), aipw(df)
    print(f"{label}:")
    print(f"   g-computation ATE {g['ATE']:6.1f}   AIPW ATE {a['ATE']:6.1f} (SE {a['ATE_se']:.1f})   truth {t['ATE']:6.1f}")
    print(f"   g-computation ATT {g['ATT']:6.1f}   AIPW ATT {a['ATT']:6.1f} (SE {a['ATT_se']:.1f})   truth {t['ATT']:6.1f}")
```

```bash
python -m day37.selection_fails
```

```
selection on observables holds:
   g-computation ATE  156.8   AIPW ATE  157.8 (SE 2.1)   truth  158.4
   g-computation ATT  183.9   AIPW ATT  184.6 (SE 2.7)   truth  186.0
hidden brand affinity:
   g-computation ATE  230.6   AIPW ATE  232.5 (SE 2.4)   truth  158.4
   g-computation ATT  255.9   AIPW ATT  256.3 (SE 3.3)   truth  183.4
```

In world B, every Days 31–36 method overstates the effect by $72–74, **with a standard error of $2.4**. That's the most dangerous output in applied work: precise and wrong. Nothing in the data warns you: balance, overlap and specification tables all look the same as in world A.

---

## 37.2 An instrument: the random invitation (50 min)

Last year, NorthMart sent half of its customers, chosen **at random**, a $20 sign-up offer. An **instrument** Z is a variable that:

| Condition | Meaning here | Checkable? |
|---|---|---|
| **Relevance** | Z changes the treatment | yes: invited customers join more often |
| **Independence** | Z is as good as random | yes, by design: it was randomised |
| **Exclusion** | Z affects the outcome **only through** the treatment | no: an argument (Day 38) |
| **Monotonicity** | nobody joins *because* they weren't invited | no: usually obvious |

Draw it as a DAG: `invited → joined → spend`, with `affinity → joined` and `affinity → spend`. There is no arrow from `affinity` to `invited` (random) and none from `invited` to `spend` (exclusion). Any difference in spending between invited and non-invited customers must therefore travel **through joining**.

---

## 37.3 The Wald estimator and the LATE (45 min)

**Concept.** Two randomised comparisons:

- **First stage:** invited vs not, in joining rates.
- **Intention-to-treat (ITT):** invited vs not, in spending. This is a clean experimental effect, of the *invitation*.

If the invitation raised joining by 16 percentage points and spending by $20 per customer, then those extra 16% of joiners must account for the whole $20: the effect per extra joiner is $20 / 0.16. That ratio is the **Wald estimator**.

Who are those 16%? **Compliers**: customers who join if invited and don't if not. The Wald estimator gives the effect for them: the **local average treatment effect (LATE)** (Imbens & Angrist, 1994). Customers who join regardless (**always-takers**) or never join (**never-takers**) contribute nothing, because the invitation didn't change what they did.

Create `day37/wald.py`:

```python
# day37/wald.py   (python -m day37.wald)
"""
The invitation was RANDOM: half of customers got a $20 sign-up offer. It can't be related to brand
affinity. Use it as an instrument: the effect of joining = (effect of the invitation on spend) /
(effect of the invitation on joining).
"""
import polars as pl

from causalkit.loyalty import public, truth, world

full = world(hidden=1.0)
df = public(full)
t = truth(full)


def mean(col: str, invited: int) -> float:
    return df.filter(pl.col("invited") == invited)[col].mean()


itt = mean("spend", 1) - mean("spend", 0)
first_stage = mean("joined", 1) - mean("joined", 0)
print(f"Joined: {mean('joined', 1):.1%} if invited, {mean('joined', 0):.1%} if not -> first stage {first_stage:.3f}")
print(f"Spend:  ${mean('spend', 1):,.0f} if invited, ${mean('spend', 0):,.0f} if not -> ITT ${itt:.1f}")
print(f"Wald estimate = ITT / first stage = {itt / first_stage:.1f}")
print(f"Truth: LATE {t['LATE']:.1f} (compliers are {t['share_compliers']:.1%} of customers);"
      f" ATE {t['ATE']:.1f}; ATT {t['ATT']:.1f}")
```

```bash
python -m day37.wald
```

```
Joined: 56.3% if invited, 40.2% if not -> first stage 0.161
Spend:  $1,056 if invited, $1,035 if not -> ITT $20.4
Wald estimate = ITT / first stage = 126.7
Truth: LATE 153.6 (compliers are 15.9% of customers); ATE 158.4; ATT 183.4
```

$127 against a true LATE of $154. Far closer than AIPW's $232, but is $27 a lot? We need a standard error.

---

## 37.4 Two-stage least squares (45 min)

**Concept.** 2SLS gives the same estimate, plus standard errors and room for covariates:

1. **Stage 1:** regress `joined` on `invited` (and covariates); keep the predicted values. They contain only the variation in joining caused by the random invitation.
2. **Stage 2:** regress `spend` on predicted joining (and the same covariates).

Type `IV_CONTROLS` and `two_sls` into `estimators.py`. Create `day37/two_sls.py`:

```python
# day37/two_sls.py   (python -m day37.two_sls)
"""
Two-stage least squares (2SLS), by hand and with linearmodels.
Stage 1: predict joining from the invitation. Stage 2: regress spend on PREDICTED joining.
"""
import statsmodels.formula.api as smf

from causalkit.estimators import two_sls
from causalkit.loyalty import public, truth, world

full = world(hidden=1.0)
df = public(full).to_pandas()
t = truth(full)

# By hand: right estimate, WRONG standard error (stage 2 ignores that stage 1 was estimated)
df["joined_hat"] = smf.ols("joined ~ invited", df).fit().fittedvalues
by_hand = smf.ols("spend ~ joined_hat", df).fit()
print(f"2SLS by hand:          {by_hand.params['joined_hat']:7.1f}  (SE {by_hand.bse['joined_hat']:.1f}  <- not valid)")

iv = two_sls(df, controls=None)
print(f"2SLS, linearmodels:    {iv.params['joined']:7.1f}  (SE {iv.std_errors['joined']:.1f})")
print(f"First-stage F statistic: {iv.first_stage.diagnostics['f.stat'].iloc[0]:.0f}")
print(f"Truth: LATE {t['LATE']:.1f}")
```

```bash
python -m day37.two_sls
```

```
2SLS by hand:            126.7  (SE 65.9  <- not valid)
2SLS, linearmodels:      126.7  (SE 63.8)
First-stage F statistic: 532
Truth: LATE 153.6
```

- Doing it by hand gives the right estimate but the **wrong standard error**: the second-stage regression treats predicted joining as data, not as an estimate. Always use an IV routine (`linearmodels.iv.IV2SLS`).
- The SE is $64. The truth is well within one SE. IV removed the $74 bias, but the price is a standard error **more than 25 times larger** than AIPW's. Only 16% of customers' behaviour carries the signal.
- First-stage F = 532: the instrument is strong (Day 38).

---

## 37.5 Day 37 mini-project: IV with covariates (70 min)

The invitation is random, so covariates aren't needed for identification. But spend is very predictable from past spend, so covariates should remove noise, just as CUPED did on Day 25. Create `day37/iv_with_covariates.py`:

```python
# day37/iv_with_covariates.py   (python -m day37.iv_with_covariates)
"""
Mini-project: the invitation is random, so covariates aren't needed for identification.
But spend is very predictable from past spend, so covariates remove noise, exactly like CUPED (Day 25).
Compare OLS, AIPW and 2SLS with and without covariates, against the truths they target.
"""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.estimators import OUTCOME_FORMULA, aipw, two_sls
from causalkit.loyalty import public, truth, world

full = world(hidden=1.0)
df = public(full).to_pandas()
t = truth(full)

ols = smf.ols(OUTCOME_FORMULA.replace("spend ~", "spend ~ joined +"), df).fit(cov_type="HC2")
a = aipw(df)
iv0, iv1 = two_sls(df, controls=None), two_sls(df)
rows = [
    {"estimator": "OLS with flexible controls", "estimate": ols.params["joined"], "se": ols.bse["joined"],
     "targets": "(a weighted average)", "truth": None},
    {"estimator": "AIPW", "estimate": a["ATE"], "se": a["ATE_se"], "targets": "ATE", "truth": t["ATE"]},
    {"estimator": "2SLS, no covariates", "estimate": iv0.params["joined"], "se": iv0.std_errors["joined"],
     "targets": "LATE", "truth": t["LATE"]},
    {"estimator": "2SLS with covariates", "estimate": iv1.params["joined"], "se": iv1.std_errors["joined"],
     "targets": "LATE", "truth": t["LATE"]},
]
table = pl.DataFrame(rows).with_columns(pl.col("estimate", "se", "truth").round(1))
with pl.Config(fmt_str_lengths=30, tbl_width_chars=110):
    print(table)
```

```bash
python -m day37.iv_with_covariates
```

```
│ estimator                  ┆ estimate ┆ se   ┆ targets              ┆ truth │
│ OLS with flexible controls ┆ 226.1    ┆ 2.4  ┆ (a weighted average) ┆ null  │
│ AIPW                       ┆ 232.5    ┆ 2.4  ┆ ATE                  ┆ 158.4 │
│ 2SLS, no covariates        ┆ 126.7    ┆ 63.8 ┆ LATE                 ┆ 153.6 │
│ 2SLS with covariates       ┆ 141.3    ┆ 13.5 ┆ LATE                 ┆ 153.6 │
```

Covariates cut the 2SLS standard error from $64 to $13.5, a fivefold gain, equivalent to having 22 times as many customers. The 95% CI [$115, $168] now excludes AIPW's $232: in this world, the data can **reject** the selection-on-observables answer.

**Try it.** Run AIPW and 2SLS (with covariates) in world A, where there is no hidden confounder. Do they agree? Create `day37/ols_vs_iv_world_a.py`.

<details>
<summary>Solution</summary>

```python
# day37/ols_vs_iv_world_a.py   (python -m day37.ols_vs_iv_world_a)
from causalkit.estimators import aipw, two_sls
from causalkit.loyalty import public, truth, world

full = world(hidden=0.0)
df = public(full).to_pandas()
t, a, iv = truth(full), aipw(df), two_sls(df)
print(f"AIPW ATE {a['ATE']:.1f} (SE {a['ATE_se']:.1f}) vs truth {t['ATE']:.1f}")
print(f"2SLS     {iv.params['joined']:.1f} (SE {iv.std_errors['joined']:.1f}) vs LATE {t['LATE']:.1f}")
```

```
AIPW ATE 157.8 (SE 2.1) vs truth 158.4
2SLS     151.9 (SE 9.6) vs LATE 156.0
```

Both are right, each for its own estimand, and AIPW is about 4.5 times more precise. When selection on observables holds, IV is the expensive option. The **comparison** between the two is informative: they agree in world A and disagree by $90 in world B. That's the logic of the **Durbin–Wu–Hausman test**: a large gap between a consistent-but-noisy estimate and an efficient-if-assumptions-hold one is evidence against the assumptions. (Here it's only suggestive: the two target different estimands, ATE vs LATE.)

</details>

---

## 37.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 37: hidden confounding, Wald estimator, 2SLS, IV with covariates"
git push
```

### Day 37 checklist

- [ ] I can recognise when an unmeasured confounder makes all of Days 31–36 fail
- [ ] I can state the four instrument conditions and say which are checkable
- [ ] I can compute the Wald estimator from a first stage and an ITT
- [ ] I can explain what a LATE is and who the compliers are
- [ ] I run 2SLS with an IV routine and use covariates to cut its standard error

---

# DAY 38 — Weak Instruments and LATE

| Time | Block |
|---|---|
| 0:00 – 0:30 | 38.1 Three ways IV goes wrong |
| 0:30 – 1:30 | 38.2 Weak instruments |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 38.3 The exclusion restriction |
| 2:40 – 3:20 | 38.4 Whose effect is the LATE? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 38.5 Day 38 mini-project: profile the compliers |
| 4:40 – 5:00 | 38.6 Save your work with Git |

---

## 38.1 Three ways IV goes wrong (30 min)

IV fixed world B, but it rests on its own assumptions. Today breaks them one at a time:

| Problem | Symptom | Checkable? |
|---|---|---|
| **Weak instrument** | small first stage; wild, skewed estimates | yes: first-stage F |
| **Exclusion violation** | the instrument affects the outcome directly | no: argue it, or design it away |
| **Wrong population** | the LATE describes compliers, not the customers the decision is about | partly: profile the compliers |

---

## 38.2 Weak instruments (60 min)

**Concept.** IV divides by the first stage. When the first stage is close to zero, small amounts of noise in the numerator become huge swings in the ratio. The rule of thumb (Staiger & Stock, 1997): **first-stage F above 10**. Recent work (Lee, McCrary, Moreira & Porter, 2022) suggests much higher, around 104, for standard 5% tests to be reliable.

`world(invite_effect=...)` controls how strongly the invitation shifts joining. Create `day38/weak_instrument.py`, which runs 60 datasets per strength:

```python
# day38/weak_instrument.py   (python -m day38.weak_instrument)
"""
Weak instruments: what if the $20 offer barely changed who joined? 60 simulated datasets per
strength. Watch the first-stage F, the drift towards the OLS bias, and the spread of the estimates.
(We report percentiles: with very weak instruments a few estimates are absurd, e.g. +16,000.)
"""
import numpy as np
import polars as pl

from causalkit.estimators import two_sls
from causalkit.loyalty import public, truth, world

rows = []
for strength in [0.9, 0.3, 0.1, 0.05]:
    est, F, cover = [], [], 0
    for s in range(60):
        full = world(seed=4000 + s, hidden=1.0, invite_effect=strength)
        iv = two_sls(public(full).to_pandas())
        b, se = iv.params["joined"], iv.std_errors["joined"]
        late = truth(full)["LATE"]
        est.append(b - late)
        F.append(iv.first_stage.diagnostics["f.stat"].iloc[0])
        cover += abs(b - late) <= 1.96 * se
    rows.append({"invitation effect (log-odds)": strength, "median first-stage F": round(np.median(F), 1),
                 "median error": round(np.median(est), 1),
                 "middle 50% of errors": f"[{np.percentile(est, 25):.0f}, {np.percentile(est, 75):.0f}]",
                 "95% CI coverage": f"{cover / 60:.0%}"})
with pl.Config(tbl_width_chars=120):
    print(pl.DataFrame(rows))
print("For reference, OLS with controls overstates the effect by about +70 in this world.")
```

```bash
python -m day38.weak_instrument
```

```
│ invitation effect (log-odds) ┆ median first-stage F ┆ median error ┆ middle 50% of errors ┆ 95% CI coverage │
│ 0.9                          ┆ 607.7                ┆ 2.4          ┆ [-10, 13]            ┆ 92%             │
│ 0.3                          ┆ 71.2                 ┆ 9.2          ┆ [-29, 36]            ┆ 93%             │
│ 0.1                          ┆ 9.3                  ┆ 17.1         ┆ [-80, 104]           ┆ 98%             │
│ 0.05                         ┆ 3.0                  ┆ 41.4         ┆ [-81, 199]           ┆ 98%             │
For reference, OLS with controls overstates the effect by about +70 in this world.
```

With F ≈ 3, half the estimates fall in a $280-wide range, and the median error ($41) is drifting towards OLS's bias. A weak instrument doesn't just add noise: **2SLS is biased towards OLS**, because the first stage starts fitting noise that's correlated with the confounder.

Why use percentiles instead of the mean and SD? With a weak first stage, some datasets divide by almost zero and produce estimates in the thousands; the mean is meaningless. (The CIs show high coverage here only because they are enormous.)

**Always report the first-stage F.** If it's below 10, report the reduced form (the ITT) and an Anderson–Rubin confidence set instead of a 2SLS point estimate.

---

## 38.3 The exclusion restriction (55 min)

**Concept.** The invitation may affect spending **only** by changing who joins. But the $20 bonus is money. Suppose invited customers spend part of the $20 at NorthMart even when they don't join. Then the instrument has a direct path to the outcome, and IV attributes that spending to joining.

`world(bonus_spend_effect=...)` adds that leak. Create `day38/exclusion_violation.py`:

```python
# day38/exclusion_violation.py   (python -m day38.exclusion_violation)
"""
The exclusion restriction: the invitation may affect spend ONLY by changing who joins.
But the $20 bonus is money to spend. If invited customers spend part of it even without joining,
the instrument has a direct path to the outcome, and IV divides that leak by a small first stage.
"""
from causalkit.estimators import two_sls
from causalkit.loyalty import public, truth, world

for leak in [0, 5, 10, 20]:
    full = world(hidden=1.0, bonus_spend_effect=leak)
    iv = two_sls(public(full).to_pandas())
    print(f"Invitation adds ${leak:>2} of spend directly: 2SLS {iv.params['joined']:6.1f} "
          f"(SE {iv.std_errors['joined']:.1f})   truth LATE {truth(full)['LATE']:.1f}")
print("Bias is roughly leak / first stage = leak / 0.16, i.e. about 6x the leak.")
```

```bash
python -m day38.exclusion_violation
```

```
Invitation adds $ 0 of spend directly: 2SLS  141.3 (SE 13.5)   truth LATE 153.6
Invitation adds $ 5 of spend directly: 2SLS  172.1 (SE 13.3)   truth LATE 153.6
Invitation adds $10 of spend directly: 2SLS  202.9 (SE 13.1)   truth LATE 153.6
Invitation adds $20 of spend directly: 2SLS  264.6 (SE 13.2)   truth LATE 153.6
Bias is roughly leak / first stage = leak / 0.16, i.e. about 6x the leak.
```

A $5 leak becomes a $31 bias; a $20 leak becomes $123. The bias equals the leak divided by the first stage, so **the weaker the instrument, the more any exclusion violation is amplified**. And the standard error doesn't change: nothing in the output warns you.

The exclusion restriction can't be tested; it has to be argued, and ideally designed in. NorthMart could have made the bonus redeemable only as programme points, so it can't be spent outside the programme. Write in `day38/NOTES.md` two other ways the invitation could affect spending directly (hint: think about what an email from NorthMart does).

---

## 38.4 Whose effect is the LATE? (40 min)

**Concept.** The LATE is the effect for **compliers**: customers whose joining was decided by the invitation. Not for always-takers (the fans who join anyway) and not for never-takers. Three effects, three groups, three decisions:

| Effect | For whom | Decision it informs |
|---|---|---|
| ATT | current members | was the programme worth it for those who joined? |
| ATE | everyone | auto-enrol all customers? |
| LATE | customers a $20 invitation persuades | is the invitation campaign worth running? |

In world B, the LATE ($154) is close to the ATE ($158), but the ATT is $183. That's not an accident of the method: compliers are less extreme than self-selected joiners, as the mini-project shows.

---

## 38.5 Day 38 mini-project: profile the compliers (65 min)

We can't list the compliers (we never see what a customer would have done under the other invitation status), but we **can** describe them. The average of any covariate X among compliers is:

```
( E[X · joined | invited] − E[X · joined | not invited] ) / first stage
```

Create `day38/compliers.py`:

```python
# day38/compliers.py   (python -m day38.compliers)
"""
Who are the compliers, the customers the LATE is about? We can't list them, but we CAN describe them:
the average of any covariate X among compliers is
    (E[X * joined | invited] - E[X * joined | not invited]) / first stage.
"""
import polars as pl

from causalkit.loyalty import public, truth, world

full = world(hidden=1.0)
df = public(full)
inv, not_inv = df.filter(pl.col("invited") == 1), df.filter(pl.col("invited") == 0)
first_stage = inv["joined"].mean() - not_inv["joined"].mean()

rows = []
for col in ["pre_spend", "engaged", "tenure_years", "age"]:
    complier_mean = ((inv[col] * inv["joined"]).mean() - (not_inv[col] * not_inv["joined"]).mean()) / first_stage
    rows.append({"covariate": col,
                 "all customers": round(df[col].mean(), 2),
                 "joined": round(df.filter(pl.col("joined") == 1)[col].mean(), 2),
                 "compliers (estimated)": round(complier_mean, 2),
                 "compliers (truth)": round(full.filter(pl.col("_complier") == 1)[col].mean(), 2)})
print(pl.DataFrame(rows))
t = truth(full)
print(f"Effects: LATE {t['LATE']:.1f}   ATE {t['ATE']:.1f}   ATT {t['ATT']:.1f}")
```

```bash
python -m day38.compliers
```

```
│ covariate    ┆ all customers ┆ joined ┆ compliers (estimated) ┆ compliers (truth) │
│ pre_spend    ┆ 714.65        ┆ 887.98 ┆ 625.82                ┆ 668.61            │
│ engaged      ┆ 0.56          ┆ 0.67   ┆ 0.61                  ┆ 0.58              │
│ tenure_years ┆ 4.01          ┆ 4.59   ┆ 3.94                  ┆ 4.0               │
│ age          ┆ 45.19         ┆ 44.4   ┆ 44.2                  ┆ 44.84             │
Effects: LATE 153.6   ATE 158.4   ATT 183.4
```

Joiners spent $888 last year; compliers about $650, slightly *below* the average customer. The $20 invitation persuades **moderate** spenders, while the big spenders were joining anyway. Since the effect grows with past spending, that's why the LATE is below the ATT.

The estimates are noisy (they divide by a first stage of 0.16), but they get the direction right. Put this table in every IV report: it tells the reader whom the number describes.

**Try it.** Without code: if NorthMart doubled the bonus to $40, the first stage would grow. Would you expect the new LATE to be larger or smaller than $154? Why?

<details>
<summary>Solution</summary>

Probably **smaller**. A bigger bonus persuades customers who were further from joining: on average, lower spenders and less engaged. Their effect from joining is smaller (the effect grows with pre_spend and engagement). Different instruments have different compliers, and so different LATEs, even for the same treatment. That's why a LATE estimated with one campaign shouldn't be assumed to hold for a different campaign.

</details>

---

## 38.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 38: weak instruments, exclusion violations, complier profiles"
git push
```

### Day 38 checklist

- [ ] I report the first-stage F with every IV estimate and know the thresholds
- [ ] I can explain why weak instruments bias 2SLS towards OLS
- [ ] I can show that an exclusion violation's bias is the leak divided by the first stage
- [ ] I can match ATT, ATE and LATE to the decisions they inform
- [ ] I can estimate the complier profile and use it to say whom a LATE describes

---

# DAY 39 — Comparing Estimators

| Time | Block |
|---|---|
| 0:00 – 0:30 | 39.1 Bias, variance, and the assumption you trust |
| 0:30 – 1:40 | 39.2 The horse race |
| 1:40 – 1:55 | Break |
| 1:55 – 2:50 | 39.3 Reading the results |
| 2:50 – 3:20 | 39.4 A decision guide |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 39.5 Day 39 mini-project: how much hidden confounding? |
| 4:40 – 5:00 | 39.6 Save your work with Git |

---

## 39.1 Bias, variance, and the assumption you trust (30 min)

Every estimator in Module 4 is a bet on an assumption:

| Estimator | Bets on | Estimand |
|---|---|---|
| OLS with controls | the linear model, and no hidden confounders | a regression-weighted average |
| g-computation | the outcome model, and no hidden confounders | ATE / ATT / ATC |
| matching | similar covariates mean similar outcomes, and no hidden confounders | ATT |
| IPW | the propensity model, and no hidden confounders | ATE / ATT |
| AIPW | one of the two models, and no hidden confounders | ATE / ATT |
| 2SLS | relevance, exclusion, monotonicity | LATE |

Notice the last column of the "bets on" list: five estimators share one assumption. Their agreement is not evidence that it holds.

---

## 39.2 The horse race (70 min)

Every estimator, 30 datasets, both worlds, each graded against the truth for its own estimand. Create `day39/horse_race.py` (it takes a few minutes):

```python
# day39/horse_race.py   (python -m day39.horse_race)
"""
Every Module 4 estimator, 30 simulated datasets, two worlds:
  A: selection on observables holds          B: hidden brand affinity drives joining and spending
Each estimate is compared with the truth for ITS estimand.
"""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from causalkit.estimators import OUTCOME_FORMULA, aipw, g_computation, ipw, match_att, two_sls
from causalkit.loyalty import public, truth, world

ESTIMATORS = {
    # name: (function returning (estimate, se or None), estimand)
    "naive difference": (lambda df: (df.spend[df.joined == 1].mean() - df.spend[df.joined == 0].mean(), None), "ATE"),
    "OLS, flexible controls": (lambda df: (smf.ols(OUTCOME_FORMULA.replace("~", "~ joined +"), df).fit()
                                           .params["joined"], None), "ATE"),
    "g-computation": (lambda df: (g_computation(df)["ATE"], None), "ATE"),
    "matching (bias-corrected)": (lambda df: (match_att(df, on="mahalanobis", bias_correct=True)["ATT"], None), "ATT"),
    "IPW": (lambda df: (ipw(df)["ATE"], None), "ATE"),
    "AIPW": (lambda df: (lambda a: (a["ATE"], a["ATE_se"]))(aipw(df)), "ATE"),
    "2SLS with covariates": (lambda df: (lambda r: (r.params["joined"], r.std_errors["joined"]))(two_sls(df)), "LATE"),
}

rows = []
for world_label, hidden in [("A: no hidden confounder", 0.0), ("B: hidden affinity", 1.0)]:
    errs = {k: [] for k in ESTIMATORS}
    cover = {k: 0 for k in ESTIMATORS}
    for s in range(30):
        full = world(seed=5000 + s, hidden=hidden)
        df = public(full).to_pandas()
        t = truth(full)
        for name, (f, estimand) in ESTIMATORS.items():
            est, se = f(df)
            errs[name].append(est - t[estimand])
            if se is not None:
                cover[name] += abs(est - t[estimand]) <= 1.96 * se
    for name, (_, estimand) in ESTIMATORS.items():
        rows.append({"world": world_label, "estimator": name, "vs": estimand,
                     "average error": round(np.mean(errs[name]), 1), "SD": round(np.std(errs[name]), 1),
                     "95% CI coverage": f"{cover[name] / 30:.0%}" if name in ("AIPW", "2SLS with covariates") else ""})

with pl.Config(tbl_rows=-1, fmt_str_lengths=30, tbl_width_chars=120):
    print(pl.DataFrame(rows))
```

```bash
python -m day39.horse_race
```

```
│ world                   ┆ estimator                 ┆ vs   ┆ average error ┆ SD   ┆ 95% CI coverage │
│ A: no hidden confounder ┆ naive difference          ┆ ATE  ┆ 426.5         ┆ 8.9  ┆                 │
│ A: no hidden confounder ┆ OLS, flexible controls    ┆ ATE  ┆ -5.8          ┆ 2.5  ┆                 │
│ A: no hidden confounder ┆ g-computation             ┆ ATE  ┆ -0.2          ┆ 2.0  ┆                 │
│ A: no hidden confounder ┆ matching (bias-corrected) ┆ ATT  ┆ -0.0          ┆ 2.3  ┆                 │
│ A: no hidden confounder ┆ IPW                       ┆ ATE  ┆ -7.0          ┆ 21.8 ┆                 │
│ A: no hidden confounder ┆ AIPW                      ┆ ATE  ┆ -0.2          ┆ 2.1  ┆ 97%             │
│ A: no hidden confounder ┆ 2SLS with covariates      ┆ LATE ┆ 3.0           ┆ 10.6 ┆ 93%             │
│ B: hidden affinity      ┆ naive difference          ┆ ATE  ┆ 443.4         ┆ 8.2  ┆                 │
│ B: hidden affinity      ┆ OLS, flexible controls    ┆ ATE  ┆ 67.8          ┆ 2.1  ┆                 │
│ B: hidden affinity      ┆ g-computation             ┆ ATE  ┆ 72.4          ┆ 1.7  ┆                 │
│ B: hidden affinity      ┆ matching (bias-corrected) ┆ ATT  ┆ 73.3          ┆ 2.2  ┆                 │
│ B: hidden affinity      ┆ IPW                       ┆ ATE  ┆ 69.7          ┆ 20.1 ┆                 │
│ B: hidden affinity      ┆ AIPW                      ┆ ATE  ┆ 73.5          ┆ 1.8  ┆ 0%              │
│ B: hidden affinity      ┆ 2SLS with covariates      ┆ LATE ┆ 0.8           ┆ 13.9 ┆ 100%            │
```

---

## 39.3 Reading the results (55 min)

**World A.** G-computation, bias-corrected matching and AIPW are all unbiased with SDs around $2. IPW is unbiased-ish but ten times noisier. OLS with flexible controls is $6 off: its coefficient targets a regression-weighted average, not the ATE. 2SLS is right, but five times noisier than AIPW.

**World B.** Every selection-on-observables method is off by $68–74, and they **agree with each other** to within a few dollars. AIPW's 95% interval covered the truth in **0 of 30** datasets. 2SLS is the only method that works.

Three takeaways:

1. **Agreement between estimators that share an assumption is not robustness.** Five methods agreed in world B and all were wrong.
2. **Precision is not accuracy.** The most precise estimators in world B had the worst coverage.
3. **The choice of strategy matters more than the choice of estimator within it.** Within selection on observables, the differences are a few dollars; between strategies, $70.

---

## 39.4 A decision guide (30 min)

| Situation | Approach |
|---|---|
| You can randomise | run an experiment (Module 3) |
| Confounders plausibly all measured; good overlap | AIPW (with cross-fitting if using ML); g-computation and matching as checks |
| Confounders measured; poor overlap | change the estimand: trimming, overlap weights, or the ATT |
| An unmeasured confounder is plausible, and a strong, credibly excluded instrument exists | 2SLS with covariates; report first-stage F and the complier profile |
| Unmeasured confounder plausible, no instrument | quasi-experiments (Module 5) or sensitivity analysis (Module 7); don't report a point estimate as "the effect" |

Write this table into `docs/ESTIMATOR_GUIDE.md` in your own words, with one NorthMart example per row.

---

## 39.5 Day 39 mini-project: how much hidden confounding? (65 min)

World B has `hidden=1.0`. Real confounding is rarely all or nothing. Run AIPW and 2SLS (with covariates) at `hidden` = 0, 0.25, 0.5 and 1.0, ten datasets each (`world(seed=6000 + s, hidden=h)`), and print the average error of each. Create `day39/how_much_hidden.py`.

<details>
<summary>Solution</summary>

```python
# day39/how_much_hidden.py   (python -m day39.how_much_hidden)
import numpy as np

from causalkit.estimators import aipw, two_sls
from causalkit.loyalty import public, truth, world

for hidden in [0.0, 0.25, 0.5, 1.0]:
    e_a, e_iv = [], []
    for s in range(10):
        full = world(seed=6000 + s, hidden=hidden)
        df, t = public(full).to_pandas(), truth(full)
        e_a.append(aipw(df)["ATE"] - t["ATE"])
        e_iv.append(two_sls(df).params["joined"] - t["LATE"])
    print(f"hidden = {hidden:4.2f}: AIPW average error {np.mean(e_a):+6.1f}   2SLS average error {np.mean(e_iv):+6.1f}")
```

```
hidden = 0.00: AIPW average error   -0.5   2SLS average error   -2.6
hidden = 0.25: AIPW average error   +4.6   2SLS average error   -2.6
hidden = 0.50: AIPW average error  +20.4   2SLS average error   -2.7
hidden = 1.00: AIPW average error  +73.8   2SLS average error   -2.3
```

AIPW's bias grows faster than linearly with the strength of the hidden confounder (it affects both joining and spending, so the bias scales roughly with the product). A little hidden confounding costs little; a lot costs everything. 2SLS doesn't care, as long as the instrument is valid. This is the idea behind **sensitivity analysis** (Module 7): ask how strong a hidden confounder would have to be to change your decision, and whether that strength is plausible.

</details>

---

## 39.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 39: estimator horse race in two worlds, estimator guide"
git push
```

### Day 39 checklist

- [ ] I can name the assumption each Module 4 estimator bets on, and its estimand
- [ ] I can explain why agreement among estimators that share an assumption isn't robustness
- [ ] I can choose a strategy from the situation, using a written decision guide
- [ ] I can show how bias grows with the strength of a hidden confounder

---

# DAY 40 — Capstone: The Loyalty Programme Effect

| Time | Block |
|---|---|
| 0:00 – 0:30 | 40.1 The brief |
| 0:30 – 1:15 | 40.2 Step 1: the DAG and the strategy |
| 1:15 – 2:10 | 40.3 Step 2: estimates and diagnostics |
| 2:10 – 2:25 | Break |
| 2:25 – 3:20 | 40.4 Step 3: from effects to money |
| 3:20 – 3:35 | Break |
| 3:35 – 4:30 | 40.5 The decision memo |
| 4:30 – 5:00 | 40.6 Answer key, final commit and Module 4 retrospective |

---

## 40.1 The brief (30 min)

NorthMart's CFO asks two questions about the loyalty programme:

1. **Does the programme pay for itself, per member?** Each member costs $45 a year in rewards. NorthMart keeps 30% of any extra spending as margin. So a member is worth `0.30 × effect − $45`.
2. **Should NorthMart keep sending the $20 sign-up invitations?** Every invited customer who joins receives the $20 bonus, including those who would have joined anyway.

The brand team insists that "our fans join and our fans spend". You have one year of data on 20,000 customers, including last year's random invitation. Generate it:

```bash
python data/make_loyalty_scenario.py
```

```python
"""
Capstone data for Module 4: NorthMart's loyalty programme. Provided.

    python data/make_loyalty_scenario.py

Writes data/scenario/loyalty.csv (what the analyst gets) and data/truth/loyalty_truth.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.loyalty import public, truth, world  # noqa: E402

full = world(seed=40, hidden=1.0)
root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
(root / "truth").mkdir(exist_ok=True)
public(full).write_csv(root / "scenario" / "loyalty.csv")
(root / "truth" / "loyalty_truth.json").write_text(json.dumps(truth(full), indent=2))
print(f"Wrote {full.height:,} customers to data/scenario/loyalty.csv")
```

You don't know whether this is world A or world B. That's the point: the analysis has to be right in either. **Don't open `data/truth/` until your memo is committed.**

---

## 40.2 Step 1: the DAG and the strategy (45 min)

Create `day40/dag.py`:

```python
# day40/dag.py   (python -m day40.dag)
"""Capstone step 1: the DAG, and what it says about each strategy."""
import networkx as nx

from causalkit.dags import draw_dag, make_dag, minimal_adjustment_sets

X = ["pre_spend", "engaged", "tenure"]
edges = [(x, "joined") for x in X] + [(x, "spend") for x in X] + [
    ("tenure", "engaged"), ("tenure", "pre_spend"),
    ("affinity", "joined"), ("affinity", "spend"),      # the brand team's worry: loyal fans join AND spend
    ("invited", "joined"),                              # the random $20 invitation
    ("joined", "spend"),
]
dag = make_dag(edges, latent={"affinity"})

if __name__ == "__main__":
    print("Valid adjustment sets for joined -> spend:", minimal_adjustment_sets(dag, "joined", "spend") or "none")

    # Instrument checks, read off the graph:
    no_effect = dag.copy()
    no_effect.remove_edge("joined", "spend")
    print("invited independent of affinity:        ", nx.is_d_separator(dag, {"invited"}, {"affinity"}, set()))
    print("invited reaches spend only via joining: ", nx.is_d_separator(no_effect, {"invited"}, {"spend"}, set()))
    print("invited affects joining:                ", dag.has_edge("invited", "joined"))
    draw_dag(dag, "figures/day40_loyalty_dag.png", "Loyalty programme: assumed DAG", treatment="joined", outcome="spend")
    print("Saved figures/day40_loyalty_dag.png")
```

```bash
python -m day40.dag
```

```
Valid adjustment sets for joined -> spend: none
invited independent of affinity:         True
invited reaches spend only via joining:  True
invited affects joining:                 True
Saved figures/day40_loyalty_dag.png
```

If the brand team is right, no adjustment set exists: every Days 31–36 method is ruled out **by the assumptions**, before any data are touched. The invitation passes all three instrument checks on the graph. Remember, the graph shows what you assume: the exclusion check is only as good as the missing arrow from `invited` to `spend` (Day 38).

---

## 40.3 Step 2: estimates and diagnostics (55 min)

Run both strategies anyway: if they agree, the brand team's worry may be overblown; if not, the gap is informative. Create `day40/estimates.py`:

```python
# day40/estimates.py   (python -m day40.estimates)
"""Capstone step 2: both strategies on the capstone data, with diagnostics."""
import pandas as pd
import polars as pl

from causalkit.estimators import aipw, two_sls

df = pd.read_csv("data/scenario/loyalty.csv")

a = aipw(df)
iv = two_sls(df)
first = df.groupby("invited").joined.mean()
fs = first[1] - first[0]

rows = [
    {"strategy": "AIPW (selection on observables)", "estimand": "ATE", "estimate": a["ATE"], "se": a["ATE_se"]},
    {"strategy": "AIPW (selection on observables)", "estimand": "ATT", "estimate": a["ATT"], "se": a["ATT_se"]},
    {"strategy": "2SLS, invitation as instrument", "estimand": "LATE", "estimate": iv.params["joined"],
     "se": iv.std_errors["joined"]},
]
table = pl.DataFrame(rows).with_columns(
    ci_low=(pl.col("estimate") - 1.96 * pl.col("se")).round(1),
    ci_high=(pl.col("estimate") + 1.96 * pl.col("se")).round(1),
    estimate=pl.col("estimate").round(1), se=pl.col("se").round(1))
with pl.Config(fmt_str_lengths=40, tbl_width_chars=110):
    print(table)
print(f"\nFirst stage: {first[1]:.1%} joined if invited vs {first[0]:.1%} if not ({fs:+.1%});"
      f" F = {iv.first_stage.diagnostics['f.stat'].iloc[0]:.0f}")
print(f"Joiners' average pre_spend ${df.pre_spend[df.joined == 1].mean():,.0f}; "
      f"compliers' (estimated) ${((df.pre_spend * df.joined)[df.invited == 1].mean() - (df.pre_spend * df.joined)[df.invited == 0].mean()) / fs:,.0f}")
```

```bash
python -m day40.estimates
```

```
│ strategy                        ┆ estimand ┆ estimate ┆ se   ┆ ci_low ┆ ci_high │
│ AIPW (selection on observables) ┆ ATE      ┆ 229.9    ┆ 2.4  ┆ 225.2  ┆ 234.5   │
│ AIPW (selection on observables) ┆ ATT      ┆ 256.4    ┆ 3.2  ┆ 250.0  ┆ 262.7   │
│ 2SLS, invitation as instrument  ┆ LATE     ┆ 146.3    ┆ 15.0 ┆ 116.9  ┆ 175.6   │

First stage: 55.7% joined if invited vs 41.1% if not (+14.6%); F = 513
Joiners' average pre_spend $881; compliers' (estimated) $696
```

- AIPW and 2SLS disagree by $84, and the 2SLS interval excludes AIPW's estimate. That's the world B pattern of Day 39.
- The instrument is strong (F = 513), so weak-instrument bias isn't the explanation.
- Compliers spent much less last year than joiners, so the LATE and the ATT describe different people. But in world A, where both strategies were valid, that difference was worth a few dollars (Day 37's Try it), not $84. Different populations can't explain a gap this size; a hidden confounder can.

---

## 40.4 Step 3: from effects to money (55 min)

Create `day40/decision.py`:

```python
# day40/decision.py   (python -m day40.decision)
"""
Capstone step 3: turn effects into money.
  margin on extra spend: 30%      reward cost: $45 per member per year      sign-up bonus: $20 per invited joiner
Q1: does the programme pay for itself, per member?            value = 0.30 x effect - 45
Q2: should NorthMart keep sending $20 invitations?            per 1,000 invitations:
    compliers x (0.30 x LATE - 45)  -  $20 x (all invited customers who join, compliers or not)
"""
import pandas as pd

from causalkit.estimators import aipw, two_sls

MARGIN, REWARDS, BONUS = 0.30, 45.0, 20.0
df = pd.read_csv("data/scenario/loyalty.csv")
a, iv = aipw(df), two_sls(df)
late, late_se = iv.params["joined"], iv.std_errors["joined"]
first = df.groupby("invited").joined.mean()
compliers_per_1000 = 1000 * (first[1] - first[0])
joiners_per_1000_invited = 1000 * first[1]


def value(effect: float) -> float:
    return MARGIN * effect - REWARDS


print("Q1: value per member per year (0.30 x effect - $45)")
print(f"   using AIPW ATT ${a['ATT']:.0f}:  ${value(a['ATT']):+.1f}   <- biased upwards if fans self-select")
print(f"   using 2SLS LATE ${late:.0f}:  ${value(late):+.1f}  "
      f"[95% CI ${value(late - 1.96 * late_se):+.1f} to ${value(late + 1.96 * late_se):+.1f}]")


def invitations(effect: float) -> float:
    return compliers_per_1000 * value(effect) - BONUS * joiners_per_1000_invited


print("\nQ2: net value of 1,000 invitations")
print(f"   {compliers_per_1000:.0f} extra joiners, but {joiners_per_1000_invited:.0f} invited joiners receive the $20 bonus")
print(f"   using 2SLS LATE: ${invitations(late):+,.0f}  "
      f"[95% CI ${invitations(late - 1.96 * late_se):+,.0f} to ${invitations(late + 1.96 * late_se):+,.0f}]")
print(f"   using AIPW ATE (wrong population, biased): ${invitations(a['ATE']):+,.0f}")
```

```bash
python -m day40.decision
```

```
Q1: value per member per year (0.30 x effect - $45)
   using AIPW ATT $256:  $+31.9   <- biased upwards if fans self-select
   using 2SLS LATE $146:  $-1.1  [95% CI $-9.9 to $+7.7]

Q2: net value of 1,000 invitations
   146 extra joiners, but 557 invited joiners receive the $20 bonus
   using 2SLS LATE: $-11,303  [95% CI $-12,586 to $-10,021]
   using AIPW ATE (wrong population, biased): $-7,649
```

**Q1.** With AIPW, the programme looks clearly profitable: +$32 per member. With the LATE, it's break-even: −$1, with a CI from −$10 to +$8. The answer to "does it pay for itself?" depends entirely on which assumption you trust.

**Q2.** This one is clear whichever effect you use. Each 1,000 invitations create 146 extra members, but pay the $20 bonus to all 557 invited joiners, 411 of whom would have joined anyway. Even the optimistic estimate loses money. Note that the LATE is the **right** effect for Q2: the invitation's extra joiners are, by definition, its compliers.

---

## 40.5 The decision memo (55 min)

Write `docs/LOYALTY_DECISION.md`, one page, for the CFO:

1. **Recommendation** (two sentences, one per question).
2. **The question and the data** (one paragraph).
3. **Why we didn't trust the simple comparison** (the $593 gap; the fan worry; what the DAG says).
4. **Results** (a small table: strategy, estimand, estimate, CI, value per member).
5. **Whose effect is this?** (the complier profile, in plain language).
6. **Risks and next steps** (exclusion: could the invitation email itself raise spending? A follow-up that would settle Q1).

<details>
<summary>Model answer for sections 1, 5 and 6</summary>

> **Recommendation.** Stop sending $20 sign-up invitations: they cost about $11,000 per 1,000 invitations, mainly because most of the bonus goes to customers who would have joined anyway. Keep the programme for now, but don't expand it: our best estimate is that it roughly breaks even ($−1 per member per year, plausibly between −$10 and +$8), not the +$32 a simple comparison suggests.
>
> **Whose effect is this?** Our reliable estimate comes from last year's random invitation, so it describes the customers the invitation persuaded to join. They are moderate spenders ($696 a year on average, against $881 for members). The effect for our current, heavier-spending members is probably somewhat larger, but we can't measure it without assuming that nothing else (like brand loyalty) explains why they joined, and the data suggest that assumption fails.
>
> **Risks and next steps.** If the invitation email itself encouraged spending, even without joining, our estimate is too high and the programme is worse than break-even. To settle the per-member question, randomise **programme benefits** among new members for six months (Module 3), for example the reward rate, which measures the value of the rewards for the members we actually have.

</details>

---

## 40.6 Answer key, final commit and Module 4 retrospective (30 min)

After the memo is committed, compare with the truth. Create `day40/answer_key.py`:

```python
# day40/answer_key.py   (python -m day40.answer_key)
"""Instructor check, after the memo: the true effects and the decisions they imply."""
import json
from pathlib import Path

import pandas as pd

MARGIN, REWARDS, BONUS = 0.30, 45.0, 20.0
t = json.loads(Path("data/truth/loyalty_truth.json").read_text())
df = pd.read_csv("data/scenario/loyalty.csv")
joiners_per_1000_invited = 1000 * df[df.invited == 1].joined.mean()

print("True effects of joining on next-year spend:")
for k in ["ATE", "ATT", "ATC", "LATE"]:
    print(f"   {k:<5} ${t[k]:6.1f}   value per member (0.30 x effect - $45): ${MARGIN * t[k] - REWARDS:+.1f}")
compliers = 1000 * t["share_compliers"]
net = compliers * (MARGIN * t["LATE"] - REWARDS) - BONUS * joiners_per_1000_invited
print(f"True net value of 1,000 invitations: ${net:+,.0f} ({compliers:.0f} compliers)")
```

```bash
python -m day40.answer_key
```

```
True effects of joining on next-year spend:
   ATE   $ 158.1   value per member (0.30 x effect - $45): $+2.4
   ATT   $ 182.5   value per member (0.30 x effect - $45): $+9.8
   ATC   $ 135.2   value per member (0.30 x effect - $45): $-4.5
   LATE  $ 156.3   value per member (0.30 x effect - $45): $+1.9
True net value of 1,000 invitations: $-10,842 (157 compliers)
```

This was world B. The 2SLS interval contains the true LATE ($156), and the invitation decision is right: the true loss is $10,842 per 1,000 invitations, inside our interval. AIPW's ATT ($256) was $74 too high, and would have made the programme look worth $32 per member when it's really worth about $10 for current members. A director acting on AIPW would have expanded a programme that barely breaks even.

```bash
git add .
git commit -m "Day 40: loyalty programme capstone: DAG, estimates, decision memo"
git push
```

**What you learned in Module 4.**

| Day | Idea | The simulator lesson |
|---|---|---|
| 31 | G-computation | One model gives ATE, ATT and ATC; the linear model got the ATC wrong by $45 |
| 32 | Propensity scores | 5 strata remove ~85% of the bias, not all; a missing confounder adds $25, silently |
| 33 | Matching | Balance under 0.1 still left $18 of bias; bias correction removed it |
| 34 | IPW | Unbiased, but 9× noisier than g-computation; simulate before concluding |
| 35 | Overlap | Poor overlap: weights of 404, ESS 2,273 of 20,000; overlap weights cut the SD from $116 to $3.5 |
| 36 | AIPW | Right if either model is right; not if a confounder is missing |
| 37 | IV, 2SLS | Hidden affinity: AIPW off by $74 with SE $2; 2SLS right, with covariates for precision |
| 38 | Weak IV, exclusion, LATE | A $5 leak becomes a $31 bias; compliers are moderate spenders |
| 39 | Horse race | Five estimators agreed in world B, and all were wrong |
| 40 | Capstone | The invitation loses $11 per invitation; the programme roughly breaks even, not +$32 |

**Coming next: Module 5, Quasi-Experiments.** IV needs a lucky random nudge. Module 5 uses the structure of *time* and *rules* instead: difference-in-differences, event studies, synthetic control and regression discontinuity, to evaluate rollouts and campaigns that were never randomised.

### Day 40 checklist

- [ ] I chose a strategy from a DAG before looking at the estimates
- [ ] I ran both strategies and used their disagreement as evidence
- [ ] I reported the first-stage F and the complier profile with the IV estimate
- [ ] I matched each business question to the right estimand (LATE for the invitation decision)
- [ ] I wrote a decision memo with money, uncertainty and the assumptions it rests on

