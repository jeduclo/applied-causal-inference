# Module 2 · Causal Thinking, Estimands & DAGs
## Days 11–13: Potential Outcomes, Estimands and Randomisation

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Module 1
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 13:** Students can state precisely which causal effect a business question asks for, decompose a naive comparison into the true effect plus selection bias, and show with repeated simulated experiments why randomisation removes that bias.

---

### Where we are

Module 1 delivered the **Causal Data Mart**: `data/published/panel_customer_week.parquet` and `customer_features.parquet`, rebuilt and checked every week. It also showed, twice, that the naive comparison says the SPRING26 email raised orders by **+106%** when the truth is about **+15%**.

Module 2 explains *why*, with two languages that every later method relies on:

| Days | Language | Question it answers |
|---|---|---|
| 11–13 | **Potential outcomes** | *What* effect do we want, and why does a comparison miss it? |
| 14–17 | **DAGs** (causal graphs) | *Which* variables must we adjust for, and which must we leave alone? |
| 18–20 | Both, with **DoWhy** | Can the effect be identified from our data at all? |

No new estimation methods yet: those start in Module 3. Module 2 is about thinking clearly enough that the methods answer the right question.

### NorthMart's hidden truth

Real data shows each customer in one world only: emailed, or not. Because NorthMart is simulated, we can rebuild **both** worlds for every customer. A new provided script does this.

<details>
<summary>data/northmart_truth.py (provided; click to expand)</summary>

```python
"""
NorthMart's hidden truth: BOTH potential outcomes for every customer. Provided for Module 2.

Run once from the repository root (after data/make_northmart.py):
    python data/northmart_truth.py

Writes data/truth/potential_outcomes.parquet with one row per customer:
    treated        1 if the customer got the SPRING26 email (what really happened)
    y_observed     completed orders in weeks 9-16 (what the data shows)
    y0, y1         orders in weeks 9-16 WITHOUT and WITH the email
                   (one of them is y_observed; the other is simulated by this script)
    tau            y1 - y0: the individual effect (never observable in real life)
    tau_expected   the expected individual effect, without Poisson noise

No real dataset has these columns. We use them to check our methods against the truth.
"""
from pathlib import Path

import numpy as np
import polars as pl

import make_northmart as nm   # the Module 1 generator, in the same folder

CANCEL_RATE = 0.03            # the generator cancels ~3% of orders at random
POST_WEEKS = range(nm.CAMPAIGN_WEEK, nm.N_WEEKS + 1)
OUT = Path(__file__).parent / "truth"


def main() -> None:
    # 1. Replay the generator's random draws to recover each customer's latent rate and treatment
    rng = np.random.default_rng(nm.SEED)
    customers = nm.make_customers(rng)
    nm.make_products(rng)
    treated = nm.assign_campaign(rng, customers)

    recorded = set(pl.read_csv(nm.OUT / "email_campaign.csv")["customer_id"])
    assert recorded == set(customers.filter(pl.Series(treated))["customer_id"]), "replay mismatch"

    # 2. Observed post-period orders (completed, deduplicated) from the raw file
    orders = (pl.read_csv(nm.OUT / "orders.csv", try_parse_dates=True)
              .unique().filter(pl.col("status") == "completed")
              .filter(pl.col("order_date") >= pl.date(2026, 3, 2)))
    y_obs = orders.group_by("customer_id").agg(y_observed=pl.len())

    # 3. Expected orders over weeks 9-16 without the email (sum of weekly Poisson means)
    base = customers["_base_rate"].to_numpy()
    season = sum(1.0 + 0.10 * np.sin(2 * np.pi * w / 16) for w in POST_WEEKS)
    mu0 = base * season * (1 - CANCEL_RATE)
    mu1 = mu0 * (1 + nm.TRUE_EFFECT)

    # 4. Draw the potential outcome that did NOT happen, coupled to the one that did:
    #    with the email, a customer makes every order they would have made anyway, plus extra ones.
    #    Treated:  y0 keeps each observed order with probability 1/1.15 (so y0 <= y1).
    #    Control:  y1 adds extra orders, Poisson with mean mu1 - mu0 (so y1 >= y0).
    rng_cf = np.random.default_rng(nm.SEED + 1)
    df = (customers.select("customer_id", "is_loyalty", "region")
          .with_columns(treated=pl.Series(treated.astype(np.int8)),
                        mu0=pl.Series(mu0), mu1=pl.Series(mu1))
          .join(y_obs, on="customer_id", how="left")
          .with_columns(pl.col("y_observed").fill_null(0).cast(pl.Int64)))
    y = df["y_observed"].to_numpy()
    kept = rng_cf.binomial(y, 1 / (1 + nm.TRUE_EFFECT))
    extra = rng_cf.poisson(mu1 - mu0)
    t = treated
    df = df.with_columns(
        y0=pl.Series(np.where(t, kept, y)),
        y1=pl.Series(np.where(t, y, y + extra)),
    ).with_columns(
        tau=pl.col("y1") - pl.col("y0"),
        tau_expected=pl.col("mu1") - pl.col("mu0"),
    )

    OUT.mkdir(exist_ok=True)
    df.select("customer_id", "treated", "is_loyalty", "region", "y_observed", "y0", "y1",
              "tau", "tau_expected", "mu0", "mu1").write_parquet(OUT / "potential_outcomes.parquet")
    print(f"Wrote {df.height:,} customers to {OUT / 'potential_outcomes.parquet'}")


if __name__ == "__main__":
    main()
```

</details>

Run it from the `data/` folder (it imports the Module 1 generator next to it):

```bash
cd data && python northmart_truth.py && cd ..
```

```
Wrote 4,000 customers to /workspaces/causal-decision-lab/data/truth/potential_outcomes.parquet
```

> **Use the truth file only to check your work.** It contains columns no real dataset has. Every method in this course must work *without* it; the truth tells us how well it did.

### Update your environment

Module 2 adds four packages. Append these lines to `requirements.txt` and to `.gitignore`, then reinstall:

```text
# Module 2: causal graphs and identification
networkx==3.7
statsmodels==0.15.0
matplotlib==3.11.2
dowhy==0.14
```

```text
# Module 2: generated truth, scenario data and figures
data/truth/
data/scenario/
figures/
```

```bash
pip install -r requirements.txt
mkdir -p figures
```

### Running scripts in Module 2

From now on, run scripts as **modules** from the repository root:

```bash
python -m day11.gods_view          # instead of: python day11/gods_view.py
```

`python -m` puts the repository root on Python's search path, so scripts in any `dayNN/` folder can import shared helpers from the `causalkit/` folder we start on Day 14. Each script's first line shows its command.

---

# DAY 11 — Potential Outcomes and the Fundamental Problem

| Time | Block |
|---|---|
| 0:00 – 0:30 | 11.1 Two worlds for every customer |
| 0:30 – 1:20 | 11.2 Individual effects and the average effect |
| 1:20 – 1:35 | Break |
| 1:35 – 2:20 | 11.3 The fundamental problem of causal inference |
| 2:20 – 3:10 | 11.4 Assumptions that make potential outcomes meaningful |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 11.5 Day 11 mini-project: naive comparison = effect + selection bias |
| 4:40 – 5:00 | 11.6 Save your work with Git |

---

## 11.1 Two worlds for every customer (30 min)

**Concept.** For each customer *i* and the email, imagine two numbers:

- **Y₁(i)**: orders in weeks 9–16 **if** customer *i* receives the email.
- **Y₀(i)**: orders in weeks 9–16 **if** customer *i* does not.

These are **potential outcomes**. Both exist before marketing decides anything; the decision only determines which one we get to see. The **causal effect** of the email on customer *i* is the difference:

```text
τ(i) = Y₁(i) − Y₀(i)
```

This is the **Rubin causal model**, the framework behind every modern policy evaluation, from minimum-wage studies to drug trials. Economists know it as the logic of counterfactuals: "compared with what?"

Every causal question names the comparison. "Did the email work?" really means: *did emailed customers buy more than they would have **without** the email?* Not more than other customers; more than themselves, in the world that didn't happen.

---

## 11.2 Individual effects and the average effect (50 min)

**Example 1 — the god's-eye view.** Create `day11/gods_view.py`:

```python
# day11/gods_view.py   (python -m day11.gods_view)
"""What only a simulation can show: both potential outcomes for the same customers."""
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")

# 1. Six customers, both worlds side by side
six = po.filter(pl.col("customer_id").is_in(["C00004", "C00009", "C00010", "C00001", "C00002", "C00003"]))
print(six.select("customer_id", "treated", "y0", "y1", "tau").sort("customer_id"))

# 2. Individual effects vary: most customers would not change at all
print(po.group_by("tau").len().sort("tau"))

# 3. The average of the individual effects is the average treatment effect (ATE)
print(f"ATE = average of y1 - y0 = {po['tau'].mean():.3f} orders per customer over 8 weeks")
```

```bash
python -m day11.gods_view
```

```
│ customer_id ┆ treated ┆ y0  ┆ y1  ┆ tau │
│ C00001      ┆ 0       ┆ 3   ┆ 4   ┆ 1   │
│ C00002      ┆ 0       ┆ 4   ┆ 4   ┆ 0   │
│ C00003      ┆ 0       ┆ 3   ┆ 3   ┆ 0   │
│ C00004      ┆ 1       ┆ 2   ┆ 2   ┆ 0   │
│ C00009      ┆ 1       ┆ 2   ┆ 2   ┆ 0   │
│ C00010      ┆ 1       ┆ 3   ┆ 4   ┆ 1   │
...
│ 0   ┆ 2952 │
│ 1   ┆ 862  │
│ 2   ┆ 150  │
│ 3   ┆ 33   │
│ 4   ┆ 2    │
│ 5   ┆ 1    │
ATE = average of y1 - y0 = 0.319 orders per customer over 8 weeks
```

Read it carefully:

- **Customer C00004** (Day 1: 7 orders before the email, 2 after) would have ordered 2 times either way. The email did nothing for them. Their drop from 7 to 2 was not caused by the email, which Day 1 could only guess.
- **Customer C00001** was *not* emailed, but would have placed one extra order if they had been.
- **Effects vary.** For 2,952 of 4,000 customers the email changes nothing; for 862 it adds one order; a few gain more.
- The **average treatment effect (ATE)** is the mean of the individual effects: **0.319** extra orders per customer over eight weeks.

> In the simulation, the email adds about 15% to each customer's *expected* orders. Because orders are counts, that shows up as "+1 order" for some customers and "0" for others. Real effects are lumpy like this too.

---

## 11.3 The fundamental problem of causal inference (45 min)

**Concept.** For every customer we observe **one** potential outcome, never both. This is called the **fundamental problem of causal inference** (Holland, 1986). It is not a data-quality problem that more data can fix: it's built into the question.

Create `day11/fundamental_problem.py`:

```python
# day11/fundamental_problem.py   (python -m day11.fundamental_problem)
"""What the data really shows: one potential outcome per customer, never both."""
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")

observed = po.select(
    "customer_id",
    "treated",
    y0=pl.when(pl.col("treated") == 0).then(pl.col("y0")),   # NULL when treated
    y1=pl.when(pl.col("treated") == 1).then(pl.col("y1")),   # NULL when not treated
)
print(observed.sort("customer_id").head(6))

missing = observed.select(pl.col("y0").is_null().sum() + pl.col("y1").is_null().sum()).item()
print(f"Missing potential outcomes: {missing:,} of {2 * po.height:,} ({missing / (2 * po.height):.0%})")
print("Individual effects we can compute from data alone:", 0)
```

```
│ customer_id ┆ treated ┆ y0   ┆ y1   │
│ C00001      ┆ 0       ┆ 3    ┆ null │
│ C00002      ┆ 0       ┆ 4    ┆ null │
│ C00003      ┆ 0       ┆ 3    ┆ null │
│ C00004      ┆ 1       ┆ null ┆ 2    │
...
Missing potential outcomes: 4,000 of 8,000 (50%)
Individual effects we can compute from data alone: 0
```

Causal inference is therefore a **missing-data problem**: exactly half the table is missing, and the missing half is the counterfactual. Every method in this course fills it in using *other* units or *other* periods, under assumptions. The assumptions are the whole game.

**Try it.** Take a pen and paper. For C00001, the data says 3 orders without the email. Write down three different ways you might *guess* their Y₁, and the assumption each guess relies on.

<details>
<summary>Solution</summary>

1. **Use similar emailed customers:** average Y of emailed customers with the same loyalty status and pre-period orders. *Assumes* that, given those characteristics, who got the email is as good as random (Module 4).
2. **Use C00001's own trend:** project their orders forward and add the average change of emailed customers. *Assumes* emailed and non-emailed customers would have followed parallel trends (Module 5).
3. **Use a model:** predict Y₁ from a machine-learning model trained on emailed customers. *Assumes* the model captures everything that drove both targeting and orders (Module 6).

None of the three is checkable from C00001's data alone. That's the fundamental problem.

</details>

---

## 11.4 Assumptions that make potential outcomes meaningful (50 min)

Writing Y₁(i) and Y₀(i) already assumes two things. They're usually stated together as **SUTVA** (the stable unit treatment value assumption):

| Part | Meaning | NorthMart threat |
|---|---|---|
| **No interference** | A customer's outcome depends on their own treatment only | Two customers in the same household: one gets the email and shops for both |
| **One version of the treatment** | "The email" means the same thing for everyone | Some recipients got it on a phone, some in a spam folder; is that the same treatment? |

And one more, which links potential outcomes to data:

- **Consistency:** a customer who received the email shows Y₁ in the data (Y = Y₁ when T = 1, Y = Y₀ when T = 0). It fails when the recorded treatment isn't the one received: for example a send log that includes bounced emails.

> Module 1's contracts protect consistency. `assert_single_send_date` and `assert_treatment_is_fixed` exist so that "treated" means one thing, at one time, for every customer.

**Try it.** For each scenario, say which assumption is threatened and how you would change the analysis.

1. NorthMart sends a referral email: "Give a friend $10".
2. 8% of SPRING26 emails bounced, but the send log lists all 1,279 customers.
3. The email had two subject lines, chosen at random per customer.

<details>
<summary>Solution</summary>

1. **No interference** is violated by design: the friend's purchases depend on *your* email. Redefine the unit (households or friend pairs), or measure spillovers explicitly.
2. **Consistency:** 8% of "treated" customers never received the treatment. The comparison then estimates the effect of *being on the send list* (an intention-to-treat effect, Day 12), not of receiving the email. Say so, or use the delivery log.
3. **One version of the treatment:** there are two treatments. Either analyse "any email" (and say it averages over subject lines), or estimate each version's effect separately.

</details>

---

## 11.5 Day 11 mini-project: naive comparison = effect + selection bias (75 min)

**Goal.** Split NorthMart's naive comparison into its two parts, exactly. Add and subtract the never-observed quantity E[Y₀ | emailed]:

```text
  E[Y₁ | T=1] − E[Y₀ | T=0]       naive difference (what the data shows)
= E[Y₁ − Y₀ | T=1]                ATT: the effect on the treated
+ E[Y₀ | T=1] − E[Y₀ | T=0]       selection bias
```

The first term is what we want. The second asks: *without* the email, would the emailed group have ordered more anyway? With targeted marketing, yes.

Create `day11/decomposition.py`:

```python
# day11/decomposition.py   (python -m day11.decomposition)
"""Mini-project: split the naive comparison into the true effect and selection bias."""
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
t = po.filter(pl.col("treated") == 1)
c = po.filter(pl.col("treated") == 0)

naive = t["y1"].mean() - c["y0"].mean()       # what the data shows
att = t["y1"].mean() - t["y0"].mean()         # true effect on the emailed (needs the truth)
bias = t["y0"].mean() - c["y0"].mean()        # how different the groups were WITHOUT the email

print(f"E[Y1 | emailed]            {t['y1'].mean():6.3f}   (observed)")
print(f"E[Y0 | not emailed]        {c['y0'].mean():6.3f}   (observed)")
print(f"E[Y0 | emailed]            {t['y0'].mean():6.3f}   (NEVER observed: from the truth file)")
print()
print(f"Naive difference           {naive:+6.3f}")
print(f"  = true effect (ATT)      {att:+6.3f}")
print(f"  + selection bias         {bias:+6.3f}")
assert abs(naive - (att + bias)) < 1e-9

print(f"\nBias is {bias / naive:.0%} of the naive difference.")
print(f"In relative terms the true effect is {att / t['y0'].mean():+.1%}; the naive gap is {naive / c['y0'].mean():+.0%}.")
```

```bash
python -m day11.decomposition
```

```
E[Y1 | emailed]             3.332   (observed)
E[Y0 | not emailed]         1.621   (observed)
E[Y0 | emailed]             2.889   (NEVER observed: from the truth file)

Naive difference           +1.711
  = true effect (ATT)      +0.443
  + selection bias         +1.268

Bias is 74% of the naive difference.
In relative terms the true effect is +15.3%; the naive gap is +106%.
```

Three quarters of the naive gap is selection bias: the emailed customers would have ordered 2.889 times anyway, against 1.621 for the others. The `assert` line checks that the decomposition is an identity, not an approximation. It always holds; the only question is how big the bias term is.

**Try it.** Does comparing *within* a segment remove the bias? Repeat the decomposition separately for loyalty members and non-members.

<details>
<summary>Solution</summary>

```python
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
for members in [True, False]:
    g = po.filter(pl.col("is_loyalty") == members)
    t, c = g.filter(pl.col("treated") == 1), g.filter(pl.col("treated") == 0)
    naive = t["y1"].mean() - c["y0"].mean()
    att = t["y1"].mean() - t["y0"].mean()
    bias = t["y0"].mean() - c["y0"].mean()
    label = "loyalty members" if members else "non-members"
    print(f"{label:<16} naive {naive:+.3f} = ATT {att:+.3f} + bias {bias:+.3f}")
```

```
loyalty members  naive +1.620 = ATT +0.482 + bias +1.137
non-members      naive +1.466 = ATT +0.404 + bias +1.062
```

Within each segment the bias shrinks only a little. Loyalty explains part of the targeting, but marketing also targeted on shopping intensity, which differs a lot *within* each segment. Day 15 shows this with a DAG.

</details>

---

## 11.6 Save your work with Git (20 min)

```bash
git add .
git status         # data/truth/ must NOT appear
git commit -m "Day 11: potential outcomes, truth file, bias decomposition"
git push
```

### Day 11 checklist

- [ ] I can define potential outcomes and an individual causal effect for a business question
- [ ] I can explain the fundamental problem of causal inference as a missing-data problem
- [ ] I can name the parts of SUTVA and consistency, with a threat to each in my own data
- [ ] I can decompose a naive comparison into the ATT plus selection bias
- [ ] I use the truth file only to check methods, never as an input to them

---

# DAY 12 — Estimands: ATE, ATT, CATE, LATE

| Time | Block |
|---|---|
| 0:00 – 0:40 | 12.1 An estimand is a precise question |
| 0:40 – 1:30 | 12.2 Same email, five true answers |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 12.3 Conditional effects (CATE) and where the effect comes from |
| 2:30 – 3:20 | 12.4 When people don't comply: ITT and LATE |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 12.5 Day 12 mini-project: estimand cards for three decisions |
| 4:45 – 5:00 | 12.6 Save your work with Git |

---

## 12.1 An estimand is a precise question (40 min)

**Concept.** Three words are easy to mix up:

| Word | What it is | NorthMart example |
|---|---|---|
| **Estimand** | The quantity you want to know | ATT: average extra orders the email caused among emailed customers, weeks 9–16 |
| **Estimator** | A recipe for computing a guess from data | difference-in-differences, matching, regression… |
| **Estimate** | The number the recipe produces | +0.46 |

Choosing the estimand comes **first**, because different decisions need different estimands, and their true values differ. A complete estimand states five things:

1. **Unit:** customer, household, store, region?
2. **Treatment and comparison:** emailed vs *not emailed*? vs *a different email*?
3. **Outcome and horizon:** orders, revenue, margin? Over 8 weeks? 12 months?
4. **Population:** all customers, the emailed ones, loyalty members?
5. **Summary:** an average difference, a ratio, a percentile?

The main population choices have names:

| Estimand | Population | Typical decision |
|---|---|---|
| **ATE** (average treatment effect) | everyone | "Should we email *all* customers next time?" |
| **ATT** (… on the treated) | those who were treated | "Was *this* campaign worth what we spent?" |
| **ATC** (… on the controls) | those who were not treated | "Should we extend it to the rest?" |
| **CATE** (conditional ATE) | a subgroup defined by characteristics | "Which segment should we target?" |
| **LATE** (local ATE) | compliers (see 12.4) | "What does *opening* the email do?" |

---

## 12.2 Same email, five true answers (50 min)

Create `day12/estimands.py`:

```python
# day12/estimands.py   (python -m day12.estimands)
"""Same email, different questions, different true answers."""
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")


def effect(df: pl.DataFrame) -> tuple[float, float]:
    """Average effect in orders, and relative to what the group would have done anyway."""
    return df["tau"].mean(), df["tau"].sum() / df["y0"].sum()


rows = []
for name, df in [
    ("ATE  (all customers)", po),
    ("ATT  (emailed customers)", po.filter(pl.col("treated") == 1)),
    ("ATC  (not emailed)", po.filter(pl.col("treated") == 0)),
    ("CATE (loyalty members)", po.filter(pl.col("is_loyalty"))),
    ("CATE (non-members)", po.filter(~pl.col("is_loyalty"))),
]:
    abs_eff, rel_eff = effect(df)
    rows.append({"estimand": name, "customers": df.height,
                 "orders_per_customer": round(abs_eff, 3), "relative": f"{rel_eff:+.1%}"})

with pl.Config(tbl_width_chars=100, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
```

```bash
python -m day12.estimands
```

```
│ estimand                 ┆ customers ┆ orders_per_customer ┆ relative │
│ ATE  (all customers)     ┆ 4000      ┆ 0.319               ┆ +15.7%   │
│ ATT  (emailed customers) ┆ 1279      ┆ 0.443               ┆ +15.3%   │
│ ATC  (not emailed)       ┆ 2721      ┆ 0.26                ┆ +16.1%   │
│ CATE (loyalty members)   ┆ 1194      ┆ 0.429               ┆ +15.9%   │
│ CATE (non-members)       ┆ 2806      ┆ 0.272               ┆ +15.6%   │
```

The **relative** effect is about +15% for every group: that is how the simulation was built. The **absolute** effect differs by a factor of almost two between the emailed (0.443) and the rest (0.260), because emailed customers are heavier shoppers: 15% of more orders is more orders.

So "the effect of the email" has no single true value. A manager who wants to know what it would add to send the email to the 2,721 other customers needs the **ATC**, 0.26 orders each, not the ATT. Reporting the ATT for that decision would overstate the gain by 70%, with no estimation error at all.

**Try it.** NorthMart earns about $160 per order. Using the true values, how much extra revenue did the email bring in from the 1,279 customers who got it? How much would emailing the other 2,721 add?

<details>
<summary>Solution</summary>

- This campaign (ATT): 1,279 × 0.443 × $160 ≈ **$90,700**.
- Extending it (ATC): 2,721 × 0.260 × $160 ≈ **$113,200**.

Using the ATT for the second question would give 2,721 × 0.443 × $160 ≈ $192,900, overstating the gain by about $80,000. (At ~$157 average order value from the Module 1 data, the figures barely change.)

</details>

---

## 12.3 Conditional effects (CATE) and where the effect comes from (45 min)

**Concept.** A **CATE** is the average effect for units with given characteristics X: CATE(x) = E[Y₁ − Y₀ | X = x]. CATEs drive **targeting**: email the segments with the largest effects. Module 6 estimates them with machine learning; today we look at the true ones.

Create `day12/cate_by_region.py`:

```python
# day12/cate_by_region.py   (python -m day12.cate_by_region)
"""Conditional effects (CATE) by region, and a chart of where the effect comes from."""
import matplotlib

matplotlib.use("Agg")                      # write files; no screen needed
import matplotlib.pyplot as plt
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")

by_region = (po.group_by("region")
             .agg(customers=pl.len(),
                  cate=pl.col("tau_expected").mean(),
                  baseline=pl.col("mu0").mean())
             .with_columns(relative=pl.col("cate") / pl.col("baseline"))
             .sort("cate", descending=True))
print(by_region)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(po["mu0"], po["tau_expected"], s=6, alpha=0.4)
ax.set_xlabel("Expected orders without the email, weeks 9-16 (mu0)")
ax.set_ylabel("Expected extra orders from the email (tau)")
ax.set_title("Same +15% for everyone, so bigger shoppers gain more orders")
fig.tight_layout()
fig.savefig("figures/day12_effect_vs_baseline.png", dpi=150)
print("Saved figures/day12_effect_vs_baseline.png")
```

```
│ region ┆ customers ┆ cate     ┆ baseline ┆ relative │
│ ATL    ┆ 485       ┆ 0.313129 ┆ 2.087529 ┆ 0.15     │
│ QC     ┆ 931       ┆ 0.311348 ┆ 2.075651 ┆ 0.15     │
│ AB     ┆ 453       ┆ 0.308103 ┆ 2.054023 ┆ 0.15     │
│ ON     ┆ 1513      ┆ 0.303939 ┆ 2.026258 ┆ 0.15     │
│ BC     ┆ 618       ┆ 0.303255 ┆ 2.021703 ┆ 0.15     │
Saved figures/day12_effect_vs_baseline.png
```

Open `figures/day12_effect_vs_baseline.png` in VS Code: a straight line. Each customer's expected extra orders is exactly 15% of their expected orders without the email. Region doesn't matter at all; what matters is how much a customer shops. These true CATEs use `tau_expected` (the effect without counting noise), which is why they look so smooth.

> The truth file uses each customer's *true* region, so there's no `UNKNOWN` group here.

**Targeting lesson.** If marketing could only email 1,279 customers, the ones with the largest *absolute* CATE are the heaviest shoppers, which is roughly whom they chose. Targeting on expected effect is good business. It's also exactly what makes the naive comparison biased. Good targeting and easy evaluation pull in opposite directions; Module 3 shows how to get both.

---

## 12.4 When people don't comply: ITT and LATE (50 min)

**Concept.** Often we control the *offer* but not the *take-up*: we send an email (Z), but customers decide whether to open it (D). Two estimands follow:

- **ITT** (intention to treat): the effect of *sending*, whatever people do with it. Easy to estimate when sending is random.
- **LATE** (local average treatment effect): the effect of *opening*, for **compliers**: customers who open the email when they receive it.

Comparing openers with non-openers is badly biased: people who open marketing emails are different. But if sending was random, the LATE can be recovered with the **Wald estimator**:

```text
LATE = ITT / first stage
     = (effect of sending on orders) / (effect of sending on opening)
```

Create `day12/late_toy.py`, a toy world built for this purpose (not NorthMart):

```python
# day12/late_toy.py   (python -m day12.late_toy)
"""
A toy world for the LATE: the effect for customers who OPEN the email because it was sent.

Z = email sent (randomised)      D = email opened      Y = orders
Only 'compliers' open an email when they get one. 'Never-takers' never open it.
Heavy shoppers are more likely to be compliers: openers are not like non-openers.
"""
import numpy as np
import polars as pl

rng = np.random.default_rng(11)
n = 100_000

intensity = rng.lognormal(0, 0.5, n)                       # unobserved shopping intensity
complier = rng.random(n) < np.clip(0.2 + 0.25 * intensity, 0, 0.9)
z = rng.random(n) < 0.5                                     # random send
d = z & complier                                            # opened
effect = 0.5                                                # true effect of OPENING, for compliers
y = rng.poisson(2 * intensity + effect * d)

df = pl.DataFrame({"z": z, "d": d, "y": y, "complier": complier})
mean = lambda e: df.filter(e)["y"].mean()

itt = mean(pl.col("z")) - mean(~pl.col("z"))                 # effect of SENDING
first_stage = df.filter(pl.col("z"))["d"].mean() - df.filter(~pl.col("z"))["d"].mean()
late = itt / first_stage                                     # Wald estimator
naive_open = mean(pl.col("d")) - mean(~pl.col("d"))          # openers vs everyone else

print(f"Share of compliers              {complier.mean():.1%}")
print(f"ITT (effect of sending)         {itt:+.3f}")
print(f"First stage (opens if sent)     {first_stage:.3f}")
print(f"LATE = ITT / first stage        {late:+.3f}   (truth for compliers: +{effect})")
print(f"Naive: openers vs non-openers   {naive_open:+.3f}   (badly biased)")
```

```
Share of compliers              47.7%
ITT (effect of sending)         +0.234
First stage (opens if sent)     0.474
LATE = ITT / first stage        +0.493   (truth for compliers: +0.5)
Naive: openers vs non-openers   +0.947   (badly biased)
```

The ITT is diluted: half the people who were sent the email never opened it. Dividing by the share of compliers scales it back up to the effect per opener. This is **instrumental variables** in its simplest form. Module 4 develops it fully; for now, notice that the LATE is an effect for *compliers only*, which you can't list by name.

**Try it.** Make the effect larger for heavy shoppers: replace `effect * d` with `0.5 * intensity * d`, and add two lines that print the true average effect for compliers, `(0.5 * intensity[complier]).mean()`, and for everyone, `(0.5 * intensity).mean()`. Does the LATE still describe all customers?

<details>
<summary>Solution</summary>

```
LATE = ITT / first stage        +0.662   (ignore the old "truth" label)
True effect for compliers       +0.652
True effect if everyone opened  +0.566
```

The Wald ratio still recovers the effect **for compliers** (0.662 vs 0.652, the gap is sampling noise). But compliers are heavier shoppers than average, so their effect is about 17% larger than the effect across everyone. The LATE answers "what does opening do for people who open when emailed", not "what would opening do for anyone". If a different email changes *who* complies, the LATE changes too.

</details>

---

## 12.5 Day 12 mini-project: estimand cards for three decisions (70 min)

**Goal.** Before estimating anything, write down exactly what you want. Create `docs/ESTIMANDS.md` with one card per decision, using this template:

```markdown
### Decision: <the business decision>
- **Unit:**
- **Treatment vs comparison:**
- **Outcome and horizon:**
- **Population:**
- **Summary measure:**
- **Name:** ATE / ATT / ATC / CATE / LATE
- **True value (from the truth file, for checking only):**
- **Data we would need to estimate it:**
```

Write cards for:

1. The CFO asks: *"Did SPRING26 pay for itself?"*
2. The CMO asks: *"Next spring, should we email everyone?"*
3. The CRM team asks: *"Should loyalty members get the email first?"*

<details>
<summary>Model answers</summary>

```markdown
### Decision: Did SPRING26 pay for itself?
- **Unit:** customer
- **Treatment vs comparison:** received SPRING26 vs the same customer not receiving it
- **Outcome and horizon:** completed orders (then margin) in weeks 9–16 of 2026
- **Population:** the 1,279 customers who were emailed
- **Summary measure:** average difference, then total (× 1,279)
- **Name:** ATT
- **True value:** +0.443 orders per customer (+15.3%)
- **Data we would need:** pre- and post-period orders for emailed and non-emailed customers,
  plus an assumption (e.g. parallel trends) to stand in for the missing Y₀ of the emailed

### Decision: Next spring, should we email everyone?
- **Population:** all 4,000 customers (or next year's base)        **Name:** ATE
- **True value:** +0.319 orders per customer (+15.7%)
- **Caveat:** next spring is a different period; the effect may not transfer

### Decision: Should loyalty members get the email first?
- **Population:** two subgroups: loyalty members and non-members   **Name:** CATE by loyalty
- **True values:** +0.429 (members) vs +0.272 (non-members) orders per customer
- **Note:** in relative terms both are ~+15.8%; members gain more orders only because they shop more
```

</details>

---

## 12.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 12: estimands, CATE, LATE, estimand cards"
git push
```

### Day 12 checklist

- [ ] I can distinguish an estimand, an estimator and an estimate
- [ ] I can state an estimand completely: unit, treatment, outcome, horizon, population, summary
- [ ] I can choose between ATE, ATT, ATC and CATE for a given decision
- [ ] I can explain why the LATE is an effect for compliers only, and compute it with the Wald ratio
- [ ] I have written estimand cards for three NorthMart decisions

---

# DAY 13 — Randomisation as the Benchmark

| Time | Block |
|---|---|
| 0:00 – 0:40 | 13.1 Why a coin flip removes selection bias |
| 0:40 – 1:30 | 13.2 Replaying the email as a randomised experiment |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 13.3 Right on average: the sampling distribution |
| 2:40 – 3:20 | 13.4 One experiment, analysed properly |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 13.5 Day 13 mini-project: balance and a permutation test |
| 4:45 – 5:00 | 13.6 Save your work with Git |

---

## 13.1 Why a coin flip removes selection bias (40 min)

**Concept.** Day 11 showed: naive difference = ATT + selection bias, where the bias is E[Y₀ | T=1] − E[Y₀ | T=0]. Selection bias exists because the treated and untreated would have differed *anyway*.

If treatment is assigned by a coin flip, it can't depend on anything about the customer, including their potential outcomes. Then on average:

```text
E[Y₀ | T=1] = E[Y₀ | T=0]      → selection bias = 0
E[Y₁ − Y₀ | T=1] = E[Y₁ − Y₀]  → ATT = ATE
```

So the simple difference in means estimates the ATE, with no model and no adjustment. That's why randomised experiments are the **benchmark** every other method in this course is compared against. In Modules 4–6, the question for every observational method will be: *how close does it get to what an experiment would have told us?*

"On average" matters. In any single experiment the groups differ a little by chance. Randomisation doesn't make the groups identical; it makes the differences **random**, so they average out and can be quantified with standard errors.

---

## 13.2 Replaying the email as a randomised experiment (50 min)

The truth file lets us do what no real company can: rerun the spring campaign with a coin flip instead of targeting, on the *same* customers.

Create `day13/one_randomization.py`:

```python
# day13/one_randomization.py   (python -m day13.one_randomization)
"""What if marketing had flipped a coin instead of targeting? The truth file lets us replay it."""
import numpy as np
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
rng = np.random.default_rng(2026)

# 1. Assign the email at random to the same number of customers (1,279)
z = np.zeros(po.height, dtype=np.int8)
z[rng.choice(po.height, size=1279, replace=False)] = 1
rand = po.with_columns(z=pl.Series(z)).with_columns(
    y=pl.when(pl.col("z") == 1).then(pl.col("y1")).otherwise(pl.col("y0"))   # what we would observe
)

# 2. The simple difference in means
diff = rand.filter(pl.col("z") == 1)["y"].mean() - rand.filter(pl.col("z") == 0)["y"].mean()
print(f"Randomised difference in means:  {diff:+.3f}")
print(f"True ATE:                        {po['tau'].mean():+.3f}")

# 3. Why it works: without the email, the two random groups look alike
bias = rand.filter(pl.col("z") == 1)["y0"].mean() - rand.filter(pl.col("z") == 0)["y0"].mean()
print(f"Selection bias with randomisation: {bias:+.3f}  (vs +1.268 with targeting)")
```

```bash
python -m day13.one_randomization
```

```
Randomised difference in means:  +0.391
True ATE:                        +0.319
Selection bias with randomisation: +0.071  (vs +1.268 with targeting)
```

The estimate is +0.391 instead of +1.711. It isn't exactly 0.319: in this particular draw, the randomly emailed group happened to contain slightly heavier shoppers (+0.071). That's chance imbalance, not selection bias, and the next section shows it averages out.

---

## 13.3 Right on average: the sampling distribution (55 min)

**Concept.** An estimator's **sampling distribution** is the spread of estimates you'd get across many repetitions of the same design. It tells you two things: whether the estimator is right on average (**unbiased**), and how far off a single estimate can be (**standard error**).

Create `day13/many_randomizations.py`:

```python
# day13/many_randomizations.py   (python -m day13.many_randomizations)
"""Repeat the coin flip 2,000 times: randomisation is right ON AVERAGE, and we can measure the noise."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
y0, y1 = po["y0"].to_numpy(), po["y1"].to_numpy()
n, n_treated, reps = po.height, 1279, 2000
rng = np.random.default_rng(13)

estimates = np.empty(reps)
for r in range(reps):
    z = np.zeros(n, dtype=bool)
    z[rng.choice(n, size=n_treated, replace=False)] = True
    estimates[r] = y1[z].mean() - y0[~z].mean()

ate = (y1 - y0).mean()
print(f"True ATE                      {ate:.3f}")
print(f"Average of 2,000 estimates    {estimates.mean():.3f}")
print(f"Standard deviation (noise)    {estimates.std():.3f}")
print(f"Middle 95% of estimates       [{np.percentile(estimates, 2.5):.3f}, {np.percentile(estimates, 97.5):.3f}]")
print("Targeted (real) naive estimate: 1.711")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(estimates, bins=50, color="#3987e5", alpha=0.85)
ax.axvline(ate, color="black", linewidth=2, label=f"true ATE = {ate:.2f}")
ax.set_xlabel("Difference in means across 2,000 random assignments (orders per customer)")
ax.set_ylabel("Number of assignments")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day13_randomization_distribution.png", dpi=150)
print("Saved figures/day13_randomization_distribution.png")
```

```
True ATE                      0.319
Average of 2,000 estimates    0.317
Standard deviation (noise)    0.073
Middle 95% of estimates       [0.180, 0.461]
Targeted (real) naive estimate: 1.711
Saved figures/day13_randomization_distribution.png
```

Open the figure. The 2,000 estimates form a bell curve centred on the true ATE. Its standard deviation, 0.073, is the **standard error** of the experiment. The real, targeted comparison (1.711) is about 19 standard errors away: no amount of noise explains it.

**Try it.** How does the noise change with the number of emailed customers? Rerun the simulation with 200, 1,279 and 2,000 emailed customers (1,000 repetitions each).

<details>
<summary>Solution</summary>

```python
import numpy as np
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
y0, y1 = po["y0"].to_numpy(), po["y1"].to_numpy()
rng = np.random.default_rng(13)

for n_treated in [200, 1279, 2000]:
    est = []
    for _ in range(1000):
        z = np.zeros(po.height, dtype=bool)
        z[rng.choice(po.height, size=n_treated, replace=False)] = True
        est.append(y1[z].mean() - y0[~z].mean())
    est = np.array(est)
    print(f"{n_treated:>5} emailed: mean {est.mean():.3f}, SD {est.std():.3f}, "
          f"share of estimates below 0.1: {(est < 0.1).mean():.1%}")
```

```
  200 emailed: mean 0.320, SD 0.163, share of estimates below 0.1: 7.7%
 1279 emailed: mean 0.313, SD 0.073, share of estimates below 0.1: 0.3%
 2000 emailed: mean 0.314, SD 0.065, share of estimates below 0.1: 0.1%
```

All three are unbiased; only the noise changes. With 200 emailed customers, about one experiment in thirteen would report an effect under 0.1, a third of the truth. A balanced split (2,000 vs 2,000) is the most precise for a given total. Module 3 turns this into **power analysis**: choosing a sample size *before* running the test.

</details>

---

## 13.4 One experiment, analysed properly (40 min)

**Concept.** In practice you run **one** experiment. Two tools give its uncertainty:

1. **Regression of the outcome on the treatment** reproduces the difference in means, and gives a standard error and confidence interval. Use **robust** (HC2) standard errors: they don't assume equal variance in both groups.
2. **Adding pre-treatment covariates** doesn't change *what* is estimated (randomisation already removed bias) but absorbs noise, giving a tighter interval. Module 3 develops this as CUPED.

Create `day13/regression_ci.py` (it uses `pre_orders` from Module 1's published `customer_features`):

```python
# day13/regression_ci.py   (python -m day13.regression_ci)
"""One randomised experiment, analysed properly: estimate, standard error, confidence interval."""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
features = pl.read_parquet("data/published/customer_features.parquet").select("customer_id", "pre_orders")

rng = np.random.default_rng(2026)
z = np.zeros(po.height, dtype=np.int8)
z[rng.choice(po.height, size=1279, replace=False)] = 1
df = (po.with_columns(z=pl.Series(z))
        .with_columns(y=pl.when(pl.col("z") == 1).then(pl.col("y1")).otherwise(pl.col("y0")))
        .join(features, on="customer_id")
        .to_pandas())

# 1. Difference in means as a regression: the coefficient on z IS the difference
m1 = smf.ols("y ~ z", data=df).fit(cov_type="HC2")      # HC2 = robust standard errors
# 2. Add a pre-treatment covariate: same target, less noise
m2 = smf.ols("y ~ z + pre_orders", data=df).fit(cov_type="HC2")

for name, m in [("y ~ z", m1), ("y ~ z + pre_orders", m2)]:
    lo, hi = m.conf_int().loc["z"]
    print(f"{name:<22} effect {m.params['z']:+.3f}  SE {m.bse['z']:.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"True ATE {po['tau'].mean():+.3f}")
```

```
y ~ z                  effect +0.391  SE 0.073  95% CI [+0.249, +0.534]
y ~ z + pre_orders     effect +0.355  SE 0.064  95% CI [+0.230, +0.481]
True ATE +0.319
```

- The robust standard error (0.073) matches the spread across 2,000 replays in 13.3: one experiment can measure its own uncertainty.
- Adding `pre_orders` shrinks the standard error by 12% and moves the estimate closer to the truth, because it corrects part of this draw's chance imbalance.
- Both intervals contain the true ATE.

> **Covariates in experiments:** only ever add variables measured **before** treatment. Day 17 shows what happens otherwise.

---

## 13.5 Day 13 mini-project: balance and a permutation test (70 min)

**Goal.** Two checks every experiment report should contain:

1. **A balance table**: are treated and control similar on pre-treatment characteristics? Measured with **standardised mean differences** (SMD): the gap in means divided by the pooled standard deviation, so different variables are comparable. |SMD| > 0.1 is the usual warning sign.
2. **A permutation test** (randomisation inference): if the email did nothing, the treatment labels are arbitrary. Shuffle them thousands of times; how often is the shuffled difference as large as the real one? That share is a p-value that needs no distributional assumptions.

Create `day13/balance_and_permutation.py`:

```python
# day13/balance_and_permutation.py   (python -m day13.balance_and_permutation)
"""Mini-project: (1) balance under targeting vs randomisation, (2) a permutation test."""
import numpy as np
import polars as pl

features = pl.read_parquet("data/published/customer_features.parquet")
po = pl.read_parquet("data/truth/potential_outcomes.parquet").select("customer_id", "y0", "y1")
df = features.join(po, on="customer_id")

rng = np.random.default_rng(2026)
z = np.zeros(df.height, dtype=np.int8)
z[rng.choice(df.height, size=1279, replace=False)] = 1
df = df.with_columns(z=pl.Series(z))

COVARIATES = ["pre_orders", "pre_revenue", "is_loyalty", "age", "tenure_days"]


def smd(data: pl.DataFrame, group: str, col: str) -> float:
    """Standardised mean difference: gap in means, in units of the pooled standard deviation."""
    a = data.filter(pl.col(group) == 1)[col]
    b = data.filter(pl.col(group) == 0)[col]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


# 1. Balance table
table = pl.DataFrame({
    "covariate": COVARIATES,
    "smd_targeted": [round(smd(df, "treated", c), 3) for c in COVARIATES],
    "smd_randomised": [round(smd(df, "z", c), 3) for c in COVARIATES],
})
print(table)
print("Rule of thumb: |SMD| above 0.1 signals imbalance.\n")

# 2. Permutation test for the randomised experiment
y = np.where(df["z"].to_numpy() == 1, df["y1"].to_numpy(), df["y0"].to_numpy())
zz = df["z"].to_numpy().astype(bool)
observed = y[zz].mean() - y[~zz].mean()

reps = 5000
null = np.empty(reps)
for r in range(reps):
    shuffled = rng.permutation(zz)             # pretend the email did nothing: labels are arbitrary
    null[r] = y[shuffled].mean() - y[~shuffled].mean()

p_value = (np.abs(null) >= abs(observed)).mean()
print(f"Observed difference      {observed:+.3f}")
shown = f"{p_value:.4f}" if p_value > 0 else f"< {1 / reps:.4f}"
print(f"Permutation p-value      {shown}  (none of {reps:,} shuffles was as extreme)" if p_value == 0
      else f"Permutation p-value      {shown}  ({reps:,} shuffles)")
```

```
│ covariate   ┆ smd_targeted ┆ smd_randomised │
│ pre_orders  ┆ 0.627        ┆ 0.037          │
│ pre_revenue ┆ 0.461        ┆ 0.005          │
│ is_loyalty  ┆ 0.621        ┆ 0.031          │
│ age         ┆ 0.038        ┆ 0.025          │
│ tenure_days ┆ -0.04        ┆ 0.01           │
Rule of thumb: |SMD| above 0.1 signals imbalance.

Observed difference      +0.391
Permutation p-value      < 0.0002  (none of 5,000 shuffles was as extreme)
```

Under targeting, three covariates are badly imbalanced (SMD above 0.6); under randomisation, every SMD is below 0.04. The randomised experiment's effect is far too large to be chance.

Finish by adding a short section to `docs/ESTIMANDS.md`: **"What if marketing had randomised?"** In four sentences: what the experiment would have estimated, how precisely, what it would have cost (emails sent to customers marketing considered less promising), and why that cost may be worth paying.

---

## 13.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 13: randomisation benchmark, sampling distribution, permutation test"
git push
```

**Coming next (Days 14–17).** Randomisation is the gold standard, but NorthMart's email *wasn't* randomised. Days 14–17 introduce **DAGs**, the tool for deciding what to adjust for when randomisation isn't available, and for spotting the adjustments that make things worse.

### Day 13 checklist

- [ ] I can explain why randomisation makes selection bias zero on average, and ATT equal ATE
- [ ] I can simulate a sampling distribution and read bias and standard error from it
- [ ] I can analyse one experiment with robust standard errors and pre-treatment covariates
- [ ] I can build a balance table with standardised mean differences
- [ ] I can run a permutation test and interpret its p-value
