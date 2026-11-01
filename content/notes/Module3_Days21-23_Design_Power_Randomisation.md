# Module 3 · Experimentation & A/B Testing
## Days 21–23: Designing Tests, Power, and Randomisation

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–2
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 23:** Students can write a pre-registered experiment plan, size a test with power analysis (by formula and by simulation), assign treatment with a deterministic stratified service, and catch a broken experiment with A/A tests and sample-ratio checks before trusting any result.

---

### Where we are

Module 2 ended with a clear lesson: NorthMart's spring email can't be evaluated from the data marketing produced, because nobody recorded *why* customers were chosen. Day 13 showed what a coin flip would have bought: an unbiased estimate, with a measurable margin of error.

Module 3 is about running that coin flip **well**. A badly run experiment is worse than none, because it looks trustworthy. Over ten days we build the toolkit an experimentation team uses every week:

| Days | Question |
|---|---|
| 21–23 | How do we design, size and assign a test so it *can* answer the question? |
| 24–27 | How do we analyse it: precisely, honestly, and without fooling ourselves by peeking or fishing? |
| 28–30 | What if customers affect each other, or the change hits everyone at once? Then the capstone. |

### The experimentation lab

Real tests take weeks. To run hundreds of them in a day, we use a **simulator**: NorthMart's experimentation lab, with 40,000 customers (more than the 4,000 of Modules 1–2, because experiments need numbers) and eight weeks of pre-period history. It runs any experiment for four weeks and returns each customer's orders, revenue, conversion and margin. Because it's simulated, it also knows the **true** effect of every experiment, so we can grade every method.

Create `causalkit/lab.py`. **Read only the docstrings for now**: the code contains the answers to the Day 30 capstone.

<details>
<summary>causalkit/lab.py (provided; click to expand)</summary>

```python
"""
NorthMart's experimentation lab (simulated). Provided for Module 3: read the docstrings, not the code,
until the capstone is done (the code contains the true effects).

    from causalkit.lab import population, run, truth

    pop = population()                      # 40,000 customers with 8 weeks of pre-period history
    out = run(pop, treated, "email_v2")     # outcomes over the 4-week test window
    truth("email_v2")                       # the true average effects (for checking only)

A real company has the first two and never the third.
"""
from functools import lru_cache

import numpy as np
import polars as pl

N_CUSTOMERS = 40_000
PRE_WEEKS = 8
TEST_WEEKS = 4
SHIPPING_FEE = 7.0
OLD_THRESHOLD, NEW_THRESHOLD = 50.0, 35.0
GROSS_MARGIN = 0.30

EXPERIMENTS = {
    # lift in order rate, by loyalty status (non-members, members)
    "email_v2":      {"lift": (0.05, 0.05)},
    "free_shipping": {"lift": (0.10, 0.03)},
    "aa_test":       {"lift": (0.0, 0.0)},
}


@lru_cache(maxsize=1)
def _population(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_CUSTOMERS
    loyalty = rng.random(n) < 0.30
    rate = rng.lognormal(-1.6, 0.6, n) * np.where(loyalty, 1.6, 1.0)        # orders per week
    basket = rng.lognormal(np.log(55), 0.45, n)                             # typical order value
    household = rng.integers(0, int(n / 2.2), n)                            # ~2.2 customers per household
    pre_orders = rng.poisson(rate * PRE_WEEKS)
    pre_revenue = _order_values(rng, pre_orders, basket).round(2)
    return pl.DataFrame({
        "customer_id": [f"L{i:05d}" for i in range(1, n + 1)],
        "household_id": [f"H{h:05d}" for h in household],
        "region": rng.choice(["ON", "QC", "BC", "AB", "ATL"], n, p=[0.38, 0.23, 0.15, 0.12, 0.12]),
        "device": rng.choice(["mobile", "desktop"], n, p=[0.62, 0.38]),
        "loyalty": loyalty.astype(np.int8),
        "pre_orders": pre_orders,
        "pre_revenue": pre_revenue,
        "_rate": rate,
        "_basket": basket,
    })


def _order_values(rng, n_orders: np.ndarray, basket: np.ndarray) -> np.ndarray:
    """Total value of n_orders[i] orders for each customer i."""
    owner = np.repeat(np.arange(len(n_orders)), n_orders)
    values = basket[owner] * rng.lognormal(-0.045, 0.3, owner.size)
    return np.bincount(owner, weights=values, minlength=len(n_orders))


def population(seed: int = 2030) -> pl.DataFrame:
    """40,000 customers: id, household, region, device, loyalty, and 8 weeks of pre-period orders/revenue.
    Columns starting with '_' are the simulator's hidden truth; don't use them in an analysis."""
    return _population(seed).clone()


def run(pop: pl.DataFrame, treated, experiment: str, seed: int = 0,
        spillover: float = 0.0) -> pl.DataFrame:
    """Run an experiment for 4 weeks. `treated` is a 0/1 array aligned with `pop`.
    Returns customer_id, treated, orders, revenue, converted (any order), and margin
    (30% of revenue, minus shipping NorthMart absorbs under free shipping).
    `spillover` (Day 28): share of a treated household member's lift that reaches the others."""
    spec = EXPERIMENTS[experiment]
    rng = np.random.default_rng(seed)
    t = np.asarray(treated).astype(bool)
    loyal = pop["loyalty"].to_numpy().astype(bool)
    lift = np.where(loyal, spec["lift"][1], spec["lift"][0])
    exposure = t.astype(float)
    if spillover:
        hh = pop["household_id"].to_numpy()
        _, inv = np.unique(hh, return_inverse=True)
        treated_in_hh = np.bincount(inv, weights=t)[inv] - t
        exposure = np.where(t, 1.0, np.minimum(treated_in_hh, 1) * spillover)
    rate = pop["_rate"].to_numpy() * (1 + lift * exposure)
    orders = rng.poisson(rate * TEST_WEEKS)

    owner = np.repeat(np.arange(len(orders)), orders)
    values = pop["_basket"].to_numpy()[owner] * rng.lognormal(-0.045, 0.3, owner.size)
    revenue = np.bincount(owner, weights=values, minlength=len(orders))
    absorbed = np.zeros(len(orders))
    if experiment == "free_shipping":
        free_now = t[owner] & (values >= NEW_THRESHOLD) & (values < OLD_THRESHOLD)
        absorbed = np.bincount(owner, weights=free_now * SHIPPING_FEE, minlength=len(orders))
    return pl.DataFrame({
        "customer_id": pop["customer_id"],
        "treated": t.astype(np.int8),
        "orders": orders,
        "revenue": revenue.round(2),
        "converted": (orders > 0).astype(np.int8),
        "margin": (GROSS_MARGIN * revenue - absorbed).round(2),
    })


def truth(experiment: str, seed: int = 2030) -> dict:
    """True average effects per customer over the 4-week window (ATE over the population)."""
    pop = _population(seed)
    rate = pop["_rate"].to_numpy()
    basket = pop["_basket"].to_numpy()
    loyal = pop["loyalty"].to_numpy().astype(bool)
    spec = EXPERIMENTS[experiment]
    lift = np.where(loyal, spec["lift"][1], spec["lift"][0])
    mu0 = rate * TEST_WEEKS
    mu1 = mu0 * (1 + lift)
    d_orders = mu1 - mu0
    d_revenue = d_orders * basket
    d_conv = np.exp(-mu0) - np.exp(-mu1)
    d_margin = GROSS_MARGIN * d_revenue
    if experiment == "free_shipping":
        # share of orders with value in [35, 50): order value = basket * lognormal(-0.045, 0.3)
        from scipy.stats import norm
        lo = (np.log(NEW_THRESHOLD / basket) + 0.045) / 0.3
        hi = (np.log(OLD_THRESHOLD / basket) + 0.045) / 0.3
        share = norm.cdf(hi) - norm.cdf(lo)
        d_margin = d_margin - SHIPPING_FEE * share * mu1
    return {
        "orders": float(d_orders.mean()),
        "revenue": float(d_revenue.mean()),
        "converted": float(d_conv.mean()),
        "margin": float(d_margin.mean()),
        "orders_relative": float(d_orders.sum() / mu0.sum()),
        "revenue_relative": float(d_revenue.sum() / (mu0 * basket).sum()),
    }


def run_weekly(pop: pl.DataFrame, treated, experiment: str, seed: int = 0, weeks: int = TEST_WEEKS) -> pl.DataFrame:
    """Like run(), but one row per customer per week (week 1..weeks), so results can be
    looked at as they accumulate (Days 26-27). Columns: customer_id, week, treated, orders, revenue."""
    frames = []
    for w in range(1, weeks + 1):
        one = run(pop, treated, experiment, seed=seed * 1000 + w)
        # run() simulates 4 weeks; scale to one week by thinning orders (each kept with prob 1/4)
        rng = np.random.default_rng(seed * 1000 + w + 500)
        orders = one["orders"].to_numpy()
        kept = rng.binomial(orders, 1 / TEST_WEEKS)
        share = np.divide(kept, orders, out=np.zeros(len(orders)), where=orders > 0)
        frames.append(pl.DataFrame({
            "customer_id": one["customer_id"], "week": np.full(len(orders), w, dtype=np.int8),
            "treated": one["treated"], "orders": kept,
            "revenue": (one["revenue"].to_numpy() * share).round(2),
        }))
    return pl.concat(frames)
```

</details>

Three experiments live in the lab:

| Experiment | What it is | Used on |
|---|---|---|
| `aa_test` | both groups get the same experience | Days 21, 22, 26 |
| `email_v2` | a redesigned version of the spring email | Days 21–29 |
| `free_shipping` | free-shipping threshold lowered from $50 to $35 | Day 30 capstone |

### New folders

```
causal-decision-lab/
├── causalkit/
│   ├── lab.py               # provided: the experimentation lab
│   ├── assignment.py        # Day 23: stratified, hash-based assignment service
│   ├── analysis.py          # Day 24: standard experiment readout
│   └── experiment_data.py   # Day 24: one run of email_v2, with covariates
├── day21/ ... day30/
└── docs/EXPERIMENT_PLAN_email_v2.md
```

Add one line to `requirements.txt` and one to `.gitignore`, then `pip install -r requirements.txt`:

```text
# Module 3: experimentation (scipy was already installed with statsmodels; pinned explicitly)
scipy==1.18.1
```

```text
data/experiments/
```

```bash
mkdir -p day21 day22 day23 day24 day25 day26 day27 day28 day29 day30
```

---

# DAY 21 — Designing an A/B Test

| Time | Block |
|---|---|
| 0:00 – 0:40 | 21.1 Anatomy of an experiment plan |
| 0:40 – 1:20 | 21.2 The lab and last month's baseline |
| 1:20 – 1:35 | Break |
| 1:35 – 2:20 | 21.3 Assignment by hashing |
| 2:20 – 3:10 | 21.4 A/A tests: test the testing machine |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 21.5 Day 21 mini-project: the plan, and a sample-ratio bug |
| 4:40 – 5:00 | 21.6 Save your work with Git |

---

## 21.1 Anatomy of an experiment plan (40 min)

**Concept.** Every decision about an experiment should be written down **before** any data arrive. Otherwise it's too easy to choose, after the fact, the metric, segment or stopping day that gives the answer you hoped for. A plan answers:

| Element | Question | email_v2 example |
|---|---|---|
| **Hypothesis** | What do we expect, and why? | The redesigned email raises orders by ~5% |
| **Unit of randomisation** | Who or what gets a coin flip? | customer |
| **Population** | Who is eligible? | all 40,000 active customers |
| **Treatment and control** | Exactly what each group experiences | new email vs the old one |
| **Primary metric** | The one number the decision rests on | revenue per customer, 4 weeks |
| **Secondary metrics** | Help explain the result | orders, conversion |
| **Guardrail metrics** | Must not get worse | margin, unsubscribes |
| **Minimum detectable effect** | Smallest effect worth knowing about | +5% revenue |
| **Sample size and duration** | From power analysis (Day 22) | 40,000 customers, 4 weeks |
| **Analysis** | Estimator, covariates, test, α | difference in means with CUPED, two-sided, α = 0.05 |
| **Decision rule** | What result leads to what action | ship if revenue CI excludes 0 and margin isn't harmed |

This is the **estimand card** of Day 12 extended into a full plan. In regulated fields it's called **pre-registration**; in tech companies, an experiment brief. The name doesn't matter. Writing it first does.

---

## 21.2 The lab and last month's baseline (40 min)

Create `day21/the_lab.py`:

```python
# day21/the_lab.py   (python -m day21.the_lab)
"""Meet the experimentation lab: 40,000 customers, and last month's metrics as a baseline."""
import numpy as np
import polars as pl

from causalkit.lab import population, run

pop = population()
print(pop.drop("_rate", "_basket").head(5))
print(f"{pop.height:,} customers in {pop['household_id'].n_unique():,} households")

# "Last month": nobody treated. This is the baseline any new experiment is planned against.
last_month = run(pop, np.zeros(pop.height), "aa_test", seed=99)
summary = last_month.select(
    pl.col("orders", "revenue", "converted", "margin").mean().name.suffix("_mean"),
    pl.col("orders", "revenue", "converted", "margin").std().name.suffix("_sd"),
).unpivot().with_columns(pl.col("value").round(3))
print(summary)
```

```bash
python -m day21.the_lab
```

```
│ customer_id ┆ household_id ┆ region ┆ device  ┆ loyalty ┆ pre_orders ┆ pre_revenue │
│ L00001      ┆ H17668       ┆ ON     ┆ mobile  ┆ 0       ┆ 2          ┆ 44.58       │
...
40,000 customers in 16,168 households
│ orders_mean    ┆ 1.147  │
│ revenue_mean   ┆ 70.217 │
│ converted_mean ┆ 0.606  │
│ margin_mean    ┆ 21.065 │
│ orders_sd      ┆ 1.351  │
│ revenue_sd     ┆ 99.03  │
│ converted_sd   ┆ 0.489  │
│ margin_sd      ┆ 29.709 │
```

"Last month" is the lab run with nobody treated: the historical baseline any plan starts from. Look at the standard deviations: **revenue's is 99 against a mean of 70**. A few customers spend a lot, most spend little. That spread, not the mean, determines how many customers a test needs (Day 22).

---

## 21.3 Assignment by hashing (45 min)

**Concept.** Large companies don't flip coins and store the results. They **hash** each customer's ID together with an experiment-specific **salt** and use the result as the coin. This gives three properties for free:

1. **Deterministic:** the same customer always lands in the same group, on every device and every visit.
2. **Independent across experiments:** different salts give unrelated assignments, so customers can be in many tests at once.
3. **Stateless:** nothing to store or sync; any server can compute the group.

Create `day21/hash_assignment.py`:

```python
# day21/hash_assignment.py   (python -m day21.hash_assignment)
"""
Assignment the way experimentation platforms do it: hash the customer id with an
experiment-specific salt. Deterministic (same customer, same group, every time),
independent across experiments (different salts), and needs no stored table.
"""
import hashlib

import numpy as np
import polars as pl
from scipy.stats import chisquare

from causalkit.lab import population


def bucket(customer_id: str, salt: str) -> int:
    """A number from 0 to 9,999, effectively random but fixed for (salt, customer)."""
    digest = hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 10_000


def assign(ids: pl.Series, salt: str, treat_share: float = 0.5) -> np.ndarray:
    return np.array([bucket(c, salt) < treat_share * 10_000 for c in ids], dtype=np.int8)


if __name__ == "__main__":
    pop = population()
    ids = pop["customer_id"]

    z = assign(ids, "email_v2")
    print("Treated share:", z.mean().round(4))
    print("Same answer the second time:", bool((assign(ids, "email_v2") == z).all()))

    z_other = assign(ids, "free_shipping")
    print("Overlap with another experiment's treated group:",
          round(((z == 1) & (z_other == 1)).mean(), 4), "(independent would be 0.25)")

    # Sample ratio mismatch (SRM) check: are the group sizes consistent with a 50/50 design?
    counts = np.bincount(z, minlength=2)
    stat, p = chisquare(counts, f_exp=[pop.height / 2] * 2)
    print(f"Group sizes {counts.tolist()}  SRM p-value {p:.3f}")
```

```bash
python -m day21.hash_assignment
```

```
Treated share: 0.5003
Same answer the second time: True
Overlap with another experiment's treated group: 0.2479 (independent would be 0.25)
Group sizes [19987, 20013]  SRM p-value 0.897
```

The last line is a **sample ratio mismatch (SRM)** check: a chi-square test of whether the group sizes are consistent with the planned 50/50 split. An SRM is one of the most common signs that an experiment is broken: a bug in assignment, logging, or eligibility. Industry rule: **if SRM p < 0.001, don't read the results; find the bug.**

---

## 21.4 A/A tests: test the testing machine (50 min)

**Concept.** In an **A/A test**, both groups get the same experience. The true effect is zero, so a correctly working pipeline should declare "significant" about 5% of the time at α = 0.05, and its p-values should be spread evenly between 0 and 1. If not, something in assignment, logging or analysis is broken, and every real test using that pipeline is suspect.

Create `day21/aa_tests.py`:

```python
# day21/aa_tests.py   (python -m day21.aa_tests)
"""
A/A tests: both groups get the SAME experience. Any 'significant' difference is a false positive.
Run 500 of them: about 5% should be significant at alpha = 0.05, and p-values should be uniform.
If not, the pipeline (assignment, logging, analysis) is broken, before any real test runs.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from causalkit.lab import population, run

pop = population()
rng = np.random.default_rng(21)
pvals = []
for r in range(500):
    z = rng.permutation(np.repeat([0, 1], pop.height // 2))
    out = run(pop, z, "aa_test", seed=r)
    y = out["revenue"].to_numpy()
    pvals.append(ttest_ind(y[z == 1], y[z == 0], equal_var=False).pvalue)
pvals = np.array(pvals)

print(f"Share of A/A tests with p < 0.05: {np.mean(pvals < 0.05):.3f}  (should be about 0.05)")
print(f"Share with p < 0.01:              {np.mean(pvals < 0.01):.3f}  (should be about 0.01)")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(pvals, bins=20, range=(0, 1), color="#3987e5", edgecolor="white")
ax.axhline(len(pvals) / 20, color="black", linestyle="--", label="expected if uniform")
ax.set_xlabel("p-value"); ax.set_ylabel("A/A tests"); ax.legend()
ax.set_title("500 A/A tests: p-values should be flat")
fig.tight_layout(); fig.savefig("figures/day21_aa_pvalues.png", dpi=150)
print("Saved figures/day21_aa_pvalues.png")
```

```bash
python -m day21.aa_tests
```

```
Share of A/A tests with p < 0.05: 0.048  (should be about 0.05)
Share with p < 0.01:              0.008  (should be about 0.01)
Saved figures/day21_aa_pvalues.png
```

Open the figure: a flat histogram. That's what a healthy pipeline looks like. Many experimentation teams run A/A tests continuously, alongside real tests, as a smoke alarm.

---

## 21.5 Day 21 mini-project: the plan, and a sample-ratio bug (75 min)

**Part 1 — write the plan.** Create `docs/EXPERIMENT_PLAN_email_v2.md` with every element of the table in 21.1, filled in for email_v2. Leave the sample-size line as "to be computed on Day 22".

**Part 2 — a bug hunt.** The new email's tracking pixel fails for 10% of treated customers who never place an order, so those customers silently disappear from the treated group's data. Create `day21/srm_bug.py`:

```python
# day21/srm_bug.py   (python -m day21.srm_bug)
"""
Mini-project, part 2: a logging bug. The new email's tracking pixel fails for 10% of treated
customers who never ordered, so they vanish from the treated group's data.
The SRM check catches it, and the estimate shows why it matters.
"""
import numpy as np
import polars as pl
from scipy.stats import chisquare

from causalkit.lab import population, run, truth
from day21.hash_assignment import assign

pop = population()
z = assign(pop["customer_id"], "email_v2")
out = run(pop, z, "email_v2", seed=7)

rng = np.random.default_rng(3)
lost = (out["treated"] == 1) & (out["orders"] == 0) & pl.Series(rng.random(out.height) < 0.10)
logged = out.filter(~lost)


def report(df: pl.DataFrame, label: str) -> None:
    counts = df.group_by("treated").len().sort("treated")["len"].to_numpy()
    p_srm = chisquare(counts, f_exp=[counts.sum() / 2] * 2).pvalue
    m = df.group_by("treated").agg(pl.col("revenue").mean()).sort("treated")["revenue"]
    print(f"{label:<22} groups {counts.tolist()}  SRM p = {p_srm:.4f}  revenue lift = {m[1] - m[0]:+.2f}")


report(out, "complete data")
report(logged, "with the logging bug")
print(f"True revenue effect: {truth('email_v2')['revenue']:+.2f} per customer")
```

```bash
python -m day21.srm_bug
```

```
complete data          groups [19987, 20013]  SRM p = 0.8966  revenue lift = +4.32
with the logging bug   groups [19987, 19278]  SRM p = 0.0003  revenue lift = +7.13
True revenue effect: +3.48 per customer
```

The bug removes non-buyers only from the treated group, so the treated group looks better than it is: the lift jumps from +4.32 to +7.13. That's collider bias from Day 16, created by a logging failure. The SRM check catches it (p = 0.0003) **before** anyone reads the lift.

**Try it.** Change the failure rate from 10% to 3% (`< 0.10` → `< 0.03`). Does the SRM check still catch it? Is the estimate still biased?

<details>
<summary>Solution</summary>

```
complete data          groups [19987, 20013]  SRM p = 0.8966  revenue lift = +4.32
with the logging bug   groups [19987, 19804]  SRM p = 0.3589  revenue lift = +5.10
True revenue effect: +3.48 per customer
```

The check misses it (p = 0.36), yet the lift is still inflated by +0.78. An SRM test has power like any other test: small leaks slip through. That's why teams also compare the number of customers *assigned* with the number *logged*, per group, every day. A 1% gap between those counts is a bug even when the chi-square test is silent.

</details>

---

## 21.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 21: experiment plan, lab, hash assignment, A/A tests, SRM"
git push
```

### Day 21 checklist

- [ ] I can write an experiment plan with hypothesis, unit, metrics, guardrails, MDE and decision rule
- [ ] I can assign treatment by hashing IDs with a salt, and explain why companies do it
- [ ] I can run a sample-ratio-mismatch check and know what to do when it fails
- [ ] I can use A/A tests to check that a pipeline's false positive rate is about α

---

# DAY 22 — Power and Sample Size

| Time | Block |
|---|---|
| 0:00 – 0:40 | 22.1 Errors, power and the minimum detectable effect |
| 0:40 – 1:30 | 22.2 The sample-size formula |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 22.3 Power by simulation |
| 2:45 – 3:20 | 22.4 Underpowered tests and what they cost |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 22.5 Day 22 mini-project: MDE curves and the guardrail problem |
| 4:45 – 5:00 | 22.6 Save your work with Git |

---

## 22.1 Errors, power and the minimum detectable effect (40 min)

**Concept.** A test can be wrong in two ways:

| | No real effect | Real effect |
|---|---|---|
| **Declare an effect** | false positive (rate α, usually 5%) | correct (rate = **power**) |
| **Declare nothing** | correct | false negative (rate β = 1 − power) |

**Power** is the probability of detecting an effect *of a given size*, if it's really there. The convention is 80%: one real effect in five goes unnoticed. The **minimum detectable effect (MDE)** is the smallest true effect a test detects with that power. Power depends on four things:

- **effect size** δ: bigger effects are easier to see;
- **noise** σ (the standard deviation of the metric): noisier metrics need more data;
- **sample size** n: precision grows with √n, so halving the MDE needs 4× the customers;
- **α**: a stricter threshold costs power.

---

## 22.2 The sample-size formula (50 min)

For two equal groups and a difference in means, the number of customers **per group** is:

```text
n = 2 × (z₁₋α/₂ + z_power)² × σ² / δ²
  = 2 × (1.96 + 0.84)² × σ² / δ²      for α = 0.05, power = 80%
```

Create `day22/power_formula.py`:

```python
# day22/power_formula.py   (python -m day22.power_formula)
"""How many customers do we need? The textbook formula, then statsmodels."""
import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ALPHA, POWER = 0.05, 0.80
z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)          # 1.96 + 0.84 = 2.80

# Baseline (Day 21, last month): means and standard deviations per customer over 4 weeks
baseline = {"revenue": (70.22, 99.03), "orders": (1.147, 1.351)}
lift = 0.05                                             # we want to detect a +5% change

print(f"z_(1-alpha/2) + z_(power) = {z:.3f}\n")
for metric, (mean, sd) in baseline.items():
    delta = lift * mean
    n_formula = 2 * z**2 * sd**2 / delta**2
    n_sm = NormalIndPower().solve_power(effect_size=delta / sd, alpha=ALPHA, power=POWER)
    print(f"{metric:<9} detect {delta:+.3f}: n per group = {n_formula:,.0f} (formula)  {n_sm:,.0f} (statsmodels)")

# A proportion: conversion from 60.6% to 60.6% x 1.025 (+2.5% relative)
p0 = 0.606
p1 = p0 * 1.025
h = proportion_effectsize(p1, p0)                       # Cohen's h for proportions
n_conv = NormalIndPower().solve_power(effect_size=h, alpha=ALPHA, power=POWER)
print(f"converted detect {p0:.3f} -> {p1:.3f}: n per group = {n_conv:,.0f}")
```

```bash
python -m day22.power_formula
```

```
z_(1-alpha/2) + z_(power) = 2.802

revenue   detect +3.511: n per group = 12,488 (formula)  12,488 (statsmodels)
orders    detect +0.057: n per group = 8,711 (formula)  8,711 (statsmodels)
converted detect 0.606 -> 0.621: n per group = 16,214
```

Detecting a 5% lift in revenue needs about 25,000 customers; in orders, 17,500. Orders need fewer because their spread relative to the mean is smaller (1.35 / 1.15 vs 99 / 70). For proportions, statsmodels uses **Cohen's h**, a variance-stabilising scale; detecting a 2.5% relative change in conversion needs 32,000 customers.

> The formula assumes a normal approximation, which holds well for means of thousands of customers even when individual revenue is very skewed.

---

## 22.3 Power by simulation (60 min)

**Concept.** Formulas need assumptions. Simulation needs none: run the experiment many times at a given sample size, with the true effect built in, and count how often the test detects it. With a lab, that's easy.

Create `day22/power_by_simulation.py`:

```python
# day22/power_by_simulation.py   (python -m day22.power_by_simulation)
"""
Power by simulation: run the experiment many times at each sample size and count how often
the test detects the (true, known) effect. No formula assumptions needed.
"""
import numpy as np
import polars as pl
from scipy.stats import ttest_ind

from causalkit.lab import population, run

pop = population()
rng = np.random.default_rng(22)
REPS = 200

rows = []
for n_per_group in [2_500, 5_000, 10_000, 20_000]:
    hits = {"revenue": 0, "orders": 0}
    for r in range(REPS):
        idx = rng.choice(pop.height, size=2 * n_per_group, replace=False)
        sub = pop[np.sort(idx)]
        z = rng.permutation(np.repeat([0, 1], n_per_group))
        out = run(sub, z, "email_v2", seed=r)
        for m in hits:
            y = out[m].to_numpy()
            hits[m] += ttest_ind(y[z == 1], y[z == 0], equal_var=False).pvalue < 0.05
    rows.append({"n_per_group": n_per_group,
                 "power_revenue": hits["revenue"] / REPS, "power_orders": hits["orders"] / REPS})

print(pl.DataFrame(rows))
print("Formula said: 80% power needs 12,488 (revenue) and 8,711 (orders) per group.")
```

```bash
python -m day22.power_by_simulation
```

```
│ n_per_group ┆ power_revenue ┆ power_orders │
│ 2500        ┆ 0.24          ┆ 0.31         │
│ 5000        ┆ 0.435         ┆ 0.595        │
│ 10000       ┆ 0.695         ┆ 0.82         │
│ 20000       ┆ 0.905         ┆ 0.975        │
Formula said: 80% power needs 12,488 (revenue) and 8,711 (orders) per group.
```

The simulation agrees with the formula: orders reach 80% power near 9,000 per group, revenue somewhere between 10,000 and 20,000. Simulation shines when the formula doesn't apply: clustered designs (Day 28), ratio metrics, sequential stopping (Day 27), or any analysis too complex for a closed form.

---

## 22.4 Underpowered tests and what they cost (35 min)

At 2,500 customers per group, email_v2 has 24% power on revenue: three times out of four, a real 5% lift goes unnoticed. That's the obvious cost. The hidden one is worse: when an underpowered test *does* come out significant, it **overstates** the effect, because only the lucky draws cross the threshold. This is called a **Type M (magnitude) error**, and it's why small tests that "found a big effect" so often fail to replicate. You'll see it directly on Day 26 (the winner's curse).

**Practical rules:**

- Decide the MDE from the **business**: the smallest effect that would change the decision (it must at least pay for the change). Then compute n.
- If n is larger than you have, don't run a weak test and hope. Change the design: a less noisy metric, CUPED (Day 25), a longer window, or a bigger effect worth testing.
- Never compute "observed power" after a test. It's a restatement of the p-value and tells you nothing new.

---

## 22.5 Day 22 mini-project: MDE curves and the guardrail problem (70 min)

**Goal.** Compute every metric's MDE at the sample we can afford, plot MDE curves, and check whether the free-shipping margin guardrail (Day 30) is testable at all.

Create `day22/mde_plan.py`:

```python
# day22/mde_plan.py   (python -m day22.mde_plan)
"""
Mini-project: the minimum detectable effect (MDE) of every metric at every affordable sample size,
and what that means for the planned tests.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from causalkit.lab import population, run

pop = population()
base = run(pop, np.zeros(pop.height), "aa_test", seed=99)        # last month
Z = norm.ppf(0.975) + norm.ppf(0.80)

metrics = ["revenue", "orders", "converted", "margin"]
stats = {m: (base[m].mean(), base[m].std()) for m in metrics}


def mde(sd: float, n_per_group: float) -> float:
    return Z * sd * np.sqrt(2 / n_per_group)


rows = []
for m, (mean, sd) in stats.items():
    rows.append({"metric": m, "baseline": round(mean, 3), "sd": round(sd, 3),
                 "MDE at 20k/group": round(mde(sd, 20_000), 3),
                 "MDE relative": f"{mde(sd, 20_000) / mean:.1%}"})
print(pl.DataFrame(rows))

# The free-shipping margin guardrail: could we even detect a loss of $0.50 per customer?
sd_margin = stats["margin"][1]
n_needed = 2 * Z**2 * sd_margin**2 / 0.50**2
print(f"\nDetecting a $0.50 margin loss needs {n_needed:,.0f} customers per group "
      f"({2 * n_needed:,.0f} total). We have {pop.height:,}.")

n = np.linspace(1_000, 20_000, 60)
fig, ax = plt.subplots(figsize=(7, 4))
for m, (mean, sd) in stats.items():
    ax.plot(n, 100 * mde(sd, n) / mean, label=m)
ax.set_xlabel("Customers per group"); ax.set_ylabel("Minimum detectable effect (% of baseline)")
ax.set_ylim(0, 25); ax.legend(); ax.set_title("MDE shrinks with the square root of sample size")
fig.tight_layout(); fig.savefig("figures/day22_mde_curves.png", dpi=150)
print("Saved figures/day22_mde_curves.png")
```

```bash
python -m day22.mde_plan
```

```
│ metric    ┆ baseline ┆ sd     ┆ MDE at 20k/group ┆ MDE relative │
│ revenue   ┆ 70.217   ┆ 99.03  ┆ 2.774            ┆ 4.0%         │
│ orders    ┆ 1.147    ┆ 1.351  ┆ 0.038            ┆ 3.3%         │
│ converted ┆ 0.606    ┆ 0.489  ┆ 0.014            ┆ 2.3%         │
│ margin    ┆ 21.065   ┆ 29.709 ┆ 0.832            ┆ 4.0%         │

Detecting a $0.50 margin loss needs 55,421 customers per group (110,841 total). We have 40,000.
Saved figures/day22_mde_curves.png
```

With all 40,000 customers, email_v2 can detect a 4% revenue lift: the planned +5% is within reach. Complete the sample-size line of `docs/EXPERIMENT_PLAN_email_v2.md`.

The margin guardrail is a different story: seeing a $0.50 loss per customer (2.4% of margin) would take 110,000 customers. Day 25 (CUPED) and Day 30 (non-inferiority) are the way out.

---

## 22.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 22: power by formula and simulation, MDE planning"
git push
```

### Day 22 checklist

- [ ] I can explain α, power and the MDE, and what each depends on
- [ ] I can compute sample sizes for means and proportions, by formula and with statsmodels
- [ ] I can estimate power by simulation
- [ ] I can explain why an underpowered significant result tends to overstate the effect
- [ ] I can decide whether a planned test is feasible, and what to change if it isn't

---

# DAY 23 — Randomisation, Stratification and Balance

| Time | Block |
|---|---|
| 0:00 – 0:40 | 23.1 Three ways to flip coins |
| 0:40 – 1:40 | 23.2 Comparing designs over 400 replays |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 23.3 Balance checks: what they can and can't tell you |
| 2:40 – 3:20 | 23.4 Building an assignment service |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 23.5 Day 23 mini-project: assign and audit email_v2 |
| 4:45 – 5:00 | 23.6 Save your work with Git |

---

## 23.1 Three ways to flip coins (40 min)

| Design | How | Group sizes | Precision |
|---|---|---|---|
| **Bernoulli** | an independent coin for each customer | vary from test to test | baseline |
| **Complete** | shuffle, then treat exactly half | fixed | about the same |
| **Stratified (blocked)** | split customers into similar groups (**strata**), then treat exactly half *within each* | fixed in every stratum | better, if strata predict the outcome |

All three are unbiased. Stratification removes the chance that, say, the treated group gets more heavy shoppers. That imbalance doesn't bias the estimate on average, but it adds noise to any single experiment. If strata are built from variables that predict the outcome (here: loyalty and pre-period revenue), the noise they remove is substantial.

---

## 23.2 Comparing designs over 400 replays (60 min)

Create `day23/three_designs.py`:

```python
# day23/three_designs.py   (python -m day23.three_designs)
"""
Three ways to randomise the same 40,000 customers. All three are unbiased;
they differ in how much the estimate jumps around from one experiment to the next.
"""
import numpy as np
import polars as pl

from causalkit.lab import population, run, truth

pop = population()
n = pop.height
rng = np.random.default_rng(23)

# Strata: loyalty x quintile of pre-period revenue (10 strata)
quint = pl.Series(pop["pre_revenue"]).rank("ordinal").to_numpy() * 5 // (n + 1)
stratum = pop["loyalty"].to_numpy() * 5 + quint


def bernoulli() -> np.ndarray:            # a coin per customer: group sizes vary
    return (rng.random(n) < 0.5).astype(np.int8)


def complete() -> np.ndarray:             # exactly half treated
    return rng.permutation(np.repeat(np.array([0, 1], dtype=np.int8), n // 2))


def stratified() -> np.ndarray:           # exactly half treated WITHIN each stratum
    z = np.zeros(n, dtype=np.int8)
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        z[rng.choice(idx, size=len(idx) // 2, replace=False)] = 1
    return z


results = {name: [] for name in ["bernoulli", "complete", "stratified"]}
sizes = {name: [] for name in results}
for r in range(400):
    for name, design in [("bernoulli", bernoulli), ("complete", complete), ("stratified", stratified)]:
        z = design()
        y = run(pop, z, "email_v2", seed=r)["revenue"].to_numpy()
        results[name].append(y[z == 1].mean() - y[z == 0].mean())
        sizes[name].append(z.sum())

true_effect = truth("email_v2")["revenue"]
rows = [{"design": k, "mean estimate": round(np.mean(v), 3), "SD of estimate": round(np.std(v), 3),
         "treated group size (min-max)": f"{min(sizes[k]):,}-{max(sizes[k]):,}"} for k, v in results.items()]
print(pl.DataFrame(rows))
print(f"True effect: {true_effect:+.3f}")
```

```bash
python -m day23.three_designs
```

```
│ design     ┆ mean estimate ┆ SD of estimate ┆ treated group size (min-max) │
│ bernoulli  ┆ 3.507         ┆ 1.018          ┆ 19,655-20,245                │
│ complete   ┆ 3.459         ┆ 0.979          ┆ 20,000-20,000                │
│ stratified ┆ 3.561         ┆ 0.874          ┆ 19,997-19,997                │
True effect: +3.480
```

- All three centre on the truth (differences of ±0.08 are simulation noise: 400 replays with SD ≈ 1).
- Stratification cuts the estimate's standard deviation by 11% compared with complete randomisation: free precision, worth about 25% more customers.
- Six of the 10 strata have an odd number of customers, and each gives its extra customer to control, so the treated group has 19,997 rather than 20,000. That's by design, not a bug.

---

## 23.3 Balance checks: what they can and can't tell you (45 min)

A **balance table** compares treated and control groups on pre-treatment characteristics, using the standardised mean difference (SMD, Day 13). In a randomised experiment it serves two purposes:

1. **Catching bugs.** If one group is clearly different (|SMD| > 0.1 with thousands of customers), assignment is probably broken: an eligibility filter applied to one group only, a cached old assignment, a bot filter that removed more of one group.
2. **Reassuring readers.** Stakeholders who see similar groups trust the result more.

What it can't do: prove the experiment worked. Balance on *observed* covariates says nothing directly about unobserved ones; in a randomised experiment, randomisation (not the table) is what protects those. And don't run significance tests on every covariate: with 20 covariates, one will be "significant" by chance (Day 26).

**Try it.** Compare simple hash assignment (Day 21) with stratified assignment (23.4) for five different salts: how big is the SMD of `pre_revenue` in each?

<details>
<summary>Solution</summary>

```python
# day23/compare_balance.py   (python -m day23.compare_balance)
import numpy as np
import polars as pl

from causalkit.assignment import assign_stratified, balance_table
from causalkit.lab import population
from day21.hash_assignment import assign

pop = population()
rows = []
for salt in ["test_a", "test_b", "test_c", "test_d", "test_e"]:
    simple = balance_table(pop, assign(pop["customer_id"], salt))
    strat = balance_table(pop, assign_stratified(pop, salt))
    pick = lambda t: t.filter(pl.col("covariate") == "pre_revenue")["smd"].item()
    rows.append({"salt": salt, "smd_simple": pick(simple), "smd_stratified": pick(strat)})
print(pl.DataFrame(rows))
```

```
│ salt   ┆ smd_simple ┆ smd_stratified │
│ test_a ┆ 0.013      ┆ 0.002          │
│ test_b ┆ -0.0004    ┆ -0.0043        │
│ test_c ┆ 0.0121     ┆ -0.0031        │
│ test_d ┆ 0.0136     ┆ -0.0027        │
│ test_e ┆ -0.007     ┆ 0.0013         │
```

Simple hashing gives SMDs up to 0.014; stratified assignment keeps every one below 0.005. Both are far from 0.1. With 40,000 customers, even simple randomisation balances well. Stratification matters most in **small** experiments (hundreds of units, or the 20 cities of Day 29).

</details>

---

## 23.4 Building an assignment service (40 min)

**Concept.** Combine Day 21's hashing with stratification: within each stratum, order customers by their hash and treat every second one. The assignment stays deterministic (same salt, same groups), needs no stored table, and is exactly balanced within strata.

Create `causalkit/assignment.py`. It also holds the SRM check and the balance table, so every later experiment uses the same audit:

```python
"""
NorthMart's assignment service (Day 23 mini-project): deterministic, stratified, auditable.

    from causalkit.assignment import assign_stratified, srm_pvalue, balance_table
"""
import hashlib

import numpy as np
import polars as pl
from scipy.stats import chisquare


def _bucket(customer_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:12], 16)


def add_strata(pop: pl.DataFrame) -> pl.DataFrame:
    """Loyalty x quintile of pre-period revenue: 10 strata."""
    return pop.with_columns(
        stratum=pl.col("loyalty").cast(pl.Int32) * 5
        + (pl.col("pre_revenue").rank("ordinal") * 5 // (pl.len() + 1)).cast(pl.Int32)
    )


def assign_stratified(pop: pl.DataFrame, salt: str) -> np.ndarray:
    """Within each stratum, order customers by their hash and treat every second one.
    Same salt, same answer; exact 50/50 inside every stratum (+/- 1 customer)."""
    df = add_strata(pop).with_columns(
        h=pl.Series([_bucket(c, salt) for c in pop["customer_id"]], dtype=pl.UInt64),
        row=pl.int_range(pl.len()),
    )
    ranked = df.sort("stratum", "h").with_columns(
        z=(pl.int_range(pl.len()).over("stratum") % 2).cast(pl.Int8))
    return ranked.sort("row")["z"].to_numpy()


def srm_pvalue(z: np.ndarray, treat_share: float = 0.5) -> float:
    counts = np.bincount(z, minlength=2)
    return chisquare(counts, f_exp=[counts.sum() * (1 - treat_share), counts.sum() * treat_share]).pvalue


def balance_table(pop: pl.DataFrame, z: np.ndarray) -> pl.DataFrame:
    """Standardised mean differences on pre-treatment covariates; |SMD| > 0.1 is flagged."""
    df = pop.with_columns(z=pl.Series(z), mobile=(pl.col("device") == "mobile").cast(pl.Int8))
    rows = []
    for col in ["pre_orders", "pre_revenue", "loyalty", "mobile"]:
        a = df.filter(pl.col("z") == 1)[col].cast(pl.Float64)
        b = df.filter(pl.col("z") == 0)[col].cast(pl.Float64)
        smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rows.append({"covariate": col, "treated": round(a.mean(), 3), "control": round(b.mean(), 3),
                     "smd": round(smd, 4), "flag": bool(abs(smd) > 0.1)})
    return pl.DataFrame(rows)
```

---

## 23.5 Day 23 mini-project: assign and audit email_v2 (70 min)

Create `day23/assignment_report.py`:

```python
# day23/assignment_report.py   (python -m day23.assignment_report)
"""Mini-project: assign the email_v2 test with the stratified service, and audit the result."""
import polars as pl

from causalkit.assignment import add_strata, assign_stratified, balance_table, srm_pvalue
from causalkit.lab import population

pop = population()
z = assign_stratified(pop, "email_v2")

print("Deterministic:", bool((assign_stratified(pop, "email_v2") == z).all()))
print(f"Treated: {z.sum():,} of {len(z):,}   SRM p-value: {srm_pvalue(z):.3f}")

per_stratum = (add_strata(pop).with_columns(z=pl.Series(z))
               .group_by("stratum").agg(customers=pl.len(), treated_share=pl.col("z").mean().round(4))
               .sort("stratum"))
print(per_stratum)
print(balance_table(pop, z))
```

```bash
python -m day23.assignment_report
```

```
Deterministic: True
Treated: 19,997 of 40,000   SRM p-value: 0.976
│ stratum ┆ customers ┆ treated_share │
│ 0       ┆ 6497      ┆ 0.4999        │
│ 1       ┆ 6126      ┆ 0.5           │
...
│ 9       ┆ 3536      ┆ 0.5           │
│ covariate   ┆ treated ┆ control ┆ smd     ┆ flag  │
│ pre_orders  ┆ 2.275   ┆ 2.262   ┆ 0.006   ┆ false │
│ pre_revenue ┆ 138.178 ┆ 138.536 ┆ -0.0021 ┆ false │
│ loyalty     ┆ 0.3     ┆ 0.3     ┆ -0.0001 ┆ false │
│ mobile      ┆ 0.621   ┆ 0.62    ┆ 0.0022  ┆ false │
```

Add an **Assignment** section to `docs/EXPERIMENT_PLAN_email_v2.md`: the salt (`email_v2`), the strata definition, and this audit. Anyone can now reproduce the exact groups from the plan alone.

Strata 0–4 are non-members by revenue quintile, 5–9 are members. Notice how unequal they are: loyalty members are 30% of customers but are concentrated in the top revenue quintiles.

---

## 23.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 23: stratified hash assignment service, balance and SRM audit"
git push
```

**Coming next (Days 24–27).** The email_v2 test is designed, sized and assigned. Days 24–27 analyse it: regression and robust standard errors, CUPED variance reduction, and the two most common ways teams fool themselves, testing many metrics and peeking early.

### Day 23 checklist

- [ ] I can explain Bernoulli, complete and stratified randomisation, and when stratification helps
- [ ] I can show by simulation that stratification reduces the estimate's variance
- [ ] I can read a balance table and explain what it can and can't prove
- [ ] I have a deterministic, stratified assignment service with SRM and balance audits
