# Module 7 · Sensitivity, Robustness & Decisions
## Days 67–70: Decisions under Uncertainty, the Value of Information, the Decision Memo and the Capstone

**Goal by end of Day 70:** Students can compare options under uncertainty with expected value, downside risk, regret and risk attitude; value flexibility (stop rules) and information (EVPI, EVSI, holdouts); write a decision memo whose numbers come from code; and deliver a complete targeting decision with robustness checks.

---

# DAY 67 — Decisions under Uncertainty

| Time | Block |
|---|---|
| 0:00 – 0:40 | 67.1 Decision criteria |
| 0:40 – 1:40 | 67.2 Four options, one set of uncertainties |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 67.3 Risk attitude and certainty equivalents |
| 2:40 – 3:20 | 67.4 What if you notice too late? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 67.5 Day 67 mini-project: how many customers to coupon? |
| 4:40 – 5:00 | 67.6 Save your work with Git |

---

## 67.1 Decision criteria (40 min)

When outcomes are uncertain, "the best option" depends on the criterion:

| Criterion | Choose the option with the... | When it fits |
|---|---|---|
| **Expected value** | highest average outcome | many similar decisions, losses affordable |
| **Probability of loss** | lowest chance of losing money | when a loss has consequences beyond its size (a manager's credibility, a covenant) |
| **Expected shortfall** (average of the worst 5%) | best bad case | when tail losses matter |
| **Maximin** | best worst case | extreme caution; uncertainty you can't put probabilities on |
| **Minimax regret** | smallest maximum regret (regret: how much worse than the best option in hindsight) | when being shown up by hindsight is costly |
| **Certainty equivalent** | highest sure-amount equivalent, given a risk tolerance | a company with a defined appetite for risk |

None is "correct". A good analysis reports several and lets the decision-maker see whether they agree.

---

## 67.2 Four options, one set of uncertainties (60 min)

Day 66's uncertain inputs, drawn 10,000 times, evaluated for four options on the **same** draws (so regret can be computed draw by draw):

1. don't build;
2. coupons for everyone;
3. targeted programme;
4. targeted programme **with a stop rule**: shut it down in the first month it stops paying.

The stop rule is the most realistic option. It's also the one most CBAs forget: a programme isn't a commitment to run for two years whatever happens. Add `stop` to Day 66's `npv` function (the version in `day66/cba.py` above already has it). Create `day67/options.py`:

```python
# day67/options.py   (python -m day67.options)
"""
Decisions under uncertainty: four options for the coupon programme, each evaluated on the SAME 10,000 draws of
the uncertain inputs (Day 66). Different decision criteria can pick different options.
  don't build | coupons for everyone | targeted | targeted, with a stop rule (shut down when it stops paying)
Run day66.cba first.
"""
import json

import numpy as np
import polars as pl

from day66.cba import npv


def draw_inputs(base, n=10_000, seed=67):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield base | {"fatigue": rng.uniform(0.96, 1.00), "pull_forward": rng.uniform(0.0, 0.30),
                      "uplift": rng.normal(base["uplift"], base["uplift"] * 0.066), "margin": rng.uniform(0.25, 0.35),
                      "coupon_cost": rng.uniform(4.0, 6.0), "customers": rng.uniform(300_000, 500_000),
                      "running_cost": rng.uniform(15_000, 50_000), "build_cost": rng.uniform(100_000, 300_000)}


def option_npvs(base):
    """NPV ($k) of each option in each draw."""
    out = {"don't build": [], "coupons for everyone": [], "targeted": [], "targeted + stop rule": []}
    for a in draw_inputs(base):
        out["don't build"].append(0.0)
        scale = base["uplift_everyone"] / base["uplift"]
        out["coupons for everyone"].append(npv(a | {"uplift": a["uplift"] * scale, "share": 1.0}, stop=False)[0] / 1000)
        out["targeted"].append(npv(a)[0] / 1000)
        out["targeted + stop rule"].append(npv(a, stop=True)[0] / 1000)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    base = json.load(open("data/day66_inputs.json"))
    v = option_npvs(base)
    best = np.max(np.column_stack(list(v.values())), axis=1)
    rows = []
    for name, x in v.items():
        worst5 = np.sort(x)[: len(x) // 20].mean()
        rows.append({"option": name, "expected NPV": round(x.mean()), "P(loss)": f"{(x < 0).mean():.0%}",
                     "average of worst 5%": round(worst5), "worst case": round(x.min()),
                     "expected regret": round((best - x).mean()), "max regret": round((best - x).max())})
    with pl.Config(tbl_width_chars=140, fmt_str_lengths=30):
        print(pl.DataFrame(rows))
    print("All $ figures in $ thousands. Regret: how much less than the best option in the same draw.")
```

```bash
python -m day67.options
```

```
│ option               ┆ expected NPV ┆ P(loss) ┆ average of worst 5% ┆ worst case ┆ expected regret ┆ max regret │
│ don't build          ┆ 0            ┆ 0%      ┆ 0                   ┆ 0          ┆ 4986            ┆ 29363      │
│ coupons for everyone ┆ -13007       ┆ 94%     ┆ -29742              ┆ -41256     ┆ 17993           ┆ 41256      │
│ targeted             ┆ 3809         ┆ 26%     ┆ -6573               ┆ -13224     ┆ 1176            ┆ 13224      │
│ targeted + stop rule ┆ 4976         ┆ 6%      ┆ -185                ┆ -299       ┆ 10              ┆ 299        │
All $ figures in $ thousands. Regret: how much less than the best option in the same draw.
```

The stop rule wins on **every** criterion. It's worth $1.2M of expected value over the plain programme, cuts the chance of a loss from 26% to 6%, and limits the worst case to roughly the build cost. That $1.2M is the **value of flexibility**: being able to stop when the evidence turns.

Notice the difference between "targeted" and "targeted + stop rule". It isn't a better model or more data; it's a monitoring commitment written into the plan. Decision quality often depends more on what you do *after* launch than on the estimate you launched with.

---

## 67.3 Risk attitude and certainty equivalents (45 min)

**Concept.** A company that can't afford a $10M loss shouldn't treat it as cancelling out a $10M gain. **Exponential utility** with risk tolerance R gives each option a **certainty equivalent (CE)**: the sure amount the company would take instead of the gamble.

```
CE = −R × log( mean( exp(−NPV / R) ) )
```

R is roughly the largest gamble the company would accept at even odds of winning R or losing R/2. For a mid-sized retailer, a few million dollars is typical. Create `day67/risk_attitude.py`:

```python
# day67/risk_attitude.py   (python -m day67.risk_attitude)
"""
Expected value treats a $10M loss and a $10M gain as cancelling out. A company that can't afford the loss
shouldn't. Exponential utility with risk tolerance R (roughly: the size of gamble the company would take at
even odds of winning R or losing R/2) gives each option a CERTAINTY EQUIVALENT: the sure amount the company
would accept instead of the gamble. CE = -R x log(mean(exp(-NPV / R))).
"""
import json

import numpy as np
import polars as pl

from day67.options import option_npvs

base = json.load(open("data/day66_inputs.json"))
v = option_npvs(base)
rows = []
for label, R in [("risk neutral", np.inf), ("R = $20M", 20_000), ("R = $5M", 5_000), ("R = $1M", 1_000)]:
    ce = {k: x.mean() if np.isinf(R) else -R * np.log(np.mean(np.exp(-x / R))) for k, x in v.items()}
    rows.append({"risk attitude": label} | {k: round(c) for k, c in ce.items()} | {"choice": max(ce, key=ce.get)})
with pl.Config(tbl_width_chars=140, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
print("Certainty equivalents in $ thousands.")
```

```bash
python -m day67.risk_attitude
```

```
│ risk attitude ┆ don't build ┆ coupons for everyone ┆ targeted ┆ targeted + stop rule ┆ choice               │
│ risk neutral  ┆ 0           ┆ -13007               ┆ 3809     ┆ 4976                 ┆ targeted + stop rule │
│ R = $20M      ┆ 0           ┆ -14673               ┆ 3028     ┆ 4512                 ┆ targeted + stop rule │
│ R = $5M       ┆ 0           ┆ -19484               ┆ 1006     ┆ 3514                 ┆ targeted + stop rule │
│ R = $1M       ┆ 0           ┆ -33041               ┆ -5543    ┆ 1613                 ┆ targeted + stop rule │
Certainty equivalents in $ thousands.
```

For a very risk-averse company (R = $1M), the plain targeted programme is worth **less than nothing** (CE −$5.5M), but with the stop rule it's still worth $1.6M. Flexibility matters most to those who can least afford to be wrong.

---

## 67.4 What if you notice too late? (40 min)

**Try it.** A stop rule assumes you see a bad month immediately. In practice, results arrive with a lag. Rerun the options with the stop acting 1, 2 or 4 months after the first loss-making month. Create `day67/late_stop.py`.

<details>
<summary>Solution</summary>

```python
# day67/late_stop.py   (python -m day67.late_stop)
import json

import numpy as np

from day67.options import draw_inputs


def npv_late_stop(a, months=24, lag=2):
    """Like npv(stop=True), but a loss-making month is only noticed `lag` months later."""
    rate = (1 + a["discount_rate"]) ** (1 / 12) - 1
    total, first_bad = -a["build_cost"], None
    for m in range(1, months + 1):
        flow = a["customers"] * (a["margin"] * a["uplift"] * (1 - a["pull_forward"]) * a["fatigue"] ** (m - 1)
                                 - a["coupon_cost"] * a["share"]) - a["running_cost"]
        if first_bad is None and flow < 0:
            first_bad = m
        if first_bad is not None and m >= first_bad + lag:
            break
        total += flow / (1 + rate) ** m
    return total / 1000


base = json.load(open("data/day66_inputs.json"))
for lag in [0, 1, 2, 4]:
    x = np.array([npv_late_stop(a, lag=lag) for a in draw_inputs(base)])
    print(f"stop rule acting {lag} months late: expected NPV ${x.mean():,.0f}k, P(loss) {(x < 0).mean():.0%}")
```

```
stop rule acting 0 months late: expected NPV $4,976k, P(loss) 6%
stop rule acting 1 months late: expected NPV $4,967k, P(loss) 7%
stop rule acting 2 months late: expected NPV $4,944k, P(loss) 7%
stop rule acting 4 months late: expected NPV $4,863k, P(loss) 9%
```

Even a four-month lag keeps almost all the benefit. In this programme, losses build up slowly (the effect fades gradually), so a late stop costs little. With a programme that could fail abruptly, the monitoring lag would matter much more. Ask: "how fast could this go wrong, and how fast would we know?"

</details>

---

## 67.5 Day 67 mini-project: how many customers to coupon? (65 min)

Finance isn't sure of the margin (25–35%) or the coupon's real cost ($4–6). Candidate policies coupon the top 20%, 35%, 50%, 65% or 80% of customers by the DR-learner's predicted effect. Create `day67/how_many.py`:

```python
# day67/how_many.py   (python -m day67.how_many)
"""
Mini-project: the finance team isn't sure of the margin (25-35%) or the coupon's real cost ($4-6).
Candidate policies: coupon the top 20%, 35%, 50%, 65% or 80% of customers by the DR-learner's predicted effect.
Evaluate each in each of 9 (margin, cost) scenarios on the experiment, then choose by three criteria:
  expected value (scenarios equally likely) | maximin (best worst case) | minimax regret.
"""
import itertools

import numpy as np
import polars as pl
from econml.dr import DRLearner

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import dr_scores
from day64.uplift_models import experiment
from day65.policy_value import outcome_models

train, test = experiment()
dr = DRLearner(model_propensity=clf(), model_regression=reg(),
               model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
               min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
rank = np.argsort(np.argsort(-dr.effect(test[X_COLS]))) / len(test)       # 0 = highest predicted effect
y, mu0, mu1 = outcome_models(test, cost=0.0)
s1, s0 = dr_scores(y / 0.30, test.coupon, 0.5, mu0 / 0.30, mu1 / 0.30)      # scores for SPEND
uplift = s1 - s0

shares = [0.20, 0.35, 0.50, 0.65, 0.80]
scenarios = list(itertools.product([0.25, 0.30, 0.35], [4.0, 5.0, 6.0]))
value = np.array([[np.mean((rank < q) * (m * uplift - c)) for (m, c) in scenarios] for q in shares])
regret = value.max(axis=0) - value
table = pl.DataFrame({"coupon the top": [f"{q:.0%}" for q in shares],
                      "expected value": value.mean(1).round(2), "worst case": value.min(1).round(2),
                      "max regret": regret.max(1).round(2)})
print(table)
for crit, col, best in [("expected value", "expected value", max), ("maximin", "worst case", max),
                        ("minimax regret", "max regret", min)]:
    i = table[col].to_list().index(best(table[col].to_list()))
    print(f"{crit:<15} chooses: coupon the top {shares[i]:.0%}")
print("Net $ per customer in the base. Best share in each scenario (margin, cost): " +
      ", ".join(f"({m:.2f}, ${c:.0f}) {shares[int(np.argmax(value[:, j]))]:.0%}" for j, (m, c) in enumerate(scenarios)))
```

```bash
python -m day67.how_many
```

```
│ coupon the top ┆ expected value ┆ worst case ┆ max regret │
│ 20%            ┆ 1.31           ┆ 0.73       ┆ 1.25       │
│ 35%            ┆ 1.78           ┆ 0.84       ┆ 0.43       │
│ 50%            ┆ 1.82           ┆ 0.6        ┆ 0.24       │
│ 65%            ┆ 1.67           ┆ 0.2        ┆ 0.64       │
│ 80%            ┆ 1.28           ┆ -0.4       ┆ 1.24       │
expected value  chooses: coupon the top 50%
maximin         chooses: coupon the top 35%
minimax regret  chooses: coupon the top 50%
```

Expected value and minimax regret pick 50%; maximin picks 35% (the best worst case, at margin 25% and cost $6). The difference between 35% and 50% is small in expected value ($0.04 per customer) but the 50% policy's worst case is lower. The script's last line shows where the best share changes: 35% is best when the coupon costs $6 or the margin is only 25%, and 50% or more otherwise. A defensible recommendation: "coupon the top 50%; if finance finds the coupon costs closer to $6, or the margin is nearer 25%, coupon the top 35%". A good analysis turns disagreement between criteria into a **conditional** recommendation.

---

## 67.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 67: decision criteria, value of flexibility, risk attitude, robust choice"
git push
```

### Day 67 checklist

- [ ] I can compare options by expected value, P(loss), expected shortfall, maximin and regret
- [ ] I evaluate options on the same simulated draws so regret is meaningful
- [ ] I can value flexibility (a stop rule) and explain why it changes the decision
- [ ] I can compute certainty equivalents for different risk tolerances
- [ ] I can turn disagreement between criteria into a conditional recommendation

---

# DAY 68 — The Value of Information

| Time | Block |
|---|---|
| 0:00 – 0:40 | 68.1 Information is worth what it changes |
| 0:40 – 1:20 | 68.2 A prior that includes the sensitivity analysis |
| 1:20 – 1:35 | Break |
| 1:35 – 2:10 | 68.3 EVPI: the ceiling |
| 2:10 – 3:10 | 68.4 EVSI: test first, or launch and learn? |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 68.5 Day 68 mini-project: what would actually have happened |
| 4:40 – 5:00 | 68.6 Save your work with Git |

---

## 68.1 Information is worth what it changes (40 min)

A study is worth running only if its result could **change the decision**, and only by how much the better decision earns. If you'd launch whatever the study found, it's worth nothing to this decision, however interesting.

The decision today is Day 61's: NorthMart's marketing team, working from last quarter's observational data, wants to send a $5 coupon to **all** 400,000 customers every month for a year. The DML estimate ($24.6) is comfortably above the $16.67 break-even. Should they launch, test first, or something in between?

Two quantities:

- **EVPI**, the expected value of perfect information: how much better, on average, we'd decide if we knew the true effect. The ceiling on what any study can be worth.
- **EVSI**, the expected value of sample information: the same for a real study with a given size and noise.

---

## 68.2 A prior that includes the sensitivity analysis (40 min)

Value of information needs a **belief** about the effect, uncertainty included. The sampling SE ($0.64) is the smallest part of the uncertainty. Day 61's benchmarks said a hidden confounder as strong as app use would inflate the estimate by about $5, and we can't rule out something stronger. So the belief: the true effect = $24.6 − (a bias somewhere between $0 and $10) ± sampling error.

Create `day68/prior.py`:

```python
# day68/prior.py   (python -m day68.prior)
"""
The value of information starts from what you believe now, uncertainty included. The decision: send a $5
coupon to all 400,000 customers every month for a year? It pays if the average effect exceeds $16.67.
What we know (Day 61): observational DML says $24.6 (SE 0.6), but a hidden confounder could inflate it.
Benchmarks said a confounder as strong as app use would move it by about $5, one as strong as deal-seeking
by about $2.5; we can't rule out something stronger. Belief: bias somewhere between $0 and $10, all values
equally plausible. Prior for the true effect = 24.6 - bias + sampling error.
"""
import numpy as np

ESTIMATE, SE, BIAS_MAX = 24.6, 0.64, 10.0
BREAK_EVEN = 5 / 0.30
CUSTOMER_MONTHS = 400_000 * 12
DOLLARS_PER_DOLLAR_OF_EFFECT = 0.30 * CUSTOMER_MONTHS     # NPV changes by this much per $1 of average effect


def prior_draws(n=200_000, seed=68):
    rng = np.random.default_rng(seed)
    return ESTIMATE - rng.uniform(0, BIAS_MAX, n) + rng.normal(0, SE, n)


def launch_value(effect):
    """Net value ($) of a year of coupons for everyone, if the true average effect is `effect`."""
    return DOLLARS_PER_DOLLAR_OF_EFFECT * (effect - BREAK_EVEN)


if __name__ == "__main__":
    theta = prior_draws()
    print(f"Prior for the true effect: mean {theta.mean():.1f}, SD {theta.std():.1f}, "
          f"90% range {np.percentile(theta, 5):.1f} to {np.percentile(theta, 95):.1f}")
    print(f"P(effect above break-even ${BREAK_EVEN:.2f}): {(theta > BREAK_EVEN).mean():.0%}")
    v = launch_value(theta)
    print(f"Launch: expected value ${v.mean() / 1e6:.1f}M, P(loss) {(v < 0).mean():.0%}, "
          f"expected loss when it loses ${-v[v < 0].mean() / 1e6:.1f}M")
```

```bash
python -m day68.prior
```

```
Prior for the true effect: mean 19.6, SD 3.0, 90% range 15.0 to 24.2
P(effect above break-even $16.67): 79%
Launch: expected value $4.2M, P(loss) 21%, expected loss when it loses $1.6M
```

This is the step that connects Days 61–63 to the decision: the sensitivity analysis becomes a probability distribution, and the "fragile" robustness value becomes a 21% chance that the launch loses money.

---

## 68.3 EVPI: the ceiling (35 min)

With perfect information we'd launch only when it pays: E[max(value, 0)]. Without it, we choose the better option on average: max(E[value], 0). Create `day68/evpi.py`:

```python
# day68/evpi.py   (python -m day68.evpi)
"""
Expected value of perfect information (EVPI): how much better would we do, on average, if we KNEW the true
effect before deciding? With perfect information we'd launch only when it pays: E[max(value, 0)].
Without it we pick the better option on average: max(E[value], 0). The difference caps what ANY study is worth.
"""
import numpy as np

from day68.prior import launch_value, prior_draws

v = launch_value(prior_draws())
with_info = np.maximum(v, 0).mean()
without = max(v.mean(), 0)
print(f"Decide now:                     ${without / 1e6:.2f}M  ({'launch' if v.mean() > 0 else 'do nothing'})")
print(f"Decide knowing the true effect: ${with_info / 1e6:.2f}M")
print(f"EVPI:                           ${(with_info - without) / 1e6:.2f}M")
```

```bash
python -m day68.evpi
```

```
Decide now:                     $4.23M  (launch)
Decide knowing the true effect: $4.57M
EVPI:                           $0.34M
```

Perfect information about the effect is worth at most $340,000 for this decision. Any study costing more isn't worth it. The EVPI is small relative to the stakes because we'd probably launch anyway (79%); information only pays in the 21% of worlds where it would stop us.

---

## 68.4 EVSI: test first, or launch and learn? (60 min)

Two ways to learn:

- **B. Test first:** a one-month experiment with n customers per arm, then decide for the remaining 11 months. Learning costs a month of the programme.
- **C. Launch with a holdout:** launch to everyone except a random holdout of n customers; after month 1, compare; stop if the evidence says it loses money. Learning costs only the holdout's month.

The math (normal approximation): a test with standard error s moves the belief to a posterior mean m′; before the test, m′ is itself uncertain, with variance = prior variance − posterior variance. The value of deciding after the test is E[max(0, k(m′ − break-even))], which has a closed form. Create `day68/evsi.py`:

```python
# day68/evsi.py   (python -m day68.evsi)
"""
Expected value of SAMPLE information: a real study is smaller than perfect information. Three ways to act:
  A. launch to everyone now, for 12 months
  B. test first: a 1-month experiment with n customers per arm, then decide for the remaining 11 months
  C. launch now, but keep a random holdout of n customers for the first month; after month 1, compare,
     and stop the programme for the remaining 11 months if the evidence says it loses money
Normal approximation (the prior is approximately normal): a test with standard error s moves our belief
to a posterior mean m'. Before the test, m' is uncertain: normal around today's mean with variance
(prior variance - posterior variance). Value of deciding after the test = E[max(0, k x (m' - break-even))].
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from day68.prior import BREAK_EVEN, DOLLARS_PER_DOLLAR_OF_EFFECT, prior_draws

RESIDUAL_SD = 38.8        # SD of spend after adjusting for the features (an experiment's analysis, Day 24)
CUSTOMERS = 400_000
FIXED_TEST_COST = 30_000  # setting up and analysing a test

theta = prior_draws()
mu0, sd0 = theta.mean(), theta.std()
per_month = DOLLARS_PER_DOLLAR_OF_EFFECT / 12              # $ per $1 of effect, per month, all customers


def expected_max0(mean, sd):
    """E[max(0, X)] for X ~ Normal(mean, sd)."""
    if sd == 0:
        return max(mean, 0.0)
    z = mean / sd
    return mean * norm.cdf(z) + sd * norm.pdf(z)


def value_after_learning(test_se, months):
    """Expected value of deciding, for `months` months, after a test with standard error test_se."""
    post_var = 1 / (1 / sd0**2 + 1 / test_se**2)
    spread = np.sqrt(sd0**2 - post_var)                     # SD of the posterior mean, before the test
    k = per_month * months
    return expected_max0(k * (mu0 - BREAK_EVEN), k * spread)


if __name__ == "__main__":
    A = per_month * 12 * (mu0 - BREAK_EVEN)
    rows, ns = [], [500, 1000, 2000, 5000, 10000, 20000, 50000]
    for n in ns:
        test_first = value_after_learning(RESIDUAL_SD * np.sqrt(2 / n), 11) - FIXED_TEST_COST
        holdout = (per_month * (mu0 - BREAK_EVEN) * (1 - n / CUSTOMERS)
                   + value_after_learning(RESIDUAL_SD * np.sqrt(1 / n + 1 / (CUSTOMERS - n)), 11) - FIXED_TEST_COST)
        rows.append((n, test_first, holdout))
    print(f"Prior: mean {mu0:.1f}, SD {sd0:.1f}.   A. launch now (12 months): ${A / 1e6:.2f}M")
    print(f"{'n per arm':>10} {'B. test first':>15} {'C. launch with holdout':>24}")
    for n, b, c in rows:
        print(f"{n:>10,} {b / 1e6:>14.2f}M {c / 1e6:>23.2f}M")
    best = max(rows, key=lambda r: r[2])
    print(f"Best: launch with a holdout of {best[0]:,} customers: ${best[2] / 1e6:.2f}M, "
          f"${(best[2] - A) / 1e6:.2f}M more than launching blind")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [r[1] / 1e6 for r in rows], marker="o", label="B. test first, then decide")
    ax.plot(ns, [r[2] / 1e6 for r in rows], marker="o", label="C. launch with a holdout")
    ax.axhline(A / 1e6, color="#64748b", ls="--", label="A. launch now")
    ax.set_xscale("log")
    ax.set_xlabel("customers per arm in the test")
    ax.set_ylabel("expected value of the year ($M)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/day68_evsi.png", dpi=120)
    print("Saved figures/day68_evsi.png")
```

```bash
python -m day68.evsi
```

```
Prior: mean 19.6, SD 3.0.   A. launch now (12 months): $4.23M
 n per arm   B. test first   C. launch with holdout
       500           3.99M                    4.41M
     1,000           4.05M                    4.46M
     2,000           4.11M                    4.49M
     5,000           4.14M                    4.51M
    10,000           4.16M                    4.51M
    20,000           4.17M                    4.51M
    50,000           4.17M                    4.48M
Best: launch with a holdout of 10,000 customers: $4.51M, $0.28M more than launching blind
Saved figures/day68_evsi.png
```

- **Testing first is worse than launching now**, at every size. The month of delay costs about $350,000 in expected value (a twelfth of $4.2M), more than the information is worth.
- **Launching with a holdout beats both.** It captures $0.28M of the $0.34M EVPI, for the cost of not couponing 5,000–20,000 customers for one month.
- Beyond about 10,000 customers, the holdout stops helping: the test is already precise enough, and more holdout customers just forgo profitable coupons.

"Test or launch?" is often a false choice. **Launch with a holdout and a stop rule** combines Day 67's flexibility with Day 68's information.

**Try it.** How does the EVPI change if you trust the observational estimate more: a possible bias of up to $5, or up to $2? Create `day68/more_trust.py`.

<details>
<summary>Solution</summary>

```python
# day68/more_trust.py   (python -m day68.more_trust)
import numpy as np

import day68.prior as prior
from day68.prior import launch_value

for bias_max in [10.0, 5.0, 2.0]:
    prior.BIAS_MAX = bias_max
    v = launch_value(prior.prior_draws())
    print(f"bias up to ${bias_max:>4.1f}: P(loss) {(v < 0).mean():.0%}, EVPI ${(np.maximum(v, 0).mean() - max(v.mean(), 0)) / 1e6:.2f}M")
```

```
bias up to $10.0: P(loss) 21%, EVPI $0.34M
bias up to $ 5.0: P(loss) 0%, EVPI $0.00M
bias up to $ 2.0: P(loss) 0%, EVPI $0.00M
```

If you're sure the bias is below $5, the effect is certainly above break-even, nothing a study could find would change the decision, and the information is worth zero. **The value of a test is set by the sensitivity analysis.** That's why the benchmarks of Days 61–62 aren't an academic add-on: they decide whether the next study is worth running.

</details>

---

## 68.5 Day 68 mini-project: what would actually have happened (65 min)

Check the formula by simulation (with the real, not-quite-normal prior), then run the holdout plan once against NorthMart's **real** effect from the simulator. Create `day68/simulate_holdout.py`:

```python
# day68/simulate_holdout.py   (python -m day68.simulate_holdout)
"""
Mini-project: check the formula by simulation, then ask what would ACTUALLY happen at NorthMart.
For each of 200,000 draws of the true effect from the (not quite normal) prior: run the month-1 holdout
(10,000 customers), update the belief, continue only if the posterior mean is above break-even, and record
the value actually earned. Then do the same once with the simulator's real effect.
"""
import numpy as np

from causalkit.coupons import coupons, truth
from day68.evsi import CUSTOMERS, FIXED_TEST_COST, RESIDUAL_SD, mu0, per_month, sd0
from day68.prior import BREAK_EVEN, prior_draws

N = 10_000
se = RESIDUAL_SD * np.sqrt(1 / N + 1 / (CUSTOMERS - N))


def year_with_holdout(theta, rng):
    estimate = theta + rng.normal(0, se, np.shape(theta))
    post_mean = (mu0 / sd0**2 + estimate / se**2) / (1 / sd0**2 + 1 / se**2)
    month1 = per_month * (theta - BREAK_EVEN) * (1 - N / CUSTOMERS)
    rest = np.where(post_mean > BREAK_EVEN, per_month * 11 * (theta - BREAK_EVEN), 0.0)
    return month1 + rest - FIXED_TEST_COST, post_mean > BREAK_EVEN


rng = np.random.default_rng(1)
theta = prior_draws()
value, cont = year_with_holdout(theta, rng)
blind = per_month * 12 * (theta - BREAK_EVEN)
print(f"Simulated over the prior: launch blind ${blind.mean() / 1e6:.2f}M; launch with holdout ${value.mean() / 1e6:.2f}M "
      f"(continues in {cont.mean():.0%} of draws)")
true_effect = truth(coupons())["ATE"]
v, c = year_with_holdout(np.array([true_effect]), np.random.default_rng(2))
print(f"NorthMart's real average effect: ${true_effect:.2f} (break-even ${BREAK_EVEN:.2f})")
print(f"   launch blind:        ${per_month * 12 * (true_effect - BREAK_EVEN) / 1e6:+.2f}M")
print(f"   launch with holdout: ${v[0] / 1e6:+.2f}M  ({'continued' if c[0] else 'stopped after month 1'})")
```

```bash
python -m day68.simulate_holdout
```

```
Simulated over the prior: launch blind $4.23M; launch with holdout $4.49M (continues in 80% of draws)
NorthMart's real average effect: $16.51 (break-even $16.67)
   launch blind:        $-0.23M
   launch with holdout: $-0.05M  (stopped after month 1)
```

The simulation matches the formula ($4.49M vs $4.51M). And in NorthMart's actual world, where the observational estimate was inflated by the browsing signal and the true effect is just below break-even, launching blind would have lost $230,000; launching with a 10,000-customer holdout loses $50,000 (one month, plus the test's cost) and stops.

That's the whole module in one line: **the estimate was wrong, the sensitivity analysis said it might be, the value-of-information analysis said how to find out cheaply, and the plan that followed lost a fifth as much.**

---

## 68.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 68: prior from sensitivity analysis, EVPI, EVSI, launch with a holdout"
git push
```

### Day 68 checklist

- [ ] I can build a prior that includes sensitivity to hidden confounding, not just sampling error
- [ ] I can compute EVPI and explain why it caps the value of any study
- [ ] I can compute EVSI for a test of a given size, with a normal approximation
- [ ] I can compare "test first" with "launch with a holdout and a stop rule"
- [ ] I can explain why the value of a test depends on the sensitivity analysis

---

# DAY 69 — Writing the Decision Memo

| Time | Block |
|---|---|
| 0:00 – 0:40 | 69.1 What a decision memo is for |
| 0:40 – 1:20 | 69.2 Numbers from code, not from memory |
| 1:20 – 1:35 | Break |
| 1:35 – 2:15 | 69.3 One chart |
| 2:15 – 3:10 | 69.4 Anatomy of a bad memo |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 69.5 Day 69 mini-project: rewrite it |
| 4:40 – 5:00 | 69.6 Save your work with Git |

---

## 69.1 What a decision memo is for (40 min)

You've written a decision memo at the end of every module. Today makes the format explicit, because it's where most analyses succeed or fail: a decision-maker reads one page, often only the first paragraph.

| Section | Contents | Common mistake |
|---|---|---|
| **Recommendation** | the action, in one or two sentences, with the expected value and the main risk | burying it under methods |
| **Why** | the key evidence in plain language, with the comparison that matters | p-values instead of money |
| **How sure we are** | ranges, probabilities, what would change the answer | false precision, or "more research needed" with no plan |
| **What we assumed** | the assumptions the answer depends on, ranked by importance | listing everything equally |
| **What happens next** | monitoring, stop rules, the next test | ending at the estimate |
| **Appendix** | methods, robustness checks, specification curve | putting this first |

Four rules that cover most of it:

1. **Lead with the decision**, not the analysis.
2. **Money and probabilities**, not coefficients and p-values. "86% chance it pays" beats "p < 0.001".
3. **State what would change your mind**, with a number. "If the coupon costs more than $6, coupon the top 35% instead."
4. **Every number traceable to code.** If the memo says $4.98M, a script printed $4.98M.

---

## 69.2 Numbers from code, not from memory (40 min)

Collect every number the memo uses in one script that writes them to a file, so the memo and the chart can't drift apart. Create `day69/memo_numbers.py` (about 20 seconds):

```python
# day69/memo_numbers.py   (python -m day69.memo_numbers)
"""
A memo's numbers should come from code, not from copying out of six notebooks. Collect every number the
coupon-programme memo needs in one place, with its source, and save them for the memo and the chart.
Run day66.cba first.
"""
import json

import numpy as np
from econml.dr import DRLearner

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import BREAK_EVEN, dr_scores, policy_value
from day64.uplift_models import experiment
from day65.policy_value import outcome_models
from day67.options import option_npvs

train, test = experiment()
dr = DRLearner(model_propensity=clf(), model_regression=reg(),
               model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
               min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
pol = (dr.effect(test[X_COLS]) > BREAK_EVEN).astype(int)
y, mu0, mu1 = outcome_models(test)
s1, s0 = dr_scores(y, test.coupon, 0.5, mu0, mu1)
rng = np.random.default_rng(0)
boot = [policy_value(pol[i], s1[i], s0[i]) for i in (rng.integers(0, len(test), len(test)) for _ in range(500))]
v = option_npvs(json.load(open("data/day66_inputs.json")))

numbers = {
    "experiment customers": 40_000,
    "share couponed by the targeted policy": round(float(pol.mean()), 3),
    "targeted: net $ per customer per campaign": round(policy_value(pol, s1, s0), 2),
    "targeted: 95% CI": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
    "everyone: net $ per customer per campaign": round(policy_value(np.ones(len(test)), s1, s0), 2),
    "programme NPV, targeted + stop rule ($k, expected)": round(float(v["targeted + stop rule"].mean())),
    "programme P(loss), targeted + stop rule": round(float((v["targeted + stop rule"] < 0).mean()), 3),
    "programme NPV, targeted without stop rule ($k, expected)": round(float(v["targeted"].mean())),
    "programme P(loss), targeted without stop rule": round(float((v["targeted"] < 0).mean()), 3),
    "programme NPV, coupons for everyone ($k, expected)": round(float(v["coupons for everyone"].mean())),
}
json.dump(numbers, open("data/day69_memo_numbers.json", "w"), indent=2)
for k, val in numbers.items():
    print(f"{k:<58} {val}")
```

```bash
python -m day69.memo_numbers
```

```
experiment customers                                       40000
share couponed by the targeted policy                      0.484
targeted: net $ per customer per campaign                  1.84
targeted: 95% CI                                           [1.63, 2.05]
everyone: net $ per customer per campaign                  0.18
programme NPV, targeted + stop rule ($k, expected)         4976
programme P(loss), targeted + stop rule                    0.063
programme NPV, targeted without stop rule ($k, expected)   3809
programme P(loss), targeted without stop rule              0.26
programme NPV, coupons for everyone ($k, expected)         -13007
```

In a real team, `data/day69_memo_numbers.json` is what gets reviewed alongside the memo. A reviewer who wants to check "$5.0M" opens one file, not six notebooks.

---

## 69.3 One chart (40 min)

A memo gets one chart. For a decision under uncertainty, show each option's **range** of outcomes and its **chance of a loss**, not a bar of the expected value alone. Create `day69/one_chart.py`:

```python
# day69/one_chart.py   (python -m day69.one_chart)
"""
One chart for the memo: for each option, the distribution of two-year NPV (median, 5th-95th percentile),
with the probability of a loss written next to it. Decision-makers read ranges and odds, not p-values.
Run day66.cba first.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from day67.options import option_npvs

v = option_npvs(json.load(open("data/day66_inputs.json")))
fig, ax = plt.subplots(figsize=(8, 3.6))
names = list(v)[::-1]
for i, name in enumerate(names):
    x = v[name] / 1000
    lo, med, hi = np.percentile(x, [5, 50, 95])
    ax.plot([lo, hi], [i, i], color="#94a3b8", lw=6, solid_capstyle="round")
    ax.plot(med, i, "o", color="#1e293b")
    ax.text(max(hi, 0) + 1.5, i, f"P(loss) {np.mean(x < 0):.0%}", va="center", fontsize=9)
ax.axvline(0, color="#dc2626", lw=1)
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("two-year NPV, $ million (dot: median; bar: 90% range)")
ax.set_xlim(-45, 30)
fig.tight_layout()
fig.savefig("figures/day69_memo_chart.png", dpi=150)
print("Saved figures/day69_memo_chart.png")
```

```bash
python -m day69.one_chart
```

```
Saved figures/day69_memo_chart.png
```

Open it: four horizontal bars (90% range of NPV), a dot for the median, a red line at zero, and P(loss) written beside each. A reader sees in five seconds that blanket coupons are a disaster, the plain targeted programme is a gamble, and the stop rule is the safe winner.

---

## 69.4 Anatomy of a bad memo (55 min)

Here's a memo, written in good faith, about the same programme. Read it as the CFO would, then list everything wrong with it before reading the critique.

> **Subject: Coupon programme analysis results**
>
> We analysed last quarter's coupon campaign using double/debiased machine learning (DML) with LightGBM nuisance learners and 5-fold cross-fitting. The interactive regression model gives an ATE of 24.61 (SE 0.64, p < 0.001), which is highly significant. We also fitted a causal forest, which found significant heterogeneity, with deal_share being the most important feature (importance 0.887). A sensitivity analysis gives a robustness value of 24.7%, so the result is robust to unobserved confounding.
>
> Given the strong and robust effect, we recommend rolling out coupons to all customers. The expected incremental revenue is $24.61 × 400,000 × 12 = $118M per year.
>
> Further research could explore heterogeneity in more detail.

<details>
<summary>The critique</summary>

1. **The recommendation is in the second paragraph**, after the methods. The CFO reads "DML with LightGBM nuisance learners" and stops.
2. **It reports revenue, not profit.** $118M of extra spend at a 30% margin, minus $5 × 400,000 × 12 = $24M of coupon cost, is a different number; and with the true effect it's negative.
3. **The robustness value is for the wrong question.** 24.7% is the strength needed to explain away *any* effect. The decision depends on beating $16.67, and that robustness value is 8.8%: fragile.
4. **"Robust to unobserved confounding" is not something a robustness value can establish.** It says how strong a confounder would need to be, not whether one exists. No benchmark was given.
5. **It ignores its own heterogeneity finding.** The forest found that effects depend on deal-seeking, and the memo recommends treating everyone anyway. Day 65: blanket coupons break even; targeting earns $1.6 per customer.
6. **No uncertainty about the money.** No range, no probability of a loss, no fatigue or pull-forward.
7. **No plan after launch.** No holdout, no stop rule, no monitoring.
8. **"Further research could explore..."** is not a next step. Which research, costing what, to decide what?

Each of these is a mistake an expert makes when writing for other experts. The fix isn't more methods; it's writing for the person who has to act.

</details>

---

## 69.5 Day 69 mini-project: rewrite it (75 min)

Write `docs/COUPON_PROGRAMME_MEMO.md`, one page, using only numbers from `data/day69_memo_numbers.json` and Days 61–68's outputs, and the chart from 69.3.

<details>
<summary>Model answer</summary>

> **Coupon programme: recommendation**
>
> **Recommendation.** Build the targeted coupon programme, sending coupons only to the half of customers our model predicts will respond most, and commit in advance to a stop rule: shut it down in any month it stops paying for itself. Expected value over two years: about **$5.0M**, with a **6%** chance of a net loss (at most the $150–300k build cost). Do **not** send coupons to all customers: that loses money in 94% of scenarios (expected −$13M).
>
> **Why.** In last spring's randomised test of 40,000 customers, coupons raised spending by about $17 per customer on average, almost exactly the $16.67 a $5 coupon needs to pay for itself at our 30% margin. Couponing everyone roughly breaks even. But the effect varies a lot: customers who buy many items on promotion spend $30–40 more with a coupon, while our biggest spenders spend slightly *less* (they'd have paid full price). Targeting the top half earns **$1.84 per customer per campaign** (95% CI $1.63–2.05) in held-out test data.
>
> **How sure we are.** The per-campaign value is well measured. The two-year value is not: it depends most on how quickly customers get used to coupons (fatigue) and how much of the extra spending is just purchases moved earlier (pull-forward), neither of which a one-month test measures. Under plausible ranges for both, the plain programme loses money in 26% of scenarios; the stop rule cuts that to 6%, because it shuts down a programme that stops working.
>
> **What we assumed.** Margin 25–35%; coupon cost $4–6 (if it's closer to $6, coupon the top 35% instead of 50%); effect fading by 0–4% a month; 0–30% of extra spend pulled forward.
>
> **What happens next.** Launch with a random holdout of 10,000 customers who get no coupons for six months. It costs about $12,000 a month in profit forgone (10,000 customers × $1.20, the first-month net value per customer), measures fatigue and pull-forward directly, and tells us within one month whether to continue. We report the holdout comparison monthly against the stop rule.
>
> *Appendix: methods (DR-learner, doubly robust policy evaluation, two-year NPV with Monte Carlo over the assumptions above), robustness checks (Days 61–63), and code references.*

</details>

---

## 69.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 69: memo numbers from code, decision chart, memo rewrite"
git push
```

### Day 69 checklist

- [ ] I lead with the decision, in money and probabilities
- [ ] Every number in my memo comes from a script that prints it
- [ ] I use one chart that shows ranges and the chance of a loss
- [ ] I state what would change the recommendation, with numbers
- [ ] My memo ends with a concrete plan: monitoring, stop rule, next test

---

# DAY 70 — Capstone: The Targeting Policy

| Time | Block |
|---|---|
| 0:00 – 0:30 | 70.1 The brief |
| 0:30 – 1:20 | 70.2 Step 1: what does last quarter say, and how robust is it? |
| 1:20 – 2:20 | 70.3 Step 2: learn policies, judge them on the pilot |
| 2:20 – 2:35 | Break |
| 2:35 – 3:15 | 70.4 Step 3: from campaign to programme |
| 3:15 – 4:25 | 70.5 Step 4: the decision memo |
| 4:25 – 5:00 | 70.6 Answer key, final commit and Module 7 retrospective |

---

## 70.1 The brief (30 min)

NorthMart's CMO wants to replace the current coupon targeting with something better, and asks three questions:

1. Should we keep sending coupons at all?
2. If so, to whom? Marketing's current rule is **"lapsing customers, or anyone who buys more than 25% of items on promotion"**.
3. Is a two-year programme worth building, and how do we limit the downside?

Economics as before: $5 per coupon, 30% margin, break-even uplift $16.67. You have two datasets:

- **last quarter**: 30,000 customers, coupons sent by the current targeting model (observational, and the targeting model used a browsing signal that wasn't stored);
- **a pilot**: 6,000 *different* customers, coupons by coin flip.

```bash
python data/make_targeting_scenario.py
```

```python
"""
Capstone data for Module 7: NorthMart's coupon targeting. Provided.

    python data/make_targeting_scenario.py

Writes data/scenario/coupons_last_quarter.csv (30,000 customers, coupons sent by the current targeting model),
data/scenario/coupons_pilot.csv (6,000 different customers, coupons by coin flip), and the true effects
to data/truth/coupon_effects.csv.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.coupons import coupons, public  # noqa: E402

root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
(root / "truth").mkdir(exist_ok=True)
obs = coupons(n=30_000, seed=70, hidden=0.3)
pilot = coupons(n=6_000, seed=71, randomized=True).with_columns(pl.col("customer_id").str.replace("C", "P"))
public(obs).write_csv(root / "scenario" / "coupons_last_quarter.csv")
public(pilot).write_csv(root / "scenario" / "coupons_pilot.csv")
pl.concat([obs.select("customer_id", "_tau"), pilot.select("customer_id", "_tau")]).rename(
    {"_tau": "true_effect"}).write_csv(root / "truth" / "coupon_effects.csv")
print("Wrote data/scenario/coupons_last_quarter.csv (30,000) and data/scenario/coupons_pilot.csv (6,000)")
```

**Don't open `data/truth/` until your memo is committed.**

---

## 70.2 Step 1: what does last quarter say, and how robust is it? (50 min)

Create `day70/sensitivity.py`:

```python
# day70/sensitivity.py   (python -m day70.sensitivity)
"""Capstone step 1: what does last quarter's data say about coupons, and how robust is it?"""
import doubleml as dml
import numpy as np
import pandas as pd
import polars as pl

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import BREAK_EVEN

np.random.seed(70)  # DoubleML draws its cross-fitting folds from numpy's global RNG
obs = pd.read_csv("data/scenario/coupons_last_quarter.csv")
pilot = pd.read_csv("data/scenario/coupons_pilot.csv")
irm = dml.DoubleMLIRM(dml.DoubleMLData(obs[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS),
                      ml_g=reg(), ml_m=clf(), n_folds=5, trimming_threshold=0.02).fit()
print(f"Last quarter (observational): ATE {irm.coef[0]:.2f}, 95% CI [{irm.confint().iloc[0, 0]:.2f}, "
      f"{irm.confint().iloc[0, 1]:.2f}]")
irm.sensitivity_analysis(null_hypothesis=BREAK_EVEN)
print(f"   robustness value for 'above break-even (${BREAK_EVEN:.2f})': {irm.sensitivity_params['rv'][0]:.1%}")
rows = []
for f in ["deal_share", "app_user", "pre_spend"]:
    b = irm.sensitivity_benchmark(benchmarking_set=[f]).iloc[0]
    rows.append({"confounder as strong as": f, "cf_y": round(b.cf_y, 3), "cf_d": round(b.cf_d, 3),
                 "moves the estimate by": round(b.delta_theta, 2)})
print(pl.DataFrame(rows))
diff = pilot.spend[pilot.coupon == 1].mean() - pilot.spend[pilot.coupon == 0].mean()
se = np.sqrt(pilot.spend[pilot.coupon == 1].var() / (pilot.coupon == 1).sum()
             + pilot.spend[pilot.coupon == 0].var() / (pilot.coupon == 0).sum())
print(f"Pilot (randomised): difference in means {diff:.2f}, 95% CI [{diff - 1.96 * se:.2f}, {diff + 1.96 * se:.2f}]")
```

```bash
python -m day70.sensitivity
```

```
Last quarter (observational): ATE 18.98, 95% CI [18.01, 19.95]
   robustness value for 'above break-even ($16.67)': 2.7%
│ confounder as strong as ┆ cf_y  ┆ cf_d  ┆ moves the estimate by │
│ deal_share              ┆ 0.045 ┆ 0.127 ┆ 2.92                  │
│ app_user                ┆ 0.185 ┆ 0.034 ┆ 5.34                  │
│ pre_spend               ┆ 1.0   ┆ 0.094 ┆ -20.54                │
Pilot (randomised): difference in means 16.13, 95% CI [13.00, 19.26]
```

Last quarter's data say coupons pay ($18.98, CI entirely above $16.67), but the robustness value for that conclusion is 2.7%: a hidden confounder far weaker than deal-seeking or app use could overturn it. The pilot, which needs no such assumption, says $16.13, with a CI from $13 to $19 that straddles break-even.

**Answer to question 1**: blanket coupons are, at best, break-even. Don't decide on last quarter's number.

---

## 70.3 Step 2: learn policies, judge them on the pilot (60 min)

Learn targeting policies from last quarter's 30,000 customers (more data, even if confounded), then judge every policy on the pilot, where coupons were random and the models never looked. Create `day70/policies.py` (about 20 seconds):

```python
# day70/policies.py   (python -m day70.policies)
"""
Capstone step 2: learn targeting policies from last quarter's data, then judge them on the pilot, which the
models never saw and where coupons were random. Candidates:
  nobody | everyone | marketing's current rule (lapsing customers, or deal_share above 25%) |
  DR-learner: effect > $16.67 | policy tree, depth 2
Also check the DR-learner's calibration on the pilot (Day 57).
"""
import numpy as np
import pandas as pd
import polars as pl
from econml.dr import DRLearner
from econml.policy import DRPolicyTree
from econml.validate import DRTester
from sklearn.model_selection import KFold

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import BREAK_EVEN, dr_scores, net, policy_value, tree_rules


def fit_policies(obs):
    dr = DRLearner(model_propensity=clf(), model_regression=reg(),
                   model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                   min_propensity=0.02, random_state=0).fit(obs.spend, obs.coupon, X=obs[X_COLS])
    tree = DRPolicyTree(model_regression=reg(), model_propensity=clf(), max_depth=2, min_samples_leaf=500,
                        random_state=0).fit(net(obs.spend, obs.coupon), obs.coupon, X=obs[X_COLS])
    return dr, tree


def decisions(dr, tree, df):
    return {
        "nobody": np.zeros(len(df)),
        "everyone": np.ones(len(df)),
        "marketing's rule": ((df.lapsing == 1) | (df.deal_share > 0.25)).astype(int).to_numpy(),
        "DR-learner > $16.67": (dr.effect(df[X_COLS]) > BREAK_EVEN).astype(int),
        "policy tree": tree.predict(df[X_COLS]),
    }


def pilot_scores(pilot):
    """Cross-fitted doubly robust scores of the net outcome on the pilot (coupon probability 0.5)."""
    y = net(pilot.spend, pilot.coupon)
    mu0, mu1 = np.zeros(len(pilot)), np.zeros(len(pilot))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(pilot):
        a = pilot.iloc[tr]
        for t, mu in [(0, mu0), (1, mu1)]:
            mu[te] = reg().fit(a[a.coupon == t][X_COLS], y[tr][a.coupon.to_numpy() == t]).predict(pilot.iloc[te][X_COLS])
    return dr_scores(y, pilot.coupon, 0.5, mu0, mu1)


if __name__ == "__main__":
    obs = pd.read_csv("data/scenario/coupons_last_quarter.csv")
    pilot = pd.read_csv("data/scenario/coupons_pilot.csv")
    dr, tree = fit_policies(obs)
    print("Policy tree rules:")
    for r in tree_rules(tree.export_graphviz(feature_names=X_COLS)):
        print("   " + r)
    s1, s0 = pilot_scores(pilot)
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(pilot), len(pilot)) for _ in range(1000)]
    pols = decisions(dr, tree, pilot)
    rows = []
    for name, pol in pols.items():
        b = [policy_value(pol[i], s1[i], s0[i]) for i in boot]
        rows.append({"policy": name, "share couponed": f"{pol.mean():.0%}",
                     "net $ per customer (pilot)": round(policy_value(pol, s1, s0), 2),
                     "95% CI": f"[{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"})
    with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
        print(pl.DataFrame(rows))
    gain = pols["DR-learner > $16.67"] - pols["marketing's rule"]
    b = [policy_value(gain[i], s1[i], s0[i]) for i in boot]
    print(f"DR-learner minus marketing's rule: {policy_value(gain, s1, s0):+.2f} per customer, "
          f"95% CI [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")
    pd.DataFrame({k: v for k, v in pols.items()}).assign(customer_id=pilot.customer_id).to_csv(
        "data/day70_pilot_decisions.csv", index=False)

    tester = DRTester(model_regression=reg(), model_propensity=clf(), cate=dr)
    tester.fit_nuisance(pilot[X_COLS].to_numpy(), pilot.coupon.to_numpy(), pilot.spend.to_numpy(),
                        obs[X_COLS].to_numpy(), obs.coupon.to_numpy(), obs.spend.to_numpy())
    s = tester.evaluate_all(pilot[X_COLS].to_numpy(), obs[X_COLS].to_numpy(), n_bootstrap=200).summary().iloc[0]
    print(f"DR-learner on the pilot: BLP slope {s.blp_est:.2f} (SE {s.blp_se:.2f}), AUTOC {s.autoc_est:.1f}, "
          f"calibration R2 {s.cal_r_squared:.2f}")
```

```bash
python -m day70.policies
```

```
Policy tree rules:
   deal_share <= 0.264 and basket <= 21.835  ->  no coupon  (558 training customers)
   deal_share <= 0.264 and basket > 21.835  ->  no coupon  (7099 training customers)
   deal_share > 0.264 and pre_spend <= 597.325  ->  coupon  (6657 training customers)
   deal_share > 0.264 and pre_spend > 597.325  ->  coupon  (686 training customers)
│ policy              ┆ share couponed ┆ net $ per customer (pilot) ┆ 95% CI        │
│ nobody              ┆ 0%             ┆ 0.0                        ┆ [0.00, 0.00]  │
│ everyone            ┆ 100%           ┆ -0.33                      ┆ [-0.92, 0.24] │
│ marketing's rule    ┆ 58%            ┆ 0.86                       ┆ [0.42, 1.29]  │
│ DR-learner > $16.67 ┆ 56%            ┆ 1.28                       ┆ [0.85, 1.70]  │
│ policy tree         ┆ 50%            ┆ 1.05                       ┆ [0.65, 1.46]  │
DR-learner minus marketing's rule: +0.42 per customer, 95% CI [+0.15, +0.70]
DR-learner on the pilot: BLP slope 0.96 (SE 0.12), AUTOC 9.2, calibration R2 0.65
```

- **Every targeted policy beats blanket coupons**, which lose $0.33 per customer on the pilot.
- **The DR-learner beats marketing's rule** by $0.42 per customer (95% CI +$0.15 to +$0.70): the comparison that matters, and it's clearly positive.
- **The DR-learner is well calibrated on the pilot** (BLP slope 0.96), even though it was trained on confounded data. The hidden browsing signal biases the *level* of effects (Step 1) more than their *ranking* across customers. Targeting depends on the ranking.
- The policy tree's rule ("coupon if deal_share > 26%") is marketing's rule without the lapsing clause, and earns $0.19 more than marketing's rule. If the CRM can't use a model score, it's the simple upgrade.

**Answer to question 2**: target with the DR-learner, coupon about 56% of customers.

---

## 70.4 Step 3: from campaign to programme (40 min)

Feed the pilot's measurements into Days 66–67's programme model. Create `day70/programme.py`:

```python
# day70/programme.py   (python -m day70.programme)
"""
Capstone step 3: from one campaign to a two-year programme (Days 66-68). Inputs measured on the pilot:
each policy's extra spend per customer in the base, and the share it coupons. Everything else as on Day 66.
Run day70.policies first.
"""
import numpy as np
import pandas as pd
import polars as pl

from day66.cba import BASE, npv
from day67.options import draw_inputs
from day70.policies import pilot_scores

pilot = pd.read_csv("data/scenario/coupons_pilot.csv")
dec = pd.read_csv("data/day70_pilot_decisions.csv")
s1, s0 = pilot_scores(pilot)
extra_spend = ((s1 - s0) + 5.0) / 0.30                      # net score difference back to $ of extra spend
rows = []
for name in ["everyone", "marketing's rule", "DR-learner > $16.67"]:
    pol = dec[name].to_numpy()
    base = BASE | {"uplift": float(np.mean(pol * extra_spend)), "share": float(pol.mean())}
    for stop in [False, True]:
        x = np.array([npv(a, stop=stop)[0] for a in draw_inputs(base | {"uplift_everyone": base["uplift"]}, n=5000)]) / 1000
        rows.append({"policy": name, "stop rule": "yes" if stop else "no",
                     "expected NPV ($k)": round(x.mean()), "P(loss)": f"{(x < 0).mean():.0%}",
                     "5th percentile ($k)": round(np.percentile(x, 5))})
with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day70.programme
```

```
│ policy              ┆ stop rule ┆ expected NPV ($k) ┆ P(loss) ┆ 5th percentile ($k) │
│ everyone            ┆ no        ┆ -16086            ┆ 98%     ┆ -29187              │
│ everyone            ┆ yes       ┆ -110              ┆ 94%     ┆ -289                │
│ marketing's rule    ┆ no        ┆ -3447             ┆ 74%     ┆ -11989              │
│ marketing's rule    ┆ yes       ┆ 1055              ┆ 50%     ┆ -274                │
│ DR-learner > $16.67 ┆ no        ┆ -556              ┆ 56%     ┆ -9321               │
│ DR-learner > $16.67 ┆ yes       ┆ 2322              ┆ 28%     ┆ -249                │
```

With the pilot's more modest numbers (and pull-forward and fatigue on top), the programme is a close call:

- Without a stop rule, **even the best policy loses money in expectation** (−$0.6M).
- With a stop rule, the DR-learner programme is worth **$2.3M** in expectation, with a 28% chance of a small loss (at most about the build cost).
- Marketing's rule with a stop rule is worth less than half as much ($1.1M) and loses in half the scenarios.

**Answer to question 3**: build it only with a stop rule, and launch with a holdout (Day 68) so the stop rule has data to act on.

---

## 70.5 Step 4: the decision memo (70 min)

Write `docs/TARGETING_POLICY_DECISION.md` using Day 69's structure: recommendation first; money and probabilities; one chart; what would change the decision; next steps with a holdout and stop rule; methods and robustness in an appendix.

<details>
<summary>Model answer for the recommendation and next steps</summary>

> **Recommendation.** Stop sending coupons to everyone and stop using the current rule of thumb. Target coupons with the new model, which selects about 56% of customers: on a randomised pilot it earned $0.42 more per customer per campaign than the current rule (95% CI $0.15–0.70), and $1.61 more than couponing everyone. Build it as a programme only with a pre-agreed stop rule: expected two-year value $2.3M, with a 28% chance of a small loss (at most about the $150–300k build cost). Without a stop rule, the programme is expected to lose money.
>
> **Why we didn't trust last quarter's numbers.** Last quarter's data say coupons raise spending by $19, comfortably profitable. But that conclusion would be overturned by a hidden factor much weaker than ones we can measure, and the targeting model used browsing data we never stored. The randomised pilot says $16, right at break-even.
>
> **Next steps.** Launch with a random holdout of 10,000 customers for six months. Compare monthly; stop the programme in the first month its measured net value turns negative. Re-train the model each quarter on holdout and pilot data, not only on targeted campaigns, so its levels stay honest.

</details>

---

## 70.6 Answer key, final commit and Module 7 retrospective (35 min)

After the memo is committed, compare with the truth. Create `day70/answer_key.py`:

```python
# day70/answer_key.py   (python -m day70.answer_key)
"""Instructor check, after the memo: each policy's TRUE net value per customer on the pilot customers."""
import pandas as pd
import polars as pl

from causalkit.policy import COST, MARGIN

truth = pd.read_csv("data/truth/coupon_effects.csv").set_index("customer_id").true_effect
dec = pd.read_csv("data/day70_pilot_decisions.csv").set_index("customer_id")
gain = MARGIN * truth.loc[dec.index] - COST
rows = [{"policy": c, "share couponed": f"{dec[c].mean():.0%}", "true net $ per customer": round((dec[c] * gain).mean(), 2)}
        for c in dec.columns]
rows.append({"policy": "oracle", "share couponed": f"{(gain > 0).mean():.0%}",
             "true net $ per customer": round(gain.clip(lower=0).mean(), 2)})
print(pl.DataFrame(rows))
print(f"True average effect, pilot customers: ${truth.loc[dec.index].mean():.2f}; "
      f"last-quarter customers: ${truth.drop(dec.index).mean():.2f} (break-even ${COST / MARGIN:.2f})")
```

```bash
python -m day70.answer_key
```

```
│ policy              ┆ share couponed ┆ true net $ per customer │
│ nobody              ┆ 0%             ┆ 0.0                     │
│ everyone            ┆ 100%           ┆ -0.06                   │
│ marketing's rule    ┆ 58%            ┆ 1.05                    │
│ DR-learner > $16.67 ┆ 56%            ┆ 1.58                    │
│ policy tree         ┆ 50%            ┆ 1.36                    │
│ oracle              ┆ 47%            ┆ 1.66                    │
True average effect, pilot customers: $16.45; last-quarter customers: $16.16 (break-even $16.67)
```

The pilot ranked the policies exactly as the truth does. The DR-learner earns $1.58 per customer per campaign in truth, 95% of the oracle's $1.66, and $0.53 more than marketing's rule. Couponing everyone loses a little, as the pilot said. The true average effect ($16.45 and $16.16) is below break-even, so last quarter's $18.98 would have justified blanket coupons that lose money. Over 400,000 customers a month, the switch from marketing's rule is worth about $210,000 a month.

```bash
git add .
git commit -m "Day 70: targeting policy capstone: sensitivity, policies on the pilot, programme, memo"
git push
```

**What you learned in Module 7.**

| Day | Idea | The simulator lesson |
|---|---|---|
| 61 | DML sensitivity | "Coupons pay" had RV 8.8%; the real hidden confounder was stronger, and it didn't pay |
| 62 | Robustness values, E-values | A confounder 3× email engagement left the loyalty effect above break-even; so did the real one, by $4 |
| 63 | Refutations, specification curves | All refutations passed; one choice (lasso) explained the whole spread of 16 specifications |
| 64 | Uplift | Qini areas ranked models like the true error; the bottom decile were sleeping dogs |
| 65 | Policy learning | Blanket coupons broke even; targeting earned $1.6 per customer, within 3% of the oracle |
| 66 | Cost-benefit | Two unmeasured assumptions dominated the NPV; expected NPV was half the base case |
| 67 | Decisions under uncertainty | A stop rule won on every criterion, worth $1.2M and cutting P(loss) from 26% to 6% |
| 68 | Value of information | Testing first was worse than launching blind; launching with a holdout beat both |
| 69 | The memo | Lead with money and probabilities; every number from code |
| 70 | Capstone | The confounded data's ranking was right and its level wrong; the pilot caught both |

**Coming next: Module 8, Decision Intelligence Delivery.** The analyses are done. Module 8 makes them last: packaging the estimators as a tested library, scheduled pipelines that re-estimate effects as data arrive, monitoring for drift, dashboards for decision-makers, and the final course capstone.

### Day 70 checklist

- [ ] I checked whether the observational conclusion survives plausible hidden confounding
- [ ] I learned policies on one dataset and judged them on a randomised one
- [ ] I compared policies with paired confidence intervals, not just separate ones
- [ ] I valued the programme with a stop rule, and recommended a holdout
- [ ] I wrote a memo a CMO can act on, and checked it against the truth

