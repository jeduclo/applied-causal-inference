# Module 7 · Sensitivity, Robustness & Decisions
## Days 64–66: Uplift Modelling, Policy Learning and Cost-Benefit Analysis

**Goal by end of Day 66:** Students can build and evaluate uplift models on a randomised campaign with Qini and uplift curves written from scratch, learn a cost-aware targeting policy and value it with doubly robust scores, and scale a campaign result into a programme-level cost-benefit analysis with NPV, a tornado chart and Monte Carlo uncertainty.

---

# DAY 64 — Uplift Modelling

| Time | Block |
|---|---|
| 0:00 – 0:40 | 64.1 Uplift is a CATE in marketing's language |
| 0:40 – 1:20 | 64.2 Uplift metrics, from scratch |
| 1:20 – 1:35 | Break |
| 1:35 – 2:30 | 64.3 Four uplift models on a 40,000-customer test |
| 2:30 – 3:10 | 64.4 Qini and uplift curves |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 64.5 Day 64 mini-project: the decile table and the sleeping dogs |
| 4:40 – 5:00 | 64.6 Save your work with Git |

---

## 64.1 Uplift is a CATE in marketing's language (40 min)

Marketing teams have estimated heterogeneous effects for decades, under the name **uplift modelling**: who buys *because* of the campaign? It's Day 55's CATE, usually estimated on randomised campaigns. Marketing sorts customers into four groups:

| | buys if contacted | doesn't buy if contacted |
|---|---|---|
| **buys if not contacted** | sure things (coupon wasted) | **sleeping dogs** (contact hurts) |
| **doesn't buy if not contacted** | **persuadables** (target these) | lost causes |

A response model targets everyone likely to buy, including sure things; an uplift model targets persuadables and avoids sleeping dogs.

The data: last spring's coupon test, 40,000 customers, coupons by coin flip. Half to train, half to evaluate.

---

## 64.2 Uplift metrics, from scratch (40 min)

**Concept.** Sort customers by predicted uplift, highest first. For the top k customers:

- **uplift curve**: the measured uplift among them (difference in means: it's an experiment) × k, the extra outcome you'd get by treating exactly those k;
- **Qini curve**: treated customers' total outcome minus control customers' total scaled to the same number of treated customers (Radcliffe, 2007). Same idea, slightly different scaling.

A good model's curve rises steeply and then flattens or falls (the last customers added are sleeping dogs). Random targeting is the straight line from zero to the overall effect. The **area between the curve and that line** scores the model.

The usual Python package for this, scikit-uplift, no longer runs with current numpy and scikit-learn. It's 40 lines; write them. Create `causalkit/uplift.py`:

```python
"""Uplift metrics for randomised campaigns (Day 64), written out because the usual package no longer runs.

Sort customers by a model's score (highest predicted uplift first). For the top k customers:
    uplift curve:  (mean outcome of treated - mean outcome of control among the top k) x k
                   = extra outcome if we had treated exactly those k customers
    Qini curve:    sum of treated outcomes - sum of control outcomes x (treated count / control count),
                   the same idea, scaled by the number of treated customers among the top k
Area under each curve, minus the area for random targeting (a straight line), is the model's score.
"""
import numpy as np
import polars as pl


def _sorted(y, t, score):
    order = np.argsort(-np.asarray(score), kind="stable")
    return np.asarray(y, float)[order], np.asarray(t, int)[order]


def uplift_curve(y, t, score, points=50):
    """Returns (share targeted, extra outcome per customer in the whole population)."""
    y, t = _sorted(y, t, score)
    n = len(y)
    ks = np.unique(np.linspace(n // points, n, points).astype(int))
    out = []
    for k in ks:
        yt, tt = y[:k], t[:k]
        if tt.sum() == 0 or (1 - tt).sum() == 0:
            out.append(0.0)
            continue
        out.append((yt[tt == 1].mean() - yt[tt == 0].mean()) * k / n)
    return ks / n, np.array(out)


def qini_curve(y, t, score, points=50):
    """Returns (share targeted, Qini value per customer in the whole population)."""
    y, t = _sorted(y, t, score)
    n = len(y)
    ks = np.unique(np.linspace(n // points, n, points).astype(int))
    cy_t, cy_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    cn_t, cn_c = np.cumsum(t), np.cumsum(1 - t)
    q = cy_t[ks - 1] - cy_c[ks - 1] * cn_t[ks - 1] / np.maximum(cn_c[ks - 1], 1)
    return ks / n, q / n


def area_above_random(x, curve):
    """Area between a curve and the straight line from (0, 0) to its end point."""
    return float(np.trapezoid(curve - x * curve[-1], x))


def uplift_at_k(y, t, score, k=0.3):
    """Measured uplift (difference in means) among the top k share of customers."""
    y, t = _sorted(y, t, score)
    top = int(len(y) * k)
    return float(y[:top][t[:top] == 1].mean() - y[:top][t[:top] == 0].mean())


def by_decile(y, t, score):
    """Measured uplift in each decile of the score, with standard errors."""
    y, t = _sorted(y, t, score)
    rows = []
    for d, idx in enumerate(np.array_split(np.arange(len(y)), 10), start=1):
        yd, td = y[idx], t[idx]
        a, b = yd[td == 1], yd[td == 0]
        rows.append({"decile": d, "customers": len(idx), "uplift": round(a.mean() - b.mean(), 1),
                     "SE": round(float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))), 1)})
    return pl.DataFrame(rows)
```

---

## 64.3 Four uplift models on a 40,000-customer test (55 min)

Three classic uplift models and the DR-learner of Day 55. The **transformed outcome** is the oldest trick in uplift modelling: with treatment probability p, Y* = Y·(T − p)/(p(1 − p)) has expectation equal to the uplift, so any regression of Y* on X estimates it. Create `day64/uplift_models.py`:

```python
# day64/uplift_models.py   (python -m day64.uplift_models)
"""
Uplift modelling is CATE estimation for randomised campaigns, in marketing's language. NorthMart ran a
40,000-customer coupon test last spring (coupons by coin flip). Train uplift models on half, score on the
other half. Three standard uplift models, plus Day 55's DR-learner:
  solo model (S-learner), two models (T-learner),
  transformed outcome: with p = 0.5, Y* = 2Y if couponed, -2Y if not, has expectation tau(x): regress it on X.
"""
import numpy as np
import polars as pl
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg
from causalkit.uplift import area_above_random, qini_curve, uplift_at_k


def experiment():
    return train_test_split(coupons(n=40_000, seed=640, randomized=True).to_pandas(), test_size=0.5, random_state=0)


def fit_uplift_models(train):
    p = train.coupon.mean()
    y_star = train.spend * (train.coupon - p) / (p * (1 - p))
    return {
        "solo model (S)": SLearner(overall_model=reg()).fit(train.spend, train.coupon, X=train[X_COLS]),
        "two models (T)": TLearner(models=reg()).fit(train.spend, train.coupon, X=train[X_COLS]),
        "transformed outcome": reg(n_estimators=100, num_leaves=4, min_child_samples=300).fit(train[X_COLS], y_star),
        "DR-learner": DRLearner(model_propensity=clf(), model_regression=reg(),
                                model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                                min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS]),
    }


def score(model, X):
    return model.effect(X) if hasattr(model, "effect") else model.predict(X)


if __name__ == "__main__":
    train, test = experiment()
    models = fit_uplift_models(train)
    rows = []
    for name, m in models.items():
        s = score(m, test[X_COLS])
        x, q = qini_curve(test.spend, test.coupon, s)
        top = np.argsort(-s)[: int(0.3 * len(s))]
        rows.append({"model": name, "Qini area": round(area_above_random(x, q), 2),
                     "uplift in top 30%": round(uplift_at_k(test.spend, test.coupon, s, 0.3), 1),
                     "true uplift in top 30%": round(test._tau.to_numpy()[top].mean(), 1),
                     "true PEHE": round(float(np.sqrt(np.mean((s - test._tau) ** 2))), 1)})
    with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
        print(pl.DataFrame(rows))
    print(f"Average uplift (everyone): {test.spend[test.coupon == 1].mean() - test.spend[test.coupon == 0].mean():.1f}")
```

```bash
python -m day64.uplift_models
```

```
│ model               ┆ Qini area ┆ uplift in top 30% ┆ true uplift in top 30% ┆ true PEHE │
│ solo model (S)      ┆ 2.09      ┆ 35.3              ┆ 32.1                   ┆ 5.4       │
│ two models (T)      ┆ 1.87      ┆ 32.3              ┆ 30.2                   ┆ 9.8       │
│ transformed outcome ┆ 1.29      ┆ 29.6              ┆ 27.4                   ┆ 14.7      │
│ DR-learner          ┆ 2.17      ┆ 35.6              ┆ 32.6                   ┆ 4.0       │
Average uplift (everyone): 17.5
```

- With 20,000 training customers from an experiment, the DR-learner and the solo model are best; the transformed outcome is worst (Y* is extremely noisy: each customer's value is ±2 × spend).
- The Qini area ranks the models the same way as the true PEHE. That's the point: **the Qini area is computable without the truth**.
- "Uplift in top 30%" is measured on the held-out half and is a few dollars above the truth: the top 30% is chosen by the model, and the measurement adds noise. With one model it's unbiased; when you compare many models and report the best, it's optimistic.

---

## 64.4 Qini and uplift curves (40 min)

Create `day64/qini_plot.py`:

```python
# day64/qini_plot.py   (python -m day64.qini_plot)
"""Qini and uplift curves for the four models, plus random targeting and the oracle (true uplift)."""
import matplotlib.pyplot as plt

from causalkit.coupons import X_COLS
from causalkit.uplift import qini_curve, uplift_curve
from day64.uplift_models import experiment, fit_uplift_models, score

train, test = experiment()
models = fit_uplift_models(train)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
scores = {name: score(m, test[X_COLS]) for name, m in models.items()} | {"oracle": test._tau.to_numpy()}
for name, s in scores.items():
    for ax, curve in zip(axes, [qini_curve, uplift_curve]):
        x, c = curve(test.spend, test.coupon, s)
        ax.plot(x, c, label=name, ls="--" if name == "oracle" else "-")
for ax, title in zip(axes, ["Qini curve", "Uplift curve"]):
    x, c = qini_curve(test.spend, test.coupon, scores["oracle"]) if title == "Qini curve" else \
        uplift_curve(test.spend, test.coupon, scores["oracle"])
    ax.plot([0, 1], [0, c[-1]], color="grey", lw=0.8, label="random")
    ax.set_title(title)
    ax.set_xlabel("share of customers targeted (highest score first)")
axes[0].set_ylabel("extra $ per customer in the base")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/day64_qini.png", dpi=120)
print("Saved figures/day64_qini.png")
```

```bash
python -m day64.qini_plot
```

```
Saved figures/day64_qini.png
```

Open the figure. The oracle (true uplift) rises fastest, then **falls** at the end: the last customers it adds have negative uplift. The DR-learner and solo model follow it closely. The uplift curve's right-hand end is the same for every model (treating everyone gives the average effect, whatever the ranking).

**Try it.** Rerun the four models on a 4,000-customer experiment instead of 40,000. Which model's Qini area falls most? Create `day64/small_experiment.py`.

<details>
<summary>Solution</summary>

```python
# day64/small_experiment.py   (python -m day64.small_experiment)
import numpy as np
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons
from causalkit.uplift import area_above_random, qini_curve
from day64.uplift_models import fit_uplift_models, score

for n in [4_000, 40_000]:
    train, test = train_test_split(coupons(n=n, seed=640, randomized=True).to_pandas(), test_size=0.5, random_state=0)
    out = []
    for name, m in fit_uplift_models(train).items():
        x, q = qini_curve(test.spend, test.coupon, score(m, test[X_COLS]))
        out.append(f"{name.split(' (')[0]} {area_above_random(x, q):.2f}")
    print(f"n = {n:>6,}: Qini area  " + " | ".join(out))
```

```
n =  4,000: Qini area  solo model 1.28 | two models 0.62 | transformed outcome 0.00 | DR-learner 1.03
n = 40,000: Qini area  solo model 2.09 | two models 1.87 | transformed outcome 1.29 | DR-learner 2.17
```

With 2,000 training customers, the transformed-outcome model learns nothing at all (Qini 0: it can't find a single split worth making in such noisy data), and the solo model wins. Same lesson as Day 55: with little data, the most regularised model is best; flexible meta-learners need data. Size your test for the model you want to fit.

</details>

---

## 64.5 Day 64 mini-project: the decile table and the sleeping dogs (75 min)

The table marketing reads: uplift measured in each decile of the model's score. Create `day64/deciles.py`:

```python
# day64/deciles.py   (python -m day64.deciles)
"""
Mini-project: the table marketing actually reads. Uplift measured in each decile of the DR-learner's score
(decile 1 = highest predicted uplift), with standard errors. Look at the bottom: customers for whom a coupon
does nothing, or reduces spending ('sleeping dogs': loyal customers who'd have paid full price).
"""
import numpy as np
import polars as pl

from causalkit.coupons import X_COLS
from causalkit.uplift import by_decile
from day64.uplift_models import experiment, fit_uplift_models, score

train, test = experiment()
s = score(fit_uplift_models(train)["DR-learner"], test[X_COLS])
table = by_decile(test.spend, test.coupon, s)
true = test.assign(score=s).sort_values("score", ascending=False)._tau.to_numpy()
table = table.with_columns(true_uplift=pl.Series([round(c.mean(), 1) for c in np.array_split(true, 10)]))
print(table)
print(f"Share of customers with a negative true uplift: {(test._tau < 0).mean():.1%}; "
      f"in the bottom two deciles: {(true[-len(true) // 5:] < 0).mean():.1%}")
```

```bash
python -m day64.deciles
```

```
│ decile ┆ customers ┆ uplift ┆ SE  ┆ true_uplift │
│ 1      ┆ 2000      ┆ 41.3   ┆ 2.7 ┆ 40.6        │
│ 2      ┆ 2000      ┆ 37.7   ┆ 2.5 ┆ 31.4        │
│ 3      ┆ 2000      ┆ 26.8   ┆ 2.5 ┆ 25.7        │
│ 4      ┆ 2000      ┆ 22.6   ┆ 2.7 ┆ 21.7        │
│ 5      ┆ 2000      ┆ 19.0   ┆ 2.9 ┆ 17.6        │
│ 6      ┆ 2000      ┆ 15.5   ┆ 2.8 ┆ 13.9        │
│ 7      ┆ 2000      ┆ 10.1   ┆ 2.7 ┆ 10.3        │
│ 8      ┆ 2000      ┆ 5.9    ┆ 2.8 ┆ 7.0         │
│ 9      ┆ 2000      ┆ -1.9   ┆ 2.9 ┆ 3.0         │
│ 10     ┆ 2000      ┆ -5.0   ┆ 3.2 ┆ -4.8        │
Share of customers with a negative true uplift: 10.5%; in the bottom two deciles: 50.0%
```

A clean staircase from $41 to −$5. The bottom decile are the **sleeping dogs**: big spenders who'd have bought anyway, and for whom a coupon just lowers what they pay. Half of the bottom fifth have a negative true uplift. Couponing them costs money twice: the coupon, and the margin given away.

With a $5 coupon and 30% margin, a coupon pays above $16.67 of uplift: deciles 1–5 clearly, decile 6 roughly at break-even, deciles 7–10 not. Tomorrow turns this reasoning into a policy.

---

## 64.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 64: uplift metrics from scratch, four uplift models, Qini curves, deciles"
git push
```

### Day 64 checklist

- [ ] I can explain uplift modelling and the four customer types
- [ ] I can compute uplift and Qini curves, areas and uplift-at-k by hand
- [ ] I can fit S, T, transformed-outcome and DR uplift models and compare them by Qini area
- [ ] I can produce and read a decile table, and spot sleeping dogs

---

# DAY 65 — Policy Learning

| Time | Block |
|---|---|
| 0:00 – 0:40 | 65.1 From effects to decisions |
| 0:40 – 1:30 | 65.2 Policy trees |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 65.3 Which policy makes the most money? |
| 2:40 – 3:20 | 65.4 How deep should the tree be? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 65.5 Day 65 mini-project: when coupons cost more |
| 4:40 – 5:00 | 65.6 Save your work with Git |

---

## 65.1 From effects to decisions (40 min)

A **policy** says, for each customer, coupon or no coupon. Its value is what it earns: each couponed customer adds margin × (their effect) − cost. With a 30% margin and a $5 coupon:

```
value of couponing customer i = 0.30 × τᵢ − 5      →  coupon if τᵢ > $16.67
```

Two ways to get a policy:

1. **Plug-in:** estimate τ(x) (Days 55–57, 64), coupon if τ̂(x) > $16.67.
2. **Policy learning** (Athey & Wager, 2021): search directly for the rule, among simple ones, that maximises the estimated value. You don't need τ(x) to be accurate everywhere, only to know which side of $16.67 each customer is on.

Both are evaluated the same way: with **doubly robust scores** (Day 36), which give an unbiased value for **any** policy from one experiment (or from observational data, with the usual assumptions). This is **off-policy evaluation**: judging a policy nobody has run yet.

Create `causalkit/policy.py`:

```python
"""Policy evaluation helpers (Days 65-70). A policy says, for each customer, coupon (1) or not (0).

Net value of a coupon for one customer: MARGIN x (extra spend) - COST.
"""
import re

import numpy as np

MARGIN, COST = 0.30, 5.0
BREAK_EVEN = COST / MARGIN            # a coupon pays if it raises spend by more than $16.67


def net(spend, coupon, margin=MARGIN, cost=COST):
    """Each customer's net outcome: margin on what they spent, minus the cost if they got a coupon."""
    return margin * np.asarray(spend) - cost * np.asarray(coupon)


def dr_scores(y, t, p, mu0, mu1):
    """Doubly robust (AIPW) scores for E[y(1)] and E[y(0)], one per customer."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    s1 = mu1 + t * (y - mu1) / p
    s0 = mu0 + (1 - t) * (y - mu0) / (1 - p)
    return s1, s0


def policy_value(policy, s1, s0):
    """Average net value per customer of a policy, relative to coupons for nobody."""
    return float(np.mean(np.asarray(policy) * (s1 - s0)))


def tree_rules(graphviz: str) -> list[str]:
    """Turn EconML's export_graphviz text into one readable rule per leaf."""
    labels = dict(re.findall(r'^(\d+) \[label="([^"]+)"', graphviz, flags=re.M))
    edges = re.findall(r"^(\d+) -> (\d+)", graphviz, flags=re.M)
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    rules = []

    def walk(node, conds):
        lab = labels[node].split("\\n")
        if node not in children:
            treat = [x for x in lab if x.startswith("Treatment")][0].split("= ")[1]
            samples = [x for x in lab if x.startswith("samples")][0].split("= ")[1]
            rules.append(f"{' and '.join(conds) or 'everyone'}  ->  {'coupon' if treat != 'None' else 'no coupon'}"
                         f"  ({samples} training customers)")
            return
        cond = lab[0]
        left, right = children[node]
        walk(left, conds + [cond])
        walk(right, conds + [cond.replace("<=", ">")])

    walk("0", [])
    return rules
```

---

## 65.2 Policy trees (50 min)

EconML's `DRPolicyTree` grows a shallow tree whose leaves are decisions, choosing splits to maximise the doubly robust value of the net outcome (0.30 × spend − $5 if couponed). Create `day65/policy_tree.py`:

```python
# day65/policy_tree.py   (python -m day65.policy_tree)
"""
Policy learning (Athey & Wager, 2021): instead of estimating effects and then thresholding them, search
directly for the simple rule that maximises the net value. EconML's DRPolicyTree uses doubly robust scores,
so it works on observational data too. Net outcome = 0.30 x spend - $5 if couponed.
"""
from econml.policy import DRPolicyTree

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import net, tree_rules
from day64.uplift_models import experiment

if __name__ == "__main__":
    train, test = experiment()
    for depth in [1, 2]:
        tree = DRPolicyTree(model_regression=reg(), model_propensity=clf(), max_depth=depth, min_samples_leaf=500,
                            random_state=0).fit(net(train.spend, train.coupon), train.coupon, X=train[X_COLS])
        print(f"Policy tree, depth {depth}:")
        for rule in tree_rules(tree.export_graphviz(feature_names=X_COLS)):
            print("   " + rule)
```

```bash
python -m day65.policy_tree
```

```
Policy tree, depth 1:
   deal_share <= 0.322  ->  no coupon  (6290 training customers)
   deal_share > 0.322  ->  coupon  (3710 training customers)
Policy tree, depth 2:
   deal_share <= 0.322 and z2 <= -1.411  ->  no coupon  (502 training customers)
   deal_share <= 0.322 and z2 > -1.411  ->  no coupon  (5788 training customers)
   deal_share > 0.322 and region_west <= 0.5  ->  coupon  (2986 training customers)
   deal_share > 0.322 and region_west > 0.5  ->  coupon  (724 training customers)
```

One rule: **coupon customers who buy more than 32% of their items on promotion**. The depth-2 tree adds splits on a noise column (z2) and on region, but both sides of each split get the same decision, so they don't change the policy. A rule a store manager can apply, and audit, from one number.

---

## 65.3 Which policy makes the most money? (55 min)

Create `day65/policy_value.py`. Its `outcome_models` function is reused through Day 70:

```python
# day65/policy_value.py   (python -m day65.policy_value)
"""
Which policy makes the most money? Evaluate five policies on the held-out half of the experiment with
doubly robust scores (in an experiment the propensity is known: 0.5), with bootstrap CIs. Value is net $
per customer in the base, relative to sending nobody a coupon.
"""
import numpy as np
import polars as pl
from econml.dr import DRLearner
from econml.policy import DRPolicyTree

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import BREAK_EVEN, MARGIN, COST, dr_scores, net, policy_value
from day64.uplift_models import experiment


def policies(train, test):
    """Fit the candidate policies on train; return each one's decisions for the test customers."""
    dr = DRLearner(model_propensity=clf(), model_regression=reg(),
                   model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                   min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
    tree = DRPolicyTree(model_regression=reg(), model_propensity=clf(), max_depth=2, min_samples_leaf=500,
                        random_state=0).fit(net(train.spend, train.coupon), train.coupon, X=train[X_COLS])
    n = len(test)
    return {
        "coupons for nobody": np.zeros(n),
        "coupons for everyone": np.ones(n),
        "policy tree (depth 2)": tree.predict(test[X_COLS]),
        "DR-learner: effect > $16.67": (dr.effect(test[X_COLS]) > BREAK_EVEN).astype(int),
        "oracle (true effects)": (test._tau.to_numpy() > BREAK_EVEN).astype(int),
    }


def outcome_models(test, cost=COST):
    """Cross-fitted models of the net outcome for the evaluation half (for doubly robust scores)."""
    from sklearn.model_selection import KFold
    mu0, mu1 = np.zeros(len(test)), np.zeros(len(test))
    y = net(test.spend, test.coupon, cost=cost)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(test):
        a = test.iloc[tr]
        for t, mu in [(0, mu0), (1, mu1)]:
            mu[te] = reg().fit(a[a.coupon == t][X_COLS], y[tr][a.coupon.to_numpy() == t]).predict(test.iloc[te][X_COLS])
    return y, mu0, mu1


if __name__ == "__main__":
    train, test = experiment()
    pols = policies(train, test)
    y, mu0, mu1 = outcome_models(test)
    s1, s0 = dr_scores(y, test.coupon, 0.5, mu0, mu1)
    true_gain = MARGIN * test._tau.to_numpy() - COST
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(test), len(test)) for _ in range(500)]
    rows = []
    for name, pol in pols.items():
        v = policy_value(pol, s1, s0)
        b = [policy_value(pol[i], s1[i], s0[i]) for i in boot]
        rows.append({"policy": name, "share couponed": f"{np.mean(pol):.0%}", "net $ per customer": round(v, 2),
                     "95% CI": f"[{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]",
                     "true net $ per customer": round(float(np.mean(pol * true_gain)), 2)})
    with pl.Config(tbl_width_chars=130, fmt_str_lengths=40):
        print(pl.DataFrame(rows))
```

```bash
python -m day65.policy_value
```

```
│ policy                      ┆ share couponed ┆ net $ per customer ┆ 95% CI        ┆ true net $ per customer │
│ coupons for nobody          ┆ 0%             ┆ 0.0                ┆ [0.00, 0.00]  ┆ 0.0                     │
│ coupons for everyone        ┆ 100%           ┆ 0.18               ┆ [-0.10, 0.46] ┆ -0.01                   │
│ policy tree (depth 2)       ┆ 38%            ┆ 1.58               ┆ [1.40, 1.75]  ┆ 1.33                    │
│ DR-learner: effect > $16.67 ┆ 48%            ┆ 1.84               ┆ [1.63, 2.05]  ┆ 1.61                    │
│ oracle (true effects)       ┆ 48%            ┆ 1.9                ┆ [1.68, 2.10]  ┆ 1.65                    │
```

- **Coupons for everyone** break even (true −$0.01 per customer): the average effect is just below $16.67, so the gains on persuadables are cancelled by the losses on everyone else.
- **Targeting** earns $1.3–1.6 per customer per campaign in truth. Over NorthMart's 400,000 customers, that's about $600,000 a month against zero for blanket coupons.
- The **plug-in DR-learner policy** beats the policy tree here and comes within 3% of the oracle: with an accurate CATE model, thresholding it is hard to beat. The tree gives up $0.28 per customer for a one-line rule. Whether that's worth it is a business question (can the CRM system apply a model score? does the team need to explain the rule?).
- The DR estimates are about $0.25 above the truth for every targeted policy, consistently: the experiment's evaluation half happened to be a little favourable. That's sampling noise, inside the CIs, and it shifts all the policies together; the **comparisons** between them are much more precise than the levels.

---

## 65.4 How deep should the tree be? (40 min)

**Try it.** Fit policy trees of depth 1 to 4 (with smaller minimum leaves for deeper trees) and evaluate each on the experiment. Create `day65/deeper_tree.py`.

<details>
<summary>Solution</summary>

```python
# day65/deeper_tree.py   (python -m day65.deeper_tree)
from econml.policy import DRPolicyTree

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import dr_scores, net, policy_value
from day64.uplift_models import experiment
from day65.policy_value import outcome_models

train, test = experiment()
y, mu0, mu1 = outcome_models(test)
s1, s0 = dr_scores(y, test.coupon, 0.5, mu0, mu1)
for depth, leaf in [(1, 500), (2, 500), (3, 200), (4, 50)]:
    tree = DRPolicyTree(model_regression=reg(), model_propensity=clf(), max_depth=depth, min_samples_leaf=leaf,
                        random_state=0).fit(net(train.spend, train.coupon), train.coupon, X=train[X_COLS])
    print(f"depth {depth}, min leaf {leaf:>3}: net $ per customer {policy_value(tree.predict(test[X_COLS]), s1, s0):.2f}")
```

```
depth 1, min leaf 500: net $ per customer 1.58
depth 2, min leaf 500: net $ per customer 1.58
depth 3, min leaf 200: net $ per customer 1.58
depth 4, min leaf  50: net $ per customer 1.56
```

No gain from depth, and a slight loss at depth 4 (overfitting to noise in small leaves). The true effect here depends mostly on one smooth feature, `deal_share`, and a tree can only approximate it with steps. When a policy tree plateaus below a plug-in CATE policy, the rule's simplicity is costing money; when the plug-in policy can't be deployed, a depth-1 or depth-2 tree is the honest compromise.

</details>

---

## 65.5 Day 65 mini-project: when coupons cost more (65 min)

The best policy depends on the cost: the threshold is cost / margin. Create `day65/cost_sensitivity.py`:

```python
# day65/cost_sensitivity.py   (python -m day65.cost_sensitivity)
"""
Mini-project: the policy depends on the coupon's cost. For costs from $2 to $10, threshold the DR-learner's
effects at cost / margin and evaluate on the experiment, against coupons for everyone.
"""
import numpy as np
import polars as pl
from econml.dr import DRLearner

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import MARGIN, dr_scores, policy_value
from day64.uplift_models import experiment
from day65.policy_value import outcome_models

train, test = experiment()
dr = DRLearner(model_propensity=clf(), model_regression=reg(),
               model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
               min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
tau_hat = dr.effect(test[X_COLS])
rows = []
for cost in [2, 4, 5, 6, 8, 10]:
    y, mu0, mu1 = outcome_models(test, cost=cost)
    s1, s0 = dr_scores(y, test.coupon, 0.5, mu0, mu1)
    pol = (tau_hat > cost / MARGIN).astype(int)
    rows.append({"coupon cost": cost, "break-even effect": round(cost / MARGIN, 1), "share couponed": f"{pol.mean():.0%}",
                 "targeted: net $ per customer": round(policy_value(pol, s1, s0), 2),
                 "everyone: net $ per customer": round(policy_value(np.ones(len(test)), s1, s0), 2)})
with pl.Config(tbl_width_chars=130):
    print(pl.DataFrame(rows))
```

```bash
python -m day65.cost_sensitivity
```

```
│ coupon cost ┆ break-even effect ┆ share couponed ┆ targeted: net $ per customer ┆ everyone: net $ per customer │
│ 2           ┆ 6.7               ┆ 76%            ┆ 3.66                         ┆ 3.18                         │
│ 4           ┆ 13.3              ┆ 58%            ┆ 2.38                         ┆ 1.18                         │
│ 5           ┆ 16.7              ┆ 48%            ┆ 1.84                         ┆ 0.18                         │
│ 6           ┆ 20.0              ┆ 40%            ┆ 1.44                         ┆ -0.82                        │
│ 8           ┆ 26.7              ┆ 24%            ┆ 0.73                         ┆ -2.82                        │
│ 10          ┆ 33.3              ┆ 10%            ┆ 0.28                         ┆ -4.82                        │
```

With cheap coupons ($2), targeting barely beats blanket coupons (most customers are worth couponing). With expensive ones ($10), blanket coupons lose $4.82 per customer while targeting still earns a little from the top 10%. **The value of a targeting model rises with the cost of the treatment.** That's the business case for building one, in a single table.

---

## 65.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 65: policy value, off-policy evaluation, policy trees, cost sensitivity"
git push
```

### Day 65 checklist

- [ ] I can define a policy's value with margins and costs, and its break-even effect
- [ ] I can evaluate any policy with doubly robust scores and bootstrap CIs
- [ ] I can learn a policy tree with EconML and read its rules
- [ ] I can compare plug-in and policy-learning approaches, and explain the trade-off
- [ ] I can show how the optimal policy changes with costs

---

# DAY 66 — Cost-Benefit Analysis

| Time | Block |
|---|---|
| 0:00 – 0:40 | 66.1 From a campaign to a programme |
| 0:40 – 1:40 | 66.2 A two-year NPV |
| 1:40 – 1:55 | Break |
| 1:55 – 2:45 | 66.3 Which assumption matters most? The tornado chart |
| 2:45 – 3:20 | 66.4 How much fatigue can it take? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 66.5 Day 66 mini-project: everything uncertain at once |
| 4:40 – 5:00 | 66.6 Save your work with Git |

---

## 66.1 From a campaign to a programme (40 min)

Day 65 valued **one** campaign. The decision on the table is a **programme**: build a targeting system and run monthly campaigns for two years. That brings in things an experiment doesn't measure:

| Element | Source | Value |
|---|---|---|
| extra spend per customer in the base, from the targeted policy | Day 65's experiment | measured (≈ $14.2, ±13%) |
| share of customers couponed | Day 65's policy | measured (48%) |
| margin on extra spend | finance | 30% (25–35%) |
| coupon cost | finance | $5 ($4–6) |
| **pull-forward**: extra spend that is just next month's purchases made earlier | assumption | 15% (0–30%) |
| **fatigue**: the coupon's effect retained each month as customers get used to it | assumption | 99% (96–100%) |
| customers reached | CRM | 400,000 (300k–500k) |
| build cost; running cost | IT | $150k; $25k a month |
| discount rate | finance | 10% a year |

The experiment pins down two inputs. The others are assumptions, some of them big. A cost-benefit analysis (CBA) makes them explicit and shows which ones the decision depends on.

---

## 66.2 A two-year NPV (60 min)

Create `day66/cba.py`. It measures the two experimental inputs and saves all inputs for the rest of the module:

```python
# day66/cba.py   (python -m day66.cba)
"""
Cost-benefit analysis: is the targeted coupon PROGRAMME worth building? Day 65 valued one campaign.
A programme runs monthly for two years, costs money to build and run, and its effect may fade.
  per customer per month:  margin x (extra spend among couponed customers, per customer in the base)
                           x (1 - pull-forward) x fatigue^month  -  coupon cost x share couponed
  pull-forward: part of the extra spend is purchases customers would have made next month anyway
  fatigue:      the coupon's effect shrinks a little every month customers keep receiving them
The two numbers that come from the experiment (Day 65's DR-learner policy, measured on held-out customers):
  uplift: extra spend per customer in the base;  share: share of customers couponed.
"""
import json

import numpy as np
from econml.dr import DRLearner

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from causalkit.policy import BREAK_EVEN, dr_scores
from day64.uplift_models import experiment
from day65.policy_value import outcome_models


def npv(a: dict, months: int = 24, stop: bool = False) -> tuple[float, int | None]:
    """Net present value ($) of the programme, and the first month cumulative cash flow turns positive.
    stop=True: shut the programme down in the first month its monthly cash flow would be negative."""
    rate = (1 + a["discount_rate"]) ** (1 / 12) - 1
    cum, total, payback = -a["build_cost"], -a["build_cost"], None
    for m in range(1, months + 1):
        per_customer = (a["margin"] * a["uplift"] * (1 - a["pull_forward"]) * a["fatigue"] ** (m - 1)
                        - a["coupon_cost"] * a["share"])
        flow = a["customers"] * per_customer - a["running_cost"]
        if stop and flow < 0:
            break
        total += flow / (1 + rate) ** m
        cum += flow
        if payback is None and cum > 0:
            payback = m
    return total, payback


BASE = {"customers": 400_000, "margin": 0.30, "coupon_cost": 5.0, "pull_forward": 0.15, "fatigue": 0.99,
        "build_cost": 150_000, "running_cost": 25_000, "discount_rate": 0.10}

if __name__ == "__main__":
    train, test = experiment()
    dr = DRLearner(model_propensity=clf(), model_regression=reg(),
                   model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                   min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
    pol = (dr.effect(test[X_COLS]) > BREAK_EVEN).astype(int)
    y, mu0, mu1 = outcome_models(test, cost=0.0)            # cost 0 and margin 1/0.3: models of spend x 0.3
    s1, s0 = dr_scores(y / 0.30, test.coupon, 0.5, mu0 / 0.30, mu1 / 0.30)
    inputs = BASE | {"uplift": float(np.mean(pol * (s1 - s0))), "share": float(pol.mean()),
                     "uplift_everyone": float(np.mean(s1 - s0))}
    json.dump(inputs, open("data/day66_inputs.json", "w"), indent=2)
    print("Inputs:", {k: round(v, 3) for k, v in inputs.items()})
    total, payback = npv(inputs)
    first = (inputs["margin"] * inputs["uplift"] * (1 - inputs["pull_forward"]) - inputs["coupon_cost"] * inputs["share"])
    print(f"Month 1: net ${first:.2f} per customer x {inputs['customers']:,} customers "
          f"= ${first * inputs['customers']:,.0f}, minus ${inputs['running_cost']:,} running cost")
    print(f"Two-year NPV: ${total:,.0f};  payback in month {payback}")
    print(f"Same programme with coupons for EVERYONE: ${npv(inputs | {'uplift': inputs['uplift_everyone'], 'share': 1.0})[0]:,.0f}")
```

```bash
python -m day66.cba
```

```
Inputs: {'customers': 400000, 'margin': 0.3, 'coupon_cost': 5.0, 'pull_forward': 0.15, 'fatigue': 0.99, 'build_cost': 150000, 'running_cost': 25000, 'discount_rate': 0.1, 'uplift': 14.192, 'share': 0.484, 'uplift_everyone': 17.283}
Month 1: net $1.20 per customer x 400,000 customers = $479,545, minus $25,000 running cost
Two-year NPV: $6,479,697;  payback in month 1
Same programme with coupons for EVERYONE: $-9,829,495
```

At the base assumptions, the targeted programme is worth about $6.5M over two years and pays back its build cost in the first month. The same programme without targeting **loses** $9.8M, even though its average uplift is higher ($17.28 vs $14.19 per customer): it coupons everyone, and pull-forward pushes the effective uplift below break-even.

The month-1 net value ($1.20 per customer) is lower than Day 65's $1.84 because of pull-forward: 15% of the extra spend would have happened next month anyway.

---

## 66.3 Which assumption matters most? The tornado chart (50 min)

Move each input to a plausible low and high value, one at a time; sort by the swing in NPV. Create `day66/tornado.py`:

```python
# day66/tornado.py   (python -m day66.tornado)
"""
Which assumption matters most? Move each input to a plausible low and high value, one at a time, and
record the NPV. Sorted by the size of the swing, that's a tornado chart. Run day66.cba first.
"""
import json

import matplotlib.pyplot as plt
import polars as pl

from day66.cba import npv

base = json.load(open("data/day66_inputs.json"))
ranges = {
    "fatigue (monthly retention of the effect)": (0.96, 1.00),
    "pull-forward share": (0.30, 0.0),
    "uplift (95% CI of the experiment)": (base["uplift"] * 0.87, base["uplift"] * 1.13),
    "margin": (0.25, 0.35),
    "coupon cost ($)": (6.0, 4.0),
    "customers reached": (300_000, 500_000),
    "running cost ($/month)": (50_000, 15_000),
    "build cost ($)": (300_000, 100_000),
}
keys = {"fatigue (monthly retention of the effect)": "fatigue", "pull-forward share": "pull_forward",
        "uplift (95% CI of the experiment)": "uplift", "margin": "margin", "coupon cost ($)": "coupon_cost",
        "customers reached": "customers", "running cost ($/month)": "running_cost", "build cost ($)": "build_cost"}
base_npv = npv(base)[0]
rows = []
for label, (lo, hi) in ranges.items():
    k = keys[label]
    rows.append({"assumption": label, "low input": lo, "high input": hi,
                 "NPV low ($k)": round(npv(base | {k: lo})[0] / 1000), "NPV high ($k)": round(npv(base | {k: hi})[0] / 1000)})
table = pl.DataFrame(rows).with_columns(swing=(pl.col("NPV high ($k)") - pl.col("NPV low ($k)")).abs()).sort("swing")
with pl.Config(tbl_width_chars=130, fmt_str_lengths=45, float_precision=2):
    print(table.sort("swing", descending=True))
print(f"Base NPV: ${base_npv / 1000:,.0f}k")

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, r in enumerate(table.iter_rows(named=True)):
    ax.barh(i, r["NPV low ($k)"] - base_npv / 1000, left=base_npv / 1000, color="#dc2626", alpha=0.8)
    ax.barh(i, r["NPV high ($k)"] - base_npv / 1000, left=base_npv / 1000, color="#16a34a", alpha=0.8)
ax.set_yticks(range(table.height), table["assumption"], fontsize=8)
ax.axvline(base_npv / 1000, color="k", lw=0.8)
ax.axvline(0, color="grey", ls="--", lw=0.8)
ax.set_xlabel("two-year NPV ($k)  (red: low input, green: high input)")
fig.tight_layout()
fig.savefig("figures/day66_tornado.png", dpi=120)
print("Saved figures/day66_tornado.png")
```

```bash
python -m day66.tornado
```

```
│ assumption                                ┆ low input ┆ high input ┆ NPV low ($k) ┆ NPV high ($k) ┆ swing │
│ fatigue (monthly retention of the effect) ┆ 0.96      ┆ 1.00       ┆ -950         ┆ 9743          ┆ 10693 │
│ pull-forward share                        ┆ 0.30      ┆ 0.00       ┆ 1496         ┆ 11464         ┆ 9968  │
│ margin                                    ┆ 0.25      ┆ 0.35       ┆ 1773         ┆ 11187         ┆ 9414  │
│ coupon cost ($)                           ┆ 6.00      ┆ 4.00       ┆ 2266         ┆ 10693         ┆ 8427  │
│ uplift (95% CI of the experiment)         ┆ 12.35     ┆ 16.04      ┆ 2808         ┆ 10151         ┆ 7343  │
│ customers reached                         ┆ 300000.00 ┆ 500000.00  ┆ 4686         ┆ 8273          ┆ 3587  │
│ running cost ($/month)                    ┆ 50000.00  ┆ 15000.00   ┆ 5936         ┆ 6697          ┆ 761   │
│ build cost ($)                            ┆ 300000.00 ┆ 100000.00  ┆ 6330         ┆ 6530          ┆ 200   │
Base NPV: $6,480k
```

Open `figures/day66_tornado.png`. The two biggest bars are **assumptions nobody has measured**: fatigue and pull-forward. The experiment's uncertainty (uplift) comes fifth. IT's costs, which usually get the most scrutiny in a budget meeting, barely matter.

**The tornado tells you what to learn next.** A one-month experiment can't measure fatigue; a holdout kept for six months can. Day 68 puts a dollar value on that.

---

## 66.4 How much fatigue can it take? (35 min)

**Try it.** Find the fatigue value at which the NPV is exactly zero (bisection: try the midpoint, keep the half where the sign changes). Create `day66/break_even_fatigue.py`.

<details>
<summary>Solution</summary>

```python
# day66/break_even_fatigue.py   (python -m day66.break_even_fatigue)
import json
import math

from day66.cba import npv

base = json.load(open("data/day66_inputs.json"))
lo, hi = 0.90, 1.0
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if npv(base | {"fatigue": mid})[0] < 0 else (lo, mid)
print(f"NPV is zero when the effect keeps {hi:.4f} of its size each month "
      f"(it halves every {math.log(0.5) / math.log(hi):.0f} months)")
```

```
NPV is zero when the effect keeps 0.9646 of its size each month (it halves every 19 months)
```

The programme loses money only if the coupon's effect halves in under about a year and a half. That's a statement a marketing director can assess from experience ("our loyalty mailers lose about half their punch within a year"), which "fatigue = 0.9646" is not. Converting a break-even parameter into a half-life, a number of customers, or a price makes it discussable.

</details>

---

## 66.5 Day 66 mini-project: everything uncertain at once (65 min)

A tornado moves one input at a time. Draw them all together, 10,000 times. Create `day66/npv_monte_carlo.py`:

```python
# day66/npv_monte_carlo.py   (python -m day66.npv_monte_carlo)
"""
Mini-project: a tornado moves one input at a time. Reality moves them all at once. Draw every uncertain
input from a distribution (uniform over the tornado's range; the uplift from its sampling distribution),
10,000 times, and look at the distribution of NPV. Run day66.cba first.
"""
import json

import numpy as np

from day66.cba import npv

base = json.load(open("data/day66_inputs.json"))
rng = np.random.default_rng(66)
draws = []
for _ in range(10_000):
    a = base | {"fatigue": rng.uniform(0.96, 1.00), "pull_forward": rng.uniform(0.0, 0.30),
                "uplift": rng.normal(base["uplift"], base["uplift"] * 0.066), "margin": rng.uniform(0.25, 0.35),
                "coupon_cost": rng.uniform(4.0, 6.0), "customers": rng.uniform(300_000, 500_000),
                "running_cost": rng.uniform(15_000, 50_000), "build_cost": rng.uniform(100_000, 300_000)}
    draws.append(npv(a)[0])
d = np.array(draws) / 1000
print(f"NPV ($k): mean {d.mean():,.0f}, median {np.median(d):,.0f}, 5th percentile {np.percentile(d, 5):,.0f}, "
      f"95th percentile {np.percentile(d, 95):,.0f}")
print(f"Probability the programme loses money over two years: {(d < 0).mean():.1%}")
```

```bash
python -m day66.npv_monte_carlo
```

```
NPV ($k): mean 3,618, median 3,128, 5th percentile -4,900, 95th percentile 13,725
Probability the programme loses money over two years: 27.4%
```

The expected NPV ($3.6M) is well **below** the base case ($6.5M). Nothing is wrong: NPV is not a straight-line function of fatigue (effects compound month after month), so averaging over uncertainty isn't the same as plugging in the average. And there's a 27% chance of losing money. A base-case NPV alone would have hidden both facts.

Tomorrow: what to do about that 27%.

---

## 66.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 66: programme cost-benefit analysis, NPV, tornado chart, Monte Carlo"
git push
```

### Day 66 checklist

- [ ] I can turn a campaign result into a multi-period programme NPV, with pull-forward and fatigue
- [ ] I separate measured inputs from assumptions, and give every assumption a range
- [ ] I can build a tornado chart and use it to decide what to learn next
- [ ] I can find and explain a break-even value for an assumption
- [ ] I know why the expected NPV differs from the base-case NPV

