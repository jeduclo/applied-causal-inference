# Module 6 · Causal Machine Learning & Heterogeneity
## Days 55–57: Heterogeneous Effects, Causal Forests and Validating CATEs

**Goal by end of Day 57:** Students can estimate conditional average treatment effects (CATEs) with S-, T-, X- and DR-learners and causal forests, turn them into a targeting policy, estimate group effects that aren't shrunk towards the average, and validate a CATE model on a randomised experiment with BLP, uplift (TOC/AUTOC) and calibration checks.

---

# DAY 55 — Heterogeneous Effects and Meta-Learners

| Time | Block |
|---|---|
| 0:00 – 0:40 | 55.1 From one effect to one effect per customer |
| 0:40 – 1:40 | 55.2 Four meta-learners |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 55.3 What each learner found |
| 2:40 – 3:20 | 55.4 How much data do they need? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 55.5 Day 55 mini-project: who should get next month's coupons? |
| 4:40 – 5:00 | 55.6 Save your work with Git |

---

## 55.1 From one effect to one effect per customer (40 min)

The **conditional average treatment effect** is τ(x) = E[spend(1) − spend(0) | X = x]: the average effect for customers with features x. In the coupon simulator, τ ranges from about −$26 to +$69: deal-seekers respond strongly, app users more, big spenders less (they'd have bought anyway, and the coupon just gives away margin).

Knowing τ(x) answers the business question directly: **coupon the customers with the largest τ(x)**. Three warnings before we start:

1. τ(x) is never observed for anyone, so no learner can be checked the way a prediction can. In real work, Day 57's validation on an experiment is the only check. Here the simulator lets us cheat and compare with the truth: **PEHE**, the root mean squared error against the true CATE.
2. CATE estimates are much noisier than an ATE: each one rests on far fewer comparable customers.
3. All Days 31–36 assumptions still apply. Heterogeneous effects estimated from confounded data are only as good as the confounders you measured.

---

## 55.2 Four meta-learners (60 min)

A **meta-learner** builds a CATE model out of ordinary ML models (Künzel et al. 2019; Nie & Wager 2021; Kennedy 2023):

| Learner | Recipe | Weakness |
|---|---|---|
| **S** (single) | one model of spend on (X, coupon); τ(x) = f(x, 1) − f(x, 0) | regularisation shrinks τ towards 0 (Day 51) |
| **T** (two) | separate models for each group; τ(x) = f₁(x) − f₀(x) | the difference of two noisy models is very noisy |
| **X** | impute each customer's effect using the *other* group's model, then model those effects | more steps, more tuning |
| **DR** (doubly robust) | compute Day 36's AIPW score for each customer, then regress it on X | the AIPW scores are very noisy, so the final model must be simple |

Create `day55/meta_learners.py`. It fits on half the customers and scores on the other half:

```python
# day55/meta_learners.py   (python -m day55.meta_learners)
"""
Who responds to the coupon? Four meta-learners (Kunzel et al., 2019; Kennedy, 2023), each building a CATE
model tau(x) out of ordinary ML models. Fit on half the customers, scored on the other half against the
simulator's true effect for each customer.
  S-learner:  one model of spend on (X, coupon); tau(x) = f(x, 1) - f(x, 0)
  T-learner:  separate models for couponed and non-couponed customers; tau(x) = f1(x) - f0(x)
  X-learner:  imputes each customer's effect from the OTHER group's model, then models those effects
  DR-learner: regresses the doubly robust (AIPW) score of Day 36 on X
"""
import numpy as np
import polars as pl
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons, public
from causalkit.learners import clf, reg

full = coupons().with_row_index("row")
train, test = train_test_split(full.to_pandas(), test_size=0.5, random_state=0)
Xtr, Xte = train[X_COLS], test[X_COLS]

models = {
    "S-learner": SLearner(overall_model=reg()),
    "T-learner": TLearner(models=reg()),
    "X-learner": XLearner(models=reg(), propensity_model=clf()),
    "DR-learner, flexible final model": DRLearner(model_propensity=clf(), model_regression=reg(),
                                                  model_final=reg(), cv=5, min_propensity=0.02, random_state=0),
    "DR-learner, simple final model": DRLearner(model_propensity=clf(), model_regression=reg(),
                                                model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300),
                                                cv=5, min_propensity=0.02, random_state=0),
}
rows, preds = [], {}
for name, m in models.items():
    m.fit(train.spend, train.coupon, X=Xtr)
    tau_hat = m.effect(Xte)
    preds[name] = tau_hat
    rows.append({"learner": name, "ATE": round(tau_hat.mean(), 2),
                 "PEHE (RMSE vs true CATE)": round(float(np.sqrt(np.mean((tau_hat - test._tau) ** 2))), 2),
                 "correlation with true CATE": round(float(np.corrcoef(tau_hat, test._tau)[0, 1]), 3)})
rows.append({"learner": "TRUTH", "ATE": round(test._tau.mean(), 2), "PEHE (RMSE vs true CATE)": 0.0,
             "correlation with true CATE": 1.0})
with pl.Config(tbl_width_chars=110, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
print(f"True CATEs in the test half: SD {test._tau.std():.1f}, from {test._tau.min():.0f} to {test._tau.max():.0f}")
np.savez("data/day55_cate_preds.npz", rows=test.row.to_numpy(),
         **{k.split(",")[0].replace("-", "_"): v for k, v in preds.items() if "flexible" not in k})
```

```bash
python -m day55.meta_learners
```

```
│ learner                          ┆ ATE   ┆ PEHE (RMSE vs true CATE) ┆ correlation with true CATE │
│ S-learner                        ┆ 14.36 ┆ 7.4                      ┆ 0.867                      │
│ T-learner                        ┆ 15.86 ┆ 13.74                    ┆ 0.677                      │
│ X-learner                        ┆ 16.14 ┆ 8.28                     ┆ 0.826                      │
│ DR-learner, flexible final model ┆ 16.24 ┆ 16.45                    ┆ 0.62                       │
│ DR-learner, simple final model   ┆ 16.05 ┆ 6.17                     ┆ 0.896                      │
│ TRUTH                            ┆ 16.47 ┆ 0.0                      ┆ 1.0                        │
True CATEs in the test half: SD 13.8, from -26 to 69
```

- The **S-learner** ranks customers well (correlation 0.87) but its average effect is biased ($14.4): shrinkage again.
- The **T-learner** is the noisiest: each group's model makes its own errors, and the difference doubles them.
- The **DR-learner** with a flexible final model is the worst of all; with a simple one (100 shallow trees, at least 300 customers per leaf), it's the best. AIPW scores are unbiased but extremely noisy, so the last step must average a lot of them.

---

## 55.3 What each learner found (45 min)

Create `day55/cate_plot.py`:

```python
# day55/cate_plot.py   (python -m day55.cate_plot)
"""What did each learner find? Predicted effect against deal_share (the main driver), with the truth.
Run day55.meta_learners first: it saves the test-half predictions."""
import matplotlib.pyplot as plt
import numpy as np

from causalkit.coupons import coupons

saved = np.load("data/day55_cate_preds.npz")
test = coupons().to_pandas().iloc[saved["rows"]]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, name in zip(axes, ["S_learner", "T_learner", "X_learner", "DR_learner"]):
    ax.scatter(test.deal_share, saved[name], s=2, alpha=0.25, color="#2563eb", label="predicted")
    ax.scatter(test.deal_share, test._tau, s=2, alpha=0.15, color="#94a3b8", label="true")
    ax.set_title(name.replace("_", "-"))
    ax.set_xlabel("deal_share")
axes[0].set_ylabel("effect of the coupon ($)")
axes[0].legend(markerscale=5)
fig.tight_layout()
fig.savefig("figures/day55_cate_by_deal_share.png", dpi=120)
print("Saved figures/day55_cate_by_deal_share.png")
```

```bash
python -m day55.cate_plot
```

```
Saved figures/day55_cate_by_deal_share.png
```

Open the figure: predicted effect against `deal_share`, with the truth in grey. The DR-learner and S-learner track the rising trend; the T-learner's predictions scatter far above and below it. The S-learner's cloud is visibly flatter than the truth: shrinkage again.

---

## 55.4 How much data do they need? (40 min)

**Try it.** Rerun the four learners with 4,000 customers instead of 20,000 (`coupons(n=4_000)`, half for training). Which learner suffers most? Create `day55/small_sample.py`.

<details>
<summary>Solution</summary>

```python
# day55/small_sample.py   (python -m day55.small_sample)
import numpy as np
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg

for n in [4_000, 20_000]:
    full = coupons(n=n).to_pandas()
    tr, te = train_test_split(full, test_size=0.5, random_state=0)
    models = {"S": SLearner(overall_model=reg()), "T": TLearner(models=reg()),
              "X": XLearner(models=reg(), propensity_model=clf()),
              "DR": DRLearner(model_propensity=clf(), model_regression=reg(),
                              model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                              min_propensity=0.02, random_state=0)}
    out = []
    for name, m in models.items():
        m.fit(tr.spend, tr.coupon, X=tr[X_COLS])
        out.append(f"{name} {np.sqrt(np.mean((m.effect(te[X_COLS]) - te._tau) ** 2)):.1f}")
    print(f"n = {n:>6,}: PEHE  " + "   ".join(out))
```

```
n =  4,000: PEHE  S 11.6   T 23.2   X 15.2   DR 16.9
n = 20,000: PEHE  S 7.4   T 13.7   X 8.3   DR 6.2
```

With 2,000 training customers, the S-learner is best: its bias is the price of stability. The DR-learner needs data: it wins at 10,000 training customers and does poorly at 2,000. No meta-learner is best everywhere, which is why Day 57's validation matters.

</details>

---

## 55.5 Day 55 mini-project: who should get next month's coupons? (65 min)

Next month's budget covers 30% of customers. Rank the test-half customers by each model, coupon the top 30%, and score each policy with the true effects. Create `day55/targeting_value.py`:

```python
# day55/targeting_value.py   (python -m day55.targeting_value)
"""
Mini-project: the reason to estimate CATEs is to decide WHO gets a coupon. Next month's budget covers
30% of customers. Rank the test-half customers by each learner's predicted effect, coupon the top 30%,
and score the policy with the simulator's true effects. Compare with coupons at random and with the
current targeting model (which ranks customers by its own score, the propensity).
Run day55.meta_learners first.
"""
import numpy as np
import polars as pl

from causalkit.coupons import coupons

saved = np.load("data/day55_cate_preds.npz")
test = coupons().to_pandas().iloc[saved["rows"]]
k = int(0.3 * len(test))
tau = test._tau.to_numpy()
scores = {"random": np.random.default_rng(0).random(len(test)), "current targeting model": test._ps.to_numpy()}
scores |= {name.replace("_", "-"): saved[name] for name in ["S_learner", "T_learner", "X_learner", "DR_learner"]}
scores["oracle (true effects)"] = tau
rows = []
for name, s in scores.items():
    top = np.argsort(-s)[:k]
    rows.append({"policy": name, "effect per coupon": round(tau[top].mean(), 2),
                 "extra $ per 1,000 coupons": round(1000 * tau[top].mean()),
                 "negative effects": f"{(tau[top] < 0).mean():.1%}"})
with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day55.targeting_value
```

```
│ policy                  ┆ effect per coupon ┆ extra $ per 1,000 coupons ┆ negative effects │
│ random                  ┆ 16.27             ┆ 16270                     ┆ 10.4%            │
│ current targeting model ┆ 26.67             ┆ 26665                     ┆ 3.3%             │
│ S-learner               ┆ 31.3              ┆ 31295                     ┆ 0.0%             │
│ T-learner               ┆ 28.11             ┆ 28107                     ┆ 0.5%             │
│ X-learner               ┆ 30.5              ┆ 30501                     ┆ 0.2%             │
│ DR-learner              ┆ 31.74             ┆ 31739                     ┆ 0.0%             │
│ oracle (true effects)   ┆ 32.97             ┆ 32973                     ┆ 0.0%             │
```

Targeting on estimated effects raises the return per coupon by about 18% over the current model (which targets the customers most *likely to get* a coupon, not the ones who *respond* most), and gets within 5% of the oracle. Notice the S-learner: biased on average, but nearly as good as the best for **ranking**. For targeting, the order matters more than the level.

---

## 55.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 55: S, T, X and DR meta-learners, PEHE, targeting policy"
git push
```

### Day 55 checklist

- [ ] I can define the CATE and explain why it can't be checked directly
- [ ] I can fit S-, T-, X- and DR-learners with EconML and explain each one's weakness
- [ ] I know that the DR-learner's final model must be simple
- [ ] I can turn CATE estimates into a targeting policy and value it

---

# DAY 56 — Causal Forests

| Time | Block |
|---|---|
| 0:00 – 0:40 | 56.1 Trees that split on effects |
| 0:40 – 1:40 | 56.2 A causal forest on the coupon data |
| 1:40 – 1:55 | Break |
| 1:55 – 2:50 | 56.3 Group effects without shrinkage |
| 2:50 – 3:20 | 56.4 What noise features cost |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 56.5 Day 56 mini-project: rules marketing can use |
| 4:40 – 5:00 | 56.6 Save your work with Git |

---

## 56.1 Trees that split on effects (40 min)

**Concept.** A regression tree splits customers where **spend** differs. A **causal tree** (Athey & Imbens, 2016) splits where the **effect** differs: it looks for the split that makes the coupon effect most different between the two sides. A **causal forest** (Wager & Athey 2018; Athey, Tibshirani & Wager 2019) averages many such trees. Three ingredients make it statistically valid:

- **Honesty:** each tree uses one half of its sample to choose splits and the other half to estimate effects in the leaves, so the estimates aren't tuned to the noise that chose the splits.
- **Subsampling:** each tree sees a random subset, which makes the forest's variance estimable and gives confidence intervals.
- **Orthogonalisation:** EconML's `CausalForestDML` first residualises spend and coupon on X (Day 52), so the forest only has to find effect differences, not untangle confounding.

---

## 56.2 A causal forest on the coupon data (60 min)

Create `day56/causal_forest.py`. Its `fit_forest` function is reused for the rest of the module (fitting takes about 40 seconds):

```python
# day56/causal_forest.py   (python -m day56.causal_forest)
"""
Causal forests (Wager & Athey, 2018; Athey, Tibshirani & Wager, 2019), in EconML's CausalForestDML:
residualise spend and coupon on X with ML (Day 52), then grow 'honest' trees that split where the
EFFECT differs, not where spend differs. Each customer gets a CATE and a confidence interval.
"""
import numpy as np
from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg


def fit_forest(df=None):
    """Fit on half of the customers; return the forest and the (train, test) halves as pandas."""
    full = (coupons() if df is None else df).with_row_index("row")
    train, test = train_test_split(full.to_pandas(), test_size=0.5, random_state=0)
    cf = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=5, n_estimators=1000,
                         min_samples_leaf=20, max_samples=0.45, honest=True, random_state=0)
    cf.fit(train.spend, train.coupon, X=train[X_COLS])
    return cf, train, test


if __name__ == "__main__":
    cf, train, test = fit_forest()
    tau_hat = cf.effect(test[X_COLS])
    lo, hi = cf.effect_interval(test[X_COLS], alpha=0.05)
    tau = test._tau.to_numpy()
    print(f"PEHE {np.sqrt(np.mean((tau_hat - tau) ** 2)):.2f}; "
          f"correlation with true CATE {np.corrcoef(tau_hat, tau)[0, 1]:.3f}")
    print(f"Customer-level 95% intervals contain the true CATE for {np.mean((lo <= tau) & (tau <= hi)):.0%} "
          f"of customers; median width ${np.median(hi - lo):.0f}")
    ate = cf.ate_inference(X=test[X_COLS])
    print(f"ATE {ate.mean_point:.2f}   truth {tau.mean():.2f}")
    imp = sorted(zip(X_COLS, cf.feature_importances_), key=lambda kv: -kv[1])
    print("Where the forest splits on effect differences (importance):")
    for name, v in imp[:5]:
        print(f"   {name:<14} {v:.3f}")
    print(f"   noise features z1-z10 together: {sum(v for n, v in imp if n.startswith('z')):.3f}")
```

```bash
python -m day56.causal_forest
```

```
PEHE 8.64; correlation with true CATE 0.838
Customer-level 95% intervals contain the true CATE for 47% of customers; median width $11
ATE 15.87   truth 16.47
Where the forest splits on effect differences (importance):
   deal_share     0.887
   app_user       0.018
   visits         0.013
   pre_spend      0.009
   z9             0.006
   noise features z1-z10 together: 0.048
```

- The forest finds the main driver: 89% of its effect-splits are on `deal_share`. The noise features get under 5%.
- Its accuracy (PEHE 8.6) is between the meta-learners of Day 55: forests are not magic.
- **The customer-level 95% intervals cover the true effect for only 47% of customers.** The forest's intervals describe uncertainty around *its own smoothed estimate*, and a forest smooths: customers with extreme true effects get estimates pulled towards their neighbours'. Individual-level intervals are the least reliable output of any CATE method. Use averages over groups instead.

---

## 56.3 Group effects without shrinkage (55 min)

**Concept.** The business already uses groups: deal-seeker bands, app users. Their **group average treatment effects (GATEs)** are what marketing can act on. Two ways to compute them:

1. Average the forest's predictions within each group. Simple, but forest predictions are shrunk towards the overall average, and so are their averages.
2. Average the **doubly robust (AIPW) scores** within each group (Chernozhukov et al.'s GATES). Each customer's score is a noisy but unbiased estimate of their effect; averaging over thousands removes the noise without shrinking anything. DoubleML's `IRM.gate()` does this, with confidence intervals.

Create `day56/gates.py`:

```python
# day56/gates.py   (python -m day56.gates)
"""
Group average treatment effects (GATEs) for groups the business already uses. Two ways:
  1. average the causal forest's CATE predictions within each group;
  2. average the doubly robust (AIPW) scores within each group: DoubleML's IRM.gate().
Forest predictions are SMOOTHED towards the overall average (that's how forests reduce noise), so
group averages of them inherit that shrinkage. Averages of doubly robust scores don't.
"""
import doubleml as dml
import numpy as np
import pandas as pd
import polars as pl

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg
from day56.causal_forest import fit_forest

np.random.seed(56)  # DoubleML draws its cross-fitting folds from numpy's global RNG

df = coupons().to_pandas()
cf, train, test = fit_forest()
irm = dml.DoubleMLIRM(dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS),
                      ml_g=reg(), ml_m=clf(), n_folds=5, trimming_threshold=0.02).fit()

partitions = [   # each partition's groups must not overlap: one gate() call per partition
    {"deal share < 15%": lambda d: d.deal_share < 0.15,
     "deal share 15-25%": lambda d: d.deal_share.between(0.15, 0.25, inclusive="left"),
     "deal share 25-40%": lambda d: d.deal_share.between(0.25, 0.40, inclusive="left"),
     "deal share 40%+": lambda d: d.deal_share >= 0.40},
    {"no app": lambda d: d.app_user == 0, "app user": lambda d: d.app_user == 1},
]
rows = []
for splits in partitions:
    gate = irm.gate(groups=pd.DataFrame({k: f(df) for k, f in splits.items()})).confint()
    for name, f in splits.items():
        rows.append({"group": name, "customers": int(f(df).sum()),
                     "forest GATE": round(cf.effect(test[f(test)][X_COLS]).mean(), 1),
                     "DR-score GATE": round(gate.loc[name, "effect"], 1),
                     "DR 95% CI": f"[{gate.loc[name, '2.5 %']:.1f}, {gate.loc[name, '97.5 %']:.1f}]",
                     "true average": round(df._tau[f(df)].mean(), 1)})
with pl.Config(tbl_width_chars=110, fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```bash
python -m day56.gates
```

```
│ group             ┆ customers ┆ forest GATE ┆ DR-score GATE ┆ DR 95% CI    ┆ true average │
│ deal share < 15%  ┆ 4421      ┆ 9.6         ┆ 4.1           ┆ [1.4, 6.7]   ┆ 2.9          │
│ deal share 15-25% ┆ 4759      ┆ 10.5        ┆ 9.5           ┆ [7.1, 11.8]  ┆ 10.6         │
│ deal share 25-40% ┆ 6066      ┆ 15.4        ┆ 18.4          ┆ [16.4, 20.5] ┆ 18.7         │
│ deal share 40%+   ┆ 4754      ┆ 27.5        ┆ 30.9          ┆ [28.2, 33.5] ┆ 32.3         │
│ no app            ┆ 11016     ┆ 14.9        ┆ 11.3          ┆ [9.7, 12.9]  ┆ 12.0         │
│ app user          ┆ 8984      ┆ 17.1        ┆ 21.9          ┆ [20.1, 23.8] ┆ 22.1         │
```

The forest says the least deal-prone customers respond by $9.6; the truth is $2.9, and the DR-score GATE gets $4.1. At the top, the forest says $27.5 against $32.3 true. The forest's **ranking** is right, but its **spread** is compressed by about half. The DR-score GATEs are within $1.4 of the truth everywhere, with honest intervals. Note the app-user gap: the forest says $2.2 between app users and others, the truth is $10.1.

Rule: **use a CATE model to form groups; use doubly robust scores to estimate the groups' effects.**

`gate()` needs non-overlapping groups, so the script calls it once per partition (deal bands, then app use).

---

## 56.4 What noise features cost (30 min)

**Try it.** Refit the forest without the ten noise columns z1–z10. Does it improve? Create `day56/no_noise_features.py`.

<details>
<summary>Solution</summary>

```python
# day56/no_noise_features.py   (python -m day56.no_noise_features)
import numpy as np
from econml.dml import CausalForestDML

from causalkit.coupons import BEHAVIOUR
from causalkit.learners import clf, reg
from day56.causal_forest import fit_forest

_, train, test = fit_forest()
cols = BEHAVIOUR + ["lapsing"]
cf = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=5, n_estimators=1000,
                     min_samples_leaf=20, max_samples=0.45, honest=True, random_state=0)
cf.fit(train.spend, train.coupon, X=train[cols])
pred = cf.effect(test[cols])
print(f"Without z1-z10: PEHE {np.sqrt(np.mean((pred - test._tau) ** 2)):.2f}, "
      f"correlation {np.corrcoef(pred, test._tau)[0, 1]:.3f}")
```

```
Without z1-z10: PEHE 7.77, correlation 0.871
```

PEHE improves from 8.64 to 7.77. Every split spent on a noise feature is a split not spent on a real driver; with 10,000 training customers, that adds up. Thinking about which features *could* modify the effect, before fitting, is still worth doing. But don't drop features because the forest says they're unimportant and then refit: that's choosing the model on the same data you evaluate it with.

</details>

---

## 56.5 Day 56 mini-project: rules marketing can use (65 min)

Marketing can't act on 20,000 individual estimates, but it can act on three or four rules. Fit a depth-2 tree to the forest's predictions (EconML's `SingleTreeCateInterpreter`), turn its leaves into segments, and estimate each segment's effect with DR scores. Create `day56/segment_rules.py`:

```python
# day56/segment_rules.py   (python -m day56.segment_rules)
"""
Mini-project: marketing can't act on 20,000 individual CATEs, but it can act on a few rules.
Fit a depth-2 tree to the causal forest's CATEs (EconML's SingleTreeCateInterpreter), read off the
segments, and check each segment's effect against the truth with doubly robust GATEs.
"""
import doubleml as dml
import numpy as np
import pandas as pd
import polars as pl
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.tree import export_text

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg
from day56.causal_forest import fit_forest

np.random.seed(56)  # DoubleML draws its cross-fitting folds from numpy's global RNG

cf, train, test = fit_forest()
interp = SingleTreeCateInterpreter(max_depth=2, min_samples_leaf=500, include_model_uncertainty=False)
interp.interpret(cf, test[X_COLS])
tree = interp.tree_model_
print(export_text(tree, feature_names=X_COLS, decimals=2))



def leaf_rules(tree, names):
    """Plain-language rule for each leaf of a fitted sklearn tree."""
    t, out = tree.tree_, {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = " and ".join(conds)
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.2f}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.2f}"])

    walk(0, [])
    return out


rules = leaf_rules(tree, X_COLS)
df = coupons().to_pandas()
leaf = tree.apply(df[X_COLS].to_numpy())
irm = dml.DoubleMLIRM(dml.DoubleMLData(df[X_COLS + ["coupon", "spend"]], "spend", "coupon", X_COLS),
                      ml_g=reg(), ml_m=clf(), n_folds=5, trimming_threshold=0.02).fit()
groups = pd.DataFrame({rules[k]: leaf == k for k in np.unique(leaf)})
gate = irm.gate(groups=groups).confint()
rows = [{"segment": name, "customers": int(groups[name].sum()), "DR-score GATE": round(gate.loc[name, "effect"], 1),
         "95% CI": f"[{gate.loc[name, '2.5 %']:.1f}, {gate.loc[name, '97.5 %']:.1f}]",
         "true average": round(df._tau[groups[name]].mean(), 1)} for name in groups]
with pl.Config(tbl_width_chars=120, fmt_str_lengths=50):
    print(pl.DataFrame(rows))
```

```bash
python -m day56.segment_rules
```

```
|--- deal_share <= 0.37
|   |--- deal_share <= 0.29
|   |   |--- value: [10.36]
|   |--- deal_share >  0.29
|   |   |--- value: [16.03]
|--- deal_share >  0.37
|   |--- deal_share <= 0.44
|   |   |--- value: [22.61]
|   |--- deal_share >  0.44
|   |   |--- value: [28.47]

│ segment                                   ┆ customers ┆ DR-score GATE ┆ 95% CI       ┆ true average │
│ deal_share <= 0.37 and deal_share <= 0.29 ┆ 10810     ┆ 7.8           ┆ [6.2, 9.4]   ┆ 8.0          │
│ deal_share <= 0.37 and deal_share > 0.29  ┆ 3646      ┆ 19.0          ┆ [16.3, 21.6] ┆ 19.6         │
│ deal_share > 0.37 and deal_share <= 0.44  ┆ 1881      ┆ 26.2          ┆ [22.7, 29.7] ┆ 24.8         │
│ deal_share > 0.37 and deal_share > 0.44   ┆ 3663      ┆ 32.4          ┆ [29.2, 35.6] ┆ 34.2         │
```

The rules are simple: four `deal_share` bands, with cut points at 29%, 37% and 44%. The tree's own leaf values (10, 16, 23, 28) are the forest's shrunk estimates again; the DR-score GATEs (8, 19, 26, 32) are within $1.8 of the truth. A brief for marketing reads: "customers who bought more than 29% of their items on promotion respond to a $10 coupon with $19–32 of extra spend; the rest, about $8".

One caution to write in `day56/NOTES.md`: the cut points were chosen by looking at the data, so testing "is segment 4 different from segment 1?" on the same data overstates the evidence. Day 57 shows how to validate on data the model never saw.

---

## 56.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 56: causal forest, DR-score GATEs, interpretable segment rules"
git push
```

### Day 56 checklist

- [ ] I can explain honesty, subsampling and orthogonalisation in a causal forest
- [ ] I know that individual-level CATE intervals are unreliable, and why
- [ ] I estimate group effects with doubly robust scores, not by averaging forest predictions
- [ ] I can turn a forest into a few rules with a shallow interpreter tree

---

# DAY 57 — Validating CATEs: RATE, Qini and Calibration

| Time | Block |
|---|---|
| 0:00 – 0:40 | 57.1 How do you test what you can't observe? |
| 0:40 – 1:40 | 57.2 Five models, one experiment |
| 1:40 – 1:55 | Break |
| 1:55 – 2:45 | 57.3 Uplift curves by hand |
| 2:45 – 3:20 | 57.4 Can you validate without an experiment? |
| 3:20 – 3:35 | Break |
| 3:35 – 4:40 | 57.5 Day 57 mini-project: calibration by hand |
| 4:40 – 5:00 | 57.6 Save your work with Git |

---

## 57.1 How do you test what you can't observe? (40 min)

In real data there is no true CATE to compare with. But a CATE model makes predictions that **can** be checked, as long as you have data where the treatment was randomised and the model never saw:

| Check | Question | Good result |
|---|---|---|
| **BLP slope** (best linear predictor; Chernozhukov et al. 2018) | Regress doubly robust effect estimates on the model's predictions. Does a $1 higher prediction mean $1 more effect? | slope near 1; 0 = no information; above 1 = predictions too compressed |
| **Uplift: TOC and AUTOC** (Yadlowsky et al. 2021, "RATE") | Do the customers the model ranks highest respond more than average? | a large positive area under the curve |
| **Calibration** | Within quartiles of predicted effect, does the measured effect match the average prediction? | points on the 45-degree line; calibration R² near 1 |

They test different things. Uplift tests **ranking** (what targeting needs). BLP and calibration test **levels** (what a budget or a price decision needs).

NorthMart has what's needed: last spring's coupon A/B test, on 10,000 customers who aren't in the training data.

---

## 57.2 Five models, one experiment (60 min)

EconML's `DRTester` runs all three checks. Create `day57/validate.py`. Its `fit_models` and `experiment` functions are reused below (about a minute):

```python
# day57/validate.py   (python -m day57.validate)
"""
In real data there's no true CATE to compare with. How do you know a CATE model works?
Test it on a RANDOMISED experiment it wasn't trained on: last spring's coupon A/B test on 10,000 other
customers. EconML's DRTester computes, on that experiment:
  BLP slope:  regress doubly robust effect estimates on the model's predictions. 1 = well calibrated;
              above 1 = predictions too compressed; 0 = no information.
  AUTOC:      how much better than average are the customers the model ranks highest? (Yadlowsky et al.)
  calibration R2: do predicted group effects match measured group effects?
Then compare with what only a simulator can tell you: the true error.
"""
import numpy as np
import polars as pl
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner
from econml.validate import DRTester

from causalkit.coupons import X_COLS, coupons
from causalkit.learners import clf, reg
from day56.causal_forest import fit_forest


class RandomScores:
    """A 'model' with no information, to calibrate what the metrics look like for noise."""
    def effect(self, X, T0=0, T1=1):
        return np.random.default_rng(len(X)).normal(16, 10, len(X))


def fit_models():
    """The five CATE models, all trained on the observational train half. Returns (models, train)."""
    cf, train, _ = fit_forest()
    models = {
        "causal forest": cf,
        "DR-learner": DRLearner(model_propensity=clf(), model_regression=reg(),
                                model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                                min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS]),
        "T-learner": TLearner(models=reg()).fit(train.spend, train.coupon, X=train[X_COLS]),
        "S-learner": SLearner(overall_model=reg()).fit(train.spend, train.coupon, X=train[X_COLS]),
        "random scores": RandomScores(),
    }
    return models, train


def experiment():
    """Last spring's randomised coupon test: 10,000 different customers."""
    return coupons(n=10_000, seed=570, randomized=True).to_pandas()


if __name__ == "__main__":
    models, train = fit_models()
    data = experiment()
    Xv, Xt = data[X_COLS].to_numpy(), train[X_COLS].to_numpy()
    rows = []
    for name, m in models.items():
        tester = DRTester(model_regression=reg(), model_propensity=clf(), cate=m)
        tester.fit_nuisance(Xv, data.coupon.to_numpy(), data.spend.to_numpy(),
                            Xt, train.coupon.to_numpy(), train.spend.to_numpy())
        s = tester.evaluate_all(Xv, Xt, n_groups=4, n_bootstrap=200).summary().iloc[0]
        pred = m.effect(Xv)
        rows.append({"model": name, "BLP slope": f"{s.blp_est:.2f} ({s.blp_se:.2f})", "AUTOC": round(s.autoc_est, 1),
                     "calibration R2": round(s.cal_r_squared, 2),
                     "true PEHE": round(float(np.sqrt(np.mean((pred - data._tau) ** 2))), 1),
                     "true correlation": round(float(np.corrcoef(pred, data._tau)[0, 1]), 2)})
    with pl.Config(tbl_width_chars=120, fmt_str_lengths=30):
        print(pl.DataFrame(rows))
```

```bash
python -m day57.validate
```

```
│ model         ┆ BLP slope   ┆ AUTOC ┆ calibration R2 ┆ true PEHE ┆ true correlation │
│ causal forest ┆ 1.64 (0.19) ┆ 9.5   ┆ 0.48           ┆ 8.7       ┆ 0.84             │
│ DR-learner    ┆ 0.89 (0.12) ┆ 9.2   ┆ 0.83           ┆ 6.2       ┆ 0.9              │
│ T-learner     ┆ 0.04 (0.07) ┆ 4.0   ┆ -2.08          ┆ 13.7      ┆ 0.68             │
│ S-learner     ┆ 0.94 (0.13) ┆ 8.2   ┆ 0.63           ┆ 7.4       ┆ 0.87             │
│ random scores ┆ 0.01 (0.14) ┆ 0.7   ┆ -26.2          ┆ 17.0      ┆ 0.01             │
```

The first three columns are what you'd have in real work; the last two only a simulator can give. Do the checks agree with the truth?

- **Random scores** fail everything: BLP slope 0, AUTOC near 0. The checks aren't fooled by noise.
- The **causal forest's** slope of 1.64 says its predictions are too compressed: a $1 difference in prediction means $1.64 of real difference. That's exactly Day 56's shrinkage, detected from experimental data alone.
- The **T-learner's** slope near 0 doesn't mean it has no information (its true correlation is 0.68): its predictions are so spread out that most of their variation is noise. Its calibration R² is negative: predicting the overall average would beat it.
- The **DR-learner** has the best slope, calibration and true error. The S-learner is close behind. The checks pick the right model.

---

## 57.3 Uplift curves by hand (50 min)

**Concept.** Rank the experiment's customers by predicted effect. For the top q%, measure the effect by a difference in means (it's an A/B test, so that's unbiased), and subtract the overall effect. Plotted against q, that's the **targeting operating characteristic (TOC)**; its average is the **AUTOC**. Create `day57/uplift_by_hand.py`:

```python
# day57/uplift_by_hand.py   (python -m day57.uplift_by_hand)
"""
The targeting operating characteristic (TOC) by hand, on the randomised experiment: rank customers by a
model's predicted effect; for the top q% measure the effect by a difference in means (it's an A/B test, so
that's unbiased) and subtract the overall effect. A useful model's top customers respond much more than
average. AUTOC = the area under that curve.
"""
import matplotlib.pyplot as plt
import numpy as np

from causalkit.coupons import X_COLS
from day57.validate import experiment, fit_models

models, _ = fit_models()
data = experiment()
y, d = data.spend.to_numpy(), data.coupon.to_numpy()
overall = y[d == 1].mean() - y[d == 0].mean()
qs = np.arange(5, 101, 5)
fig, ax = plt.subplots(figsize=(8, 4))
print(f"Overall effect in the experiment: {overall:.1f}")
for name, m in models.items():
    order = np.argsort(-m.effect(data[X_COLS].to_numpy()))
    toc = []
    for q in qs:
        top = order[: int(len(order) * q / 100)]
        toc.append(y[top][d[top] == 1].mean() - y[top][d[top] == 0].mean() - overall)
    print(f"{name:<14} top 10%: {toc[1] + overall:5.1f}   top 30%: {toc[5] + overall:5.1f}   AUTOC {np.mean(toc):5.1f}")
    ax.plot(qs, toc, marker="o", ms=3, label=name)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("share of customers targeted, ranked by the model (%)")
ax.set_ylabel("effect in top q% minus overall ($)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day57_toc.png", dpi=120)
print("Saved figures/day57_toc.png")
```

```bash
python -m day57.uplift_by_hand
```

```
Overall effect in the experiment: 15.4
causal forest  top 10%:  37.3   top 30%:  31.9   AUTOC  10.8
DR-learner     top 10%:  37.6   top 30%:  30.7   AUTOC  10.7
T-learner      top 10%:  36.7   top 30%:  29.5   AUTOC   8.6
S-learner      top 10%:  39.8   top 30%:  31.2   AUTOC  10.5
random scores  top 10%:  15.6   top 30%:  15.9   AUTOC  -0.7
Saved figures/day57_toc.png
```

For **targeting**, the forest, DR-learner and S-learner are almost equally good: each one's top 30% responds about twice as much as average. The forest's compression doesn't hurt the ranking at all. The T-learner is somewhat worse. (These numbers differ slightly from DRTester's AUTOC, which uses doubly robust scores and a finer grid; the conclusions are the same.)

Open the figure: every real model's curve starts high and falls towards zero as q grows to 100% (targeting everyone = the average effect).

---

## 57.4 Can you validate without an experiment? (35 min)

**Try it.** Run the same DRTester checks on the **observational** test half of the coupon data (no experiment). DRTester uses doubly robust scores, which adjust for the targeting. Create `day57/validate_observational.py`.

<details>
<summary>Solution</summary>

```python
# day57/validate_observational.py   (python -m day57.validate_observational)
from econml.validate import DRTester

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from day56.causal_forest import fit_forest
from day57.validate import fit_models

models, train = fit_models()
_, _, test = fit_forest()
for name in ["causal forest", "DR-learner", "T-learner"]:
    tester = DRTester(model_regression=reg(), model_propensity=clf(), cate=models[name])
    tester.fit_nuisance(test[X_COLS].to_numpy(), test.coupon.to_numpy(), test.spend.to_numpy(),
                        train[X_COLS].to_numpy(), train.coupon.to_numpy(), train.spend.to_numpy())
    s = tester.evaluate_all(test[X_COLS].to_numpy(), train[X_COLS].to_numpy(), n_bootstrap=200).summary().iloc[0]
    print(f"{name:<14} BLP slope {s.blp_est:.2f} ({s.blp_se:.2f})   AUTOC {s.autoc_est:.1f}   "
          f"calibration R2 {s.cal_r_squared:.2f}")
```

```
causal forest  BLP slope 1.55 (0.12)   AUTOC 9.2   calibration R2 0.53
DR-learner     BLP slope 1.03 (0.07)   AUTOC 10.0   calibration R2 0.88
T-learner      BLP slope 0.52 (0.05)   AUTOC 7.3   calibration R2 -0.02
```

The same conclusions: the forest is compressed, the DR-learner is well calibrated, the T-learner is noisy. That works because every confounder is measured in this dataset. In observational data, validation inherits the same assumption as estimation: if a confounder is missing, the check is biased in the same direction as the model, and it can't warn you. Only a randomised test is independent of that assumption.

</details>

---

## 57.5 Day 57 mini-project: calibration by hand (65 min)

Cut the experiment's customers into quartiles of predicted effect (cut points from the training data), measure each quartile's effect by a difference in means, and plot it against the average prediction. Create `day57/calibration.py`:

```python
# day57/calibration.py   (python -m day57.calibration)
"""
Mini-project: calibration by hand. Cut the experiment's customers into quartiles of predicted effect
(cut points from the training data), measure each quartile's effect by difference in means, and plot it
against the average prediction. On the 45-degree line = predictions mean what they say.
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from causalkit.coupons import X_COLS
from day57.validate import experiment, fit_models

models, train = fit_models()
data = experiment()
y, d = data.spend.to_numpy(), data.coupon.to_numpy()
fig, ax = plt.subplots(figsize=(5.5, 5))
rows = []
for name in ["causal forest", "DR-learner", "T-learner"]:
    m = models[name]
    cuts = np.quantile(m.effect(train[X_COLS].to_numpy()), [0.25, 0.5, 0.75])
    pred = m.effect(data[X_COLS].to_numpy())
    q = np.digitize(pred, cuts)
    xs, ys = [], []
    for g in range(4):
        sel = q == g
        diff = y[sel][d[sel] == 1].mean() - y[sel][d[sel] == 0].mean()
        se = np.sqrt(y[sel][d[sel] == 1].var() / (d[sel] == 1).sum() + y[sel][d[sel] == 0].var() / (d[sel] == 0).sum())
        xs.append(pred[sel].mean())
        ys.append(diff)
        rows.append({"model": name, "quartile": g + 1, "predicted": round(pred[sel].mean(), 1),
                     "measured in experiment": round(diff, 1), "SE": round(se, 1)})
    ax.plot(xs, ys, marker="o", label=name)
lim = [-10, 45]
ax.plot(lim, lim, color="grey", ls="--", lw=0.8)
ax.set_xlabel("average predicted effect ($)")
ax.set_ylabel("effect measured in the experiment ($)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day57_calibration.png", dpi=120)
with pl.Config(tbl_rows=12):
    print(pl.DataFrame(rows))
print("Saved figures/day57_calibration.png")
```

```bash
python -m day57.calibration
```

```
│ model         ┆ quartile ┆ predicted ┆ measured in experiment ┆ SE  │
│ causal forest ┆ 1        ┆ 8.5       ┆ 3.9                    ┆ 2.5 │
│ causal forest ┆ 2        ┆ 11.5      ┆ 6.0                    ┆ 2.4 │
│ causal forest ┆ 3        ┆ 16.3      ┆ 21.1                   ┆ 2.4 │
│ causal forest ┆ 4        ┆ 27.2      ┆ 32.4                   ┆ 2.5 │
│ DR-learner    ┆ 1        ┆ 2.9       ┆ 3.2                    ┆ 2.7 │
│ DR-learner    ┆ 2        ┆ 11.3      ┆ 6.6                    ┆ 2.3 │
│ DR-learner    ┆ 3        ┆ 18.1      ┆ 20.6                   ┆ 2.3 │
│ DR-learner    ┆ 4        ┆ 32.3      ┆ 32.4                   ┆ 2.3 │
│ T-learner     ┆ 1        ┆ -7.5      ┆ 7.7                    ┆ 2.8 │
│ T-learner     ┆ 2        ┆ 9.3       ┆ 11.3                   ┆ 2.3 │
│ T-learner     ┆ 3        ┆ 21.8      ┆ 13.2                   ┆ 2.4 │
│ T-learner     ┆ 4        ┆ 40.1      ┆ 31.5                   ┆ 2.3 │
Saved figures/day57_calibration.png
```

Open the figure. The DR-learner's points sit on the 45-degree line. The forest's line is **steeper** than 45 degrees (predicted 8.5 to 27.2; measured 3.9 to 32.4): compressed. The T-learner's is **flatter** (predicted −7.5 to 40.1; measured 7.7 to 31.5): over-dispersed. Its bottom quartile is predicted to *lose* $7.50 per coupon, and actually gains $7.70.

That matters for decisions. If marketing used the T-learner to stop couponing customers with "negative" predicted effects, it would drop customers who respond.

---

## 57.6 Save your work with Git (20 min)

```bash
git add .
git commit -m "Day 57: CATE validation on an experiment: BLP, TOC/AUTOC, calibration"
git push
```

**Coming next (Days 58–60).** Everything so far had a yes/no treatment. Prices are continuous, set by a pricing team that reacts to demand, and they differ by segment. Days 58–60 put DML and heterogeneity to work on NorthMart's most consequential question.

### Day 57 checklist

- [ ] I can validate a CATE model on a randomised experiment it wasn't trained on
- [ ] I can read a BLP slope, an AUTOC and a calibration R², and say what each one tests
- [ ] I can draw a TOC curve and a calibration plot by hand
- [ ] I know that validation on observational data inherits the no-hidden-confounders assumption

