# Module 2 · Causal Thinking, Estimands & DAGs
## Days 18–20: DoWhy, Simpson's Paradox and Defending the DAG

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 11–17
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 20:** Students take a new business problem from narrative to DAG, test the DAG against data, derive and defend an adjustment set, estimate the effect, and explain in a written memo why the alternatives would mislead.

---

### Where we are

You can now state an estimand (Days 11–13), draw a DAG, test it, and find adjustment sets by hand with `causalkit` (Days 14–17). The NorthMart email taught the hardest lesson first: with intensity unobserved, **no** adjustment set exists.

Days 18–20 add a standard library for the whole workflow (**DoWhy**), the classic paradox that DAGs resolve (**Simpson's**), and a capstone on a new problem where adjustment **does** work, provided you choose correctly.

### New files

```
causal-decision-lab/
├── data/make_pricing_scenario.py     # provided: the capstone scenario
├── day18/  dowhy_workflow.py, omitted_variable_bias.py
├── day19/  simpson_two_stories.py, northmart_segments.py
├── day20/  my_dag.py, test_my_dag.py, estimate.py, answer_key.py
└── docs/DAG_MEMO.md                  # the capstone memo
```

---

# DAY 18 — Encoding Assumptions with DoWhy

| Time | Block |
|---|---|
| 0:00 – 0:40 | 18.1 DoWhy's four steps |
| 0:40 – 1:30 | 18.2 Model and identify: when DoWhy says "no" |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 18.3 Estimate under an explicit, stronger assumption |
| 2:30 – 3:15 | 18.4 Refute: what the checks can and can't see |
| 3:15 – 3:30 | Break |
| 3:30 – 4:45 | 18.5 Day 18 mini-project: omitted-variable bias, exactly |
| 4:45 – 5:00 | 18.6 Save your work with Git |

---

## 18.1 DoWhy's four steps (40 min)

**Concept.** **DoWhy** (originally from Microsoft Research, now part of the PyWhy project) organises every causal analysis into four explicit steps:

| Step | What happens | Who decides |
|---|---|---|
| 1. **Model** | You give the data and a DAG | you: domain knowledge |
| 2. **Identify** | DoWhy checks whether the DAG allows the effect to be computed from the data, and how (backdoor, front door, instrument) | the mathematics |
| 3. **Estimate** | A statistical method computes the identified quantity | you: method choice |
| 4. **Refute** | Automated checks try to break the result | you: interpret them |

Its value is the separation: the *assumptions* (step 1) and the *statistics* (step 3) are written in different places, so a reviewer can disagree with one without re-reading the other. It also refuses to estimate when step 2 fails, unlike a regression, which always returns a number.

DoWhy works with pandas DataFrames, so we convert with `.to_pandas()`. It prints many library warnings; the scripts hide them so you can read the results.

---

## 18.2 Model and identify: when DoWhy says "no" (50 min)

Create `day18/dowhy_workflow.py`. Read it top to bottom; we'll discuss each step's output in turn.

```python
# day18/dowhy_workflow.py   (python -m day18.dowhy_workflow)
"""DoWhy's four steps: model, identify, estimate, refute."""
import warnings

import networkx as nx
import polars as pl
from dowhy import CausalModel

from causalkit.data import load_customers

warnings.filterwarnings("ignore")      # DoWhy prints many library warnings; we read the results instead

df = (load_customers()
      .with_columns(pl.col("email").cast(pl.Int8), loyalty=pl.col("is_loyalty").cast(pl.Int8))
      .select("email", "post_orders", "loyalty", "pre_orders")
      .to_pandas())

# STEP 1 - MODEL. Honest DAG: intensity is unobserved (it is not a column in df).
honest = nx.DiGraph([("loyalty", "intensity"), ("loyalty", "email"), ("intensity", "email"),
                     ("intensity", "pre_orders"), ("intensity", "post_orders"), ("email", "post_orders")])
m_honest = CausalModel(data=df, treatment="email", outcome="post_orders", graph=honest)

# STEP 2 - IDENTIFY.
print("HONEST DAG - backdoor variables:", m_honest.identify_effect().get_backdoor_variables() or "none: not identified")

# A stronger, explicit assumption: pre_orders captures everything about intensity that drove targeting
proxy = nx.DiGraph([("loyalty", "email"), ("pre_orders", "email"),
                    ("loyalty", "post_orders"), ("pre_orders", "post_orders"), ("email", "post_orders")])
m = CausalModel(data=df, treatment="email", outcome="post_orders", graph=proxy)
estimand = m.identify_effect()
print("PROXY DAG  - backdoor variables:", estimand.get_backdoor_variables())

# STEP 3 - ESTIMATE, two ways, same estimand
lin = m.estimate_effect(estimand, method_name="backdoor.linear_regression")
ipw = m.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting",
                        target_units="att")
print(f"\nLinear regression (ATE)       {lin.value:+.3f}")
print(f"Propensity weighting (ATT)    {ipw.value:+.3f}")
print("Truth: ATE +0.319, ATT +0.443")

# STEP 4 - REFUTE
for name, kwargs in [("placebo_treatment_refuter", {"placebo_type": "permute"}),
                     ("random_common_cause", {}),
                     ("data_subset_refuter", {"subset_fraction": 0.8})]:
    r = m.refute_estimate(estimand, lin, method_name=name, random_seed=18, **kwargs)
    print(f"{name:<28} new effect {r.new_effect:+.3f}")
```

```bash
python -m day18.dowhy_workflow
```

The first line of output:

```
HONEST DAG - backdoor variables: none: not identified
```

We gave DoWhy the Day 14 DAG, including `intensity`. Because `intensity` is not a column of the data, DoWhy treats it as **unobserved** and concludes, as `causalkit` did on Day 15, that no backdoor adjustment set exists. A regression would have given an estimate anyway; DoWhy won't.

---

## 18.3 Estimate under an explicit, stronger assumption (45 min)

To proceed, the script writes down a **different, stronger** DAG: "everything about intensity that drove targeting is captured by `pre_orders` and `loyalty`". In that DAG, `pre_orders` is a confounder itself, and DoWhy identifies the effect:

```
PROXY DAG  - backdoor variables: ['pre_orders', 'loyalty']

Linear regression (ATE)       +1.062
Propensity weighting (ATT)    +0.928
Truth: ATE +0.319, ATT +0.443
```

The point isn't the number (still 2–3 times too large: the stronger assumption is false). The point is that the assumption is now **written down**, in code, where a reviewer can read it and say "I don't believe pre-period orders capture intensity". Compare with a typical analysis, where the same assumption is hidden in a list of regression controls.

Two estimators of the same identified estimand give different numbers (1.06 vs 0.93) partly because they target different populations (ATE vs ATT) and partly because they weigh customers differently. Module 4 covers both.

---

## 18.4 Refute: what the checks can and can't see (45 min)

The last three lines:

```
placebo_treatment_refuter    new effect -0.000
random_common_cause          new effect +1.062
data_subset_refuter          new effect +1.062
```

| Refuter | What it does | Expected if the analysis is sound | Result |
|---|---|---|---|
| Placebo treatment | replaces the email with a random fake | effect ≈ 0 | passes |
| Random common cause | adds a random noise variable as a confounder | effect unchanged | passes |
| Data subset | re-estimates on random 80% subsets | effect unchanged | passes |

**All three pass. The estimate is still 2.4 times the truth.**

These refuters detect *bugs* (a leaky pipeline, an estimate driven by a few customers, code that finds effects everywhere). None of them can detect an **unobserved confounder**: by definition, it's not in the data they test. Passing refuters is necessary, not sufficient. For hidden confounding, you need **sensitivity analysis**: asking how strong a hidden confounder would have to be to change the conclusion.

---

## 18.5 Day 18 mini-project: omitted-variable bias, exactly (75 min)

**Concept.** Economists know the **omitted-variable bias** (OVB) formula from introductory econometrics. If the true model includes U but you leave it out:

```text
bias = δ × γ
  δ = how much more U the treated have, after controlling for the included variables
  γ = the effect of U on the outcome
```

It's also the simplest sensitivity analysis: you don't know δ or γ, but you can ask how large they would need to be to explain your estimate away. Usually that's an argument. In NorthMart, the truth file contains U (intensity, `mu0`), so we can compute it exactly.

Create `day18/omitted_variable_bias.py`:

```python
# day18/omitted_variable_bias.py   (python -m day18.omitted_variable_bias)
"""
Mini-project, part 1: explain the gap between the estimate and the truth with the
omitted-variable-bias formula:   bias = delta x gamma
    delta = how much more intensity the emailed have, AFTER controlling for loyalty and pre_orders
    gamma = how many extra orders one unit of intensity causes
Only possible because the truth file reveals intensity (mu0). In real data, you'd vary delta and gamma.
"""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.data import load_customers

truth = pl.read_parquet("data/truth/potential_outcomes.parquet").select("customer_id", "mu0", "tau")
df = (load_customers().join(truth, on="customer_id")
      .with_columns(pl.col("email").cast(pl.Int8), loyalty=pl.col("is_loyalty").cast(pl.Int8))
      .to_pandas())

short = smf.ols("post_orders ~ email + loyalty + pre_orders", df).fit()
long = smf.ols("post_orders ~ email + loyalty + pre_orders + mu0", df).fit()
delta = smf.ols("mu0 ~ email + loyalty + pre_orders", df).fit().params["email"]
gamma = long.params["mu0"]

print(f"Estimate without intensity          {short.params['email']:+.3f}")
print(f"Estimate with intensity             {long.params['email']:+.3f}")
print(f"delta (intensity gap, adjusted)     {delta:+.3f}")
print(f"gamma (orders per unit intensity)   {gamma:+.3f}")
print(f"bias = delta x gamma                {delta * gamma:+.3f}")
print(f"short - long                        {short.params['email'] - long.params['email']:+.3f}   (identical: OVB is an identity)")

print("\nSensitivity grid: how much bias would different hidden confounders create?")
grid = pl.DataFrame({"delta": [0.1, 0.25, 0.5, 0.75]}).with_columns(
    **{f"gamma={g}": (pl.col("delta") * g).round(3) for g in [0.5, 1.0, 1.5]})
print(grid)
```

```bash
python -m day18.omitted_variable_bias
```

```
Estimate without intensity          +1.062
Estimate with intensity             +0.353
delta (intensity gap, adjusted)     +0.651
gamma (orders per unit intensity)   +1.090
bias = delta x gamma                +0.709
short - long                        +0.709   (identical: OVB is an identity)

Sensitivity grid: how much bias would different hidden confounders create?
│ delta ┆ gamma=0.5 ┆ gamma=1.0 ┆ gamma=1.5 │
│ 0.1   ┆ 0.05      ┆ 0.1       ┆ 0.15      │
│ 0.25  ┆ 0.125     ┆ 0.25      ┆ 0.375     │
│ 0.5   ┆ 0.25      ┆ 0.5       ┆ 0.75      │
│ 0.75  ┆ 0.375     ┆ 0.75      ┆ 1.125     │
```

- Even after controlling for loyalty and pre-period orders, emailed customers have 0.65 more expected orders of underlying intensity (δ). Each unit of intensity adds about 1.09 orders (γ). Together: 0.71 orders of bias, exactly the gap between the two estimates.
- The **grid** is how you'd argue without a truth file: "for our estimate of 1.06 to fall below the break-even effect of 0.3, a hidden confounder would need δ × γ ≥ 0.76, for example an intensity gap of 0.75 with γ around 1. Given how marketing targeted, that's plausible." That sentence, not the refuters, is what a reviewer needs.

Module 7 develops formal sensitivity tools (E-values, robustness values). Today's habit: **every observational estimate comes with a sentence about what hidden confounding would overturn it.**

---

## 18.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 18: DoWhy workflow, refuters, omitted-variable bias"
git push
```

### Day 18 checklist

- [ ] I can run DoWhy's model, identify, estimate and refute steps
- [ ] I can explain why DoWhy refuses to estimate with NorthMart's honest DAG
- [ ] I can explain what the standard refuters detect, and why they can't detect hidden confounding
- [ ] I can decompose a bias with the OVB formula and write a sensitivity sentence for an estimate

---

# DAY 19 — Simpson's Paradox in Business Data

| Time | Block |
|---|---|
| 0:00 – 0:40 | 19.1 When the total disagrees with every part |
| 0:40 – 1:40 | 19.2 Same numbers, two stories |
| 1:40 – 1:55 | Break |
| 1:55 – 2:40 | 19.3 Standardisation: averaging the parts properly |
| 2:40 – 3:20 | 19.4 Simpson's paradox as a DAG question |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 19.5 Day 19 mini-project: NorthMart's segments |
| 4:45 – 5:00 | 19.6 Save your work with Git |

---

## 19.1 When the total disagrees with every part (40 min)

**Concept.** **Simpson's paradox**: a relationship that holds in every subgroup reverses (or vanishes) when the subgroups are combined. Famous cases include university admissions (each department admitted women at similar or higher rates, yet overall rates favoured men) and medical treatments that look better in every patient group but worse overall.

It's not a statistical trick. It happens whenever the subgroups differ both in how often they're treated and in their baseline outcome. The interesting question isn't *why* it happens but **which table to believe**, and the data alone can't answer that.

---

## 19.2 Same numbers, two stories (60 min)

Create `day19/simpson_two_stories.py`:

```python
# day19/simpson_two_stories.py   (python -m day19.simpson_two_stories)
"""
The same four numbers, two different stories. The data cannot tell you which table to believe;
the DAG can.
"""
import polars as pl

# Free-shipping offer: conversions out of visitors, split by a third variable
counts = pl.DataFrame({
    "group":       ["A", "A", "B", "B"],
    "offer":       [1, 0, 1, 0],
    "visitors":    [800, 200, 200, 800],
    "conversions": [160, 30, 80, 280],
})


def rates(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (df.group_by(by).agg(pl.col("conversions").sum(), pl.col("visitors").sum())
              .with_columns(rate=(pl.col("conversions") / pl.col("visitors")).round(3)).sort(by))


print("Pooled:")
print(rates(counts, ["offer"]))
print("By group:")
print(rates(counts, ["group", "offer"]))

pooled = rates(counts, ["offer"])["rate"]
print(f"\nPooled effect of the offer:    {pooled[1] - pooled[0]:+.3f}")
for g in ["A", "B"]:
    r = rates(counts.filter(pl.col("group") == g), ["offer"])["rate"]
    print(f"Effect within group {g}:        {r[1] - r[0]:+.3f}")

# Standardisation: average the within-group effects, weighting by each group's share of ALL visitors
share = counts.group_by("group").agg(pl.col("visitors").sum()).with_columns(w=pl.col("visitors") / 2000)
eff = {g: rates(counts.filter(pl.col("group") == g), ["offer"])["rate"].diff()[1] for g in ["A", "B"]}
standardised = sum(eff[g] * share.filter(pl.col("group") == g)["w"].item() for g in ["A", "B"])
print(f"Standardised effect:           {standardised:+.3f}")

print("""
Story 1 - group = REGION (A = low-converting region). Marketing showed the offer mostly in A.
          region -> offer, region -> conversion: a confounder.  Believe the BY-GROUP table.
Story 2 - group = DELIVERY SPEED CHOSEN AT CHECKOUT (A = standard). The offer pushed buyers
          to standard delivery: offer -> group -> conversion, a mediator. Believe the POOLED table.""")
```

```bash
python -m day19.simpson_two_stories
```

```
Pooled:
│ offer ┆ conversions ┆ visitors ┆ rate │
│ 0     ┆ 310         ┆ 1000     ┆ 0.31 │
│ 1     ┆ 240         ┆ 1000     ┆ 0.24 │
By group:
│ group ┆ offer ┆ conversions ┆ visitors ┆ rate │
│ A     ┆ 0     ┆ 30          ┆ 200      ┆ 0.15 │
│ A     ┆ 1     ┆ 160         ┆ 800      ┆ 0.2  │
│ B     ┆ 0     ┆ 280         ┆ 800      ┆ 0.35 │
│ B     ┆ 1     ┆ 80          ┆ 200      ┆ 0.4  │

Pooled effect of the offer:    -0.070
Effect within group A:        +0.050
Effect within group B:        +0.050
Standardised effect:           +0.050
```

The offer raises conversion by 5 points in **both** groups, yet pooled it appears to *lower* conversion by 7 points. The reason is in the visitor counts: 80% of offer visitors are in group A, where everyone converts less.

Now the two stories, with **identical** numbers:

| | Story 1: group = region | Story 2: group = delivery speed chosen at checkout |
|---|---|---|
| DAG | region → offer, region → conversion | offer → delivery choice → conversion |
| Role of group | **confounder** | **mediator** |
| Right table | **by group**: +5 points | **pooled**: −7 points |
| Why | marketing showed the offer mostly in the weak region | the offer pushed people towards standard delivery, which converts worse: that's part of the offer's effect |

In Story 2 the offer really does reduce conversion overall: it works on everyone, but by changing their delivery choice it also triggers a mechanism that hurts. Stratifying by delivery choice would hide that harm.

---

## 19.3 Standardisation: averaging the parts properly (45 min)

**Concept.** When the subgroup variable is a confounder, the overall effect is the within-group effects averaged over the **whole population's** mix of groups, not the treated group's mix. This is **standardisation** (also called the adjustment formula, or g-formula):

```text
effect = Σ over groups g:  (effect within g) × (share of all units in g)
```

In the script, both groups are 50% of visitors and both effects are +0.05, so the standardised effect is +0.05. With different within-group effects, the weights would matter, and choosing *whose* mix to weight by is choosing the estimand: weight by all units for the ATE, by treated units for the ATT.

Standardisation is backdoor adjustment written as arithmetic. Regression with group dummies does the same thing approximately, with weights you don't choose.

**Try it.** Change group A's offer conversions from 160 to 120 (so the offer *lowers* conversion in A) and rerun. What are the within-group effects, the pooled effect, and the standardised effect?

<details>
<summary>Solution</summary>

Within A: 120/800 − 30/200 = 0.15 − 0.15 = **0.000**. Within B: **+0.050**. Pooled: 200/1000 − 310/1000 = **−0.110**. Standardised: 0.5 × 0 + 0.5 × 0.05 = **+0.025**.

Now the groups disagree with each other, and the pooled number is further from both. Under Story 1, the honest summary is "+2.5 points on average, entirely from region B".

</details>

---

## 19.4 Simpson's paradox as a DAG question (40 min)

The resolution, due to Judea Pearl: **stratify by a variable if and only if it's in a valid adjustment set.**

- If the third variable is a **confounder** (it causes the treatment and the outcome): stratify.
- If it's a **mediator** or a **collider** (the treatment causes it): don't.
- If it's a **pre-treatment collider** (M-bias, Day 15): don't, even though it comes first in time.

No statistical test distinguishes Story 1 from Story 2: the counts are identical. What distinguishes them is **when the group variable was determined, and by what**. That's knowledge about the process, written down as a DAG. This is the strongest argument for drawing DAGs before looking at results: once you've seen that the pooled table says "−7" and the stratified table says "+5", it's very tempting to pick the story that gives the answer you hoped for.

---

## 19.5 Day 19 mini-project: NorthMart's segments (70 min)

**Goal.** NorthMart's naive email comparison, pooled and by loyalty segment, has a milder version of the paradox. Explain it, then decide whether stratifying by loyalty fixes the bias.

Create `day19/northmart_segments.py`:

```python
# day19/northmart_segments.py   (python -m day19.northmart_segments)
"""
Mini-project: NorthMart's naive email comparison, pooled and by loyalty segment.
The pooled gap is LARGER than the gap in either segment. Why, and which number is closer to the truth?
"""
import polars as pl

from causalkit.data import load_customers

truth = pl.read_parquet("data/truth/potential_outcomes.parquet").select("customer_id", "tau")
df = load_customers().join(truth, on="customer_id")


def naive(d: pl.DataFrame) -> float:
    return d.filter(pl.col("email") == 1)["post_orders"].mean() - d.filter(pl.col("email") == 0)["post_orders"].mean()


rows = [{"segment": "all customers", "customers": df.height,
         "share_emailed": round(df["email"].mean(), 3), "naive_gap": round(naive(df), 3)}]
for flag, label in [(True, "loyalty members"), (False, "non-members")]:
    seg = df.filter(pl.col("is_loyalty") == flag)
    rows.append({"segment": label, "customers": seg.height,
                 "share_emailed": round(seg["email"].mean(), 3), "naive_gap": round(naive(seg), 3)})
table = pl.DataFrame(rows)
print(table)

# Standardise the segment gaps to the whole customer base
w = table.filter(pl.col("segment") != "all customers")
standardised = (w["naive_gap"] * w["customers"]).sum() / w["customers"].sum()
print(f"\nStandardised over segments:   {standardised:+.3f}")
print(f"True ATE:                     {df['tau'].mean():+.3f}")
```

```bash
python -m day19.northmart_segments
```

```
│ segment         ┆ customers ┆ share_emailed ┆ naive_gap │
│ all customers   ┆ 4000      ┆ 0.32          ┆ 1.711     │
│ loyalty members ┆ 1194      ┆ 0.526         ┆ 1.62      │
│ non-members     ┆ 2806      ┆ 0.232         ┆ 1.466     │

Standardised over segments:   +1.512
True ATE:                     +0.319
```

Answer in `docs/ESTIMANDS.md`, under **"Segments and Simpson"**:

1. Why is the pooled gap (1.71) larger than the gap in **either** segment?
2. Is loyalty a confounder or a mediator of the email's effect? So should we stratify?
3. Stratifying moves the estimate from 1.71 to 1.51. Why is it still nearly five times the truth?

<details>
<summary>Model answers</summary>

1. Loyalty members are both more often emailed (53% vs 23%) and heavier buyers. Pooling mixes "emailed vs not" with "member vs not", adding a between-segment gap on top of the within-segment gaps.
2. Loyalty is decided before the email and influences who got it: a **confounder**. Stratifying by it (or adjusting for it) is right.
3. Loyalty blocks only one of the two backdoor paths (Day 15). The other, `email ← intensity → post_orders`, runs *within* each segment, and no observed variable blocks it. Stratifying on a valid-but-insufficient variable removes some bias, never all of it.

</details>

---

## 19.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 19: Simpson's paradox, standardisation, NorthMart segments"
git push
```

### Day 19 checklist

- [ ] I can explain how Simpson's paradox arises from group sizes and baselines
- [ ] I can compute a standardised effect and explain how the weights define the estimand
- [ ] I can decide, with a DAG, whether to report the pooled or the stratified result
- [ ] I can explain why stratifying on one confounder may not be enough

---

# DAY 20 — Capstone: Defending the DAG

| Time | Block |
|---|---|
| 0:00 – 0:30 | 20.1 The brief and how it is assessed |
| 0:30 – 1:15 | 20.2 Read the scenario and the data |
| 1:15 – 1:30 | Break |
| 1:30 – 2:15 | 20.3 Draw the DAG and find the adjustment set |
| 2:15 – 2:50 | 20.4 Confront the DAG with the data |
| 2:50 – 3:05 | Break |
| 3:05 – 3:45 | 20.5 Estimate, and show what the alternatives do |
| 3:45 – 4:40 | 20.6 Write the memo |
| 4:40 – 5:00 | 20.7 Answer key, final commit and Module 2 retrospective |

---

## 20.1 The brief and how it is assessed (30 min)

**The brief.** NorthMart's pricing lead writes:

> *"For the last month our pricing algorithm has offered personalised discounts to about a third of customers. Finance looked at the numbers: discounted customers spent about the same as everyone else, so they want to switch the programme off. I think the comparison is unfair. Before we decide, I need someone to tell me (a) what the discount actually did to spending, (b) which comparison is the fair one, and (c) why I should trust your answer over Finance's. Two pages, please."*

**Deliverables.**

| # | Deliverable | Where |
|---|---|---|
| D1 | A DAG with one written justification per arrow | `day20/my_dag.py`, `figures/day20_pricing_dag.png` |
| D2 | Backdoor paths and the minimal adjustment set | output of `day20/my_dag.py` |
| D3 | A test of every implication of the DAG against the data | `day20/test_my_dag.py` |
| D4 | The estimate, plus what four tempting alternatives give | `day20/estimate.py` |
| D5 | A two-page memo for the pricing lead | `docs/DAG_MEMO.md` |

**Assessment rubric.**

| Criterion | Weight | Excellent |
|---|---|---|
| Estimand | 10% | States the estimand completely (Day 12 card) and matches it to the decision |
| DAG | 25% | Every arrow justified from the brief; latent variables included; no unjustified arrows |
| Identification | 20% | Correct backdoor paths and adjustment set; explains why each excluded variable is excluded |
| Testing | 15% | Implications tested; results interpreted (what "not rejected" does and doesn't mean) |
| Estimate | 15% | Correct specification, with uncertainty; alternatives shown and explained |
| Memo | 15% | A non-technical reader can follow the argument and act on it |

---

## 20.2 Read the scenario and the data (45 min)

Generate the data (the script is provided; **don't read its code** before you've drawn your DAG, it contains the answer):

```bash
python data/make_pricing_scenario.py
```

```
Wrote 6,000 customers to data/scenario/pricing.csv
```

**What we know about the business** (from interviews with the pricing, CRM and support teams):

1. **The pricing algorithm** decides who gets a discount. Its documentation lists three inputs: the customer's **cart abandonment rate**, their **loyalty** status (members get fewer discounts: "they buy anyway"), and whether they live in a **high-traffic region**.
2. **Cart abandonment** reflects how **price-sensitive** a customer is. Price-sensitive customers also spend less in general. Price sensitivity itself is not measured anywhere.
3. **Loyalty** membership is more common among long-tenured customers. Members spend more.
4. A separate **spring email** was sent mostly to loyalty members. Emails raise spending.
5. High-traffic regions have higher spending (more stores nearby, faster delivery).
6. A discount is delivered as a **coupon**; only customers offered a discount can use one. The pricing team believes the discount works by getting people to redeem the coupon.
7. **Customer support** contacts happen after purchases. Support sees more contacts from discounted customers (coupon questions) and from big spenders.

**The data** (`data/scenario/pricing.csv`), one row per customer, 30-day window:

| Column | Meaning |
|---|---|
| `tenure_years` | years since signup |
| `loyalty` | 1 if loyalty member |
| `high_traffic_region` | 1 if the customer's region is high-traffic |
| `cart_abandon_rate` | share of carts abandoned in the previous 90 days |
| `email` | 1 if the customer got the spring email |
| `discount` | **treatment**: 1 if offered a personalised discount |
| `coupon_used` | 1 if a discount coupon was redeemed |
| `support_contact` | 1 if the customer contacted support during the window |
| `spend_30d` | **outcome**: spending in the 30-day window ($) |

**Before any code:** write the estimand card (Day 12 template) for the pricing lead's question (a). Then sketch the DAG on paper.

<details>
<summary>Model estimand card</summary>

```markdown
### Decision: keep or stop the personalised discount programme
- **Unit:** customer
- **Treatment vs comparison:** offered a personalised discount vs not offered one
- **Outcome and horizon:** spending in the 30-day window, in dollars
- **Population:** customers the algorithm chose (ATT), since the decision is about this programme
- **Summary measure:** average difference in dollars per customer
- **Note:** the effect in the simulation is the same for everyone, so ATT = ATE here;
  in general, keep them apart.
```

</details>

---

## 20.3 Draw the DAG and find the adjustment set (45 min)

Translate each numbered fact from 20.2 into arrows. Each arrow needs a comment pointing to its fact; each *missing* arrow is a claim too (for example: does the email affect who gets a discount? The documentation says the algorithm has three inputs, so no).

Create `day20/my_dag.py`. This model answer follows the brief point by point:

```python
# day20/my_dag.py   (python -m day20.my_dag)
"""Capstone step 1: the DAG, written from the brief. One comment per arrow = one assumption."""
from causalkit.dags import (backdoor_paths, describe_path, draw_dag, make_dag,
                            minimal_adjustment_sets)

T, Y = "discount", "spend_30d"

dag = make_dag([
    ("tenure_years", "loyalty"),            # longer-tenured customers join the programme
    ("tenure_years", "spend_30d"),          # ...and spend more
    ("loyalty", "email"),                   # the spring email targeted loyalty members
    ("loyalty", "discount"),                # the pricing algorithm uses loyalty (documented)
    ("loyalty", "spend_30d"),
    ("high_traffic_region", "discount"),    # ...and regional traffic (documented)
    ("high_traffic_region", "spend_30d"),
    ("price_sensitivity", "cart_abandon_rate"),   # sensitive shoppers abandon carts
    ("price_sensitivity", "spend_30d"),           # ...and spend less
    ("cart_abandon_rate", "discount"),      # ...and the algorithm targets abandoners (documented)
    ("email", "spend_30d"),
    ("discount", "coupon_used"),            # a coupon exists only if a discount was offered
    ("coupon_used", "spend_30d"),           # the discount works through the coupon
    ("discount", "support_contact"),        # discounted customers contact support more
    ("spend_30d", "support_contact"),       # ...and so do big spenders (contact happens AFTER buying)
], latent={"price_sensitivity"})

if __name__ == "__main__":
    print("Backdoor paths from discount to spend:")
    for p in backdoor_paths(dag, T, Y):
        print("  ", describe_path(dag, p))
    print("\nMinimal adjustment sets:", minimal_adjustment_sets(dag, T, Y))
    draw_dag(dag, "figures/day20_pricing_dag.png", "Discount programme: assumed DAG", treatment=T, outcome=Y)
    print("Saved figures/day20_pricing_dag.png")
```

```bash
python -m day20.my_dag
```

```
Backdoor paths from discount to spend:
   discount <- loyalty <- tenure_years -> spend_30d
   discount <- loyalty -> email -> spend_30d
   discount <- loyalty -> spend_30d
   discount <- high_traffic_region -> spend_30d
   discount <- cart_abandon_rate <- price_sensitivity -> spend_30d

Minimal adjustment sets: [{'cart_abandon_rate', 'high_traffic_region', 'loyalty'}]
Saved figures/day20_pricing_dag.png
```

Every backdoor path goes through one of the algorithm's three documented inputs. That's Day 15's lesson in action: **because the assignment rule is known and its inputs are recorded**, the effect is identifiable even though price sensitivity is latent. Compare with NorthMart's email, where marketing's judgement of "heavy shoppers" was never recorded.

Note what's *not* in the set: `coupon_used` (a mediator), `support_contact` (a collider and descendant of the treatment), `email` and `tenure_years` (blocked through loyalty; allowed but not needed).

---

## 20.4 Confront the DAG with the data (35 min)

Create `day20/test_my_dag.py`:

```python
# day20/test_my_dag.py   (python -m day20.test_my_dag)
"""Capstone step 2: confront the DAG with the data. Every rejected claim is an arrow to rethink."""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.dags import implied_independencies
from day20.my_dag import dag

df = pl.read_csv("data/scenario/pricing.csv").to_pandas()

claims = implied_independencies(dag, max_conditioning=2)
print(f"The DAG implies {len(claims)} testable independences. Testing each (alpha = 0.01):\n")
rejected = 0
for a, b, z in claims:
    fit = smf.ols(f"{b} ~ {a}" + "".join(f" + {v}" for v in sorted(z)), df).fit(cov_type="HC2")
    p = fit.pvalues[a]
    rejected += p < 0.01
    claim = f"{a} _||_ {b}" + (f" | {', '.join(sorted(z))}" if z else "")
    print(f"  {claim:<70} p = {p:.3f} {'REJECTED' if p < 0.01 else ''}")
print(f"\n{rejected} of {len(claims)} claims rejected.")
```

```bash
python -m day20.test_my_dag
```

```
The DAG implies 21 testable independences. Testing each (alpha = 0.01):

  cart_abandon_rate _||_ coupon_used | discount                          p = 0.979
  cart_abandon_rate _||_ email                                           p = 0.790
  ...
  discount _||_ email | loyalty                                          p = 0.194
  discount _||_ tenure_years | loyalty                                   p = 0.270
  ...
  support_contact _||_ tenure_years | discount, spend_30d                p = 0.691

0 of 21 claims rejected.
```

Two of these deserve a sentence each in your memo:

- `discount _||_ email | loyalty`: the data agree that the algorithm doesn't use the email (once loyalty is accounted for). That supports leaving `email` out of the adjustment set.
- `discount _||_ tenure_years | loyalty`: tenure matters for discounts only through loyalty, as the documentation implies.

"Not rejected" means the data **don't contradict** the DAG. It doesn't prove it: a DAG with extra arrows would pass too, and the latent confounder's arrows can't be tested at all.

**Try it.** A colleague insists "the algorithm ignores loyalty". Remove the arrow `loyalty → discount` and test only the **new** claims that version makes. Is the colleague's DAG consistent with the data?

<details>
<summary>Solution</summary>

```python
# day20/colleague_dag.py   (python -m day20.colleague_dag)
import polars as pl
import statsmodels.formula.api as smf

from causalkit.dags import implied_independencies
from day20.my_dag import dag

colleague = dag.copy()
colleague.remove_edge("loyalty", "discount")      # "the algorithm doesn't look at loyalty"

df = pl.read_csv("data/scenario/pricing.csv").to_pandas()
new_claims = set(implied_independencies(colleague, 2)) - set(implied_independencies(dag, 2))
for a, b, z in sorted(new_claims):
    fit = smf.ols(f"{b} ~ {a}" + "".join(f" + {v}" for v in sorted(z)), df).fit(cov_type="HC2")
    claim = f"{a} _||_ {b}" + (f" | {', '.join(sorted(z))}" if z else "")
    print(f"{claim:<55} p = {fit.pvalues[a]:.4f}")
```

```
coupon_used _||_ email                                  p = 0.3385
coupon_used _||_ loyalty                                p = 0.0000
coupon_used _||_ tenure_years                           p = 0.0000
discount _||_ email                                     p = 0.0051
discount _||_ loyalty                                   p = 0.0000
discount _||_ tenure_years                              p = 0.0000
```

Five of the six new claims are rejected at the 1% level. Discounts are clearly associated with loyalty, even unconditionally. The colleague's DAG is refuted, and with it the adjustment set it would imply (which would leave out loyalty).

</details>

---

## 20.5 Estimate, and show what the alternatives do (40 min)

Create `day20/estimate.py`:

```python
# day20/estimate.py   (python -m day20.estimate)
"""Capstone step 3: estimate the effect with the DAG's adjustment set, and show what the alternatives do."""
import polars as pl
import statsmodels.formula.api as smf

df = pl.read_csv("data/scenario/pricing.csv").to_pandas()
VALID = "cart_abandon_rate + loyalty + high_traffic_region"

specs = {
    "naive (no controls)": "spend_30d ~ discount",
    "DAG adjustment set": f"spend_30d ~ discount + {VALID}",
    "DAG set + precision controls": f"spend_30d ~ discount + {VALID} + email + tenure_years",
    "forgot cart_abandon_rate": "spend_30d ~ discount + loyalty + high_traffic_region",
    "+ coupon_used (mediator)": f"spend_30d ~ discount + {VALID} + coupon_used",
    "+ support_contact (collider)": f"spend_30d ~ discount + {VALID} + support_contact",
}
rows = []
for name, f in specs.items():
    fit = smf.ols(f, df).fit(cov_type="HC2")
    lo, hi = fit.conf_int().loc["discount"]
    rows.append({"specification": name, "effect ($)": round(fit.params["discount"], 2),
                 "SE": round(fit.bse["discount"], 2), "95% CI": f"[{lo:.1f}, {hi:.1f}]"})
with pl.Config(fmt_str_lengths=40, tbl_width_chars=110):
    print(pl.DataFrame(rows))
```

```bash
python -m day20.estimate
```

```
│ specification                ┆ effect ($) ┆ SE   ┆ 95% CI       │
│ naive (no controls)          ┆ -0.36      ┆ 0.72 ┆ [-1.8, 1.1]  │
│ DAG adjustment set           ┆ 12.83      ┆ 0.73 ┆ [11.4, 14.3] │
│ DAG set + precision controls ┆ 12.84      ┆ 0.69 ┆ [11.5, 14.2] │
│ forgot cart_abandon_rate     ┆ 2.37       ┆ 0.67 ┆ [1.1, 3.7]   │
│ + coupon_used (mediator)     ┆ 0.69       ┆ 0.97 ┆ [-1.2, 2.6]  │
│ + support_contact (collider) ┆ 9.31       ┆ 0.71 ┆ [7.9, 10.7]  │
```

Every row is a plausible analysis someone might present:

- **Naive (Finance's number):** −$0.36, "discounts do nothing". The algorithm sends discounts to price-sensitive, non-member customers who spend less anyway.
- **The DAG's set:** +$12.83 [11.4, 14.3]. Adding `email` and `tenure_years` (outcome predictors, allowed by the DAG) narrows the interval slightly without moving the estimate.
- **Forgetting one input:** +$2.37. Omitting cart abandonment leaves the price-sensitivity path open.
- **Controlling for the coupon:** +$0.69. The coupon *is* the mechanism; controlling for it removes the effect.
- **Controlling for support contacts:** +$9.31. A collider: biased, and confidently so.

**Extension: confirm with DoWhy.** Hand the same DAG to DoWhy (Day 18) and check that it finds the same adjustment set and estimate.

<details>
<summary>Solution</summary>

```python
# day18/dowhy_pricing.py   (python -m day18.dowhy_pricing)
import warnings

import polars as pl
from dowhy import CausalModel

from day20.my_dag import dag

warnings.filterwarnings("ignore")
df = pl.read_csv("data/scenario/pricing.csv").drop("customer_id").to_pandas()
m = CausalModel(data=df, treatment="discount", outcome="spend_30d", graph=dag)
estimand = m.identify_effect()
print("Backdoor variables:", sorted(estimand.get_backdoor_variables()))
est = m.estimate_effect(estimand, method_name="backdoor.linear_regression")
print(f"Effect: {est.value:+.2f}")
```

```
Backdoor variables: ['cart_abandon_rate', 'high_traffic_region', 'loyalty']
Effect: +12.84
```

DoWhy treats `price_sensitivity` as latent (it isn't a column) and still identifies the effect: same tool as Day 18, opposite verdict, because the DAG is different.

</details>

---

## 20.6 Write the memo (55 min)

Write `docs/DAG_MEMO.md` for the pricing lead: two pages at most, plain language, numbers rounded. Use these headings.

1. **The answer.** One paragraph: the effect, with its interval, and the recommendation.
2. **Why Finance's comparison misleads.** Who the algorithm targets, and why those customers would spend less anyway.
3. **The fair comparison.** The DAG (insert `figures/day20_pricing_dag.png`) and, in words, the three variables we compare like-for-like on, and why.
4. **What we deliberately did not control for.** Coupon use and support contacts, each in one sentence.
5. **How we checked.** The 21 tests, and what passing them means and doesn't mean.
6. **What could still be wrong.** At least two specific threats (for example: an undocumented input to the algorithm; spillovers between household members; a 30-day window that may miss pulled-forward purchases), with what would detect each.

<details>
<summary>Model answer for section 1</summary>

> Offering a personalised discount raised spending by about **$13 per customer** over 30 days (95% interval: $11 to $14). Finance's comparison shows no difference because the algorithm, by design, targets customers who spend less to begin with: price-sensitive customers and non-members. Compared like-for-like on the algorithm's own inputs, discounted customers spent about $13 more. Whether that's profitable depends on the discount's cost per customer, which we should now compare against $13 of extra spending (Module 7). We recommend **not** stopping the programme on the basis of the raw comparison.

</details>

---

## 20.7 Answer key, final commit and Module 2 retrospective (20 min)

Once your memo is written, compare your DAG with the process that generated the data. Create `day20/answer_key.py`:

```python
# day20/answer_key.py   (python -m day20.answer_key)
"""Instructor check: compare a student's DAG and adjustment set with the true data-generating process."""
import json
from pathlib import Path

from causalkit.dags import is_valid_adjustment, make_dag, minimal_adjustment_sets
from day20.my_dag import dag as student_dag

truth = json.loads(Path("data/truth/pricing_truth.json").read_text())
true_dag = make_dag([tuple(e) for e in truth["edges"]], latent=set(truth["latent"]))
T, Y = "discount", "spend_30d"

missing = set(true_dag.edges) - set(student_dag.edges)
extra = set(student_dag.edges) - set(true_dag.edges)
print(f"True effect: +${truth['true_effect_of_discount_on_spend']:.0f} per customer over 30 days")
print("Arrows missing from the student DAG:", sorted(missing) or "none")
print("Arrows the true process does not have:", sorted(extra) or "none")
for z in minimal_adjustment_sets(student_dag, T, Y):
    print(f"Student's set {sorted(z)} valid in the TRUE DAG? {is_valid_adjustment(true_dag, T, Y, z)}")
```

```bash
python -m day20.answer_key
```

```
True effect: +$12 per customer over 30 days
Arrows missing from the student DAG: none
Arrows the true process does not have: none
Student's set ['cart_abandon_rate', 'high_traffic_region', 'loyalty'] valid in the TRUE DAG? True
```

If your DAG differs, the last line is what matters most: **an imperfect DAG can still give a valid adjustment set.** Extra arrows that don't create new backdoor paths are harmless; a missing arrow into the treatment is not.

```bash
git add .
git commit -m "Day 20: Defending the DAG capstone"
git push
```

**What you learned in Module 2.**

| Days | Idea | The NorthMart lesson |
|---|---|---|
| 11 | Potential outcomes | Naive +1.71 = effect +0.44 + selection bias +1.27 |
| 12 | Estimands | ATT 0.44 ≠ ATE 0.32 ≠ ATC 0.26: the decision picks the estimand |
| 13 | Randomisation | A coin flip would have estimated 0.32 ± 0.07 |
| 14 | DAGs and d-separation | Data can refute "the email was random", not prove our DAG |
| 15 | Backdoor criterion | No valid set among observed variables; proxies leave 2.4× bias |
| 16 | Colliders | "Active customers only" cuts the effect by 18%, even in an experiment |
| 17 | Mediators, bad controls | A bad control can land closer to the truth by accident |
| 18 | DoWhy | Refuters pass while the estimate is 2.4× the truth; OVB explains the gap |
| 19 | Simpson's paradox | The DAG, not the data, says which table to report |
| 20 | Capstone | With a documented assignment rule, adjustment works: +$12.8 vs +$12 |

**Coming next: Module 3, Experimentation & A/B Testing.** Day 13 showed what randomisation buys. Module 3 shows how to run experiments well: power analysis, stratification, CUPED variance reduction (the `pre_orders` trick from 13.4, done properly), peeking, and interference.

### Day 20 checklist

- [ ] I translated a business narrative into a DAG with a justification for every arrow
- [ ] I derived and defended a minimal adjustment set, including what I left out and why
- [ ] I tested every implication of my DAG and can explain what "not rejected" means
- [ ] I estimated the effect and showed how four tempting alternatives mislead
- [ ] I wrote a memo a non-technical decision-maker can act on
