# Module 2 · Causal Thinking, Estimands & DAGs
## Days 14–17: DAGs, the Backdoor Criterion, Colliders and Bad Controls

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 11–13
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 17:** Students draw a DAG for a business problem, test its implications against data, find valid adjustment sets with the backdoor criterion, and recognise colliders, mediators and bad controls, all checked against NorthMart's known truth.

---

### Where we are

Days 11–13 defined *what* we want (estimands) and showed that randomisation delivers it. NorthMart's email wasn't randomised, so we need to know which differences between emailed and non-emailed customers to adjust for. Regression alone can't tell us: it estimates whatever you put in it. A **DAG** (directed acyclic graph) records our assumptions about *how the data were generated*, and from those assumptions it tells us which adjustments are valid.

### New folders (built over four days)

```
causal-decision-lab/
├── causalkit/                     # shared helpers for Module 2+
│   ├── __init__.py                # empty: makes causalkit importable
│   ├── dags.py                    # build, draw and query DAGs
│   ├── data.py                    # one row per customer from the published mart
│   └── northmart_dag.py           # the NorthMart targeting DAG
├── day14/  first_dag.py, three_structures.py, test_the_dag.py
├── day15/  backdoor_paths.py, adjustment_estimates.py, four_dags.py
├── day16/  berkson.py, active_customers_only.py, selection_audit.py
└── day17/  mediator.py, frontdoor.py, bad_controls.py, control_audit.py
```

---

# DAY 14 — Drawing DAGs

| Time | Block |
|---|---|
| 0:00 – 0:40 | 14.1 What a DAG says, and what it doesn't |
| 0:40 – 1:30 | 14.2 Tools: causalkit and the customer table |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 14.3 NorthMart's targeting DAG |
| 2:30 – 3:20 | 14.4 Three building blocks: chain, fork, collider |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 14.5 Day 14 mini-project: let the data test your DAG |
| 4:45 – 5:00 | 14.6 Save your work with Git |

---

## 14.1 What a DAG says, and what it doesn't (40 min)

**Concept.** A DAG is a set of variables (**nodes**) and arrows (**edges**). An arrow A → B says "A is a direct cause of B" (for some units, by some amount). *Directed*: arrows have a direction. *Acyclic*: you can't follow arrows back to where you started; nothing causes itself.

The most important information in a DAG is the arrows that are **missing**. A missing arrow is a strong claim: "A has no direct effect on B". Missing arrows are what make causal effects estimable, and they're what data can sometimes refute (14.5).

Vocabulary, with NorthMart examples:

| Term | Meaning | Example |
|---|---|---|
| parent / child | direct cause / direct effect | `loyalty` is a parent of `email` |
| ancestor / descendant | cause / effect through any chain | `loyalty` is an ancestor of `post_orders` |
| path | any route along edges, ignoring direction | `email ← intensity → post_orders` |
| latent node | a variable that matters but isn't in the data | `intensity`: how much a customer likes to shop |

A DAG says nothing about the *size* or *shape* of effects (linear, multiplicative…), only which variables affect which.

> **dbt also draws a DAG** (Day 7): there, arrows mean "is built from"; here they mean "causes". Same mathematics, different meaning.

---

## 14.2 Tools: causalkit and the customer table (50 min)

We'll use DAGs every day from now on, so we put the helpers in a small shared package, `causalkit`. Create the folder and an empty `causalkit/__init__.py`:

```bash
mkdir -p causalkit day14 day15 day16 day17
touch causalkit/__init__.py
```

**`causalkit/dags.py`** builds DAGs with networkx, draws them, and (from Day 15) finds backdoor paths and adjustment sets. Read the docstring of each function; you'll use them all this week.

```python
"""
Small DAG helpers for Module 2, built on networkx. Written to be read, not just used.

    from causalkit.dags import make_dag, draw_dag, backdoor_paths, is_valid_adjustment, minimal_adjustment_sets
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def make_dag(edges: list[tuple[str, str]], latent: set[str] | None = None) -> nx.DiGraph:
    """Build a DAG from (cause, effect) pairs. Latent (unobserved) nodes are marked."""
    g = nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"Not a DAG: there is a cycle {nx.find_cycle(g)}")
    for node in g.nodes:
        g.nodes[node]["latent"] = node in (latent or set())
    return g


def observed(g: nx.DiGraph) -> set[str]:
    return {n for n in g.nodes if not g.nodes[n]["latent"]}


def draw_dag(g: nx.DiGraph, path: str, title: str = "", pos: dict | None = None,
             treatment: str | None = None, outcome: str | None = None) -> None:
    """Save a picture of the DAG. Latent nodes are dashed; treatment and outcome are coloured."""
    pos = pos or _layered_layout(g)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colours = []
    for n in g.nodes:
        colours.append("#3987e5" if n == treatment else "#d95926" if n == outcome else "#e9eef5")
    nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=18, width=1.6, node_size=2600,
                           edge_color="#4a5568", connectionstyle="arc3,rad=0.05")
    for n, c in zip(g.nodes, colours):
        nx.draw_networkx_nodes(g, pos, nodelist=[n], node_color=c, node_size=2600, ax=ax,
                               edgecolors="#1f2937", linewidths=1.5,
                               node_shape="o")
        if g.nodes[n]["latent"]:
            ax.scatter(*pos[n], s=2600, facecolors="white", edgecolors="#1f2937",
                       linewidths=1.5, linestyle="--", zorder=3)
    labels = {n: n.replace("_", "\n") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=8.5, ax=ax,
                            font_color="#111827")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _layered_layout(g: nx.DiGraph) -> dict:
    """Left-to-right layout by topological generation (causes on the left)."""
    for layer, nodes in enumerate(nx.topological_generations(g)):
        for node in nodes:
            g.nodes[node]["layer"] = layer
    return nx.multipartite_layout(g, subset_key="layer", align="vertical", scale=2)


def backdoor_paths(g: nx.DiGraph, treatment: str, outcome: str) -> list[list[str]]:
    """Every path from treatment to outcome that starts with an arrow INTO the treatment."""
    skeleton = g.to_undirected()
    paths = []
    for p in nx.all_simple_paths(skeleton, treatment, outcome):
        if g.has_edge(p[1], p[0]):            # first step goes against an arrow: p[1] -> treatment
            paths.append(p)
    return paths


def path_is_blocked(g: nx.DiGraph, path: list[str], z: set[str]) -> bool:
    """d-separation for one path: blocked by a non-collider in Z, or a collider not in Z (nor its descendants)."""
    for a, m, b in zip(path, path[1:], path[2:]):
        collider = g.has_edge(a, m) and g.has_edge(b, m)
        if collider:
            if m not in z and not (nx.descendants(g, m) & z):
                return True
        elif m in z:
            return True
    return False


def describe_path(g: nx.DiGraph, path: list[str]) -> str:
    out = path[0]
    for a, b in zip(path, path[1:]):
        out += f" -> {b}" if g.has_edge(a, b) else f" <- {b}"
    return out


def is_valid_adjustment(g: nx.DiGraph, treatment: str, outcome: str, z: set[str]) -> bool:
    """Backdoor criterion: Z has no descendant of the treatment and blocks every backdoor path."""
    if nx.descendants(g, treatment) & z:
        return False
    return all(path_is_blocked(g, p, z) for p in backdoor_paths(g, treatment, outcome))


def minimal_adjustment_sets(g: nx.DiGraph, treatment: str, outcome: str) -> list[set[str]]:
    """All smallest valid adjustment sets made only of OBSERVED variables (brute force: fine for small DAGs)."""
    candidates = sorted(observed(g) - {treatment, outcome} - nx.descendants(g, treatment))
    found: list[set[str]] = []
    for k in range(len(candidates) + 1):
        for combo in combinations(candidates, k):
            z = set(combo)
            if any(f <= z for f in found):
                continue                      # a smaller valid set is already inside
            if is_valid_adjustment(g, treatment, outcome, z):
                found.append(z)
    return found


def implied_independencies(g: nx.DiGraph, max_conditioning: int = 1) -> list[tuple[str, str, frozenset]]:
    """Testable claims of the DAG: pairs of OBSERVED, non-adjacent variables that are
    d-separated given some small set of observed variables."""
    obs = sorted(observed(g))
    claims = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if g.has_edge(a, b) or g.has_edge(b, a):
                continue
            others = [v for v in obs if v not in (a, b)]
            for k in range(max_conditioning + 1):
                hit = next((frozenset(z) for z in combinations(others, k)
                            if nx.is_d_separator(g, {a}, {b}, set(z))), None)
                if hit is not None:
                    claims.append((a, b, hit))
                    break
    return claims
```

**`causalkit/data.py`** builds one row per customer from Module 1's **published** mart: pre-period features plus post-period outcomes (weeks 9–16):

```python
"""One row per customer for Module 2, built from Module 1's published Causal Data Mart."""
from pathlib import Path

import polars as pl

PUBLISHED = Path("data/published")


def load_customers() -> pl.DataFrame:
    """Pre-period features + post-period outcomes (weeks 9-16) for every customer."""
    features = pl.read_parquet(PUBLISHED / "customer_features.parquet")
    post = (pl.read_parquet(PUBLISHED / "panel_customer_week.parquet")
            .filter(pl.col("post") == 1)
            .group_by("customer_id")
            .agg(post_orders=pl.col("n_orders").sum(),
                 post_revenue=pl.col("revenue").sum().round(2),
                 post_active_weeks=(pl.col("n_orders") > 0).sum()))
    return (features.join(post, on="customer_id")
            .rename({"treated": "email"})
            .sort("customer_id"))
```

> Module 2 reads only `data/published/`. If it's missing, run the Module 1 flow first: `python pipeline/northmart_flow.py` (with the API running).

---

## 14.3 NorthMart's targeting DAG (45 min)

**Concept.** Write the DAG from **knowledge**, not from data: how did marketing choose recipients? What drives orders? Here's what NorthMart's marketing team told us (and the Module 1 generator confirms):

- Loyalty members shop more, and marketing favoured them.
- Marketing also favoured heavy shoppers, judged by recent activity.
- How much someone likes to shop (**intensity**) drives orders both before and after the email. Nobody measures intensity directly.

Create `day14/first_dag.py`:

```python
# day14/first_dag.py   (run with:  python -m day14.first_dag)
"""Write down how NorthMart's email was targeted, as a DAG."""
import networkx as nx

from causalkit.dags import draw_dag, make_dag

g = make_dag(
    [
        ("loyalty", "intensity"),     # loyalty members shop more
        ("loyalty", "email"),         # marketing favoured loyalty members
        ("intensity", "email"),       # ...and heavy shoppers
        ("intensity", "pre_orders"),  # heavy shoppers ordered more before
        ("intensity", "post_orders"), # ...and after
        ("email", "post_orders"),     # the effect we want
    ],
    latent={"intensity"},             # nobody observes "shopping intensity" directly
)

print("Nodes:", sorted(g.nodes))
print("Parents of email:      ", sorted(g.predecessors("email")))
print("Children of intensity: ", sorted(g.successors("intensity")))
print("Ancestors of post_orders:", sorted(nx.ancestors(g, "post_orders")))
print("Descendants of email:    ", sorted(nx.descendants(g, "email")))

draw_dag(g, "figures/day14_northmart_dag.png", "NorthMart: how the SPRING26 email was targeted",
         treatment="email", outcome="post_orders")
print("Saved figures/day14_northmart_dag.png")
```

```bash
python -m day14.first_dag
```

```
Nodes: ['email', 'intensity', 'loyalty', 'post_orders', 'pre_orders']
Parents of email:       ['intensity', 'loyalty']
Children of intensity:  ['email', 'post_orders', 'pre_orders']
Ancestors of post_orders: ['email', 'intensity', 'loyalty']
Descendants of email:     ['post_orders']
Saved figures/day14_northmart_dag.png
```

Open the figure. The text version:

```
                 ┌──────────► pre_orders
                 │
  loyalty ──► intensity ─────────────────► post_orders
     │           │                            ▲
     │           ▼                            │
     └───────► email ─────────────────────────┘
```

`intensity` is dashed in the figure: it's **latent**. Notice that `pre_orders` is a *child* of intensity, not a cause of the email: marketing didn't email people *because of* their pre-period order count, but because of how much they shop, which pre-period orders reflect imperfectly. That distinction will matter tomorrow.

---

## 14.4 Three building blocks: chain, fork, collider (50 min)

**Concept.** Every path in every DAG is built from three-node pieces. What happens to the association between the ends when you **control for** (condition on) the middle node?

| Structure | Picture | x and y associated? | After controlling for m |
|---|---|---|---|
| **Chain** | x → m → y | yes (x causes y through m) | **blocked** |
| **Fork** | x ← m → y | yes (common cause) | **blocked** |
| **Collider** | x → m ← y | **no** | **opened**: spurious association |

The first two match intuition. The collider is the surprise, and the source of many published mistakes.

Create `day14/three_structures.py`:

```python
# day14/three_structures.py   (python -m day14.three_structures)
"""Chain, fork, collider: what happens to an association when you condition on the middle node."""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

rng = np.random.default_rng(14)
n = 20_000


def slope(df: pl.DataFrame, formula: str, var: str = "x") -> float:
    return smf.ols(formula, data=df.to_pandas()).fit().params[var]


# Chain:    x -> m -> y      (email -> site visits -> orders)
x = rng.normal(size=n); m = x + rng.normal(size=n); y = m + rng.normal(size=n)
chain = pl.DataFrame({"x": x, "m": m, "y": y})

# Fork:     x <- m -> y      (email <- intensity -> orders), no arrow from x to y
m = rng.normal(size=n); x = m + rng.normal(size=n); y = m + rng.normal(size=n)
fork = pl.DataFrame({"x": x, "m": m, "y": y})

# Collider: x -> m <- y      (email -> "ordered in March" <- intensity), x and y unrelated
x = rng.normal(size=n); y = rng.normal(size=n); m = x + y + rng.normal(size=n)
collider = pl.DataFrame({"x": x, "m": m, "y": y})

rows = []
for name, df in [("chain    x -> m -> y", chain), ("fork     x <- m -> y", fork), ("collider x -> m <- y", collider)]:
    rows.append({"structure": name,
                 "y on x": round(slope(df, "y ~ x"), 3),
                 "y on x, controlling m": round(slope(df, "y ~ x + m"), 3)})
with pl.Config(fmt_str_lengths=30):
    print(pl.DataFrame(rows))
```

```
│ structure            ┆ y on x ┆ y on x, controlling m │
│ chain    x -> m -> y ┆ 1.023  ┆ 0.021                 │
│ fork     x <- m -> y ┆ 0.497  ┆ 0.004                 │
│ collider x -> m <- y ┆ 0.001  ┆ -0.5                  │
```

- **Chain:** controlling for the mediator removes x's effect (it all went through m).
- **Fork:** controlling for the common cause removes the spurious association. This is why we control for confounders.
- **Collider:** x and y are independent, but among units with the same m, a high x must be offset by a low y, so a strong negative association appears out of nothing.

These three rules are **d-separation**. A path is blocked if it contains a chain or fork whose middle is controlled for, or a collider whose middle (and all its descendants) is *not* controlled for. `path_is_blocked` in causalkit implements exactly these rules.

---

## 14.5 Day 14 mini-project: let the data test your DAG (70 min)

**Concept.** A DAG can't be proven by data, but it can be **refuted**. Every missing arrow implies that some variables are independent, possibly given others. If the data show a clear association where the DAG says there is none, the DAG is wrong.

`implied_independencies` (end of `causalkit/dags.py`) lists every such claim between observed variables. We test each with a regression: under the claim, the coefficient should be zero.

We compare two candidate DAGs: ours (A), and the one a hurried analyst implicitly assumes when comparing groups directly (B: "the email was sent at random"). Both include `age`, which we claim affects nothing here.

Create `day14/test_the_dag.py`:

```python
# day14/test_the_dag.py   (python -m day14.test_the_dag)
"""Mini-project: a DAG makes testable claims. Check them against NorthMart's data."""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.dags import implied_independencies, make_dag
from causalkit.data import load_customers

df = load_customers().with_columns(pl.col("email").cast(pl.Int8), pl.col("is_loyalty").alias("loyalty")).to_pandas()

CANDIDATES = {
    "A: targeted email (our DAG)": make_dag(
        [("loyalty", "intensity"), ("loyalty", "email"), ("intensity", "email"),
         ("intensity", "pre_orders"), ("intensity", "post_orders"), ("email", "post_orders")],
        latent={"intensity"}),
    "B: 'the email was random'": make_dag(
        [("loyalty", "intensity"), ("intensity", "pre_orders"), ("intensity", "post_orders"),
         ("email", "post_orders")],
        latent={"intensity"}),
}
for g in CANDIDATES.values():
    g.add_node("age", latent=False)            # age is in the data but (we claim) affects nothing here


def test(a: str, b: str, z: frozenset) -> tuple[float, float]:
    """Regress b on a (+ z). Under the claim 'a independent of b given z', a's coefficient is 0."""
    formula = f"{b} ~ {a}" + "".join(f" + {v}" for v in sorted(z))
    fit = smf.ols(formula, data=df).fit(cov_type="HC2")
    return fit.params[a], fit.pvalues[a]


for name, g in CANDIDATES.items():
    print(f"\n{name}")
    for a, b, z in implied_independencies(g):
        coef, p = test(a, b, z)
        claim = f"{a} _||_ {b}" + (f" | {', '.join(sorted(z))}" if z else "")
        verdict = "consistent" if p > 0.01 else "REJECTED by the data"
        print(f"  {claim:<28} coef {coef:+.4f}  p = {p:.3f}  -> {verdict}")
```

```bash
python -m day14.test_the_dag
```

```
A: targeted email (our DAG)
  age _||_ email               coef +0.0005  p = 0.268  -> consistent
  age _||_ loyalty             coef +0.0001  p = 0.795  -> consistent
  age _||_ post_orders         coef +0.0013  p = 0.522  -> consistent
  age _||_ pre_orders          coef +0.0029  p = 0.150  -> consistent

B: 'the email was random'
  ...
  email _||_ loyalty           coef +0.2830  p = 0.000  -> REJECTED by the data
  email _||_ pre_orders        coef +1.3822  p = 0.000  -> REJECTED by the data
```

The "random email" DAG is refuted: emailed customers had 1.38 more pre-period orders. Our DAG survives, but notice how *few* claims it makes. Because intensity is latent, most of its independences involve intensity and can't be tested. **A DAG with a latent confounder is hard to refute and hard to use**: tomorrow shows the second half of that sentence.

**Try it.** Add an arrow `age → email` to DAG A (as if marketing targeted by age). Which claim disappears, and does the data say anything about whether the arrow is needed?

<details>
<summary>Solution</summary>

Add `("age", "email")` to A's edge list. Two claims disappear, and only `age _||_ loyalty` and `age _||_ pre_orders` remain:

- `age _||_ email` goes, obviously: there is now an arrow.
- `age _||_ post_orders` goes too: age now affects orders *through* the email.
- You might expect `age _||_ post_orders | email` to appear instead. It doesn't, because `email` is a **collider** on the path `age → email ← intensity → post_orders`: controlling for it opens that path.

The data had p = 0.268 for `age _||_ email`: no evidence that marketing used age. Adding arrows never makes a DAG easier to refute; it only removes claims. Keep arrows out unless knowledge or data says they belong.

</details>

---

## 14.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 14: causalkit, NorthMart DAG, d-separation, testable implications"
git push
```

### Day 14 checklist

- [ ] I can read a DAG: parents, children, ancestors, descendants, paths, latent nodes
- [ ] I can write a DAG from domain knowledge and justify each arrow (and each missing arrow)
- [ ] I can state what controlling for the middle of a chain, fork or collider does
- [ ] I can list and test a DAG's implied independences, and explain why a test can refute but not prove a DAG

---

# DAY 15 — The Backdoor Criterion and Adjustment Sets

| Time | Block |
|---|---|
| 0:00 – 0:45 | 15.1 Backdoor paths: the routes bias travels |
| 0:45 – 1:30 | 15.2 Which sets block NorthMart's backdoors? |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 15.3 What adjustment does to the estimate |
| 2:45 – 3:20 | 15.4 Record how treatment was assigned |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 15.5 Day 15 mini-project: four classic DAGs |
| 4:45 – 5:00 | 15.6 Save your work with Git |

---

## 15.1 Backdoor paths: the routes bias travels (45 min)

**Concept.** Association between treatment T and outcome Y flows along every **open** path between them. Paths that start with an arrow *out of* T (T → … → Y) carry the causal effect. Paths that start with an arrow *into* T (T ← … Y) are **backdoor paths**: they carry association that isn't causation, which is exactly the selection bias of Day 11.

**The backdoor criterion** (Pearl). A set of variables Z is a valid adjustment set for the effect of T on Y if:

1. no variable in Z is a **descendant** of T, and
2. Z **blocks every backdoor path** from T to Y.

If such a Z exists and is observed, comparing treated and untreated units *with the same Z* (and averaging over Z) identifies the causal effect. The criterion turns "what should I control for?" from a matter of taste into a question with an answer, *given the DAG*.

Add the shared NorthMart DAG to `causalkit/northmart_dag.py`, so every later script uses the same one:

```python
"""The NorthMart targeting DAG from Day 14, shared by later days."""
from causalkit.dags import make_dag

EDGES = [
    ("loyalty", "intensity"), ("loyalty", "email"), ("intensity", "email"),
    ("intensity", "pre_orders"), ("intensity", "post_orders"), ("email", "post_orders"),
]


def northmart_dag():
    return make_dag(EDGES, latent={"intensity"})
```

---

## 15.2 Which sets block NorthMart's backdoors? (45 min)

Create `day15/backdoor_paths.py`:

```python
# day15/backdoor_paths.py   (python -m day15.backdoor_paths)
"""List the backdoor paths from email to post_orders and check which sets block them."""
from causalkit.dags import backdoor_paths, describe_path, is_valid_adjustment, minimal_adjustment_sets, path_is_blocked
from causalkit.northmart_dag import northmart_dag

g = northmart_dag()
paths = backdoor_paths(g, "email", "post_orders")
print("Backdoor paths:")
for p in paths:
    print("  ", describe_path(g, p))

print("\nWhich adjustment sets block them?")
for z in [set(), {"loyalty"}, {"pre_orders"}, {"loyalty", "pre_orders"}, {"intensity"}]:
    status = ["blocked" if path_is_blocked(g, p, z) else "OPEN" for p in paths]
    label = "{" + ", ".join(sorted(z)) + "}"
    print(f"  {label:<26} {status}  valid: {is_valid_adjustment(g, 'email', 'post_orders', z)}")

print("\nMinimal valid sets using OBSERVED variables only:", minimal_adjustment_sets(g, "email", "post_orders"))
```

```bash
python -m day15.backdoor_paths
```

```
Backdoor paths:
   email <- loyalty -> intensity -> post_orders
   email <- intensity -> post_orders

Which adjustment sets block them?
  {}                         ['OPEN', 'OPEN']  valid: False
  {loyalty}                  ['blocked', 'OPEN']  valid: False
  {pre_orders}               ['OPEN', 'OPEN']  valid: False
  {loyalty, pre_orders}      ['blocked', 'OPEN']  valid: False
  {intensity}                ['blocked', 'blocked']  valid: True

Minimal valid sets using OBSERVED variables only: []
```

This is the central result of the module for NorthMart:

- `email ← intensity → post_orders` can only be blocked by controlling for **intensity**, which nobody observes.
- `pre_orders` is not on that path: it hangs off intensity as a *child*. Controlling for it doesn't block the path in the DAG.
- **No valid adjustment set exists among the observed variables.** Under our DAG, no regression on NorthMart's columns, however many you add, identifies the email's effect.

That's not a failure of the DAG; it's what the DAG is for. It tells us to stop hoping that "more controls" will fix the problem, and to look for a different design (Module 5's difference-in-differences uses pre-period orders very differently: as a baseline for each customer's *change*).

---

## 15.3 What adjustment does to the estimate (60 min)

**Concept.** In practice, analysts adjust for proxies of the latent confounder anyway. What happens? The truth file contains each customer's expected orders without the email (`mu0`): that *is* their intensity. So we can compare imperfect adjustment with the impossible ideal.

Create `day15/adjustment_estimates.py`:

```python
# day15/adjustment_estimates.py   (python -m day15.adjustment_estimates)
"""What adjustment does to the estimate: each set from the DAG, against the truth."""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.data import load_customers

truth = pl.read_parquet("data/truth/potential_outcomes.parquet").select("customer_id", "mu0", "tau")
df = (load_customers().join(truth, on="customer_id")
      .with_columns(pl.col("email").cast(pl.Int8), loyalty=pl.col("is_loyalty").cast(pl.Int8))
      .to_pandas())

att = df.loc[df.email == 1, "tau"].mean()
ate = df["tau"].mean()

specs = {
    "nothing (naive)": "post_orders ~ email",
    "loyalty": "post_orders ~ email + loyalty",
    "loyalty + pre_orders": "post_orders ~ email + loyalty + pre_orders",
    "loyalty + pre_orders + pre_revenue + pre_active_weeks":
        "post_orders ~ email + loyalty + pre_orders + pre_revenue + pre_active_weeks",
    "intensity itself (impossible in real life)": "post_orders ~ email + mu0",
}
rows = []
for name, f in specs.items():
    fit = smf.ols(f, data=df).fit(cov_type="HC2")
    lo, hi = fit.conf_int().loc["email"]
    rows.append({"adjusting for": name, "estimate": round(fit.params["email"], 3),
                 "ci_low": round(lo, 3), "ci_high": round(hi, 3)})

with pl.Config(fmt_str_lengths=60, tbl_width_chars=120):
    print(pl.DataFrame(rows))
print(f"Truth: ATE {ate:.3f}, ATT {att:.3f} extra orders per customer (weeks 9-16)")
```

```bash
python -m day15.adjustment_estimates
```

```
│ adjusting for                                         ┆ estimate ┆ ci_low ┆ ci_high │
│ nothing (naive)                                       ┆ 1.711    ┆ 1.555  ┆ 1.866   │
│ loyalty                                               ┆ 1.523    ┆ 1.362  ┆ 1.684   │
│ loyalty + pre_orders                                  ┆ 1.062    ┆ 0.916  ┆ 1.209   │
│ loyalty + pre_orders + pre_revenue + pre_active_weeks ┆ 1.061    ┆ 0.915  ┆ 1.208   │
│ intensity itself (impossible in real life)            ┆ 0.35     ┆ 0.229  ┆ 0.472   │
Truth: ATE 0.319, ATT 0.443 extra orders per customer (weeks 9-16)
```

Read it down the column:

- Proxies help: from 1.71 to 1.06. But the estimate is still **2.4 times the ATT**, with a confidence interval nowhere near the truth.
- Adding two more pre-period variables changes **nothing**: they measure the same thing (8 noisy weeks of the same intensity) no better than `pre_orders` does.
- Controlling for the confounder itself gets close to the truth (0.35, with an interval containing both ATE and ATT). The DAG was right about what's needed; the data just don't contain it.

> **The confidence interval doesn't warn you.** [0.92, 1.21] looks precise and excludes the truth entirely. Standard errors measure noise, not bias. The DAG is how you learn about bias.

---

## 15.4 Record how treatment was assigned (35 min)

**Concept.** The cheapest fix for confounding happens *before* the campaign: record the information that drove the decision. If marketing had stored the **targeting score** it used, the DAG would change.

**Try it.** Write `day15/targeting_score.py`: a DAG where marketing computes a `score` from loyalty and intensity, and emails customers above a cut-off (so `score` is the email's only parent). Find the minimal adjustment sets.

<details>
<summary>Solution</summary>

```python
# day15/targeting_score.py   (python -m day15.targeting_score)
from causalkit.dags import make_dag, minimal_adjustment_sets

g = make_dag([
    ("loyalty", "intensity"), ("loyalty", "score"), ("intensity", "score"),
    ("score", "email"),                       # marketing emailed customers ABOVE a score cut-off
    ("intensity", "pre_orders"), ("intensity", "post_orders"), ("email", "post_orders"),
], latent={"intensity"})
print(minimal_adjustment_sets(g, "email", "post_orders"))
```

```
[{'score'}]
```

With the score recorded, `{score}` is a valid adjustment set even though intensity is still unobserved: every backdoor path into `email` now passes through `score`. **Logging the assignment rule turns an unidentifiable effect into an identifiable one.** It costs one column. Day 20's pricing scenario relies on exactly this: the algorithm's inputs are documented.

</details>

---

## 15.5 Day 15 mini-project: four classic DAGs (70 min)

**Goal.** Build intuition on four small DAGs that appear again and again in business data. For each, *predict* the minimal adjustment set on paper first, then check with code.

| # | Story | Trap |
|---|---|---|
| 1 | Seasons drive both price cuts and sales | the textbook confounder |
| 2 | **M-bias:** store traffic is caused by ad budget and competitor activity; ad budget affects price cuts, competitors affect sales | traffic is *pre-treatment*, yet adjusting for it is wrong |
| 3 | Loyalty affects price cuts and sales; price cuts raise basket size, which raises sales | basket size is a mediator |
| 4 | Unobserved demand drives a forecast; the forecast drives price cuts | demand is latent, but the forecast blocks the path |

Create `day15/four_dags.py`:

```python
# day15/four_dags.py   (python -m day15.four_dags)
"""Mini-project: find the adjustment sets for four classic DAGs, then draw them."""
from causalkit.dags import draw_dag, make_dag, minimal_adjustment_sets

DAGS = {
    "1_confounder": make_dag([("season", "price_cut"), ("season", "sales"), ("price_cut", "sales")]),
    "2_m_bias": make_dag([
        ("ad_budget", "price_cut"), ("ad_budget", "store_traffic"),
        ("competitor", "store_traffic"), ("competitor", "sales"), ("price_cut", "sales")],
        latent={"ad_budget", "competitor"}),
    "3_mediator": make_dag([
        ("loyalty", "price_cut"), ("loyalty", "sales"),
        ("price_cut", "basket_size"), ("basket_size", "sales"), ("price_cut", "sales")]),
    "4_proxy_chain": make_dag([
        ("demand", "forecast"), ("forecast", "price_cut"), ("demand", "sales"), ("price_cut", "sales")],
        latent={"demand"}),
}

for name, g in DAGS.items():
    sets = minimal_adjustment_sets(g, "price_cut", "sales")
    pretty = [("{" + ", ".join(sorted(s)) + "}") for s in sets] or ["none exists"]
    print(f"{name:<14} minimal adjustment sets: {', '.join(pretty)}")
    draw_dag(g, f"figures/day15_{name}.png", name.replace("_", " "), treatment="price_cut", outcome="sales")

# The M-bias trap: store_traffic happens BEFORE the price cut, yet adjusting for it opens a path
from causalkit.dags import is_valid_adjustment  # noqa: E402

print("\nM-bias: is {store_traffic} a valid adjustment set?",
      is_valid_adjustment(DAGS["2_m_bias"], "price_cut", "sales", {"store_traffic"}))
```

```bash
python -m day15.four_dags
```

```
1_confounder   minimal adjustment sets: {season}
2_m_bias       minimal adjustment sets: {}
3_mediator     minimal adjustment sets: {loyalty}
4_proxy_chain  minimal adjustment sets: {forecast}

M-bias: is {store_traffic} a valid adjustment set? False
```

- **M-bias** is the counterexample to "control for everything measured before treatment". `store_traffic` is a collider of two latent causes; controlling for it opens `price_cut ← ad_budget → store_traffic ← competitor → sales`.
- **DAG 4** is the NorthMart lesson in reverse: here the latent confounder affects treatment *only through* an observed variable, so the effect is identifiable. Compare with NorthMart, where intensity affects the email directly.

Open the four figures in `figures/` and add one sentence under each in `docs/ESTIMANDS.md` explaining its adjustment set in plain words.

---

## 15.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 15: backdoor criterion, adjustment sets, M-bias"
git push
```

### Day 15 checklist

- [ ] I can find every backdoor path from a treatment to an outcome
- [ ] I can check the backdoor criterion for a candidate adjustment set
- [ ] I can explain why NorthMart has no valid adjustment set among its observed variables
- [ ] I can explain why a tight confidence interval says nothing about bias
- [ ] I can recognise M-bias and explain why "control for all pre-treatment variables" can fail

---

# DAY 16 — Colliders and Selection Bias

| Time | Block |
|---|---|
| 0:00 – 0:40 | 16.1 Conditioning on a collider |
| 0:40 – 1:20 | 16.2 Berkson's paradox in a VIP programme |
| 1:20 – 1:35 | Break |
| 1:35 – 2:45 | 16.3 "Active customers only": collider bias inside an experiment |
| 2:45 – 3:20 | 16.4 Where colliders hide in business data |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 16.5 Day 16 mini-project: audit four sample filters |
| 4:45 – 5:00 | 16.6 Save your work with Git |

---

## 16.1 Conditioning on a collider (40 min)

**Concept.** Day 14 showed that controlling for a collider *creates* association. "Controlling" includes more than adding a regression variable. You also condition on a variable when you:

- **filter** the sample on it (`WHERE active = 1`),
- **stratify** by it (report results by segment),
- **only have data** for some of its values (only customers who answered a survey).

When the variable is a collider (or a descendant of one) between treatment and outcome, any of these produces **selection bias**, even if treatment was randomised. Unlike confounding, no control variable fixes it: the damage is done by who is in the data.

---

## 16.2 Berkson's paradox in a VIP programme (40 min)

**Story.** NorthMart's VIP status goes to customers who spend a lot *or* engage a lot with emails. Among VIPs, an analyst notices that heavy spenders open fewer emails, and concludes "email engagement cannibalises spending".

Create `day16/berkson.py`:

```python
# day16/berkson.py   (python -m day16.berkson)
"""Berkson's paradox: two unrelated traits look related once you only study a selected group."""
import numpy as np
import polars as pl

rng = np.random.default_rng(16)
n = 50_000
spend = rng.normal(size=n)          # how much a customer spends
engagement = rng.normal(size=n)     # how often they open emails: unrelated to spend
vip = (spend + engagement + rng.normal(scale=0.5, size=n)) > 1.5   # VIP status needs either

df = pl.DataFrame({"spend": spend, "engagement": engagement, "vip": vip})
print(f"All customers:   corr(spend, engagement) = {df.select(pl.corr('spend', 'engagement')).item():+.3f}")
vips = df.filter(pl.col("vip"))
print(f"VIPs only ({vips.height:,}): corr(spend, engagement) = {vips.select(pl.corr('spend', 'engagement')).item():+.3f}")
```

```bash
python -m day16.berkson
```

```
All customers:   corr(spend, engagement) = +0.001
VIPs only (7,875): corr(spend, engagement) = -0.543
```

Spend and engagement are unrelated by construction. VIP status is their collider (`spend → vip ← engagement`): a VIP with low spend *must* have high engagement to have qualified. Studying VIPs only manufactures a strong negative correlation.

**Try it.** Raise the VIP threshold from 1.5 to 2.5. Does the spurious correlation get weaker or stronger? Why?

<details>
<summary>Solution</summary>

Stronger: about −0.63 (on 2,367 VIPs), against −0.54 before. A stricter cut-off selects a thinner sliver of customers near the boundary `spend + engagement ≈ threshold`, where one trait almost exactly determines the other. The more selective the sample, the stronger the collider bias.

</details>

---

## 16.3 "Active customers only": collider bias inside an experiment (70 min)

**Story.** An analyst says: *"Inactive customers just add noise. Let's measure the email's effect only on customers who ordered at least once after it was sent."*

Being active after the email is caused by the email **and** by shopping intensity: `email → active ← intensity → orders`. Filtering on it opens a path from email to orders through intensity. To isolate this bias from targeting, we randomise the email (via the truth file) 1,000 times.

Create `day16/active_customers_only.py`:

```python
# day16/active_customers_only.py   (python -m day16.active_customers_only)
"""
'Let's only look at customers who were active after the email.'
Activity after the email is CAUSED by the email and by shopping intensity: a collider.
We randomise the email 1,000 times (using the truth file), so any bias is pure selection.
"""
import numpy as np
import polars as pl

po = pl.read_parquet("data/truth/potential_outcomes.parquet")
y0, y1, mu0 = po["y0"].to_numpy(), po["y1"].to_numpy(), po["mu0"].to_numpy()
rng = np.random.default_rng(16)

everyone, active_only, intensity_gap, n_active = [], [], [], []
for _ in range(1000):
    z = np.zeros(po.height, dtype=bool)
    z[rng.choice(po.height, size=1279, replace=False)] = True
    y = np.where(z, y1, y0)
    a = y > 0                                             # "active after the email"
    everyone.append(y[z].mean() - y[~z].mean())
    active_only.append(y[z & a].mean() - y[~z & a].mean())
    intensity_gap.append(mu0[z & a].mean() - mu0[~z & a].mean())
    n_active.append(a.sum())

print(f"True ATE                                   {(y1 - y0).mean():+.3f}")
print(f"Everyone, average of 1,000 experiments     {np.mean(everyone):+.3f}")
print(f"Active customers only, average             {np.mean(active_only):+.3f}   (~{np.mean(n_active):,.0f} customers)")
print(f"Intensity gap among the active (emailed - not): {np.mean(intensity_gap):+.3f} expected orders")
```

```bash
python -m day16.active_customers_only
```

```
True ATE                                   +0.319
Everyone, average of 1,000 experiments     +0.321
Active customers only, average             +0.263   (~3,137 customers)
Intensity gap among the active (emailed - not): -0.037 expected orders
```

Even with a perfect experiment, the filter cuts the estimated effect by 18%. The mechanism is in the last line: among active customers, the emailed ones are slightly *lighter* shoppers. The email pulled some light shoppers into activity; without it, only heavier shoppers would have made the cut. The comparison is no longer like for like.

> This is why Module 1 kept customers with zero orders (Days 2–3): **zero is an outcome**, and dropping it is conditioning on a collider.

---

## 16.4 Where colliders hide in business data (35 min)

| Filter or dataset | The collider | What goes wrong |
|---|---|---|
| Survey responses about a campaign | responding ← satisfaction, ← campaign | respondents are not a random slice of the treated |
| Customers who are still subscribed | still subscribed ← treatment, ← loyalty | the treatment changed who is left to compare |
| Weeks with at least one order | active week ← email, ← intensity | Day 3's zero-filled panel prevents this |
| Products still on sale | still listed ← price change, ← demand | discontinued products would have shown the failures |
| Job applicants who got an interview | interview ← referral, ← CV quality | referral looks unrelated (or negatively related) to quality |

The rule: **never filter on, stratify by, or control for anything the treatment can affect.** Filtering on pre-treatment characteristics changes *who* you study (the estimand's population), which is fine when you say so.

---

## 16.5 Day 16 mini-project: audit four sample filters (70 min)

**Goal.** For four filters an analyst might apply, compute the average estimate across 1,000 randomised replays, and compare it with the true effect *for the population the filter selects*. Classify each filter as **safe** (changes the population, no bias) or **collider** (biased).

Create `day16/selection_audit.py`:

```python
# day16/selection_audit.py   (python -m day16.selection_audit)
"""
Mini-project: audit three sample filters an analyst might apply.
A filter on something measured BEFORE the email changes WHO you study (fine, if you say so).
A filter on something the email can AFFECT creates collider bias.
"""
import numpy as np
import polars as pl

from causalkit.data import load_customers

po = pl.read_parquet("data/truth/potential_outcomes.parquet").select("customer_id", "y0", "y1")
df = load_customers().select("customer_id", "pre_orders").join(po, on="customer_id")
y0, y1 = df["y0"].to_numpy(), df["y1"].to_numpy()
pre_active = df["pre_orders"].to_numpy() > 0
rng = np.random.default_rng(160)

FILTERS = {
    "everyone": lambda y: np.ones_like(y, dtype=bool),
    "ordered BEFORE the email (pre_orders > 0)": lambda y: pre_active,
    "ordered AFTER the email (post_orders > 0)": lambda y: y > 0,
    "heavy buyers after the email (post_orders >= 3)": lambda y: y >= 3,
}
results = {name: [] for name in FILTERS}
for _ in range(1000):
    z = np.zeros(df.height, dtype=bool)
    z[rng.choice(df.height, size=1279, replace=False)] = True
    y = np.where(z, y1, y0)
    for name, rule in FILTERS.items():
        keep = rule(y)
        results[name].append(y[z & keep].mean() - y[~z & keep].mean())

rows = []
for name, est in results.items():
    target = (y1 - y0)[pre_active].mean() if "BEFORE" in name else (y1 - y0).mean()
    rows.append({"sample": name, "average estimate": round(np.mean(est), 3),
                 "true effect for that population": round(target, 3)})
with pl.Config(fmt_str_lengths=60, tbl_width_chars=120):
    print(pl.DataFrame(rows))
```

```bash
python -m day16.selection_audit
```

```
│ sample                                          ┆ average estimate ┆ true effect for that population │
│ everyone                                        ┆ 0.317            ┆ 0.318                           │
│ ordered BEFORE the email (pre_orders > 0)       ┆ 0.352            ┆ 0.355                           │
│ ordered AFTER the email (post_orders > 0)       ┆ 0.26             ┆ 0.318                           │
│ heavy buyers after the email (post_orders >= 3) ┆ 0.168            ┆ 0.318                           │
```

- **Pre-period filter: safe.** The estimate (0.352) matches the true effect for pre-period buyers (0.355). It's a different estimand, a CATE for a subgroup, and it's correctly estimated.
- **Post-period filters: collider bias**, growing with selectivity: −18% for "any order", −47% for "three or more".

Write your conclusions in `docs/ESTIMANDS.md` under **"Sample filters"**: one line per filter, with its verdict and the reason.

---

## 16.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 16: colliders, Berkson's paradox, selection audit"
git push
```

### Day 16 checklist

- [ ] I can explain why filtering, stratifying or limited data can all mean conditioning
- [ ] I can recognise Berkson's paradox and explain it with a DAG
- [ ] I can show collider bias inside a randomised experiment
- [ ] I can tell a safe pre-treatment filter from a biased post-treatment one

---

# DAY 17 — Front-Door Paths, Mediators and Bad Controls

| Time | Block |
|---|---|
| 0:00 – 0:45 | 17.1 Mediators: total versus direct effects |
| 0:45 – 1:30 | 17.2 The front-door criterion |
| 1:30 – 1:45 | Break |
| 1:45 – 2:40 | 17.3 Bad controls in NorthMart |
| 2:40 – 3:20 | 17.4 A field guide to controls |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 17.5 Day 17 mini-project: let the DAG classify every column |
| 4:45 – 5:00 | 17.6 Save your work with Git |

---

## 17.1 Mediators: total versus direct effects (45 min)

**Concept.** A **mediator** lies on a causal path from treatment to outcome: email → site visits → orders. The **total effect** of the email includes everything it causes, through visits and otherwise. Controlling for the mediator leaves only the **direct effect**: what the email does *other than* through visits.

Business questions almost always want the total effect ("did the email increase orders?"). Controlling for a mediator answers a different, narrower question, usually without anyone noticing.

Create `day17/mediator.py`:

```python
# day17/mediator.py   (python -m day17.mediator)
"""A mediator carries part of the effect. Controlling for it removes that part."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(17)
n = 50_000
email = rng.integers(0, 2, n)                          # randomised: no confounding at all
visits = 1.0 + 0.8 * email + rng.normal(size=n)        # email -> site visits
orders = 0.5 * visits + 0.2 * email + rng.normal(size=n)   # visits -> orders, plus a direct effect

df = pd.DataFrame({"email": email, "visits": visits, "orders": orders})
total = smf.ols("orders ~ email", df).fit().params["email"]
direct = smf.ols("orders ~ email + visits", df).fit().params["email"]

print(f"True total effect   0.8 x 0.5 + 0.2 = {0.8 * 0.5 + 0.2:.2f}")
print(f"orders ~ email                     {total:.3f}   <- the total effect")
print(f"orders ~ email + visits            {direct:.3f}   <- only the direct part")
```

```bash
python -m day17.mediator
```

```
True total effect   0.8 x 0.5 + 0.2 = 0.60
orders ~ email                     0.631   <- the total effect
orders ~ email + visits            0.223   <- only the direct part
```

Even in a randomised experiment, "controlling for visits to be safe" throws away two thirds of the effect. (And it can do worse: if anything unobserved affects both visits and orders, visits becomes a collider too, and the "direct effect" is biased.)

---

## 17.2 The front-door criterion (45 min)

**Concept.** Sometimes a mediator *rescues* identification. Suppose an unobserved confounder drives both treatment and outcome (no backdoor fix possible, like NorthMart's intensity), but:

1. the treatment affects the outcome **only** through an observed mediator M, and
2. the confounder does **not** affect M directly.

Then the effect can be computed in two steps, each free of confounding: T → M (no backdoor path into T affects M), and M → Y controlling for T (T blocks the backdoor path from M). Multiply them (in linear models).

Create `day17/frontdoor.py`:

```python
# day17/frontdoor.py   (python -m day17.frontdoor)
"""
Front-door adjustment: an unobserved confounder blocks the backdoor route,
but the effect flows entirely through an observed mediator we can use instead.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(170)
n = 50_000
u = rng.normal(size=n)                                  # unobserved intensity
email = (u + rng.normal(size=n) > 0).astype(int)        # targeted on u
visits = 1.0 + 0.8 * email + rng.normal(size=n)         # email -> visits (u does NOT affect visits)
orders = 0.5 * visits + 1.0 * u + rng.normal(size=n)    # visits -> orders, u -> orders

df = pd.DataFrame({"email": email, "visits": visits, "orders": orders})
naive = smf.ols("orders ~ email", df).fit().params["email"]
a = smf.ols("visits ~ email", df).fit().params["email"]              # step 1: email -> visits
b = smf.ols("orders ~ visits + email", df).fit().params["visits"]    # step 2: visits -> orders, holding email
print(f"True effect of email          {0.8 * 0.5:.3f}")
print(f"Naive (confounded by u)       {naive:.3f}")
print(f"Front door: {a:.3f} x {b:.3f}       = {a * b:.3f}")
```

```
True effect of email          0.400
Naive (confounded by u)       1.545
Front door: 0.796 x 0.503       = 0.400
```

**Try it.** Break assumption 2: let intensity also raise site visits (`visits = 1.0 + 0.8 * email + 0.5 * u + rng.normal(size=n)`). What happens to the front-door estimate?

<details>
<summary>Solution</summary>

```
True effect of email          0.400
Naive (confounded by u)       1.830
Front door: 1.367 x 0.791       = 1.081
```

Step 1 (email → visits) is now confounded by u, and the estimate is badly off. The front door needs a mediator that is fully shielded from the confounder, which is rare. Use it when you can defend that, mostly in settings where the mechanism is technical (a recommendation algorithm, a delivery system) rather than behavioural.

</details>

---

## 17.3 Bad controls in NorthMart (55 min)

**Concept.** A **bad control** is a variable affected by the treatment. Controlling for it removes part of the effect (if it's a mediator), creates collider bias (if it shares causes with the outcome), or both. In NorthMart, two columns are measured *after* the email: `post_active_weeks` and `post_revenue`.

Create `day17/bad_controls.py`:

```python
# day17/bad_controls.py   (python -m day17.bad_controls)
"""Controlling for something the email affects: the estimate collapses."""
import polars as pl
import statsmodels.formula.api as smf

from causalkit.data import load_customers

df = load_customers().with_columns(pl.col("email").cast(pl.Int8),
                                   loyalty=pl.col("is_loyalty").cast(pl.Int8)).to_pandas()
base = "post_orders ~ email + loyalty + pre_orders"
for name, f in [("pre-treatment controls only", base),
                ("+ post_active_weeks (BAD)", base + " + post_active_weeks"),
                ("+ post_revenue (BAD)", base + " + post_revenue")]:
    fit = smf.ols(f, data=df).fit(cov_type="HC2")
    print(f"{name:<30} email coefficient {fit.params['email']:+.3f}")
print("Truth (ATT): +0.443")
```

```
pre-treatment controls only    email coefficient +1.062
+ post_active_weeks (BAD)      email coefficient +0.111
+ post_revenue (BAD)           email coefficient +0.573
Truth (ATT): +0.443
```

Look at the second line: adding a bad control moved the estimate from 1.06 to 0.11, and 0.11 is *closer* to the truth than 1.06. That's a coincidence of two wrongs: confounding pushed the estimate up, the bad control pushed it down. Judge controls by the DAG, **never** by whether the estimate "looks more reasonable".

---

## 17.4 A field guide to controls (40 min)

| Kind | DAG position | Control for it? | NorthMart example |
|---|---|---|---|
| **Confounder** | causes T and Y | yes | `loyalty` |
| **Proxy of a latent confounder** | child of a confounder | helps, never fully | `pre_orders` |
| **Outcome predictor** | causes Y only | optional: improves precision | (none here; `email` in Day 20) |
| **Treatment predictor** | causes T only | no: inflates variance (sometimes bias) | a random targeting tie-breaker |
| **Neutral** | unconnected | harmless | `age`, `region` |
| **Mediator** | T → M → Y | no (for total effects) | site visits |
| **Collider / descendant of T** | T → C ← … | **never** | `post_active_weeks` |
| **M-bias collider** | pre-treatment collider of two latents | no | store traffic (Day 15) |

The two questions to ask about every candidate control:

1. **Could the treatment have changed it?** If yes, leave it out.
2. **Does it sit on a backdoor path?** If yes, include it (or something else that blocks that path).

---

## 17.5 Day 17 mini-project: let the DAG classify every column (70 min)

**Goal.** Extend the NorthMart DAG to every column of the customer table, then let code apply the field guide: descendants of the email are bad controls; variables connected to both email and outcome are useful; the rest are neutral.

Create `day17/control_audit.py`:

```python
# day17/control_audit.py   (python -m day17.control_audit)
"""Mini-project: classify every column of the customer table as a control, using the DAG."""
import networkx as nx
import polars as pl

from causalkit.dags import make_dag

g = make_dag([
    # how the email was targeted (Day 14)
    ("loyalty", "intensity"), ("loyalty", "email"), ("intensity", "email"),
    # pre-period measurements of intensity
    ("intensity", "pre_orders"), ("pre_orders", "pre_revenue"), ("pre_orders", "pre_active_weeks"),
    ("intensity", "days_since_last_order"),
    # outcomes
    ("email", "post_orders"), ("intensity", "post_orders"),
    ("post_orders", "post_revenue"), ("post_orders", "post_active_weeks"),
], latent={"intensity"})
for neutral in ["age", "region", "tenure_days", "pre_discount_share"]:
    g.add_node(neutral, latent=False)       # in the data; no arrows in or out (the generator ignores them)

T, Y = "email", "post_orders"
no_effect = g.copy()
no_effect.remove_edges_from(list(g.out_edges(T)))   # the graph without the email's effect

rows = []
for v in sorted(n for n in g.nodes if not g.nodes[n]["latent"] and n not in (T, Y)):
    if v in nx.descendants(g, T):
        verdict, why = "BAD control", "affected by the email (post-treatment)"
    elif not nx.is_d_separator(no_effect, {v}, {T}, set()) and not nx.is_d_separator(no_effect, {v}, {Y}, set()):
        verdict, why = "useful control", "helps block a backdoor path (tied to email and outcome)"
    else:
        verdict, why = "neutral", "unrelated to the email or the outcome"
    rows.append({"column": v, "verdict": verdict, "why": why})

with pl.Config(tbl_rows=-1, fmt_str_lengths=60, tbl_width_chars=120):
    print(pl.DataFrame(rows))
```

```bash
python -m day17.control_audit
```

```
│ column                ┆ verdict        ┆ why                                                     │
│ age                   ┆ neutral        ┆ unrelated to the email or the outcome                   │
│ days_since_last_order ┆ useful control ┆ helps block a backdoor path (tied to email and outcome) │
│ loyalty               ┆ useful control ┆ helps block a backdoor path (tied to email and outcome) │
│ post_active_weeks     ┆ BAD control    ┆ affected by the email (post-treatment)                  │
│ post_revenue          ┆ BAD control    ┆ affected by the email (post-treatment)                  │
│ pre_active_weeks      ┆ useful control ┆ helps block a backdoor path (tied to email and outcome) │
│ pre_discount_share    ┆ neutral        ┆ unrelated to the email or the outcome                   │
│ pre_orders            ┆ useful control ┆ helps block a backdoor path (tied to email and outcome) │
│ pre_revenue           ┆ useful control ┆ helps block a backdoor path (tied to email and outcome) │
│ region                ┆ neutral        ┆ unrelated to the email or the outcome                   │
│ tenure_days           ┆ neutral        ┆ unrelated to the email or the outcome                   │
```

Paste the table into `docs/ESTIMANDS.md` under **"Control audit"**. Add one sentence explaining why "useful control" doesn't mean "sufficient": Day 15 showed that all five useful controls together still leave a bias of more than 0.6 orders.

---

## 17.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 17: mediators, front door, bad controls, control audit"
git push
```

**Coming next (Days 18–20).** Day 18 hands the DAG to **DoWhy**, a library that performs identification, estimation and refutation for you, and shows what its checks can and can't catch. Day 19 resolves **Simpson's paradox** with a DAG. Day 20 is the Module 2 capstone: **Defending the DAG** for NorthMart's discount programme.

### Day 17 checklist

- [ ] I can distinguish total and direct effects, and explain what controlling for a mediator does
- [ ] I can state the front-door conditions and show what breaks when one fails
- [ ] I can spot bad controls and explain why "the estimate looks more reasonable" is not evidence
- [ ] I can classify every column of a dataset as a control, using a DAG
