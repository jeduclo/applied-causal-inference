# Module 8 · Decision Intelligence Delivery
## Days 71–73: Serving Causal Models, Scenario Design and Communicating Uncertainty

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Modules 1–7
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 73:** Students can package causal estimates as a versioned parameters file, serve a scenario simulator through a FastAPI service, design a scenario menu that refuses to extrapolate quietly, and describe uncertainty in words and pictures an executive can act on.

---

### Where we are

Seven modules produced estimates: price elasticities per store segment and category (Module 6), and a coupon uplift curve (Module 7). They live in notebooks and scripts that only their author can run. Module 8 turns them into something the business uses every week: a **scenario simulator**. A category manager moves a price slider and sees projected profit next to the counterfactual (what happens if nothing changes), with an honest range.

| Days | Question | Tools |
|---|---|---|
| 71–73 | How do other people's programs and people use our estimates? | parameters file, FastAPI, pydantic, scenario presets, plain-language uncertainty |
| 74–76 | How do decision-makers use them, and how do we keep it working? | Streamlit, Altair, pytest, GitHub Actions, Docker |
| 77–80 | How do we know when the estimates are out of date? | rolling monitoring, CUSUM, value of information, final capstone |

### The architecture, in one picture

```
data/*.csv ──► engine/build.py ──► models/causal_parameters.json ──► engine/simulate.py
 (Modules 6–7 estimators)            (versioned, committed)            │
                                                                       ├──► service/app.py   (FastAPI: other programs)
                                                                       └──► app/streamlit_app.py (people)
```

The key design decision: **estimation and simulation are separate.** Estimation is slow (minutes of DoubleML and DR-learners) and runs when new data arrive. Simulation is fast (milliseconds of arithmetic) and runs every time someone moves a slider. Between them sits one small JSON file with a version string. Anyone can ask "which estimates produced this number?" and get a precise answer.

### Setup

```bash
mkdir -p engine service app tests models day71 day72 day73 day74 day75 day76 day77 day78 day79 day80
touch engine/__init__.py service/__init__.py app/__init__.py tests/__init__.py
```

Three new packages. Add them to `requirements.txt` and install (`pip install -r requirements.txt`):

```
# Module 8: delivery
streamlit==1.66.1
pytest==9.2.0
httpx==0.28.1          # FastAPI's TestClient uses it
```

FastAPI, uvicorn and requests are already pinned from Module 1. Altair comes with Streamlit.

**A small change to `causalkit/demand.py`.** The simulator gains three optional arguments for Days 77–79: `drift_week`, `drift_segment` and `drift_factor` (from that week on, one segment's elasticities are multiplied by the factor), and `price_test` (a randomised price test in half of one segment's stores, which adds a `price_test` column). The hidden `_elasticity` column is now per row. With the defaults nothing changes: every earlier result is identical. Copy the updated file from the course repository.

**What gets committed.** Unlike earlier modules, `models/causal_parameters.json` **is** committed: it is small, it's the thing the app serves, and Git's history of it is the history of the business's beliefs. Data, figures and secrets stay out as before.

---

# DAY 71 — Serving Causal Models with FastAPI

| Time | Block |
|---|---|
| 0:00 – 0:40 | 71.1 From estimates to a parameters file |
| 0:40 – 1:30 | 71.2 The simulation engine |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 71.3 A scenario API with FastAPI |
| 2:30 – 3:10 | 71.4 Calling the API, and what it refuses |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 71.5 Day 71 mini-project: a real server, and how fast it is |
| 4:40 – 5:00 | 71.6 Save your work with Git |

---

## 71.1 From estimates to a parameters file (40 min)

**Concept.** A model in production is a contract: inputs, outputs, and a version. For a causal simulator the "model" is small: 20 elasticities with standard errors, the prices and volumes they apply to, and a coupon curve. We store exactly that, plus where it came from.

First, the data the parameters are estimated from. Create `data/make_delivery_data.py`:

```python
"""
Module 8 data: the inputs the simulator's parameters are estimated from. Provided.

    python data/make_delivery_data.py

Writes data/scenario/weekly_demand.csv (Module 6's capstone year) and data/scenario/coupon_experiment.csv
(Module 7's 40,000-customer randomised coupon test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalkit.coupons import coupons  # noqa: E402
from causalkit.coupons import public as coupons_public  # noqa: E402
from causalkit.demand import demand, public  # noqa: E402

root = Path(__file__).parent
(root / "scenario").mkdir(exist_ok=True)
public(demand(seed=60)).write_csv(root / "scenario" / "weekly_demand.csv")
coupons_public(coupons(n=40_000, seed=640, randomized=True)).write_csv(root / "scenario" / "coupon_experiment.csv")
print("Wrote data/scenario/weekly_demand.csv and data/scenario/coupon_experiment.csv")
```

Then the builder. It reuses Module 6's DML and 2SLS and Module 7's DR-learner. Each pricing cell also records `valid_change`: the largest price move (95th percentile) that supplier costs actually caused in the data. That number comes back on Day 72. Create `engine/parameters.py` and `engine/build.py`:

```python
"""
The causal parameters file: everything the simulator knows, in one versioned JSON file.

    {"version": "...", "created": "...", "sources": {...},
     "pricing": [{"segment", "category", "elasticity", "se", "margin", "price", "weekly_units",
                  "valid_change"}, ...],
     "coupons": {"customers", "campaigns_per_year", "margin", "shares": [...], "uplift": [...], "uplift_se": [...]},
     "assumptions": {"pull_forward": ...}}

elasticity: DML point estimate; se: the honest (2SLS) standard error (Module 6, Day 59).
valid_change: the size of price change that supplier costs actually produced in the data (95th percentile).
The 2SLS elasticity is learned from changes of about that size; beyond it the simulator extrapolates
and says so.
uplift: extra spend per customer in the base, per campaign, when the targeting model coupons the top
share of customers (Module 7, Day 65), with standard errors.
"""
import json
from pathlib import Path

REQUIRED_CELL = {"segment", "category", "elasticity", "se", "margin", "price", "weekly_units", "valid_change"}


def validate(p: dict) -> dict:
    """Fail loudly on a malformed file: a simulator should never run on half a parameter set."""
    for key in ["version", "pricing", "coupons", "assumptions"]:
        if key not in p:
            raise ValueError(f"parameters file has no '{key}'")
    for cell in p["pricing"]:
        missing = REQUIRED_CELL - cell.keys()
        if missing:
            raise ValueError(f"pricing cell {cell.get('segment')}/{cell.get('category')} is missing {sorted(missing)}")
        if cell["elasticity"] >= 0:
            raise ValueError(f"non-negative elasticity for {cell['segment']}/{cell['category']}: {cell['elasticity']}")
    c = p["coupons"]
    if not (len(c["shares"]) == len(c["uplift"]) == len(c["uplift_se"])):
        raise ValueError("coupon shares, uplift and uplift_se must have the same length")
    return p


def load_parameters(path: str | Path = "models/causal_parameters.json") -> dict:
    return validate(json.loads(Path(path).read_text()))


def save_parameters(p: dict, path: str | Path = "models/causal_parameters.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(validate(p), indent=2))
```

```python
"""
Build the causal parameters file from data (Modules 6 and 7, packaged): run it whenever new data arrive.

    python -m engine.build data/scenario/weekly_demand.csv data/scenario/coupon_experiment.csv

Pricing: one elasticity per segment x category cell. DML gives the point estimate; 2SLS with the supplier
cost instrument gives the honest standard error (Module 6, Day 59).
Coupons: a DR-learner ranks customers (trained on half the experiment); doubly robust scores on the other
half give the extra spend per customer in the base when the top share is couponed (Module 7, Day 65).
"""
import hashlib
import sys
from datetime import date
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
import pyfixest as pf
from econml.dr import DRLearner
from sklearn.model_selection import KFold, train_test_split

from causalkit.coupons import X_COLS
from causalkit.learners import clf, reg
from engine.parameters import save_parameters

DRIVERS = ["promo", "temperature", "holiday", "local_event", "competitor_index", "week"]
SHARES = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def pricing_cells(demand: pd.DataFrame, recent_weeks: int = 13) -> list[dict]:
    g = demand.groupby(["store_id", "category"])
    demand = demand.assign(y=demand.log_units - g.log_units.transform("mean"),
                           d=demand.log_price - g.log_price.transform("mean"),
                           store=demand.store_id.str[1:].astype(int))
    last = demand.week.max()
    cells = []
    for (seg, cat), s in demand.groupby(["segment", "category"]):
        np.random.seed(58)
        data = dml.DoubleMLData(s[DRIVERS + ["y", "d", "store"]], "y", "d", DRIVERS, cluster_cols="store")
        learner = reg(n_estimators=500, num_leaves=31, min_child_samples=20)
        point = dml.DoubleMLPLR(data, ml_l=learner, ml_m=learner, n_folds=5).fit().coef[0]
        iv = pf.feols("log_units ~ promo | store_id + week | log_price ~ cost_index", s, vcov={"CRV1": "store_id"})
        first = pf.feols("log_price ~ cost_index + promo | store_id + week", s).coef()["cost_index"]
        cost_moves = first * (s.cost_index - s.groupby("store_id").cost_index.transform("mean"))
        recent = s[s.week > last - recent_weeks]
        weekly = recent.groupby("week").units.sum()
        cells.append({"segment": seg, "category": cat, "elasticity": round(float(point), 3),
                      "se": round(float(iv.se()["log_price"]), 3), "margin": float(s.margin.iloc[0]),
                      "price": round(float((recent.price * recent.units).sum() / recent.units.sum()), 3),
                      "weekly_units": round(float(weekly.mean()), 1),
                      "valid_change": round(float(np.expm1(np.percentile(np.abs(cost_moves), 95))), 3)})
    return cells


def coupon_curve(exp: pd.DataFrame, boot: int = 200) -> dict:
    train, test = train_test_split(exp, test_size=0.5, random_state=0)
    dr = DRLearner(model_propensity=clf(), model_regression=reg(),
                   model_final=reg(n_estimators=100, num_leaves=4, min_child_samples=300), cv=5,
                   min_propensity=0.02, random_state=0).fit(train.spend, train.coupon, X=train[X_COLS])
    rank = np.argsort(np.argsort(-dr.effect(test[X_COLS]))) / len(test)
    y, t = test.spend.to_numpy(), test.coupon.to_numpy()
    p = t.mean()
    mu0, mu1 = np.zeros(len(test)), np.zeros(len(test))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(test):
        a = test.iloc[tr]
        for arm, mu in [(0, mu0), (1, mu1)]:
            mu[te] = reg().fit(a[a.coupon == arm][X_COLS], y[tr][t[tr] == arm]).predict(test.iloc[te][X_COLS])
    score = (mu1 + t * (y - mu1) / p) - (mu0 + (1 - t) * (y - mu0) / (1 - p))
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(test), len(test)) for _ in range(boot)]
    uplift, se = [], []
    for s in SHARES:
        pol = rank < s
        uplift.append(round(float(np.mean(pol * score)), 3))
        se.append(round(float(np.std([np.mean(pol[i] * score[i]) for i in idx])), 3))
    return {"customers": 400_000, "campaigns_per_year": 12, "margin": 0.30, "shares": SHARES,
            "uplift": uplift, "uplift_se": se}


def build(demand_csv: str, coupons_csv: str, out: str = "models/causal_parameters.json") -> dict:
    demand = pd.read_csv(demand_csv)
    exp = pd.read_csv(coupons_csv)
    digest = hashlib.sha256(Path(demand_csv).read_bytes() + Path(coupons_csv).read_bytes()).hexdigest()[:8]
    params = {
        "version": f"{date.today().isoformat()}-{digest}",
        "created": date.today().isoformat(),
        "sources": {"demand": demand_csv, "demand_weeks": [int(demand.week.min()), int(demand.week.max())],
                    "coupons": coupons_csv, "coupon_customers": len(exp),
                    "pricing_method": "DML point estimate, 2SLS (supplier cost) standard error, per cell",
                    "coupon_method": "DR-learner ranking, doubly robust uplift on a held-out half"},
        "pricing": pricing_cells(demand),
        "coupons": coupon_curve(exp),
        "assumptions": {"pull_forward": 0.15},
    }
    save_parameters(params, out)
    return params


if __name__ == "__main__":
    args = sys.argv[1:] or ["data/scenario/weekly_demand.csv", "data/scenario/coupon_experiment.csv"]
    p = build(*args)
    print(f"Wrote parameters version {p['version']}: {len(p['pricing'])} pricing cells, "
          f"coupon curve at {len(p['coupons']['shares'])} shares")
```

```bash
python data/make_delivery_data.py
python -m engine.build
```

```
Wrote parameters version 2026-11-05-e117a0ef: 20 pricing cells, coupon curve at 11 shares
```

(Your date will differ. The 8 characters after it are a hash of the input data, so the same data always gives the same suffix.) Open `models/causal_parameters.json`. It's readable, and every number has a source line.

---

## 71.2 The simulation engine (50 min)

**Concept.** For a price change d in a cell with elasticity e, constant-elasticity demand gives units × (1 + d)^e. Profit per unit rises from m·p to (m + d)·p. The counterfactual is d = 0. Uncertainty comes from drawing each elasticity from N(estimate, SE²) 2,000 times.

A scenario is validated before anything runs. pydantic rejects unknown keys and impossible values with a readable message. Create `engine/scenario.py`, `engine/simulate.py` and `engine/__init__.py`:

```python
"""What a user can ask the simulator: a scenario. Pydantic validates it (FastAPI uses the same class)."""
from pydantic import BaseModel, Field, field_validator


class Scenario(BaseModel):
    """price_change: share changes in price, by key. Keys: "all", a segment ("student"), or a cell
    ("student/snacks"). The most specific key wins: {"all": 0.05, "student": 0.0} raises every price 5%
    except for students. coupon_share: share of customers the targeting model coupons (0 = no programme).
    elasticity_correlation: 0 = each cell's estimation error is independent; 1 = all cells are wrong in the
    same direction together (e.g. a bias shared by every estimate). Day 73."""
    price_change: dict[str, float] = Field(default_factory=dict)
    coupon_share: float = Field(0.0, ge=0.0, le=1.0)
    coupon_cost: float = Field(5.0, ge=0.0, le=50.0)
    pull_forward: float | None = Field(None, ge=0.0, le=0.9)
    elasticity_correlation: float = Field(0.0, ge=0.0, le=1.0)
    draws: int = Field(2000, ge=100, le=20000)
    seed: int = 0

    @field_validator("price_change")
    @classmethod
    def plausible_changes(cls, v):
        for key, change in v.items():
            if not -0.5 <= change <= 0.5:
                raise ValueError(f"price change for '{key}' is {change:.0%}; the simulator accepts -50% to +50%")
        return v

    def change_for(self, segment: str, category: str) -> float:
        for key in (f"{segment}/{category}", segment, "all"):
            if key in self.price_change:
                return self.price_change[key]
        return 0.0
```

```python
"""
The simulator: for a scenario, the projected annual revenue and gross profit, against the counterfactual
(what happens if nothing changes), with uncertainty from the causal estimates.

Pricing, for each segment x category cell with elasticity e, margin m (share of price), price change d:
    units      -> units x (1 + d)^e
    revenue    -> price x (1 + d) x units x (1 + d)^e
    gross profit per unit -> (m + d) x price   (unit cost doesn't change)
Coupons: extra spend = customers x campaigns x uplift(share) x (1 - pull_forward);
         gross profit = margin x extra spend - cost x share x customers x campaigns.
Uncertainty: every draw takes each cell's elasticity from Normal(estimate, se) (independently across cells,
or partly shared: Scenario.elasticity_correlation) and the uplift from Normal(estimate, se). Assumed: cells and coupons don't interact (a price rise doesn't change who responds
to coupons); that's a modelling choice, and the result says so.
"""
import numpy as np

from engine.scenario import Scenario

WEEKS = 52


def _uplift(coupons: dict, share: float):
    """Interpolate the uplift curve (and its SE) at a coupon share."""
    s = np.asarray(coupons["shares"])
    return float(np.interp(share, s, coupons["uplift"])), float(np.interp(share, s, coupons["uplift_se"]))


def simulate(params: dict, scenario: Scenario) -> dict:
    rng = np.random.default_rng(scenario.seed)
    n = scenario.draws
    cells, warnings = [], []
    total_rev = np.zeros(n)
    total_profit = np.zeros(n)
    base_rev = base_profit = 0.0
    rho = scenario.elasticity_correlation
    common = rng.normal(0, 1, n)                     # the part of the estimation error all cells share
    for c in params["pricing"]:
        d = scenario.change_for(c["segment"], c["category"])
        rev0 = c["price"] * c["weekly_units"] * WEEKS
        profit0 = c["margin"] * rev0
        e = c["elasticity"] + c["se"] * (np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(0, 1, n))
        units_ratio = (1 + d) ** e
        rev = rev0 * (1 + d) * units_ratio
        profit = (c["margin"] + d) * c["price"] * c["weekly_units"] * WEEKS * units_ratio
        base_rev += rev0
        base_profit += profit0
        total_rev += rev
        total_profit += profit
        if abs(d) > c["valid_change"] + 1e-9:
            warnings.append(f"{c['segment']}/{c['category']}: a {d:+.0%} change is outside the range seen in the "
                            f"data (±{c['valid_change']:.0%}); the projection extrapolates")
        cells.append({"segment": c["segment"], "category": c["category"], "price_change": d,
                      "revenue_base": rev0, "profit_base": profit0,
                      "profit_change_median": float(np.median(profit - profit0)),
                      "profit_change_p5": float(np.percentile(profit - profit0, 5)),
                      "profit_change_p95": float(np.percentile(profit - profit0, 95)),
                      "prob_profit_up": float(np.mean(profit > profit0)) if d != 0 else None})

    co = params["coupons"]
    pull = params["assumptions"]["pull_forward"] if scenario.pull_forward is None else scenario.pull_forward
    coupon_rev = np.zeros(n)
    coupon_profit = np.zeros(n)
    if scenario.coupon_share > 0:
        u, se = _uplift(co, scenario.coupon_share)
        extra = co["customers"] * co["campaigns_per_year"] * rng.normal(u, se, n) * (1 - pull)
        coupon_rev = extra
        coupon_profit = co["margin"] * extra - scenario.coupon_cost * scenario.coupon_share * co["customers"] * co["campaigns_per_year"]
    total_rev += coupon_rev
    total_profit += coupon_profit

    def band(x):
        return {"median": float(np.median(x)), "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}

    return {
        "parameters_version": params["version"],
        "counterfactual": {"revenue": base_rev, "gross_profit": base_profit},
        "projected": {"revenue": band(total_rev), "gross_profit": band(total_profit)},
        "change": {"revenue": band(total_rev - base_rev), "gross_profit": band(total_profit - base_profit),
                   "prob_profit_up": (float(np.mean(total_profit > base_profit))
                                      if scenario.price_change or scenario.coupon_share > 0 else None),
                   "prob_loss_over_1m": float(np.mean(total_profit - base_profit < -1e6))},
        "coupons": {"share": scenario.coupon_share, "profit_change": band(coupon_profit),
                    "prob_profit_up": float(np.mean(coupon_profit > 0)) if scenario.coupon_share > 0 else None},
        "cells": cells,
        "draws": {"gross_profit_change": (total_profit - base_profit)[:500].round(0).tolist()},
        "warnings": warnings,
        "assumptions": {"pull_forward": pull, "cells_and_coupons_interact": False},
    }
```

```python
"""NorthMart scenario engine (Module 8): causal parameters in, projected vs counterfactual outcomes out."""
from engine.parameters import load_parameters, save_parameters
from engine.scenario import Scenario
from engine.simulate import simulate

__all__ = ["Scenario", "load_parameters", "save_parameters", "simulate"]
```

Note what `simulate` returns: the counterfactual, the projection, the **change** (the causal quantity), the probability that profit goes up, per-cell results, warnings and the assumptions it used. A number without its assumptions can't be checked.

---

## 71.3 A scenario API with FastAPI (45 min)

**Concept.** Other teams' programs (the pricing system, a planning spreadsheet, tomorrow's app) shouldn't import our Python. They send JSON over HTTP and get JSON back. FastAPI turns the pydantic `Scenario` into request validation and documentation for free.

Create `service/app.py`:

```python
"""
The NorthMart scenario API (Module 8, Day 71): the causal parameters and the simulator behind HTTP.

    uvicorn service.app:app --port 8001            # then open http://127.0.0.1:8001/docs

Endpoints
    GET  /health       is it up, and which parameters version is it serving?
    GET  /parameters   the causal estimates it uses (so anyone can audit them)
    POST /scenario     one scenario -> projected vs counterfactual, with uncertainty
    POST /compare      several named scenarios -> one comparison table
"""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import Scenario, load_parameters, simulate

app = FastAPI(title="NorthMart scenario API", version="1.0.0",
              description="Projected vs counterfactual revenue and gross profit for pricing and coupon scenarios.")


@lru_cache
def params() -> dict:
    """Load the parameters file once (path from the PARAMS_PATH environment variable)."""
    return load_parameters(os.environ.get("PARAMS_PATH", "models/causal_parameters.json"))


@app.get("/health")
def health():
    return {"status": "ok", "parameters_version": params()["version"]}


@app.get("/parameters")
def get_parameters():
    p = params()
    return {"version": p["version"], "created": p["created"], "sources": p["sources"],
            "pricing": p["pricing"], "coupons": p["coupons"], "assumptions": p["assumptions"]}


@app.post("/scenario")
def run_scenario(scenario: Scenario, include_draws: bool = False):
    p = params()
    known = {c["segment"] for c in p["pricing"]} | {f"{c['segment']}/{c['category']}" for c in p["pricing"]} | {"all"}
    unknown = sorted(set(scenario.price_change) - known)
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown price_change keys: {unknown}; use 'all', a segment, "
                                                    "or 'segment/category'")
    result = simulate(p, scenario)
    if not include_draws:
        result.pop("draws")
    return result


class NamedScenario(BaseModel):
    name: str
    scenario: Scenario


@app.post("/compare")
def compare(scenarios: list[NamedScenario]):
    rows = []
    for s in scenarios:
        r = run_scenario(s.scenario)
        g = r["change"]["gross_profit"]
        rows.append({"name": s.name, "profit_change_median": round(g["median"]), "p5": round(g["p5"]),
                     "p95": round(g["p95"]), "prob_profit_up": r["change"]["prob_profit_up"],
                     "warnings": len(r["warnings"])})
    return {"parameters_version": params()["version"], "rows": rows}
```

Run it and open the interactive documentation:

```bash
uvicorn service.app:app --reload --port 8001
```

In Codespaces, open the forwarded port and go to `/docs`. Every endpoint is there, with a "Try it out" button. Stop the server with Ctrl+C.

---

## 71.4 Calling the API, and what it refuses (40 min)

Create `day71/call_the_api.py`:

```python
# day71/call_the_api.py   (python -m day71.call_the_api)
"""
Call the scenario API the way another team's code would. FastAPI's TestClient runs the app in-process,
so this works without starting a server; swap in `requests` and the URL to call a running one.
"""
import json

from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)
print("GET /health ->", client.get("/health").json())

p = client.get("/parameters").json()
print(f"GET /parameters -> version {p['version']}, {len(p['pricing'])} cells; first cell:")
print("   ", p["pricing"][0])

r = client.post("/scenario", json={"price_change": {"all": 0.05}})
print("POST /scenario {'price_change': {'all': 0.05}} ->")
print("    gross profit change:", {k: round(v) for k, v in r.json()["change"]["gross_profit"].items()})

bad = client.post("/scenario", json={"price_change": {"students": 0.05}})
print("POST /scenario with a misspelt segment ->", bad.status_code, bad.json()["detail"])
bad = client.post("/scenario", json={"price_change": {"all": 0.8}})
print("POST /scenario with an 80% price rise ->", bad.status_code, bad.json()["detail"][0]["msg"])

cmp = client.post("/compare", json=[
    {"name": "5% everywhere", "scenario": {"price_change": {"all": 0.05}}},
    {"name": "coupons, top 50%", "scenario": {"coupon_share": 0.5}},
    {"name": "both", "scenario": {"price_change": {"all": 0.05}, "coupon_share": 0.5}},
]).json()
print("POST /compare ->")
print(json.dumps(cmp["rows"], indent=1))
```

```bash
python -m day71.call_the_api
```

```
GET /health -> {'status': 'ok', 'parameters_version': '2026-11-05-e117a0ef'}
GET /parameters -> version 2026-11-05-e117a0ef, 20 cells; first cell:
    {'segment': 'rural value', 'category': 'fresh', 'elasticity': -3.054, 'se': 0.101, 'margin': 0.28, 'price': 6.663, 'weekly_units': 5278.2, 'valid_change': 0.216}
POST /scenario {'price_change': {'all': 0.05}} ->
    gross profit change: {'median': 458929, 'p5': 418879, 'p95': 501707}
POST /scenario with a misspelt segment -> 422 unknown price_change keys: ['students']; use 'all', a segment, or 'segment/category'
POST /scenario with an 80% price rise -> 422 Value error, price change for 'all' is 80%; the simulator accepts -50% to +50%
POST /compare ->
[ ... "5% everywhere" 458929 ... "coupons, top 50%" 5624836 ... "both" 6086041 ... ]
```

A misspelt segment is the most dangerous error a simulator can make silently: "students" matching nothing would return "no change", and someone would believe it. **An API that refuses bad input loudly is a causal-inference safeguard,** not just good engineering.

**Try it.** Ask the API what +8% in suburban family stores does, and check what happens if a caller sends `coupon_share: 1.5`. Create `day71/try_suburban.py`.

<details>
<summary>Solution</summary>

```python
# day71/try_suburban.py   (python -m day71.try_suburban)
from fastapi.testclient import TestClient

from service.app import app

with TestClient(app) as client:
    r = client.post("/scenario", json={"price_change": {"suburban family": 0.08}}).json()
    g = r["change"]["gross_profit"]
    print(f"+8% suburban family: median {g['median']:,.0f} (90%: {g['p5']:,.0f} to {g['p95']:,.0f})")
    print(f"{len(r['warnings'])} warnings, e.g. {r['warnings'][0] if r['warnings'] else 'none'}")
    bad = client.post("/scenario", json={"price_change": {"suburban family": 0.08}, "coupon_share": 1.5})
    print("coupon_share 1.5 ->", bad.status_code)
```

```
+8% suburban family: median 186,698 (90%: 154,835 to 218,592)
0 warnings, e.g. none
coupon_share 1.5 -> 422
```

No warnings: +8% is inside every suburban cell's supported range. A 150% coupon share is rejected before any arithmetic runs.

</details>

---

## 71.5 Day 71 mini-project: a real server, and how fast it is (75 min)

A slider app calls the simulator every time a slider moves. Above about 200 ms people notice the lag. Create `day71/latency.py`, which starts uvicorn in a background process, calls it over real HTTP, and times it:

```python
# day71/latency.py   (python -m day71.latency)
"""
Mini-project: run the API as a real server (uvicorn, in a background process), call it over HTTP like
another team's app would, and measure how long a scenario takes. A slider-driven app needs answers in well
under a second.
"""
import statistics
import subprocess
import sys
import time

import requests

URL = "http://127.0.0.1:8001"
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:app", "--port", "8001"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):                                   # wait for it to start
        try:
            requests.get(f"{URL}/health", timeout=1)
            break
        except requests.ConnectionError:
            time.sleep(0.2)
    print("health:", requests.get(f"{URL}/health").json())
    for draws in [500, 2000, 10000]:
        times = []
        for i in range(20):
            t0 = time.perf_counter()
            r = requests.post(f"{URL}/scenario", json={"price_change": {"all": 0.05}, "coupon_share": 0.5,
                                                       "draws": draws, "seed": i})
            r.raise_for_status()
            times.append((time.perf_counter() - t0) * 1000)
        print(f"{draws:>6} draws: median {statistics.median(times):5.1f} ms, slowest {max(times):5.1f} ms")
finally:
    server.terminate()
```

```bash
python -m day71.latency
```

```
health: {'status': 'ok', 'parameters_version': '2026-11-05-e117a0ef'}
   500 draws: median   6.6 ms, slowest  26.6 ms
  2000 draws: median  11.0 ms, slowest  17.9 ms
 10000 draws: median  21.9 ms, slowest  24.2 ms
```

Around 10 ms for 2,000 draws: fast enough to call on every slider move, because the expensive work (estimation) happened once, in `engine.build`. Had we re-fitted DoubleML per request, each answer would take minutes. **Separating estimation from simulation is what makes interactive causal tools possible.**

Your timings will differ by machine; the ratio between them is the point.

---

## 71.6 Save your work with Git (20 min)

```bash
git add engine service data/make_delivery_data.py models/causal_parameters.json day71 requirements.txt causalkit/demand.py
git commit -m "Day 71: parameters file, simulation engine and scenario API"
git push
```

### Day 71 checklist

- [ ] I built a versioned parameters file from the Module 6 and 7 estimators
- [ ] I can explain why estimation and simulation are separate steps
- [ ] My API validates scenarios and rejects misspelt segments and impossible values
- [ ] I called the API in-process (TestClient) and over HTTP (uvicorn)
- [ ] I measured how long a scenario takes, and know why it's fast


---

# DAY 72 — Scenario and What-If Design

| Time | Block |
|---|---|
| 0:00 – 0:40 | 72.1 A scenario is a counterfactual with a name |
| 0:40 – 1:30 | 72.2 The preset menu |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 72.3 Where the estimates stop being valid |
| 2:30 – 3:10 | 72.4 Combining scenarios |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 72.5 Day 72 mini-project: the optimiser that leaves the data |
| 4:40 – 5:00 | 72.6 Save your work with Git |

---

## 72.1 A scenario is a counterfactual with a name (40 min)

**Concept.** Every number the simulator shows is a comparison of two potential outcomes (Module 2): profit if we do X, against profit if we do nothing. The "nothing" must be stated. Here it's "this quarter's prices and volumes continue for a year", and the app says so.

Three design rules for a what-if tool:

1. **Name scenarios after decisions, not parameters.** "Coupons to the top 50%" rather than "coupon_share = 0.5".
2. **Always show the change**, the causal quantity, next to the level.
3. **Say when a scenario leaves the data.** Day 72.3.

---

## 72.2 The preset menu (50 min)

The business asks the same five questions every quarter. Put them in one place, `engine/presets.py`. The "raise" cells come from Module 6's capstone recommendation (Day 60).

```python
"""Named scenarios the business keeps asking about (Day 72). The simulator's menu, in one place."""
from engine.scenario import Scenario

MODULE6_RAISE = ["urban premium", "suburban family/fresh", "suburban family/pantry", "suburban family/personal care",
                 "student/pantry", "rural value/pantry", "rural value/fresh"]

PRESETS = {
    "status quo": Scenario(),
    "5% everywhere": Scenario(price_change={"all": 0.05}),
    "5% in the 'raise' cells": Scenario(price_change={k: 0.05 for k in MODULE6_RAISE}),
    "coupons to the top 50%": Scenario(coupon_share=0.5),
    "'raise' cells + coupons": Scenario(price_change={k: 0.05 for k in MODULE6_RAISE}, coupon_share=0.5),
    "coupons to everyone": Scenario(coupon_share=1.0),
}
```

Create `day72/presets.py`:

```python
# day72/presets.py   (python -m day72.presets)
"""
A scenario simulator is only useful if its menu matches the questions people ask. Run the named presets
through the engine and compare them in one table: change in annual gross profit against the counterfactual
(nothing changes), with a 90% range and the probability that profit goes up.
"""
import polars as pl

from engine import load_parameters, simulate
from engine.presets import PRESETS

params = load_parameters()
rows = []
for name, sc in PRESETS.items():
    r = simulate(params, sc)
    g, rev = r["change"]["gross_profit"], r["change"]["revenue"]
    rows.append({"scenario": name, "revenue change ($k)": round(rev["median"] / 1000),
                 "profit change ($k)": round(g["median"] / 1000),
                 "90% range ($k)": f"{int(round(g['p5'] / 1000)):,} to {int(round(g['p95'] / 1000)):,}",
                 "P(profit up)": "-" if r["change"]["prob_profit_up"] is None else f"{r['change']['prob_profit_up']:.0%}"})
print(f"Parameters version {params['version']}; counterfactual gross profit "
      f"${simulate(params, PRESETS['status quo'])['counterfactual']['gross_profit'] / 1e6:.1f}M a year")
with pl.Config(tbl_width_chars=120, fmt_str_lengths=40):
    print(pl.DataFrame(rows))
```

```bash
python -m day72.presets
```

```
Parameters version 2026-11-05-e117a0ef; counterfactual gross profit $15.3M a year
│ scenario                ┆ revenue change ($k) ┆ profit change ($k) ┆ 90% range ($k)   ┆ P(profit up) │
│ status quo              ┆ 0                   ┆ 0                  ┆ 0 to 0           ┆ -            │
│ 5% everywhere           ┆ -3700               ┆ 459                ┆ 419 to 502       ┆ 100%         │
│ 5% in the 'raise' cells ┆ -1097               ┆ 517                ┆ 489 to 544       ┆ 100%         │
│ coupons to the top 50%  ┆ 58749               ┆ 5625               ┆ 4,824 to 6,361   ┆ 100%         │
│ 'raise' cells + coupons ┆ 57650               ┆ 6142               ┆ 5,334 to 6,877   ┆ 100%         │
│ coupons to everyone     ┆ 70577               ┆ -2827              ┆ -3,902 to -1,840 ┆ 0%           │
```

Three lessons in one table:

- **Revenue and profit can move in opposite directions.** 5% everywhere loses $3.7M of revenue and gains $459k of profit. A dashboard that showed only revenue would recommend the opposite.
- **Targeting beats blanket rises:** the 7 "raise" cells earn more than all 20 together, because in the others demand is elastic enough that a rise loses money.
- **Coupons to everyone** raise revenue the most and lose $2.8M: Module 7's lesson, now one click away.

---

## 72.3 Where the estimates stop being valid (45 min)

**Concept.** The elasticities were identified from price moves caused by supplier costs. Those were rarely more than ±10%. Constant elasticity is a fine approximation inside that range and an assumption outside it. This is the delivery version of Module 4's overlap and Module 5's bandwidth: **don't let a tool answer questions the data can't.**

Create `day72/validity.py`:

```python
# day72/validity.py   (python -m day72.validity)
"""
Every causal estimate has a range where it was learned. The elasticities come from price moves that
supplier costs caused: mostly within about ±10%. A 30% price rise asks the model about a world it has never
seen. The simulator still answers, but it should say so, loudly.
"""
from engine import Scenario, load_parameters, simulate

params = load_parameters()
print("Range of price changes the data support, by cell:")
for c in sorted(params["pricing"], key=lambda c: c["valid_change"]):
    print(f"   {c['segment']:<16} {c['category']:<14} ±{c['valid_change']:.0%}")
for change in [0.05, 0.10, 0.20]:
    r = simulate(params, Scenario(price_change={"all": change}))
    g = r["change"]["gross_profit"]
    print(f"\n+{change:.0%} everywhere: profit change ${g['median'] / 1e3:,.0f}k; {len(r['warnings'])} warnings")
    for w in r["warnings"][:3]:
        print("   " + w)
```

```bash
python -m day72.validity
```

```
   rural value      pantry         ±10%
   ...
   rural value      fresh          ±22%

+5% everywhere: profit change $459k; 0 warnings

+10% everywhere: profit change $722k; 10 warnings
   rural value/pantry: a +10% change is outside the range seen in the data (±10%); the projection extrapolates
   ...

+20% everywhere: profit change $867k; 19 warnings
```

Profit keeps "rising" as the price rises: +20% looks better than +10%. Nineteen warnings say why not to trust it. The simulator doesn't refuse (people legitimately ask about big moves), but it never answers silently.

---

## 72.4 Combining scenarios (40 min)

Price and coupon effects are added in the engine: coupon profit is computed on the base, not on the post-price-change base. That's an assumption (no interaction), and it's listed in the result's `assumptions`. With a 5% price change and a 0.3 margin the interaction is a few percent of the coupon effect: small, but say it.

**Try it.** Compare "5% everywhere" with "10% in urban premium stores only". Which earns more, and what do the warnings say? Create `day72/try_urban.py`.

<details>
<summary>Solution</summary>

```python
# day72/try_urban.py   (python -m day72.try_urban)
from engine import Scenario, load_parameters, simulate

params = load_parameters()
for name, sc in {"5% everywhere": Scenario(price_change={"all": 0.05}),
                 "10% urban premium": Scenario(price_change={"urban premium": 0.10})}.items():
    r = simulate(params, sc)
    g = r["change"]["gross_profit"]
    print(f"{name:18} profit {g['median'] / 1e3:7.0f} $k (90%: {g['p5'] / 1e3:.0f} to {g['p95'] / 1e3:.0f}), "
          f"revenue {r['change']['revenue']['median'] / 1e3:7.0f} $k, warnings {len(r['warnings'])}")
```

```
5% everywhere      profit     459 $k (90%: 419 to 502), revenue   -3700 $k, warnings 0
10% urban premium  profit     666 $k (90%: 623 to 707), revenue    -414 $k, warnings 2
```

Raising prices only where demand is least elastic earns more profit and loses far less revenue. But two urban cells' supported range is below ±10% (pantry ±10%, personal care ±9%), so part of the $666k is extrapolation. A recommendation would be "+10% in urban premium, starting with fresh and snacks, where the data cover it".

</details>

---

## 72.5 Day 72 mini-project: the optimiser that leaves the data (75 min)

Someone will ask: "what's the **best** price?" With constant elasticity, profit has a maximum at a finite price change, and an optimiser will find it. Create `day72/best_price.py`:

```python
# day72/best_price.py   (python -m day72.best_price)
"""
Mini-project: 'what price change maximises profit?' With a constant elasticity e and margin m, the
profit ratio (m + d)(1 + d)^e / m has a maximum, and an optimiser will find it. Find the best change for
each cell on a grid, and compare it with the range the data support.
"""
import numpy as np
import polars as pl

from engine import load_parameters

params = load_parameters()
grid = np.linspace(-0.3, 0.6, 901)
rows = []
for c in params["pricing"]:
    ratio = (c["margin"] + grid) * (1 + grid) ** c["elasticity"] / c["margin"]
    best = grid[np.argmax(ratio)]
    rows.append({"segment": c["segment"], "category": c["category"], "elasticity": c["elasticity"],
                 "best change": f"{best:+.0%}", "profit gain at best": f"{ratio.max() - 1:+.1%}",
                 "supported range": f"±{c['valid_change']:.0%}",
                 "outside the data?": "YES" if abs(best) > c["valid_change"] else ""})
with pl.Config(tbl_rows=20, tbl_width_chars=130, fmt_str_lengths=20):
    print(pl.DataFrame(rows).sort("segment", "category"))
```

```bash
python -m day72.best_price
```

```
│ segment         ┆ category      ┆ elasticity ┆ best change ┆ profit gain at best ┆ supported range ┆ outside the data? │
│ rural value     ┆ fresh         ┆ -3.054     ┆ +7%         ┆ +1.7%               ┆ ±22%            ┆                   │
│ rural value     ┆ pantry        ┆ -2.832     ┆ +21%        ┆ +13.9%              ┆ ±10%            ┆ YES               │
│ ...
│ urban premium   ┆ pantry        ┆ -1.262     ┆ +60%        ┆ +106.0%             ┆ ±10%            ┆ YES               │
```

13 of 20 "optimal" prices lie outside what the data support, and all five urban premium cells hit the +60% cap of the search grid. Urban premium elasticities near −1.3 imply that profit rises with price almost without limit, which no shopper behaves like. **Optimisers find the edge of your model's validity and report it as an answer.** The honest output is the 7 cells where the optimum is inside the data, plus "test larger moves before trusting them" for the rest. Day 78 is about those tests.

---

## 72.6 Save your work with Git (20 min)

```bash
git add engine/presets.py day72
git commit -m "Day 72: scenario presets, validity warnings, optimiser vs supported range"
git push
```

### Day 72 checklist

- [ ] Every scenario I show names its counterfactual
- [ ] I can explain why revenue and profit moved in opposite directions
- [ ] The simulator warns when a price change leaves the range the data support
- [ ] I know why an unconstrained optimiser is dangerous with constant elasticities


---

# DAY 73 — Communicating Uncertainty to Executives

| Time | Block |
|---|---|
| 0:00 – 0:40 | 73.1 What executives do with a range |
| 0:40 – 1:30 | 73.2 Numbers into sentences |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 73.3 Show the distribution |
| 2:30 – 3:10 | 73.4 Which uncertainty is in the range, and which isn't |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 73.5 Day 73 mini-project: correlated errors |
| 4:40 – 5:00 | 73.6 Save your work with Git |

---

## 73.1 What executives do with a range (40 min)

**Concept.** A decision-maker reads a forecast to answer three questions: *which way does it go? how much? can it hurt us?* A confidence interval answers none of them directly. Translate:

| The analyst's output | The executive's question | What to show |
|---|---|---|
| median of the change | how much? | "about $460k a year", 2–3 significant figures |
| 5th–95th percentile | how sure? | "in 9 of 10 plausible outcomes, between $420k and $500k" |
| P(change > 0) | which way? | a probability **and** a word: "very likely (93%)" |
| P(loss > threshold) | can it hurt us? | only when material: "a 12% chance of losing more than $1M" |

Words alone are dangerous: people read "likely" as anything from 55% to 90%. Numbers alone get skimmed. Give both, using a fixed vocabulary (here the IPCC's calibrated bands), so "likely" always means 66–90%.

Precision is a message too. "$458,929" claims accuracy to the dollar from estimates with 5% standard errors. Round.

---

## 73.2 Numbers into sentences (50 min)

Create `engine/narrate.py`:

```python
"""Turn a simulation result into sentences an executive can read (Day 73)."""


def money(x: float) -> str:
    """$1.2M, $340k, -$5.1k: two or three significant figures is all the precision a forecast deserves."""
    if abs(x) < 0.5:
        return "$0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M" if x < 1e7 else f"{sign}${x / 1e6:.0f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}k" if x < 1e4 else f"{sign}${x / 1e3:.0f}k"
    return f"{sign}${x:.0f}"


def likelihood(p: float) -> str:
    """Words for probabilities, with the number alongside (words alone are read very differently by
    different people; numbers alone are skimmed). Bands follow the IPCC's calibrated language."""
    for cut, word in [(0.99, "virtually certain"), (0.9, "very likely"), (0.66, "likely"), (0.33, "about as likely as not"),
                      (0.1, "unlikely"), (0.01, "very unlikely")]:
        if p >= cut:
            return word
    return "exceptionally unlikely"


def summary(name: str, r: dict) -> str:
    g = r["change"]["gross_profit"]
    p = r["change"]["prob_profit_up"]
    lines = [f"{name}: gross profit changes by about {money(g['median'])} a year "
             f"(in 9 of 10 plausible outcomes, between {money(g['p5'])} and {money(g['p95'])})."]
    if p is not None:
        lines.append(f"Profit going up is {likelihood(p)} ({p:.0%}).")
    if r["change"]["prob_loss_over_1m"] > 0.01:
        lines.append(f"There is a {r['change']['prob_loss_over_1m']:.0%} chance of losing more than $1M.")
    if r["warnings"]:
        lines.append(f"Caution: {len(r['warnings'])} of the price changes go beyond what the data have seen.")
    return " ".join(lines)
```

Create `day73/exec_summary.py`:

```python
# day73/exec_summary.py   (python -m day73.exec_summary)
"""
The same results, written for a CFO: rounded to two significant figures, ranges as 'in 9 of 10 plausible
outcomes', probabilities in words AND numbers, and a warning when the scenario leaves the data.
"""
from engine import Scenario, load_parameters, simulate
from engine.narrate import summary
from engine.presets import PRESETS

params = load_parameters()
for name in ["5% everywhere", "5% in the 'raise' cells", "coupons to the top 50%", "coupons to everyone"]:
    print(summary(name, simulate(params, PRESETS[name])))
    print()
print(summary("10% in every student cell", simulate(params, Scenario(price_change={"student": 0.10}))))
```

```bash
python -m day73.exec_summary
```

```
5% everywhere: gross profit changes by about $459k a year (in 9 of 10 plausible outcomes, between $419k and $502k). Profit going up is virtually certain (100%).

coupons to the top 50%: gross profit changes by about $5.6M a year (in 9 of 10 plausible outcomes, between $4.8M and $6.4M). Profit going up is virtually certain (100%).

coupons to everyone: gross profit changes by about -$2.8M a year (in 9 of 10 plausible outcomes, between -$3.9M and -$1.8M). Profit going up is exceptionally unlikely (0%). There is a 100% chance of losing more than $1M.

10% in every student cell: gross profit changes by about -$143k a year (in 9 of 10 plausible outcomes, between -$178k and -$107k). Profit going up is exceptionally unlikely (0%). Caution: 2 of the price changes go beyond what the data have seen.
```

Every sentence is generated from the same result the charts use. Nobody types a number into a slide, so the memo and the app can't disagree.

**Try it.** Write the summary for coupons to the top 30% and the top 80%. Which would you recommend, and why is 80% still profitable when 100% isn't? Create `day73/try_summary.py`.

<details>
<summary>Solution</summary>

```python
# day73/try_summary.py   (python -m day73.try_summary)
from engine import Scenario, load_parameters, simulate
from engine.narrate import summary

params = load_parameters()
for share in [0.3, 0.8]:
    print(summary(f"Coupons to the top {share:.0%}", simulate(params, Scenario(coupon_share=share))), "\n")
```

```
Coupons to the top 30%: gross profit changes by about $5.7M a year (in 9 of 10 plausible outcomes, between $5.0M and $6.3M). Profit going up is virtually certain (100%).

Coupons to the top 80%: gross profit changes by about $2.4M a year (in 9 of 10 plausible outcomes, between $1.4M and $3.3M). Profit going up is virtually certain (100%).
```

30% earns more than 80% (and about as much as 50%) at a fraction of the coupon cost. Going from 80% to 100% adds the customers whose response is lowest, and some respond negatively (Module 7's sleeping dogs); their coupons cost more than they bring. The curve rises, flattens and falls: the right share is where it's flat.

</details>

---

## 73.3 Show the distribution (45 min)

A sentence gives three numbers. A picture shows the shape: whether the range is symmetric, whether a scenario straddles zero. Create `day73/fan_chart.py`:

```python
# day73/fan_chart.py   (python -m day73.fan_chart)
"""
Show the whole distribution, not just a number: for each scenario, the spread of plausible changes in
annual gross profit, with zero marked. The eye sees at once which options can lose money.
"""
import matplotlib.pyplot as plt
import numpy as np

from engine import load_parameters, simulate
from engine.presets import PRESETS

params = load_parameters()
names = ["5% everywhere", "5% in the 'raise' cells", "coupons to the top 50%", "coupons to everyone"]
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, name in enumerate(names):
    r = simulate(params, PRESETS[name].model_copy(update={"draws": 5000}))
    x = np.array(r["draws"]["gross_profit_change"]) / 1e6
    parts = ax.violinplot(x, positions=[i], vert=False, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor("#16a34a" if np.median(x) > 0 else "#dc2626")
        b.set_alpha(0.5)
    ax.plot(np.median(x), i, "o", color="#1e293b")
    ax.text(np.percentile(x, 95) + 0.2, i, f"P(up) {np.mean(x > 0):.0%}", va="center", fontsize=8)
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("change in annual gross profit ($M), vs doing nothing")
fig.tight_layout()
fig.savefig("figures/day73_distributions.png", dpi=130)
print("Saved figures/day73_distributions.png")
```

```bash
python -m day73.fan_chart
```

```
Saved figures/day73_distributions.png
```

Each violin is one scenario's 2,000 simulated profit changes; the red line is zero. Any violin that crosses the line can lose money, and you see how much of it does. Put this chart next to the sentences, not instead of them.

---

## 73.4 Which uncertainty is in the range, and which isn't (40 min)

The range covers **estimation uncertainty**: the elasticities' standard errors and the coupon curve's bootstrap. It doesn't cover:

- **Assumption uncertainty**: constant elasticity outside the data, the 15% coupon pull-forward, no interaction between price and coupons. Handle these with sliders (Day 74) and with the warnings: "if pull-forward is 30% rather than 15%, the coupon gain falls to $X".
- **Model drift**: the world changing after the estimates were made. Days 77–78.

The app's caption says this in one sentence. **A range that claims to cover everything is worse than one that says what it leaves out.**

---

## 73.5 Day 73 mini-project: correlated errors (75 min)

The engine draws each cell's elasticity independently, so 20 cells' errors partly cancel and the total looks precise. But Module 6 (Day 59) found that DML was about 3% too close to zero in **every** cell. Errors that share a cause don't cancel. The scenario has an `elasticity_correlation` setting: each draw is the estimate + SE × (√ρ · common shock + √(1−ρ) · own shock). Create `day73/correlated_errors.py`:

```python
# day73/correlated_errors.py   (python -m day73.correlated_errors)
"""
Mini-project: the simulator draws each cell's elasticity independently, so errors partly cancel across
20 cells and the total looks precise. But if every estimate shares a bias (Module 6, Day 59: DML was ~3%
too close to zero in EVERY cell), errors add up instead. How much wider is the range if they're correlated?
"""
from engine import load_parameters, simulate
from engine.narrate import money
from engine.presets import PRESETS

params = load_parameters()
sc = PRESETS["5% everywhere"]
for rho in [0.0, 0.5, 1.0]:
    r = simulate(params, sc.model_copy(update={"elasticity_correlation": rho}))
    g = r["change"]["gross_profit"]
    print(f"correlation {rho:.1f}: median {money(g['median'])}, 90% range {money(g['p5'])} to {money(g['p95'])} "
          f"(width {money(g['p95'] - g['p5'])})")
```

```bash
python -m day73.correlated_errors
```

```
correlation 0.0: median $459k, 90% range $419k to $502k (width $83k)
correlation 0.5: median $456k, 90% range $338k to $579k (width $241k)
correlation 1.0: median $455k, 90% range $292k to $622k (width $330k)
```

Same median, four times the width. With independent errors the simulator claims ±$40k; if the cells share a bias, ±$165k is honest. We don't know ρ, so the app exposes it as a slider, defaulting to 0 and explained in the caption. In a memo, quote the ρ = 0.5 range: it still shows that profit goes up, and nobody can say you hid the risk.

---

## 73.6 Save your work with Git (20 min)

```bash
git add engine/narrate.py day73
git commit -m "Day 73: executive summaries, distributions, correlated estimation errors"
git push
```

### Day 73 checklist

- [ ] My summaries give a rounded median, a 90% range in words, and a calibrated probability
- [ ] The text and the charts come from the same simulation result
- [ ] I can say which uncertainties the range covers and which it doesn't
- [ ] I know why correlated estimation errors widen the range, and by how much here

