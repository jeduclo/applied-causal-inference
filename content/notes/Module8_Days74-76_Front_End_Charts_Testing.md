# Module 8 · Decision Intelligence Delivery
## Days 74–76: The Simulator Front End, Counterfactual Charts, Testing and Deployment

**Goal by end of Day 76:** Students can build a slider-driven scenario simulator in Streamlit, chart counterfactuals without misleading, test the engine, the API and the app automatically, and package the API for deployment with a smoke test.

---

# DAY 74 — Building the Simulator Front End

| Time | Block |
|---|---|
| 0:00 – 0:40 | 74.1 Who uses the simulator, and for what |
| 0:40 – 1:30 | 74.2 A Streamlit app in one file |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 74.3 Defaults, empty states and warnings |
| 2:30 – 3:10 | 74.4 Shareable scenarios and the API switch |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 74.5 Day 74 mini-project: testing the app without a browser |
| 4:40 – 5:00 | 74.6 Save your work with Git |

---

## 74.1 Who uses the simulator, and for what (40 min)

The users are a category manager preparing a pricing review and a CFO checking the plan. Neither knows what an elasticity is; both know what a price is. So:

- **Inputs are decisions** (price change by segment, coupon share and cost), never parameters.
- **Assumptions are inputs too**, but labelled as assumptions: pull-forward and error correlation sit at the bottom of the sidebar with plain-language labels.
- **The first screen is the counterfactual.** Before anyone moves a slider, the app says what happens if nothing changes.

**Why Streamlit?** The course website is Next.js, which is the right choice for a product used by thousands. An internal decision tool for twenty people needs to be built by the analyst, in Python, in a day. Streamlit reruns the script top to bottom on every interaction; no front-end code at all. Its limits (layout control, many concurrent users) don't matter here.

---

## 74.2 A Streamlit app in one file (50 min)

The app uses three charts from `app/charts.py`. Tomorrow is about their design; for now, create the file as it is:

```python
"""Charts for counterfactuals (Day 75). Each function takes a simulation result and returns an Altair chart."""
import altair as alt
import numpy as np
import pandas as pd

BLUE, GREY, GREEN, RED = "#2563eb", "#94a3b8", "#16a34a", "#dc2626"


def projected_vs_counterfactual(r: dict, measure: str = "gross_profit") -> alt.Chart:
    """Two bars: what happens if nothing changes, and the scenario (median, with its 90% range)."""
    label = "Gross profit" if measure == "gross_profit" else "Revenue"
    p = r["projected"][measure]
    df = pd.DataFrame({"case": ["No change", "Scenario"],
                       "value": [r["counterfactual"][measure] / 1e6, p["median"] / 1e6],
                       "low": [r["counterfactual"][measure] / 1e6, p["p5"] / 1e6],
                       "high": [r["counterfactual"][measure] / 1e6, p["p95"] / 1e6]})
    base = alt.Chart(df).encode(y=alt.Y("case:N", title=None, sort=None, axis=alt.Axis(labelLimit=200)))
    bars = base.mark_bar(size=28).encode(
        x=alt.X("value:Q", title=f"{label}, $M a year"),
        color=alt.Color("case:N", scale=alt.Scale(range=[GREY, BLUE]), legend=None),
        tooltip=[alt.Tooltip("value:Q", format=",.2f", title="$M (median)"),
                 alt.Tooltip("low:Q", format=",.2f", title="5th pct"), alt.Tooltip("high:Q", format=",.2f", title="95th pct")])
    ranges = base.mark_rule(strokeWidth=2, color="#1e293b").encode(x="low:Q", x2="high:Q")
    return (bars + ranges).properties(height=120)


def change_by_segment(r: dict) -> alt.Chart:
    """Where the profit change comes from: each segment's pricing effect, plus the coupon programme."""
    cells = pd.DataFrame(r["cells"])
    seg = cells.groupby("segment", as_index=False).profit_change_median.sum()
    rows = list(seg.itertuples(index=False, name=None))
    if r["coupons"]["share"] > 0:
        rows.append(("coupon programme", r["coupons"]["profit_change"]["median"]))
    df = pd.DataFrame(rows, columns=["part", "change"])
    df["change"] = df["change"] / 1e3
    df["sign"] = df["change"].ge(0).map({True: "gain", False: "loss"})
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("part:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        x=alt.X("change:Q", title="change in annual gross profit, $k (median)"),
        color=alt.Color("sign:N", scale=alt.Scale(domain=["gain", "loss"], range=[GREEN, RED]), legend=None),
        tooltip=["part", alt.Tooltip("change:Q", format=",.0f", title="$k")]).properties(height=42 * len(df))


def outcome_distribution(r: dict) -> alt.Chart:
    """The spread of plausible outcomes, with the line that matters: zero."""
    x = np.array(r["draws"]["gross_profit_change"]) / 1e3
    counts, edges = np.histogram(x, bins=30)
    df = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
    hist = alt.Chart(df).mark_bar(color=BLUE, opacity=0.7).encode(
        x=alt.X("start:Q", title="change in annual gross profit, $k"), x2="end:Q",
        y=alt.Y("count:Q", title="simulated outcomes"), y2=alt.datum(0))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=RED, strokeWidth=2).encode(x="x:Q")
    return (hist + zero).properties(height=200)
```

Create `app/streamlit_app.py`:

```python
"""
NorthMart scenario simulator (Module 8, Days 74-75).

    streamlit run app/streamlit_app.py

By default it runs the engine in-process. Set API_URL (e.g. http://127.0.0.1:8001) to call the scenario
API instead, which is how it runs once the API is deployed.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from app.charts import change_by_segment, outcome_distribution, projected_vs_counterfactual  # noqa: E402
from engine import Scenario, load_parameters, simulate  # noqa: E402
from engine.narrate import money, summary  # noqa: E402


def md(text: str) -> str:
    """Streamlit reads $...$ as LaTeX; escape dollar signs in anything that shows money."""
    return text.replace("$", "\\$")

st.set_page_config(page_title="NorthMart scenario simulator", layout="wide")
API_URL = os.environ.get("API_URL")


@st.cache_data
def get_parameters():
    if API_URL:
        return requests.get(f"{API_URL}/parameters", timeout=10).json()
    return load_parameters(os.environ.get("PARAMS_PATH", "models/causal_parameters.json"))


@st.cache_data
def run(scenario_json: str):
    sc = Scenario.model_validate_json(scenario_json)
    if API_URL:
        resp = requests.post(f"{API_URL}/scenario", data=scenario_json, params={"include_draws": True}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    return simulate(get_parameters(), sc)


params = get_parameters()
segments = sorted({c["segment"] for c in params["pricing"]})

st.title("NorthMart scenario simulator")
st.caption(f"Causal parameters version {params['version']} · projections are a year ahead, against the "
           "counterfactual of changing nothing")

# Every scenario is shareable: its settings live in the URL, e.g. ?price=5&coupons=50
q = st.query_params


def from_url(key, default, cast=int):
    try:
        return cast(q.get(key, default))
    except ValueError:
        return default


with st.sidebar:
    st.header("Pricing")
    everywhere = st.slider("Price change everywhere (%)", -10, 15, from_url("price", 0), 1)
    overrides = {}
    with st.expander("Override by segment"):
        for s in segments:
            if st.checkbox(s, key=f"use_{s}"):
                overrides[s] = st.slider(f"{s} (%)", -10, 15, everywhere, 1, key=f"pct_{s}")
    st.header("Coupons")
    share = st.slider("Share of customers couponed (top by predicted response, %)", 0, 100, from_url("coupons", 0), 10)
    cost = st.slider("Cost per coupon ($)", 2.0, 10.0, from_url("cost", 5.0, float), 0.5)
    pull = st.slider("Pull-forward: extra spend that is just earlier purchases (%)", 0, 40,
                     int(params["assumptions"]["pull_forward"] * 100), 5)
    st.header("Uncertainty")
    rho = st.slider("How much do estimation errors move together? (0 = independent)", 0.0, 1.0, 0.0, 0.25)

changes = {"all": everywhere / 100} | {s: v / 100 for s, v in overrides.items()}
scenario = Scenario(price_change={k: v for k, v in changes.items() if v != 0 or k != "all"},
                    coupon_share=share / 100, coupon_cost=cost, pull_forward=pull / 100,
                    elasticity_correlation=rho)
r = run(scenario.model_dump_json())
st.query_params.update({"price": everywhere, "coupons": share, "cost": cost})

if not scenario.price_change and scenario.coupon_share == 0:
    st.info(md(f"Counterfactual: gross profit {money(r['counterfactual']['gross_profit'])} and revenue "
               f"{money(r['counterfactual']['revenue'])} a year if nothing changes. Move a slider to build a scenario."))
    st.stop()

g, rev = r["change"]["gross_profit"], r["change"]["revenue"]
c1, c2, c3 = st.columns(3)
c1.metric("Gross profit, change per year", md(money(g["median"])), help="Median of the simulated outcomes")
c2.metric("90% range", md(f"{money(g['p5'])} to {money(g['p95'])}"))
p_up = r["change"]["prob_profit_up"]
c3.metric("Chance profit goes up", "-" if p_up is None else f"{p_up:.0%}")
st.write(md(summary("This scenario", r)))
if r["warnings"]:
    st.warning(f"{len(r['warnings'])} price changes go beyond what the data have seen: those projections extrapolate.")
    with st.expander("Which ones?"):
        for w in r["warnings"]:
            st.write(md(w))

left, right = st.columns(2)
with left:
    st.subheader("Projected vs counterfactual")
    st.altair_chart(projected_vs_counterfactual(r, "gross_profit"), width="stretch")
    st.altair_chart(projected_vs_counterfactual(r, "revenue"), width="stretch")
with right:
    st.subheader("Where the change comes from")
    st.altair_chart(change_by_segment(r), width="stretch")
    st.subheader("Plausible outcomes")
    st.altair_chart(outcome_distribution(r), width="stretch")

with st.expander("Every segment x category cell"):
    cells = pd.DataFrame(r["cells"])
    st.dataframe(cells.assign(price_change=cells.price_change.map("{:+.0%}".format),
                              profit_change_median=(cells.profit_change_median / 1e3).round(1))
                 [["segment", "category", "price_change", "profit_change_median", "prob_profit_up"]]
                 .rename(columns={"profit_change_median": "profit change ($k)", "prob_profit_up": "P(up)"}),
                 hide_index=True)
st.caption("Uncertainty shown is from the causal estimates (and their correlation, if set). It does not include "
           "uncertainty in assumptions such as pull-forward or coupon fatigue: change those with the sliders.")
```

```bash
streamlit run app/streamlit_app.py
```

Codespaces offers to open port 8501. Move the "Price change everywhere" slider to +5%: the metrics, sentence and charts update in about a tenth of a second. Then tick "student" and set it to +10%. The warning appears, and the expander lists the two student cells whose supported range is below 10%.

Two details worth copying:

- `@st.cache_data` on the parameters load and on `run`: the same scenario is never simulated twice.
- `md()` escapes dollar signs. Streamlit reads `$...$` as LaTeX, so "$459k to $502k" would otherwise render as a garbled formula. Every money string goes through it.

---

## 74.3 Defaults, empty states and warnings (45 min)

Good tools are designed for the moment nobody has done anything yet, and for the moment someone asks something unreasonable.

| State | What the app does |
|---|---|
| Nothing moved | `st.info` with the counterfactual, then `st.stop()`: no "change: $0" charts |
| Price change outside the data | One `st.warning` with the count, details in an expander (not 19 yellow boxes) |
| Unknown P(profit up) | "-" rather than "0%" |
| Tiny numbers | `money()` prints "$0", never "-$0" |

Each of these was a bug in the first version of this app. Test them by hand now: set everything to zero; set +15% everywhere; set coupons to 0% and price to 0% but correlation to 1.

---

## 74.4 Shareable scenarios and the API switch (40 min)

**Shareable scenarios.** `st.query_params` writes the main sliders into the URL. Copy the address bar and send it: the recipient opens the same scenario. This turns a tool into a conversation: "here's what I mean, click this".

**The API switch.** By default the app imports the engine. Set `API_URL` and it calls the Day 71 service instead:

```bash
uvicorn service.app:app --port 8001 &
API_URL=http://127.0.0.1:8001 streamlit run app/streamlit_app.py
```

Same answers, but now the app and the pricing system use one deployed copy of the parameters. When the parameters are rebuilt, both change together. Stop the background server afterwards with `kill %1`.

---

## 74.5 Day 74 mini-project: testing the app without a browser (75 min)

Clicking through an app after every change doesn't scale. Streamlit's `AppTest` runs the script headless, lets you set widgets and read the outputs.

**Try it.** With AppTest, set coupons to the top 50% and print the three metrics; then raise the coupon cost to $9 and print the profit change again. Create `day74/try_apptest.py`. (AppTest resolves paths relative to the calling file.)

<details>
<summary>Solution</summary>

```python
# day74/try_apptest.py   (python -m day74.try_apptest)
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
at.slider[1].set_value(50).run()          # coupons to the top 50%
for m in at.metric:
    print(f"{m.label}: {m.value}")
at.slider[2].set_value(9.0).run()         # dearer coupons
print("at $9 a coupon:", at.metric[0].value)
```

```
Gross profit, change per year: \$5.6M
90% range: \$4.8M to \$6.4M
Chance profit goes up: 100%
at $9 a coupon: -\$4.0M
```

(The backslashes are the `md()` escapes; on screen they show as plain dollar signs. A warning about a missing ScriptRunContext is normal in headless mode.)

At $5 a coupon, couponing half the base earns $5.6M; at $9 it loses $4.0M. The coupon cost is a contract term the CFO controls, and the app shows its break-even in seconds. On Day 76 this becomes an automated test.

</details>

---

## 74.6 Save your work with Git (20 min)

```bash
git add app day74
git commit -m "Day 74: Streamlit scenario simulator"
git push
```

### Day 74 checklist

- [ ] My app's inputs are decisions and labelled assumptions, not model parameters
- [ ] The first screen shows the counterfactual
- [ ] Extrapolation warnings are visible but not overwhelming
- [ ] A scenario can be shared as a link
- [ ] I can drive the app from a script with AppTest


---

# DAY 75 — Charts for Counterfactuals

| Time | Block |
|---|---|
| 0:00 – 0:40 | 75.1 What a counterfactual chart must show |
| 0:40 – 1:30 | 75.2 Reading `app/charts.py` |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 75.3 Charts outside the app |
| 2:30 – 3:10 | 75.4 Decomposing a change |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 75.5 Day 75 mini-project: the misleading chart |
| 4:40 – 5:00 | 75.6 Save your work with Git |

---

## 75.1 What a counterfactual chart must show (40 min)

A causal effect is a difference between something we will see and something we never will. Charts of effects fail in three predictable ways:

1. **The counterfactual is missing.** "Profit next year: $15.8M" with nothing to compare it to.
2. **The uncertainty is missing.** One bar, one number, and the reader assumes it's exact.
3. **The scale exaggerates.** An axis that starts at $15M turns a 3% change into a tower.

So every chart in the app has the counterfactual in grey, the scenario in colour, a 90% range on the scenario, and a zero-based axis (or, for changes, a marked zero line).

---

## 75.2 Reading `app/charts.py` (50 min)

Reopen `app/charts.py` from Day 74 and go through the three functions:

| Function | Answers | Design choices |
|---|---|---|
| `projected_vs_counterfactual` | how big is the change? | horizontal bars from zero; grey = no change, blue = scenario; a black rule spans the 5th–95th percentile; tooltips give exact values |
| `change_by_segment` | where does it come from? | one bar per segment plus coupons; green gains, red losses; height grows with the number of rows so no label is hidden |
| `outcome_distribution` | can it lose money? | histogram of the 500 returned draws (binned with numpy, drawn as rectangles from zero); a red line at zero |

Two bugs from building these, both worth knowing:

- Altair's automatic binning drew the histogram as dashes until each bar was given an explicit bottom (`y2=alt.datum(0)`) and explicit left/right edges (`x`, `x2` from `np.histogram`).
- Altair hides overlapping axis labels by default; with five rows it silently dropped two segment names. `labelOverlap=False` plus a taller chart fixed it.

---

## 75.3 Charts outside the app (45 min)

Charts also go into memos and slide decks, where nobody will run Streamlit. Altair saves standalone HTML (interactive, tooltips work) without a server. Create `day75/save_charts.py`:

```python
# day75/save_charts.py   (python -m day75.save_charts)
"""
The app's three counterfactual charts, saved as standalone HTML pages for a scenario, so they can be
reviewed (or dropped into a memo) without running the app. Open the files in a browser.
"""
from app.charts import change_by_segment, outcome_distribution, projected_vs_counterfactual
from engine import load_parameters, simulate
from engine.presets import PRESETS

r = simulate(load_parameters(), PRESETS["'raise' cells + coupons"])
for name, chart in [("projected_vs_counterfactual", projected_vs_counterfactual(r)),
                    ("change_by_segment", change_by_segment(r)),
                    ("outcome_distribution", outcome_distribution(r))]:
    chart.properties(width=520).save(f"figures/day75_{name}.html")
    print(f"Saved figures/day75_{name}.html")
```

```bash
python -m day75.save_charts
```

```
Saved figures/day75_projected_vs_counterfactual.html
Saved figures/day75_change_by_segment.html
Saved figures/day75_outcome_distribution.html
```

Open them in the browser (right-click → Open with Live Server, or download from the file explorer). They're the same charts as the app, because they're the same functions.

---

## 75.4 Decomposing a change (40 min)

"Where does the $6.1M come from?" is the second question every executive asks. The segment chart answers it, and its sum equals the headline because the engine's profit change is a sum over cells plus the coupon programme. Decompositions are only honest when they add up. (A non-additive model would need Shapley-style attributions, with their own assumptions.)

**Try it.** Save the segment chart for "5% in the 'raise' cells" and count how many cells change price and how many of those gain profit. `MODULE6_RAISE` has 7 entries: why do more cells change? Create `day75/try_segment_chart.py`.

<details>
<summary>Solution</summary>

```python
# day75/try_segment_chart.py   (python -m day75.try_segment_chart)
from pathlib import Path

from app.charts import change_by_segment
from engine import load_parameters, simulate
from engine.presets import PRESETS

Path("figures").mkdir(exist_ok=True)
r = simulate(load_parameters(), PRESETS["5% in the 'raise' cells"])
change_by_segment(r).save("figures/day75_raise_cells.html")
changed = [c for c in r["cells"] if c["price_change"] != 0]
print(f"Saved figures/day75_raise_cells.html; {len(changed)} of {len(r['cells'])} cells change price, "
      f"{sum(c['profit_change_median'] > 0 for c in changed)} of them gain; "
      f"cells with 'profit up' counted on the outcome alone: {sum(c['profit_change_median'] > 0 for c in r['cells'])}")
```

```
Saved figures/day75_raise_cells.html; 11 of 20 cells change price, 11 of them gain; cells with 'profit up' counted on the outcome alone: 14
```

One of the 7 entries is a whole segment, "urban premium", which covers its 5 categories: 6 single cells + 5 urban cells = 11 price rises, and all 11 gain. Counting "profit up" on the outcome alone gives 14: three untouched cells have medians that are positive by a floating-point hair (around 1e-10). Two lessons: one scenario key can move many cells (check the app's cell table to see what a scenario actually changed), and count on the decision variable, not on a noisy outcome.

</details>

---

## 75.5 Day 75 mini-project: the misleading chart (75 min)

Make the same numbers into two charts and decide which belongs in front of a CFO. Create `day75/misleading_chart.py`:

```python
# day75/misleading_chart.py   (python -m day75.misleading_chart)
"""
Mini-project: the same numbers, two charts. A bar chart whose axis starts at $15M makes a 3% gain look
like a doubling; one that starts at zero shows it as it is. And a chart without the range hides that the
'gain' could be smaller. Make both and decide which belongs in front of a CFO.
"""
import altair as alt
import pandas as pd

from engine import load_parameters, simulate
from engine.presets import PRESETS

r = simulate(load_parameters(), PRESETS["5% in the 'raise' cells"])
df = pd.DataFrame({"case": ["No change", "Scenario"],
                   "profit": [r["counterfactual"]["gross_profit"] / 1e6, r["projected"]["gross_profit"]["median"] / 1e6]})
misleading = alt.Chart(df, title="Profit soars!").mark_bar().encode(
    x="case:N", y=alt.Y("profit:Q", scale=alt.Scale(domain=[15.2, 15.9]), title="$M"))
honest = alt.Chart(df, title="Gross profit, $M a year: +3%").mark_bar().encode(
    x="case:N", y=alt.Y("profit:Q", scale=alt.Scale(zero=True), title="$M"))
(misleading | honest).save("figures/day75_misleading.html")
print(f"No change ${df.profit[0]:.2f}M, scenario ${df.profit[1]:.2f}M "
      f"(+{df.profit[1] / df.profit[0] - 1:.1%}). Saved figures/day75_misleading.html")
```

```bash
python -m day75.misleading_chart
```

```
No change $15.33M, scenario $15.85M (+3.4%). Saved figures/day75_misleading.html
```

On the left, the "Profit soars!" chart's axis runs from $15.2M to $15.9M: the scenario bar is five times the height of the no-change bar. On the right, from zero: two nearly equal bars, titled with the actual change. Neither chart has the range, so both overstate certainty; the app's version has it.

Write three sentences in `day75/NOTES.md`: which chart you'd show, what's still missing, and how you'd show a 3% change so it doesn't look like nothing (answer: chart the **change** with its range and a zero line, as `outcome_distribution` does, rather than the levels).

---

## 75.6 Save your work with Git (20 min)

```bash
git add day75
git commit -m "Day 75: counterfactual charts, exports, misleading-chart comparison"
git push
```

### Day 75 checklist

- [ ] Every chart shows the counterfactual, the scenario and its range
- [ ] Axes of levels start at zero; charts of changes mark zero
- [ ] My decomposition adds up to the headline number
- [ ] I can export the charts for a memo without running the app


---

# DAY 76 — Testing and Deploying the App

| Time | Block |
|---|---|
| 0:00 – 0:40 | 76.1 What can break in a causal tool |
| 0:40 – 1:30 | 76.2 Testing the engine with pytest |
| 1:30 – 1:45 | Break |
| 1:45 – 2:30 | 76.3 Testing the API and the app |
| 2:30 – 3:10 | 76.4 Continuous integration with GitHub Actions |
| 3:10 – 3:25 | Break |
| 3:25 – 4:40 | 76.5 Day 76 mini-project: a container and a smoke test |
| 4:40 – 5:00 | 76.6 Save your work with Git |

---

## 76.1 What can break in a causal tool (40 min)

Ordinary software breaks loudly: it crashes. A causal tool usually breaks quietly: it returns a plausible wrong number. The tests worth writing are about **properties the economics guarantees**:

| Property | Why it must hold |
|---|---|
| No change → change of exactly zero | the counterfactual is the scenario with nothing done |
| Price rise with elastic demand → revenue falls | sign of the elasticity |
| Higher coupon cost → lower profit | monotone in a cost |
| More specific key wins ("student/pantry" over "student" over "all") | otherwise a scenario means something else than it says |
| Same seed → same answer | reproducibility: a memo's numbers can be regenerated |
| Bad input → error, not a number | Day 71's misspelt segment |

Tests use a **tiny parameters fixture** with two cells, not the real file: they must pass whatever tomorrow's estimates are.

---

## 76.2 Testing the engine with pytest (50 min)

Create `pytest.ini` in the repository root, then `tests/conftest.py` (the fixture) and `tests/test_engine.py`:

```ini
[pytest]
testpaths = tests
filterwarnings = ignore::DeprecationWarning
```

```python
"""Shared test fixtures: a tiny, hand-checkable parameter set, so tests don't depend on estimation."""
import pytest


@pytest.fixture
def tiny_params():
    return {
        "version": "test", "created": "2026-01-01", "sources": {},
        "pricing": [
            {"segment": "a", "category": "x", "elasticity": -2.0, "se": 0.0, "margin": 0.3, "price": 10.0,
             "weekly_units": 100.0, "valid_change": 0.1},
            {"segment": "b", "category": "x", "elasticity": -1.0, "se": 0.5, "margin": 0.3, "price": 10.0,
             "weekly_units": 100.0, "valid_change": 0.1},
        ],
        "coupons": {"customers": 1000, "campaigns_per_year": 12, "margin": 0.3, "shares": [0.0, 0.5, 1.0],
                    "uplift": [0.0, 10.0, 15.0], "uplift_se": [0.0, 1.0, 1.0]},
        "assumptions": {"pull_forward": 0.0},
    }
```

```python
"""Unit tests for the scenario engine (Day 76). Run: pytest"""
import pytest
from pydantic import ValidationError

from engine import Scenario, simulate
from engine.parameters import validate


def test_no_change_means_no_change(tiny_params):
    r = simulate(tiny_params, Scenario())
    assert r["change"]["gross_profit"]["median"] == 0
    assert r["change"]["revenue"]["p95"] == 0
    assert r["change"]["prob_profit_up"] is None


def test_pricing_matches_the_formula(tiny_params):
    # cell a has se = 0, so it's deterministic: profit = (m + d) * p * units * 52 * (1 + d) ** e
    r = simulate(tiny_params, Scenario(price_change={"a": 0.05}))
    cell = next(c for c in r["cells"] if c["segment"] == "a")
    expected = (0.3 + 0.05) * 10 * 100 * 52 * 1.05 ** -2.0 - 0.3 * 10 * 100 * 52
    assert cell["profit_change_median"] == pytest.approx(expected)


def test_most_specific_key_wins():
    sc = Scenario(price_change={"all": 0.05, "a": 0.0, "b/x": 0.02})
    assert sc.change_for("a", "x") == 0.0
    assert sc.change_for("b", "x") == 0.02
    assert sc.change_for("c", "y") == 0.05


def test_coupon_cost_lowers_profit(tiny_params):
    cheap = simulate(tiny_params, Scenario(coupon_share=0.5, coupon_cost=2, seed=1))
    dear = simulate(tiny_params, Scenario(coupon_share=0.5, coupon_cost=8, seed=1))
    assert dear["change"]["gross_profit"]["median"] < cheap["change"]["gross_profit"]["median"]


def test_same_seed_same_answer(tiny_params):
    sc = Scenario(price_change={"all": 0.05}, coupon_share=0.3, seed=7)
    assert simulate(tiny_params, sc) == simulate(tiny_params, sc)


def test_extrapolation_is_flagged(tiny_params):
    assert simulate(tiny_params, Scenario(price_change={"all": 0.05}))["warnings"] == []
    assert len(simulate(tiny_params, Scenario(price_change={"all": 0.15}))["warnings"]) == 2


def test_correlated_errors_widen_the_range(tiny_params):
    tiny_params["pricing"][0]["se"] = 0.5
    width = {}
    for rho in [0.0, 1.0]:
        g = simulate(tiny_params, Scenario(price_change={"all": 0.05}, elasticity_correlation=rho))["change"]["gross_profit"]
        width[rho] = g["p95"] - g["p5"]
    assert width[1.0] > width[0.0]


def test_bad_scenarios_are_rejected():
    with pytest.raises(ValidationError):
        Scenario(price_change={"all": 0.8})
    with pytest.raises(ValidationError):
        Scenario(coupon_share=1.5)


def test_bad_parameters_are_rejected(tiny_params):
    tiny_params["pricing"][0]["elasticity"] = 0.5
    with pytest.raises(ValueError, match="non-negative elasticity"):
        validate(tiny_params)
```

```bash
python -m pytest -q
```

The engine tests run in under a second. When one fails, the name says which promise was broken.

**Try it.** Add `tests/test_more_properties.py` with two more promises: a 5% rise in a cell with elasticity −2 lowers revenue but raises profit; and more pull-forward always means less coupon profit.

<details>
<summary>Solution</summary>

```python
# tests/test_more_properties.py   (python -m pytest -q)
from engine import Scenario, simulate


def test_price_rise_lowers_revenue_when_demand_is_elastic(tiny_params):
    r = simulate(tiny_params, Scenario(price_change={"a": 0.05}))       # cell a: elasticity -2, no noise
    assert r["change"]["revenue"]["median"] < 0
    assert r["change"]["gross_profit"]["median"] > 0                   # ... while profit rises


def test_pull_forward_lowers_coupon_profit(tiny_params):
    # spend that is only pulled forward isn't extra: the more of it, the less coupons earn (same draws, seed 0)
    profits = [simulate(tiny_params, Scenario(coupon_share=0.5, pull_forward=p))["change"]["gross_profit"]["median"]
               for p in [0.0, 0.3, 0.6, 0.9]]
    assert profits == sorted(profits, reverse=True)
```

```bash
python -m pytest -q
```

```
16 passed
```

(16 once the API and app tests below are in place.) The first test pins down Day 72's lesson that revenue and profit can move in opposite directions. The second checks a whole curve at once: the list of profits must already be sorted from high to low.

</details>

---

## 76.3 Testing the API and the app (45 min)

The API tests point the service at the tiny fixture by setting `PARAMS_PATH` with pytest's `monkeypatch` and clearing the cached parameters. The app test is Day 74's AppTest, with assertions. Create `tests/test_api.py` and `tests/test_app.py`:

```python
"""API tests (Day 76): the contract other teams rely on."""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tiny_params, tmp_path, monkeypatch):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(tiny_params))
    monkeypatch.setenv("PARAMS_PATH", str(path))
    from service import app as app_module
    app_module.params.cache_clear()
    return TestClient(app_module.app)


def test_health_reports_the_parameters_version(client):
    assert client.get("/health").json() == {"status": "ok", "parameters_version": "test"}


def test_scenario_returns_projection_and_counterfactual(client):
    body = client.post("/scenario", json={"price_change": {"all": 0.05}}).json()
    assert {"counterfactual", "projected", "change", "cells", "warnings"} <= body.keys()
    assert "draws" not in body


def test_unknown_segment_is_a_422(client):
    r = client.post("/scenario", json={"price_change": {"zzz": 0.05}})
    assert r.status_code == 422
    assert "zzz" in r.json()["detail"]


def test_compare_keeps_order(client):
    rows = client.post("/compare", json=[{"name": "one", "scenario": {}},
                                         {"name": "two", "scenario": {"coupon_share": 0.5}}]).json()["rows"]
    assert [r["name"] for r in rows] == ["one", "two"]
```

```python
"""Streamlit app tests (Day 76), with Streamlit's own AppTest: runs the script without a browser."""
from streamlit.testing.v1 import AppTest


def test_app_runs_and_reacts_to_a_slider():
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()   # relative to this file
    assert not at.exception
    assert "Move a slider" in at.info[0].value
    at.slider[0].set_value(5).run()                      # price change everywhere: +5%
    assert not at.exception
    assert at.metric[0].label == "Gross profit, change per year"
    assert at.metric[2].value == "100%"
```

```bash
python -m pytest -q
```

```
16 passed, 1 warning in 8.41s
```

(The warning is a deprecation notice from the test client's HTTP library, not from our code.)

---

## 76.4 Continuous integration with GitHub Actions (40 min)

Tests only protect you if they run. GitHub Actions runs them on every push and pull request, on a clean machine, with the pinned requirements. Create `.github/workflows/ci.yml`:

```yaml
# Runs the tests on every push and pull request (Day 76).
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python -m pytest -q
```

Commit and push, then open the repository's **Actions** tab: a green tick means a fresh machine could install everything and every promise held. Break one on purpose (make `simulate` return revenue for profit), push, watch it go red, and revert. From now on, a red cross on a pull request blocks the merge.

---

## 76.5 Day 76 mini-project: a container and a smoke test (75 min)

**Deployment** means running the API somewhere other teams can reach. Whatever the platform (Cloud Run, Azure Container Apps, Fly.io, an internal Kubernetes cluster), the unit you hand over is a **container image**. The API needs only FastAPI, numpy and pydantic at run time, not DoubleML or LightGBM, so it gets its own small requirements file. Create `requirements-serve.txt` and `Dockerfile`:

```text
# Only what the API needs at run time (Day 76): the estimation stack stays out of the container.
fastapi==0.150.0
uvicorn==0.54.0
numpy==2.4.6
pydantic==2.14.2
```

```dockerfile
# The scenario API as a container (Day 76).
#   docker build -t northmart-scenarios .
#   docker run -p 8001:8001 northmart-scenarios
FROM python:3.13-slim
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY engine/ engine/
COPY service/ service/
COPY models/causal_parameters.json models/causal_parameters.json
EXPOSE 8001
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

Codespaces has Docker:

```bash
docker build -t northmart-scenarios .
docker run -d -p 8001:8001 --name scenarios northmart-scenarios
```

Note what goes into the image: the engine, the service and **one** parameters file. The image is the model version. Rebuilding the parameters means building and deploying a new image, which is exactly the audit trail you want.

After every deployment, run a **smoke test** against the live URL: is it up, is it serving the parameters version we expect, and does a known scenario give the known answer? Create `day76/smoke_test.py`:

```python
# day76/smoke_test.py   (python -m day76.smoke_test https://your-api.example.com)
"""
A smoke test for a DEPLOYED API: is it up, is it serving the parameters version we expect, and does a
known scenario give the known answer? Run it after every deployment (and from the scheduler, daily).
Exits with an error code if anything is wrong, so CI or a scheduler can alert on it.
"""
import sys

import requests

from engine import Scenario, load_parameters, simulate

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
expected = load_parameters()
problems = []
health = requests.get(f"{url}/health", timeout=10).json()
if health.get("parameters_version") != expected["version"]:
    problems.append(f"serving parameters {health.get('parameters_version')}, expected {expected['version']}")
known = Scenario(price_change={"all": 0.05}, seed=0)
local = simulate(expected, known)["change"]["gross_profit"]["median"]
remote = requests.post(f"{url}/scenario", json=known.model_dump(), timeout=30).json()["change"]["gross_profit"]["median"]
if abs(local - remote) > 1:
    problems.append(f"known scenario gives {remote:,.0f}, expected {local:,.0f}")
print(f"{url}: " + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
sys.exit(1 if problems else 0)
```

```bash
python -m day76.smoke_test            # defaults to http://127.0.0.1:8001
```

```
http://127.0.0.1:8001: OK
```

(If nothing is listening on the port you'll get a connection error: start the container, or `uvicorn service.app:app --port 8001 &`.) Now rebuild the parameters (`python -m engine.build`) on different data and rerun the smoke test without rebuilding the image: it reports that the server is serving an older version than the repository expects. That mismatch is the most common production bug in model serving, and now it's caught in seconds. Stop the container with `docker rm -f scenarios`.

---

## 76.6 Save your work with Git (20 min)

```bash
git add tests pytest.ini .github Dockerfile requirements-serve.txt day76
git commit -m "Day 76: engine, API and app tests; CI; container and smoke test"
git push
```

### Day 76 checklist

- [ ] My tests check properties the economics guarantees, on a tiny fixture
- [ ] The API and the app are tested without starting a server or a browser
- [ ] Tests run automatically on every push
- [ ] The API runs in a container with only its run-time dependencies
- [ ] A smoke test confirms the deployed version and a known answer

