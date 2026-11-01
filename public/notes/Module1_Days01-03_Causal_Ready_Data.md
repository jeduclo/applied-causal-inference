# Module 1 · Causal-Ready Data Engineering
## Days 1–3: Setup, Polars, DuckDB and the Customer × Week Panel

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts with basic Python and statistics
**Environment:** GitHub Codespaces (VS Code in the browser) or Google Colab
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 3:** Students turn four messy raw files from a simulated online retailer into a tested customer × week panel with treatment, timing, outcome and pre-period covariate columns, ready for causal analysis.

---

### How to use this guide

Each day is split into timed blocks. Every block follows the same rhythm:

1. **Concept**: a short explanation, always tied to a causal question.
2. **Examples**: small, progressive code. Type it yourself; don't copy-paste.
3. **Try it**: a quick exercise. Solutions are hidden in collapsible blocks.

### The running case: NorthMart

NorthMart is a simulated Canadian online retailer. On **Monday 2 March 2026** marketing sent the **SPRING26** email to about a third of customers. Management asks one question:

> **Did the spring email make customers buy more?**

We will answer that question properly in Modules 2–7. Module 1 builds the data that makes an honest answer possible.

Because the data is simulated, we know the truth: the email raised order frequency by **about 15%**. We will use that truth to show how far naive analysis can be from it. One more detail matters: marketing did **not** pick recipients at random. It favoured heavy shoppers and loyalty members.

### Tested versions

Every script in these notes was run on 31 October 2026 with Python 3.13, Polars 2.1.0 and DuckDB 1.5.6 (full list in `requirements.txt`). DuckDB 2.0 exists, but the ingestion tool we use from Day 5 (dlt) does not support it yet, so the whole course stays on 1.5.6. Answers you find online may target other versions; trust the pinned ones.

### Target folder structure (built up over three days)

```
causal-decision-lab/
├── .gitignore
├── requirements.txt
├── data/
│   ├── make_northmart.py        # provided: generates the raw data
│   ├── raw/                     # 4 CSV files (not committed)
│   ├── clean/                   # 4 Parquet files (not committed)
│   ├── marts/                   # analysis-ready tables (not committed)
│   └── northmart.duckdb         # local warehouse (not committed)
├── docs/
│   └── data_dictionary.md
├── sql/
│   └── panel_customer_week.sql
├── day01/
│   ├── hello_causal.py
│   ├── roles.py
│   ├── first_look.py
│   └── data_dictionary.py
├── day02/
│   ├── expressions.py
│   ├── clean.py
│   ├── summaries.py
│   ├── lazy_demo.py
│   └── naive_comparison.py
└── day03/
    ├── duck_intro.py
    ├── build_warehouse.py
    ├── calendar_spine.py
    ├── windows.py
    └── build_panel.py
```

---

# DAY 1 — From a Causal Question to a Dataset

| Time | Block |
|---|---|
| 0:00 – 0:40 | 1.1 Codespaces and your course repository |
| 0:40 – 1:20 | 1.2 Virtual environments and pinned packages |
| 1:20 – 1:45 | 1.3 Generating the NorthMart data |
| 1:45 – 2:00 | Break |
| 2:00 – 2:50 | 1.4 The causal vocabulary in plain Python |
| 2:50 – 3:45 | 1.5 A first look with Polars |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 1.6 Day 1 mini-project: a data dictionary with causal roles |
| 4:45 – 5:00 | 1.7 Save your work with Git |

---

## 1.1 Codespaces and your course repository (40 min)

**Concept.** A causal estimate is only credible if someone else can rerun it and get the same number. That starts on Day 1: one repository, one environment, every step in a script.

**Step 1 — Create a repository.** On GitHub, click **New repository**, name it `causal-decision-lab`, tick **Add a README**, and click **Create repository**.

**Step 2 — Launch a Codespace.** On the repository page click **Code → Codespaces → Create codespace on main**. After a minute you'll see VS Code in your browser.

**Step 3 — Find the terminal.** At the bottom of the screen there is a **Terminal** panel (if not, press `` Ctrl+` ``). Check Python:

```bash
python --version
```

Any Python from 3.11 upward works. Now create the folders we will fill over three days:

```bash
mkdir -p data docs sql day01 day02 day03
ls
```

**Step 4 — Your first script.** In the Explorer (left panel), create `day01/hello_causal.py`:

```python
# day01/hello_causal.py
question = "Did the spring email make customers buy more?"
treatment = "received the SPRING26 email"
outcome = "number of orders in the 8 weeks after 2 March 2026"

print("Causal question:", question)
print("Treatment:", treatment)
print("Outcome:", outcome)
```

Run it from the repository root:

```bash
python day01/hello_causal.py
```

```
Causal question: Did the spring email make customers buy more?
Treatment: received the SPRING26 email
Outcome: number of orders in the 8 weeks after 2 March 2026
```

> Always run scripts **from the repository root** (`causal-decision-lab/`). All file paths in this course, like `data/raw/orders.csv`, are relative to it.

> **Using Colab instead?** Start each notebook with `!git clone https://github.com/<you>/causal-decision-lab.git` and `%cd causal-decision-lab`, then run scripts with `!python day01/hello_causal.py`.

**Try it.** Edit the script so it also prints a variable `unit = "customer"` and a variable `period = "weekly"`.

<details>
<summary>Solution</summary>

```python
unit = "customer"
period = "weekly"
print("Unit of analysis:", unit)
print("Time period:", period)
```

</details>

---

## 1.2 Virtual environments and pinned packages (40 min)

**Concept.** Polars 1 and Polars 2 give different answers to some calls. If two analysts use different versions, they can get different numbers from the same code. A **virtual environment** is a private set of packages for this project, and `requirements.txt` pins the exact versions.

**Example 1 — Create and activate the environment.**

```bash
python -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. On Windows (outside Codespaces) the second line is `.venv\Scripts\activate`.

**Example 2 — Pin the packages.** Create `requirements.txt` in the repository root:

```text
# Pinned so every student gets identical results (tested 31 Oct 2026, Python 3.13)
# Days 1-3: data wrangling
polars==2.1.0
duckdb==1.5.6          # not 2.x yet: dlt (Day 5) supports DuckDB 1.x only
numpy==2.5.4
pyarrow==26.0.0
# Days 4-5: APIs and ingestion
requests==2.34.2
fastapi==0.150.0
uvicorn==0.54.0
dlt[duckdb]==1.32.0
# Days 6-7: transformation
dbt-core==1.12.6
dbt-duckdb==1.11.1
# Day 9: orchestration
prefect==3.9.1
```

Install them:

```bash
pip install -r requirements.txt
python -c "import polars, duckdb; print(polars.__version__, duckdb.__version__)"
```

```
2.1.0 1.5.6
```

The file already lists every package Module 1 needs (dlt, dbt, Prefect…), so you install once today and never wonder later whether your environment matches your classmates'.

**Example 3 — Tell Git what not to save.** Data files are rebuilt from scripts, and secrets must never be shared, so neither belongs in Git. Create `.gitignore` (some lines only matter from Day 4 on):

```text
# Environments and caches
.venv/
__pycache__/

# Data is rebuilt from code, never committed
data/raw/
data/clean/
data/marts/
data/api/
data/published/
data/*.duckdb
data/*.duckdb.wal

# Secrets and local state
.dlt/secrets.toml
.dlt_pipelines/
api/clock.txt

# dbt build output
transform/target/
transform/logs/
transform/dbt_packages/
transform/.user.yml
logs/
```

> Rule of thumb: commit **code** and **documentation**; regenerate **data**. That way the repository *is* the recipe.

**Try it.** Deactivate the environment (`deactivate`), try `python -c "import polars"`, and note what happens. Then reactivate it.

<details>
<summary>Solution</summary>

Outside the environment you get `ModuleNotFoundError: No module named 'polars'` (unless it happens to be installed system-wide). Reactivate with `source .venv/bin/activate`. If VS Code asks which interpreter to use, choose the one inside `.venv`.

</details>

---

## 1.3 Generating the NorthMart data (25 min)

**Concept.** We use simulated data so that we know the true answer. Students do **not** need to understand every line of the generator yet. Read the docstring at the top; by Module 4 you will be able to write such simulations yourself.

Create `data/make_northmart.py`:

<details>
<summary>data/make_northmart.py (provided; click to expand)</summary>

```python
"""
Generate the NorthMart course dataset (simulated online retailer).

Run once from the repository root:
    python data/make_northmart.py

It writes four CSV files to data/raw/:
    customers.csv       one row per customer
    products.csv        one row per product
    orders.csv          one row per order line (deliberately a little messy)
    email_campaign.csv  one row per customer who received the spring email

Everything is simulated, so we KNOW the true causal effect of the email:
treated customers place about 15% more orders in the 8 weeks after it is sent.
Marketing did not send it at random: heavy shoppers were more likely to get it.
That targeting is what makes naive comparisons misleading later in the course.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 2026
N_CUSTOMERS = 4_000
FIRST_MONDAY = date(2026, 1, 5)     # week 1 starts here
N_WEEKS = 16                        # weeks 1-16, through Sunday 2026-04-26
CAMPAIGN_WEEK = 9                   # email sent Monday 2026-03-02
TRUE_EFFECT = 0.15                  # +15% order rate after the email

OUT = Path(__file__).parent / "raw"


def make_customers(rng: np.random.Generator) -> pl.DataFrame:
    n = N_CUSTOMERS
    regions = ["ON", "QC", "BC", "AB", "ATL"]
    region = rng.choice(regions, size=n, p=[0.38, 0.23, 0.15, 0.12, 0.12])
    is_loyalty = rng.random(n) < 0.30
    age = rng.integers(18, 76, size=n)
    signup_offset = rng.integers(0, 3 * 365, size=n)
    signup_date = [date(2023, 1, 1) + timedelta(days=int(d)) for d in signup_offset]

    # Latent shopping intensity (orders per week). Never shown to students directly.
    base_rate = rng.lognormal(mean=-1.6, sigma=0.6, size=n) * np.where(is_loyalty, 1.6, 1.0)

    return pl.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "signup_date": signup_date,
        "region": region,
        "age": age,
        "is_loyalty": is_loyalty,
        "_base_rate": base_rate,
    })


def make_products(rng: np.random.Generator) -> pl.DataFrame:
    categories = {
        "Grocery": (4, 25), "Home": (15, 120), "Electronics": (40, 400),
        "Apparel": (20, 90), "Beauty": (8, 60),
    }
    rows = []
    pid = 1
    for cat, (lo, hi) in categories.items():
        for _ in range(8):
            rows.append({"product_id": f"P{pid:03d}", "category": cat,
                         "base_price": round(float(rng.uniform(lo, hi)), 2)})
            pid += 1
    return pl.DataFrame(rows)


def assign_campaign(rng: np.random.Generator, customers: pl.DataFrame) -> np.ndarray:
    """Targeted, not random: heavier shoppers and loyalty members are more likely to get the email."""
    z = np.log(customers["_base_rate"].to_numpy())
    z = (z - z.mean()) / z.std()
    logit = -1.2 + 1.1 * z + 0.6 * customers["is_loyalty"].to_numpy()
    p = 1 / (1 + np.exp(-logit))
    return rng.random(len(p)) < p


def make_orders(rng, customers, products, treated) -> pl.DataFrame:
    base = customers["_base_rate"].to_numpy()
    cust_ids = customers["customer_id"].to_numpy()
    prod_ids = products["product_id"].to_numpy()
    prices = products["base_price"].to_numpy()

    rows = []
    order_no = 1
    for week in range(1, N_WEEKS + 1):
        season = 1.0 + 0.10 * np.sin(2 * np.pi * week / 16)        # mild seasonality
        lift = np.where(treated & (week >= CAMPAIGN_WEEK), 1 + TRUE_EFFECT, 1.0)
        n_orders = rng.poisson(base * season * lift)
        week_start = FIRST_MONDAY + timedelta(weeks=week - 1)
        for i in np.nonzero(n_orders)[0]:
            for _ in range(n_orders[i]):
                k = rng.integers(len(prod_ids))
                discount = rng.choice([0, 0, 0, 10, 20])
                rows.append({
                    "order_id": f"O{order_no:06d}",
                    "customer_id": cust_ids[i],
                    "order_date": (week_start + timedelta(days=int(rng.integers(7)))).isoformat(),
                    "product_id": prod_ids[k],
                    "quantity": int(rng.integers(1, 4)),
                    "unit_price": round(prices[k] * (1 - discount / 100), 2),
                    "discount_pct": int(discount),
                    "status": "cancelled" if rng.random() < 0.03 else "completed",
                })
                order_no += 1
    orders = pl.DataFrame(rows)

    # Real data is messy: add ~1% exact duplicate rows (a double-loaded batch).
    dupes = orders.sample(fraction=0.01, seed=SEED)
    return pl.concat([orders, dupes]).sort("order_id")


def messy_regions(rng, customers: pl.DataFrame) -> pl.DataFrame:
    """Lower-case some codes, pad some with spaces, blank out ~2%."""
    region = customers["region"].to_list()
    for i in range(len(region)):
        u = rng.random()
        if u < 0.02:
            region[i] = None
        elif u < 0.06:
            region[i] = region[i].lower()
        elif u < 0.09:
            region[i] = f" {region[i]} "
    return customers.with_columns(pl.Series("region", region))


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    customers = make_customers(rng)
    products = make_products(rng)
    treated = assign_campaign(rng, customers)
    orders = make_orders(rng, customers, products, treated)

    campaign = (customers.filter(pl.Series(treated))
                .select("customer_id")
                .with_columns(campaign_id=pl.lit("SPRING26"),
                              sent_date=pl.lit(FIRST_MONDAY + timedelta(weeks=CAMPAIGN_WEEK - 1))))

    messy_regions(rng, customers).drop("_base_rate").write_csv(OUT / "customers.csv")
    products.write_csv(OUT / "products.csv")
    orders.write_csv(OUT / "orders.csv")
    campaign.write_csv(OUT / "email_campaign.csv")

    print(f"customers:      {customers.height:>7,}")
    print(f"products:       {products.height:>7,}")
    print(f"order lines:    {orders.height:>7,}")
    print(f"emails sent:    {campaign.height:>7,}")
    print(f"Files written to {OUT}")


if __name__ == "__main__":
    main()
```

</details>

Run it:

```bash
python data/make_northmart.py
```

```
customers:        4,000
products:            40
order lines:     18,542
emails sent:      1,279
Files written to /workspaces/causal-decision-lab/data/raw
```

Because the generator uses a fixed **seed** (`SEED = 2026`), everyone gets the same 18,542 rows. Peek at a file:

```bash
head -3 data/raw/orders.csv
```

```
order_id,customer_id,order_date,product_id,quantity,unit_price,discount_pct,status
O000001,C00003,2026-01-08,P011,1,61.62,20,completed
O000002,C00006,2026-01-10,P010,1,75.36,10,completed
```

**Try it.** In the generator, find the three lines that make the email **non-random**. Which customer characteristics raise the chance of receiving it?

<details>
<summary>Solution</summary>

In `assign_campaign`:

```python
logit = -1.2 + 1.1 * z + 0.6 * customers["is_loyalty"].to_numpy()
p = 1 / (1 + np.exp(-logit))
return rng.random(len(p)) < p
```

`z` is the customer's (standardised) shopping intensity and `is_loyalty` is loyalty membership. Both raise the probability of getting the email. Both also raise orders directly. That makes them **confounders**, and it is why a naive comparison will mislead us on Day 2.

</details>

---

## 1.4 The causal vocabulary in plain Python (50 min)

**Concept.** Before any data work, every column needs a **causal role**:

| Role | Meaning | NorthMart example |
|---|---|---|
| Unit | Who or what is treated | a customer |
| Treatment | The intervention | received the SPRING26 email |
| Outcome | What the treatment might change | orders after 2 March |
| Time | When things happen | order date, email date |
| Covariate | Characteristic fixed **before** treatment | region, age, loyalty |
| Pre-treatment outcome | The outcome measured before treatment | orders before 2 March |

The last row of covariates hides the most common mistake in applied work: using a variable measured **after** the treatment as a control. We call these **post-treatment variables**, and Module 2 explains why they bias estimates.

**Example 1 — One customer as a dictionary.** Create `day01/roles.py`:

```python
# day01/roles.py
# One real NorthMart customer, written by hand as a Python dictionary
customer = {
    "customer_id": "C00004",
    "region": "ON",
    "age": 22,
    "is_loyalty": True,
    "orders_before": 7,     # weeks 1-8
    "orders_after": 2,      # weeks 9-16
    "got_email": True,
}

# What role does each column play in a causal question?
roles = {
    "customer_id": "unit",
    "region": "covariate",
    "age": "covariate",
    "is_loyalty": "covariate",
    "orders_before": "pre-treatment outcome",
    "orders_after": "outcome",
    "got_email": "treatment",
}

for column, role in roles.items():
    print(f"{column:<15} {role:<22} value = {customer[column]}")

change = customer["orders_after"] - customer["orders_before"]
print(f"\nChange in orders: {change:+d}")
```

```bash
python day01/roles.py
```

```
customer_id     unit                   value = C00004
region          covariate              value = ON
age             covariate              value = 22
is_loyalty      covariate              value = True
orders_before   pre-treatment outcome  value = 7
orders_after    outcome                value = 2
got_email       treatment              value = True

Change in orders: -5
```

Three Python ideas are doing the work: a **dictionary** (`{key: value}`), a **for loop** over `.items()`, and an **f-string** (`f"{column:<15}"` pads the text to 15 characters so the columns line up).

**Example 2 — The fundamental problem, in one line.** Customer C00004 got the email and then ordered *less* (7 orders before, 2 after). Did the email hurt? We can't tell: what they would have ordered *without* it does not exist in any dataset. Maybe they would have ordered zero. Every method in this course is a way of estimating that missing number from *other* customers or *other* periods.

```python
orders_after_with_email = 2
orders_after_without_email = None   # never observed: the counterfactual
```

**Try it.**

1. Write a function `role_of(column)` that returns the role from a dictionary, or `"unknown"` if the column isn't listed.
2. Write `pct_change(before, after)` returning the percentage change, or `None` when `before` is 0.
3. *(Thinking)* A colleague proposes controlling for "number of discount codes used in March". Is that a covariate? Why or why not?

<details>
<summary>Solution</summary>

```python
ROLES = {"customer_id": "unit", "got_email": "treatment", "orders_after": "outcome"}

def role_of(column):
    return ROLES.get(column, "unknown")

def pct_change(before, after):
    if before == 0:
        return None
    return (after - before) / before * 100

print(role_of("got_email"))
print(role_of("favourite_colour"))
print(pct_change(4, 5))
print(pct_change(0, 3))
```

```
treatment
unknown
25.0
None
```

3. No. Discount codes used in March are measured **after** the email was sent on 2 March, and the email may have caused people to use more codes. It is a post-treatment variable (possibly a *mediator*). Controlling for it would remove part of the very effect we want to measure.

</details>

---

## 1.5 A first look with Polars (55 min)

**Concept.** **Polars** is a fast DataFrame library: a table with named, typed columns. It is the modern replacement for pandas in this course because it is quick on large files and makes types explicit.

**Example 1 — Read, count, look.** Create `day01/first_look.py`:

```python
# day01/first_look.py
import polars as pl

customers = pl.read_csv("data/raw/customers.csv", try_parse_dates=True)
orders = pl.read_csv("data/raw/orders.csv", try_parse_dates=True)
campaign = pl.read_csv("data/raw/email_campaign.csv", try_parse_dates=True)

print("customers:", customers.shape)
print("orders:   ", orders.shape)
print("campaign: ", campaign.shape)

print(customers.head())
print(orders.schema)
print(orders.select("quantity", "unit_price", "discount_pct").describe())

share = campaign.height / customers.height
print(f"\nShare of customers who got the email: {share:.1%}")
print("First order date:", orders["order_date"].min())
print("Last order date: ", orders["order_date"].max())
```

```bash
python day01/first_look.py
```

Key parts of the output:

```
customers: (4000, 5)
orders:    (18542, 8)
campaign:  (1279, 3)
...
Schema({'order_id': String, 'customer_id': String, 'order_date': Date, 'product_id': String,
        'quantity': Int64, 'unit_price': Float64, 'discount_pct': Int64, 'status': String})
...
Share of customers who got the email: 32.0%
First order date: 2026-01-05
Last order date:  2026-04-26
```

Notes:

- `.shape` is *(rows, columns)*.
- `try_parse_dates=True` turns `"2026-01-08"` into a real `Date`, so we can compare and sort dates.
- `.describe()` gives count, nulls, mean, quartiles: the same summary you'd report in Table 1 of a paper.
- The data covers **16 weeks**, 8 before the email and 8 after. That pre/post split matters all course.

**Example 2 — Single columns.** A column is a `Series`; it has its own methods.

```python
orders["customer_id"].n_unique()     # distinct customers who ordered
orders["status"].value_counts()      # completed vs cancelled
customers["age"].mean()
```

**Try it.** Using `data/raw/orders.csv`:

1. How many distinct customers placed at least one order? Compare with the 4,000 customers.
2. What share of order lines are cancelled?
3. Are there duplicate rows? (Hint: `orders.is_duplicated()`.)

<details>
<summary>Solution</summary>

```python
import polars as pl
orders = pl.read_csv("data/raw/orders.csv", try_parse_dates=True)
print("Customers who ordered:", orders["customer_id"].n_unique())
cancelled_share = (orders["status"] == "cancelled").mean()
print(f"Cancelled share: {cancelled_share:.1%}")
print("Duplicate rows:", orders.is_duplicated().sum() // 2)
```

```
Customers who ordered: 3738
Cancelled share: 2.8%
Duplicate rows: 183
```

262 customers never ordered. They still count: a customer who received the email and bought nothing is a real outcome (zero orders), not missing data. We'll protect them on Days 2 and 3. `is_duplicated()` flags *both* copies of a duplicated row, hence `// 2`.

</details>

---

## 1.6 Day 1 mini-project: a data dictionary with causal roles (45 min)

**Goal.** Write `day01/data_dictionary.py` that profiles every raw file and records, for each column, its type, nulls, distinct values, an example and its **causal role**. It also saves `docs/data_dictionary.md` so your team can read it on GitHub.

```python
# day01/data_dictionary.py
"""Profile every raw file and record the causal role of each column."""
from pathlib import Path

import polars as pl

RAW = Path("data/raw")

ROLES = {
    "customer_id": "unit",
    "signup_date": "covariate (pre-treatment)",
    "region": "covariate (pre-treatment)",
    "age": "covariate (pre-treatment)",
    "is_loyalty": "covariate (pre-treatment)",
    "order_id": "identifier",
    "order_date": "time",
    "product_id": "identifier",
    "quantity": "outcome ingredient",
    "unit_price": "outcome ingredient",
    "discount_pct": "possible mediator / other treatment",
    "status": "filter",
    "category": "covariate",
    "base_price": "covariate",
    "campaign_id": "treatment label",
    "sent_date": "treatment timing",
}


def profile(name: str, df: pl.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        rows.append({
            "table": name,
            "column": col,
            "dtype": str(df[col].dtype),
            "nulls": df[col].null_count(),
            "unique": df[col].n_unique(),
            "example": str(df[col].drop_nulls()[0]),
            "role": ROLES.get(col, "?"),
        })
    return rows


def main() -> None:
    all_rows = []
    for path in sorted(RAW.glob("*.csv")):
        df = pl.read_csv(path, try_parse_dates=True)
        all_rows += profile(path.stem, df)

    dictionary = pl.DataFrame(all_rows)
    with pl.Config(tbl_rows=-1, tbl_width_chars=140):
        print(dictionary)

    # Save a Markdown version for the docs/ folder
    lines = ["| table | column | dtype | nulls | unique | example | role |",
             "|---|---|---|---|---|---|---|"]
    for r in dictionary.iter_rows(named=True):
        lines.append(f"| {r['table']} | {r['column']} | {r['dtype']} | {r['nulls']} | "
                     f"{r['unique']} | {r['example']} | {r['role']} |")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data_dictionary.md").write_text("# NorthMart data dictionary\n\n" + "\n".join(lines) + "\n")
    print("\nSaved docs/data_dictionary.md")


if __name__ == "__main__":
    main()
```

```bash
python day01/data_dictionary.py
```

```
│ customers      ┆ region       ┆ String  ┆ 74    ┆ 16     ┆ ON         ┆ covariate (pre-treatment)       │
...
│ orders         ┆ order_id     ┆ String  ┆ 0     ┆ 18359  ┆ O000001    ┆ identifier                      │
...
Saved docs/data_dictionary.md
```

Read the profile like a detective:

- `region` has **16** distinct values, but Canada has 5 regions in our data. Something is messy.
- `region` has **74** nulls.
- `order_id` has 18,359 distinct values in 18,542 rows. IDs should be unique, so there are duplicates.

These three findings become Day 2's cleaning tasks.

**Extension.** Add a `pct_null` column (nulls as a share of rows) and flag any column above 1%.

<details>
<summary>Solution</summary>

Inside `profile`, add one entry to the dictionary:

```python
"pct_null": round(df[col].null_count() / df.height * 100, 2),
```

and after building `dictionary` in `main`:

```python
print(dictionary.filter(pl.col("pct_null") > 1).select("table", "column", "pct_null"))
```

Only `customers.region` (1.85%) is flagged.

</details>

---

## 1.7 Save your work with Git (15 min)

```bash
git add .
git status          # data/raw should NOT appear: .gitignore works
git commit -m "Day 1: environment, NorthMart data generator, data dictionary"
git push
```

Open your repository on GitHub and click `docs/data_dictionary.md`: the table renders as a web page.

### Day 1 checklist

- [ ] I can open a Codespace and run a script from the repository root
- [ ] I can create and activate a virtual environment and install pinned packages
- [ ] I can explain unit, treatment, outcome, time, covariate and post-treatment variable
- [ ] I can read a CSV with Polars and inspect its shape, schema and summary
- [ ] I have a data dictionary that records the causal role of every column

---

# DAY 2 — Cleaning and Shaping Data for a Causal Question

| Time | Block |
|---|---|
| 0:00 – 1:00 | 2.1 Polars expressions: select, filter, with_columns |
| 1:00 – 1:50 | 2.2 From raw to clean: duplicates, text, cancellations |
| 1:50 – 2:05 | Break |
| 2:05 – 3:00 | 2.3 Group-by and joins: one row per customer |
| 3:00 – 3:30 | 2.4 Lazy queries for bigger data |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 2.5 Day 2 mini-project: the naive comparison |
| 4:45 – 5:00 | 2.6 Write it down and save with Git |

---

## 2.1 Polars expressions: select, filter, with_columns (60 min)

**Concept.** Almost everything in Polars is built from **expressions**: `pl.col("unit_price") > 200`, `pl.col("quantity") * pl.col("unit_price")`. You pass expressions to four verbs:

| Verb | Does | Economist's translation |
|---|---|---|
| `select` | keep or compute columns | choose variables |
| `filter` | keep rows | restrict the sample |
| `with_columns` | add or replace columns | generate new variables |
| `group_by().agg()` | summarise by group | collapse (Stata) |

**Examples.** Create `day02/expressions.py` and run it after each step:

```python
# day02/expressions.py
import polars as pl

orders = pl.read_csv("data/raw/orders.csv", try_parse_dates=True)

# 1. select: keep some columns
print(orders.select("order_id", "customer_id", "unit_price").head(3))

# 2. filter: keep some rows
big = orders.filter(pl.col("unit_price") > 200)
print("Lines over $200:", big.height)

# 3. with_columns: add a new column built from others
orders = orders.with_columns(
    revenue=pl.col("quantity") * pl.col("unit_price"),
)
print(orders.select("quantity", "unit_price", "revenue").head(3))

# 4. Several conditions at once
discounted_big = orders.filter(
    (pl.col("discount_pct") > 0) & (pl.col("unit_price") > 100)
)
print("Discounted lines over $100:", discounted_big.height)

# 5. Conditional column: when / then / otherwise
orders = orders.with_columns(
    period=pl.when(pl.col("order_date") >= pl.date(2026, 3, 2))
             .then(pl.lit("post"))
             .otherwise(pl.lit("pre"))
)
print(orders.group_by("period").len().sort("period"))
```

```
Lines over $200: 2044
...
Discounted lines over $100: 1554
...
│ post   ┆ 9020 │
│ pre    ┆ 9522 │
```

Notes:

- Combine conditions with `&` (and) and `|` (or), each condition in brackets.
- `pl.when(...).then(...).otherwise(...)` is the Polars version of `if/else` for a whole column, like `gen ... if ...` / `replace` in Stata.
- `pl.date(2026, 3, 2)` is a date literal. The email date splits the sample into **pre** and **post**.

**Try it.** Starting from the raw orders, keep only completed orders, add `revenue` and a Boolean column `discounted` (discount above 0), keep only `order_id`, `revenue`, `discounted`, and print the share of discounted lines.

<details>
<summary>Solution</summary>

```python
import polars as pl
orders = pl.read_csv("data/raw/orders.csv", try_parse_dates=True)
out = (orders
       .filter(pl.col("status") == "completed")
       .with_columns(
           revenue=pl.col("quantity") * pl.col("unit_price"),
           discounted=pl.col("discount_pct") > 0,
       )
       .select("order_id", "revenue", "discounted"))
print(out.head(3))
print("Share discounted:", round(out["discounted"].mean(), 3))
```

About 40% of lines carry a discount. Keep that in mind: if discounts rose after the email, they could be part of how the email works (a mediator), not something to control for.

</details>

---

## 2.2 From raw to clean: duplicates, text, cancellations (50 min)

**Concept.** Yesterday's dictionary found three problems. Each one would bias a causal estimate if we ignored it:

| Problem | If ignored… |
|---|---|
| 183 duplicated order lines | some customers' outcomes are inflated |
| `" on "`, `"on"`, `"ON"` treated as different regions | covariate balance checks and region effects are wrong |
| Cancelled orders counted as sales | the outcome measures intent, not purchases |

We write cleaning as **functions** in a script, never as manual edits to files, so anyone can rerun it. And we finish with **assertions**: checks that stop the script loudly if a rule is broken.

Create `day02/clean.py`:

```python
# day02/clean.py
"""Clean the raw NorthMart files and save them as Parquet in data/clean/."""
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
CLEAN = Path("data/clean")


def clean_customers(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        region=pl.col("region").str.strip_chars().str.to_uppercase().fill_null("UNKNOWN")
    )


def clean_orders(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.unique(maintain_order=True)                 # drop exact duplicate rows
          .filter(pl.col("status") == "completed")     # cancelled orders never happened
          .with_columns(revenue=(pl.col("quantity") * pl.col("unit_price")).round(2))
    )


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)

    customers_raw = pl.read_csv(RAW / "customers.csv", try_parse_dates=True)
    orders_raw = pl.read_csv(RAW / "orders.csv", try_parse_dates=True)
    products = pl.read_csv(RAW / "products.csv")
    campaign = pl.read_csv(RAW / "email_campaign.csv", try_parse_dates=True)

    customers = clean_customers(customers_raw)
    orders = clean_orders(orders_raw)

    print("Regions before:", sorted(customers_raw["region"].drop_nulls().unique().to_list()))
    print("Regions after: ", sorted(customers["region"].unique().to_list()))
    print(f"Order lines: {orders_raw.height:,} raw -> {orders.height:,} clean")

    # Checks: stop loudly if something is wrong
    assert orders["order_id"].is_unique().all(), "order_id should be unique after cleaning"
    assert customers["customer_id"].is_unique().all(), "customer_id should be unique"
    assert orders["revenue"].min() > 0, "revenue should be positive"

    customers.write_parquet(CLEAN / "customers.parquet")
    orders.write_parquet(CLEAN / "orders.parquet")
    products.write_parquet(CLEAN / "products.parquet")
    campaign.write_parquet(CLEAN / "email_campaign.parquet")
    print("Saved 4 Parquet files to", CLEAN)


if __name__ == "__main__":
    main()
```

```bash
python day02/clean.py
```

```
Regions before: [' AB ', ' ATL ', ' BC ', ' ON ', ' QC ', 'AB', 'ATL', 'BC', 'ON', 'QC', 'ab', 'atl', 'bc', 'on', 'qc']
Regions after:  ['AB', 'ATL', 'BC', 'ON', 'QC', 'UNKNOWN']
Order lines: 18,542 raw -> 17,839 clean
Saved 4 Parquet files to data/clean
```

Why these choices:

- **`.unique()`** removes exact duplicates. Removing rows by `order_id` alone could hide a real data error, so we only drop rows that are identical in every column.
- **Missing region → `"UNKNOWN"`**, not dropped. Dropping customers with missing region would silently change the sample. A visible category keeps them and lets us check later whether missingness relates to treatment.
- **Parquet** stores types (dates stay dates) and is much faster and smaller than CSV. From now on we read `data/clean/*.parquet`.

**Try it.** Add one more assertion: every order's `customer_id` must exist in `customers`. (Hint: `is_in`.)

<details>
<summary>Solution</summary>

```python
assert orders["customer_id"].is_in(customers["customer_id"].implode()).all(), \
    "orders contain unknown customers"
```

It passes. In real data this check often fails (test accounts, deleted customers), and you must decide what to do *before* estimating anything.

</details>

---

## 2.3 Group-by and joins: one row per customer (55 min)

**Concept.** Causal questions have a **unit**. Ours is the customer, but orders are recorded per *line*. We collapse lines to customers with `group_by().agg()` and bring in other tables with `join`.

Two joining rules protect the analysis:

1. **Start from the unit table** (`customers`), then left-join everything onto it. Starting from `orders` silently drops the 271 customers who never bought.
2. **Treatment lives in its own table.** `email_campaign` lists only recipients, so after a left join non-recipients have nulls. We turn that into `treated = 0`.

Create `day02/summaries.py`:

```python
# day02/summaries.py
import polars as pl

orders = pl.read_parquet("data/clean/orders.parquet")
customers = pl.read_parquet("data/clean/customers.parquet")
campaign = pl.read_parquet("data/clean/email_campaign.parquet")

# 1. One number for the whole table
print(f"Total revenue: ${orders['revenue'].sum():,.0f}")

# 2. group_by + agg: one row per region... but region lives in customers, so join first
orders_c = orders.join(customers, on="customer_id", how="left")
by_region = (orders_c.group_by("region")
             .agg(n_orders=pl.len(), revenue=pl.col("revenue").sum().round(0).cast(pl.Int64))
             .sort("revenue", descending=True))
print(by_region)

# 3. Customer-level outcome table
per_customer = orders.group_by("customer_id").agg(
    n_orders=pl.col("order_id").n_unique(),
    revenue=pl.col("revenue").sum(),
    first_order=pl.col("order_date").min(),
)
print(per_customer.head())
print("Customers with at least one order:", per_customer.height, "of", customers.height)

# 4. The treatment lives in its own table: left join + fill the gaps with 0
customers_t = (customers
               .join(campaign.select("customer_id", "sent_date"), on="customer_id", how="left")
               .with_columns(treated=pl.col("sent_date").is_not_null().cast(pl.Int8)))
print(customers_t.group_by("treated").len().sort("treated"))

# 5. Keep customers who never ordered! Start from customers, not orders.
analysis = (customers_t
            .join(per_customer, on="customer_id", how="left")
            .with_columns(pl.col("n_orders", "revenue").fill_null(0)))
print("Rows in analysis table:", analysis.height)
```

```
Total revenue: $2,810,574
│ ON      ┆ 6608     ┆ 1013648 │
...
Customers with at least one order: 3729 of 4000
│ 0       ┆ 2721 │
│ 1       ┆ 1279 │
Rows in analysis table: 4000
```

The final table has **4,000 rows**: every customer, buyers or not.

**Try it.** Join orders to `products` and compute revenue and the average discount by product category, sorted by revenue.

<details>
<summary>Solution</summary>

```python
import polars as pl
orders = pl.read_parquet("data/clean/orders.parquet")
products = pl.read_parquet("data/clean/products.parquet")
print(orders.join(products, on="product_id", how="left")
      .group_by("category")
      .agg(revenue=pl.col("revenue").sum().round(0).cast(pl.Int64),
           avg_discount=pl.col("discount_pct").mean().round(1))
      .sort("revenue", descending=True))
```

```
│ Electronics ┆ 1678960 ┆ 6.1          │
│ Home        ┆ 435289  ┆ 5.9          │
...
```

</details>

---

## 2.4 Lazy queries for bigger data (30 min)

**Concept.** `read_parquet` loads the whole file at once. `scan_parquet` instead builds a **plan**, optimises it, and reads only what the plan needs. For our 18,000 lines it makes no difference; for the 200 million lines of a real retailer, it is the difference between seconds and a crash.

Create `day02/lazy_demo.py`:

```python
# day02/lazy_demo.py
import polars as pl

# scan_parquet builds a plan; nothing is read yet
plan = (
    pl.scan_parquet("data/clean/orders.parquet")
      .filter(pl.col("order_date") >= pl.date(2026, 3, 2))
      .group_by("customer_id")
      .agg(post_revenue=pl.col("revenue").sum())
)
print(plan.explain())       # see the optimised plan
result = plan.collect()     # now Polars does the work
print(result.sort("post_revenue", descending=True).head(5))
```

```
AGGREGATE[maintain_order: false]
  [col("revenue").sum().alias("post_revenue")] BY [col("customer_id")]
  FROM
  simple π 2/2 ["customer_id", "revenue"]
    Parquet SCAN [data/clean/orders.parquet]
    PROJECT 3/9 COLUMNS
    SELECTION: col("order_date") >= 2026-03-02
```

Read the plan from the bottom up: Polars reads only **3 of 9 columns** and applies the date filter **while reading**. You wrote the steps in a natural order; Polars rearranged them to be efficient.

> Habit for the rest of the course: `scan_*` + `.collect()` in pipelines, `read_*` for quick exploration.

---

## 2.5 Day 2 mini-project: the naive comparison (60 min)

**Goal.** Answer management's question the way most dashboards would: compare emailed and non-emailed customers after 2 March. Then check whether the two groups were comparable **before** the email.

Create `day02/naive_comparison.py`:

```python
# day02/naive_comparison.py
"""Mini-project: how different do emailed and non-emailed customers look, before and after?"""
import polars as pl

SENT = pl.date(2026, 3, 2)

customers = pl.read_parquet("data/clean/customers.parquet")
orders = pl.read_parquet("data/clean/orders.parquet")
campaign = pl.read_parquet("data/clean/email_campaign.parquet")

# 1. Orders per customer, before and after the email date
per_customer = orders.group_by("customer_id").agg(
    orders_pre=(pl.col("order_date") < SENT).sum(),
    orders_post=(pl.col("order_date") >= SENT).sum(),
)

# 2. One row per customer, treatment flag, zeros for non-buyers
analysis = (
    customers
    .join(campaign.select("customer_id", treated=pl.lit(1)), on="customer_id", how="left")
    .join(per_customer, on="customer_id", how="left")
    .with_columns(
        pl.col("treated").fill_null(0),
        pl.col("orders_pre", "orders_post").fill_null(0),
    )
)

# 3. Compare group means
summary = (analysis.group_by("treated")
           .agg(n=pl.len(),
                orders_pre=pl.col("orders_pre").mean(),
                orders_post=pl.col("orders_post").mean(),
                loyalty_share=pl.col("is_loyalty").mean())
           .sort("treated"))
print(summary)

t = summary.filter(pl.col("treated") == 1).row(0, named=True)
c = summary.filter(pl.col("treated") == 0).row(0, named=True)

naive = t["orders_post"] / c["orders_post"] - 1
pre_gap = t["orders_pre"] / c["orders_pre"] - 1
growth_t = t["orders_post"] / t["orders_pre"] - 1
growth_c = c["orders_post"] / c["orders_pre"] - 1

print(f"\nNaive 'effect' (post-period gap):    {naive:+.0%}")
print(f"Gap BEFORE the email was even sent:  {pre_gap:+.0%}")
print(f"Growth, emailed customers:           {growth_t:+.1%}")
print(f"Growth, other customers:             {growth_c:+.1%}")
print(f"Difference in growth:                {growth_t - growth_c:+.1%}")
print("\nThe simulation's TRUE effect is +15% more orders.")
```

```bash
python day02/naive_comparison.py
```

```
│ 0       ┆ 2721 ┆ 1.850055   ┆ 1.620728    ┆ 0.208012      │
│ 1       ┆ 1279 ┆ 3.232213   ┆ 3.331509    ┆ 0.491009      │

Naive 'effect' (post-period gap):    +106%
Gap BEFORE the email was even sent:  +75%
Growth, emailed customers:           +3.1%
Growth, other customers:             -12.4%
Difference in growth:                +15.5%

The simulation's TRUE effect is +15% more orders.
```

**Interpretation.** This is the most important output of Module 1:

1. The naive dashboard says the email **doubled** orders (+106%). The truth is +15%.
2. Emailed customers already ordered **75% more** before the email existed, and were more than twice as likely to be loyalty members. The groups were never comparable. That is **selection bias**, caused by targeting.
3. Comparing each group *with itself* over time (growth) and then comparing the growth rates gets close to the truth. That idea is **difference-in-differences**, and Module 5 makes it rigorous (with standard errors and a check of its key assumption).

None of steps 2–3 is possible without clean pre-period data for both groups. That is why causal work starts with data engineering.

**Try it.** Repeat the comparison using **revenue** instead of order counts, split by `is_loyalty`. Within each loyalty group, are treated and control customers comparable before the email?

<details>
<summary>Solution</summary>

```python
import polars as pl
SENT = pl.date(2026, 3, 2)
customers = pl.read_parquet("data/clean/customers.parquet")
orders = pl.read_parquet("data/clean/orders.parquet")
campaign = pl.read_parquet("data/clean/email_campaign.parquet")
per_customer = orders.group_by("customer_id").agg(
    revenue_pre=pl.col("revenue").filter(pl.col("order_date") < SENT).sum(),
    revenue_post=pl.col("revenue").filter(pl.col("order_date") >= SENT).sum(),
)
analysis = (customers
            .join(campaign.select("customer_id", treated=pl.lit(1)), on="customer_id", how="left")
            .join(per_customer, on="customer_id", how="left")
            .with_columns(pl.col("treated").fill_null(0),
                          pl.col("revenue_pre", "revenue_post").fill_null(0)))
print(analysis.group_by("treated", "is_loyalty")
      .agg(n=pl.len(), revenue_pre=pl.col("revenue_pre").mean().round(0),
           revenue_post=pl.col("revenue_post").mean().round(0))
      .sort("is_loyalty", "treated"))
```

```
│ 0       ┆ false      ┆ 2155 ┆ 265.0       ┆ 247.0        │
│ 1       ┆ false      ┆ 651  ┆ 430.0       ┆ 480.0        │
│ 0       ┆ true       ┆ 566  ┆ 374.0       ┆ 315.0        │
│ 1       ┆ true       ┆ 628  ┆ 585.0       ┆ 569.0        │
```

No. Even *within* loyalty groups, treated customers spent far more before the email ($430 vs $265; $585 vs $374). Loyalty explains part of the targeting, but not all of it: marketing also targeted on shopping intensity. Adjusting for one observed covariate is not enough; Module 4 shows how to adjust for pre-period behaviour directly.

`pl.col("revenue").filter(...)` inside `agg` sums only the rows meeting the condition: a compact way to build pre and post outcomes in one pass.

</details>

---

## 2.6 Write it down and save with Git (15 min)

Add a short section to `docs/data_dictionary.md` (or a new `docs/day02_notes.md`) answering, in three sentences: *What did the naive comparison say, why is it wrong, and what data would we need to do better?* Writing the reasoning down is part of every capstone in this course.

```bash
git add .
git commit -m "Day 2: cleaning pipeline, customer table, naive comparison"
git push
```

### Day 2 checklist

- [ ] I can use `select`, `filter`, `with_columns` and `when/then/otherwise`
- [ ] I can clean duplicates, messy text and cancelled rows in a rerunnable script with assertions
- [ ] I start joins from the unit table so units with no events are kept as zeros
- [ ] I can explain the difference between `read_parquet` and `scan_parquet`
- [ ] I can explain why the naive +106% is wrong, using the pre-period gap

---

# DAY 3 — DuckDB and the Customer × Week Panel

| Time | Block |
|---|---|
| 0:00 – 0:50 | 3.1 DuckDB: SQL directly on files |
| 0:50 – 1:20 | 3.2 A persistent local warehouse |
| 1:20 – 1:35 | Break |
| 1:35 – 2:30 | 3.3 Weeks and the calendar spine |
| 2:30 – 3:20 | 3.4 Window functions: lags, running totals, pre-period values |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 3.5 Day 3 mini-project: the customer × week panel |
| 4:45 – 5:00 | 3.6 Save with Git and look ahead |

---

## 3.1 DuckDB: SQL directly on files (50 min)

**Concept.** **DuckDB** is an analytical database that runs inside Python: no server, no installation beyond `pip`. It speaks SQL, the language of every company data warehouse, and it can query CSV and Parquet files directly. In this course, Polars and DuckDB are partners: Polars for Python-style transformations, DuckDB for SQL models (and, from Day 6, dbt).

Create `day03/duck_intro.py`:

```python
# day03/duck_intro.py
import duckdb

# 1. SQL straight on a file: no loading step
print(duckdb.sql("SELECT COUNT(*) AS n_lines FROM 'data/clean/orders.parquet'"))

# 2. A familiar GROUP BY, this time on the raw CSV
print(duckdb.sql("""
    SELECT status, COUNT(*) AS n, ROUND(SUM(quantity * unit_price)) AS revenue
    FROM 'data/raw/orders.csv'
    GROUP BY status
"""))

# 3. Join two files
print(duckdb.sql("""
    SELECT c.region, COUNT(DISTINCT o.customer_id) AS buyers
    FROM 'data/clean/orders.parquet' AS o
    JOIN 'data/clean/customers.parquet' AS c USING (customer_id)
    GROUP BY c.region
    ORDER BY buyers DESC
"""))

# 4. Hand a result to Polars
df = duckdb.sql("SELECT * FROM 'data/clean/customers.parquet' WHERE is_loyalty").pl()
print(type(df), df.shape)
```

```
┌───────────┬───────┬───────────┐
│  status   │   n   │  revenue  │
├───────────┼───────┼───────────┤
│ completed │ 18020 │ 2838607.0 │
│ cancelled │   522 │   83637.0 │
└───────────┴───────┴───────────┘
...
<class 'polars.dataframe.frame.DataFrame'> (1194, 5)
```

Notes:

- `FROM 'file.parquet'` treats a file as a table. Example 2 runs on the **raw** CSV, which still contains cancellations and duplicates.
- `.pl()` hands a DuckDB result to Polars with no copying. You can move between SQL and Polars freely.

**Try it.** Write a DuckDB query on the clean orders that returns lines and revenue **by month**. (Hint: `strftime(order_date, '%Y-%m')`.)

<details>
<summary>Solution</summary>

```python
import duckdb
print(duckdb.sql("""
    SELECT strftime(order_date, '%Y-%m') AS month,
           COUNT(*) AS n_lines,
           ROUND(SUM(revenue)) AS revenue
    FROM 'data/clean/orders.parquet'
    GROUP BY month
    ORDER BY month
"""))
```

```
│ 2026-01 │    4399 │ 689407.0 │
│ 2026-02 │    4622 │ 712361.0 │
│ 2026-03 │    4800 │ 763991.0 │
│ 2026-04 │    4018 │ 644815.0 │
```

</details>

---

## 3.2 A persistent local warehouse (30 min)

**Concept.** Querying files is great for exploration. For a pipeline we want named, organised tables. We create one database file, `data/northmart.duckdb`, with **schemas** (folders for tables): `clean` for cleaned sources and, later, `marts` for analysis-ready tables.

Create `day03/build_warehouse.py`:

```python
# day03/build_warehouse.py
"""Load the clean Parquet files into a persistent DuckDB database: data/northmart.duckdb"""
import duckdb

TABLES = ["customers", "orders", "products", "email_campaign"]

con = duckdb.connect("data/northmart.duckdb")
con.sql("CREATE SCHEMA IF NOT EXISTS clean")
for t in TABLES:
    con.sql(f"CREATE OR REPLACE TABLE clean.{t} AS SELECT * FROM 'data/clean/{t}.parquet'")
    n = con.sql(f"SELECT COUNT(*) FROM clean.{t}").fetchall()[0][0]
    print(f"clean.{t:<15} {n:>7,} rows")
con.close()
```

```
clean.customers         4,000 rows
clean.orders           17,839 rows
clean.products             40 rows
clean.email_campaign    1,279 rows
```

> **Getting a single value.** `con.sql(...).fetchall()` returns a list of rows, each a tuple, so `[0][0]` is the first column of the first row. This form works in both DuckDB 1.x and 2.x.

> Only one program can write to a DuckDB file at a time. If you get a "lock" error, close other scripts or notebooks that have the database open.

---

## 3.3 Weeks and the calendar spine (55 min)

**Concept.** Most causal methods in this course (DiD, event studies, synthetic control) need a **panel**: one row per unit per period. Raw data only has rows when something *happened*. A customer who bought nothing in week 5 has no row for week 5, but their true outcome that week is **0**, not missing.

If we forget this, averages are computed only over active weeks and every estimate is biased upwards. The fix is a **calendar spine**: a table of all periods, crossed with all units, onto which we left-join the events.

Create `day03/calendar_spine.py`:

```python
# day03/calendar_spine.py
import duckdb

con = duckdb.connect("data/northmart.duckdb")

# 1. Weekly orders: date_trunc('week', ...) gives the Monday of each week
print(con.sql("""
    SELECT date_trunc('week', order_date)::DATE AS week_start, COUNT(*) AS n_orders
    FROM clean.orders
    GROUP BY week_start
    ORDER BY week_start
    LIMIT 5
"""))

# 2. The problem: a customer with no orders in a week has NO ROW for that week
print(con.sql("""
    SELECT customer_id, COUNT(DISTINCT date_trunc('week', order_date)) AS weeks_with_orders
    FROM clean.orders
    GROUP BY customer_id
    ORDER BY weeks_with_orders
    LIMIT 3
"""))

# 3. The fix: a calendar spine of every week, crossed with every customer
con.sql("""
    CREATE OR REPLACE TABLE clean.weeks AS
    SELECT
        week_start::DATE AS week_start,
        ROW_NUMBER() OVER (ORDER BY week_start) AS week_no
    FROM generate_series(DATE '2026-01-05', DATE '2026-04-20', INTERVAL 7 DAY) AS g(week_start)
""")
print(con.sql("SELECT * FROM clean.weeks ORDER BY week_no"))

n = con.sql("SELECT COUNT(*) FROM clean.customers CROSS JOIN clean.weeks").fetchall()[0][0]
print(f"Customers x weeks = {n:,} rows (4,000 x 16)")
con.close()
```

```
┌────────────┬──────────┐
│ week_start │ n_orders │
├────────────┼──────────┤
│ 2026-01-05 │     1050 │
...
│ C00040      │                 1 │      <- ordered in only 1 of 16 weeks
...
Customers x weeks = 64,000 rows (4,000 x 16)
```

- `date_trunc('week', d)` returns the Monday of the week, so our weeks run Monday to Sunday. Week 9 starts on 2 March, the email date.
- `generate_series` creates every Monday from 5 January to 20 April: 16 weeks.
- `CROSS JOIN` pairs every customer with every week.

**Try it.** For customer C00004, list all 16 weeks with their number of orders, showing **0** for weeks with no orders. (Hint: left-join that customer's weekly orders onto `clean.weeks` and use `COALESCE`.)

<details>
<summary>Solution</summary>

```python
import duckdb
con = duckdb.connect("data/northmart.duckdb")
print(con.sql("""
    WITH weekly AS (
        SELECT customer_id, date_trunc('week', order_date)::DATE AS week_start, COUNT(*) AS n_orders
        FROM clean.orders GROUP BY ALL
    )
    SELECT w.week_start, COALESCE(o.n_orders, 0) AS n_orders
    FROM clean.weeks AS w
    LEFT JOIN (SELECT * FROM weekly WHERE customer_id = 'C00004') AS o USING (week_start)
    ORDER BY w.week_start
"""))
con.close()
```

The customer ordered in 7 of 16 weeks. Without the spine you would see 7 rows and might compute an average of 1.3 orders per week; the true average is 9 / 16 = 0.56.

</details>

---

## 3.4 Window functions: lags, running totals, pre-period values (50 min)

**Concept.** A **window function** computes a value from *other rows* related to the current one, without collapsing the table. `PARTITION BY customer_id` means "look only at this customer's rows"; `ORDER BY week_start` sets the order within that customer. Causal work uses windows constantly: lagged outcomes, cumulative exposure, and pre-period averages repeated on every row.

Create `day03/windows.py`:

```python
# day03/windows.py
import duckdb

con = duckdb.connect("data/northmart.duckdb")

weekly = """
    SELECT customer_id,
           date_trunc('week', order_date)::DATE AS week_start,
           COUNT(*) AS n_orders
    FROM clean.orders
    WHERE customer_id = 'C00004'
    GROUP BY ALL
"""

# 1. LAG: last week's value next to this week's
print(con.sql(f"""
    SELECT *, LAG(n_orders) OVER (PARTITION BY customer_id ORDER BY week_start) AS prev_orders
    FROM ({weekly})
    ORDER BY week_start
"""))

# 2. Running total
print(con.sql(f"""
    SELECT *, SUM(n_orders) OVER (PARTITION BY customer_id ORDER BY week_start) AS cumulative
    FROM ({weekly})
    ORDER BY week_start
"""))

# 3. A pre-period average repeated on every row of the customer (a filtered window)
print(con.sql(f"""
    SELECT *,
           AVG(n_orders) FILTER (WHERE week_start < DATE '2026-03-02')
               OVER (PARTITION BY customer_id) AS pre_avg_active_weeks
    FROM ({weekly})
    ORDER BY week_start
"""))
con.close()
```

```
│ C00004      │ 2026-01-19 │        3 │        NULL │
│ C00004      │ 2026-02-02 │        1 │           3 │
...
│ C00004      │ 2026-01-19 │        3 │                  1.4 │
```

**Look carefully at example 3.** The "pre-period average" of 1.4 is **wrong**: it averages only the 5 weeks with orders, not all 8 pre-period weeks (the true figure is 7 / 8 = 0.875). This is the missing-zeros problem from 3.3 again. Window functions are only correct when they run on the **complete** panel. In the mini-project we compute them after the spine.

- `LAG(x)` gives the previous row's value; the first row has no previous row, hence `NULL`.
- `AGG(...) FILTER (WHERE ...)` restricts which rows enter the calculation: here, pre-period weeks only.

**Try it.** Rank customers by total revenue and keep the top 5. (Hints: `RANK() OVER (ORDER BY SUM(revenue) DESC)`; DuckDB's `QUALIFY` filters on a window result the way `HAVING` filters on an aggregate.)

<details>
<summary>Solution</summary>

```python
import duckdb
con = duckdb.connect("data/northmart.duckdb")
print(con.sql("""
    SELECT customer_id,
           ROUND(SUM(revenue), 2) AS total_revenue,
           RANK() OVER (ORDER BY SUM(revenue) DESC) AS revenue_rank
    FROM clean.orders
    GROUP BY customer_id
    QUALIFY revenue_rank <= 5
    ORDER BY revenue_rank
"""))
con.close()
```

</details>

---

## 3.5 Day 3 mini-project: the customer × week panel (70 min)

**Goal.** Build the table every later module starts from: `marts.panel_customer_week`, with one row per customer per week and these columns:

| Column | Role |
|---|---|
| `customer_id`, `week_start`, `week_no` | unit and time |
| `event_week` | weeks relative to the email (0 = email week) |
| `treated`, `post`, `treated_post` | treatment group, period, and their interaction |
| `n_orders`, `revenue` | outcomes, 0 when nothing was bought |
| `region`, `age`, `is_loyalty` | pre-treatment covariates |
| `pre_orders`, `pre_revenue` | pre-period outcomes (weeks 1–8), repeated on each row |

The SQL lives in its own file so it can be read, reviewed and (from Day 6) moved into dbt. Create `sql/panel_customer_week.sql`:

```sql
-- sql/panel_customer_week.sql
-- One row per customer per week (4,000 x 16 = 64,000 rows).
-- Grain: customer_id x week_start. Treatment: SPRING26 email, sent 2026-03-02 (week 9).

CREATE OR REPLACE TABLE marts.panel_customer_week AS
WITH weekly_orders AS (          -- 1. outcomes, only where orders exist
    SELECT customer_id,
           date_trunc('week', order_date)::DATE AS week_start,
           COUNT(DISTINCT order_id)             AS n_orders,
           SUM(revenue)                         AS revenue
    FROM clean.orders
    GROUP BY ALL
),
spine AS (                       -- 2. every customer in every week
    SELECT c.customer_id, w.week_start, w.week_no
    FROM clean.customers AS c
    CROSS JOIN clean.weeks AS w
),
filled AS (                      -- 3. attach outcomes; no order = 0, not NULL
    SELECT s.*,
           COALESCE(o.n_orders, 0) AS n_orders,
           COALESCE(o.revenue, 0)  AS revenue
    FROM spine AS s
    LEFT JOIN weekly_orders AS o USING (customer_id, week_start)
)
SELECT
    f.customer_id,
    f.week_start,
    f.week_no,
    f.week_no - 9                                       AS event_week,   -- 0 = email week
    -- treatment
    (e.customer_id IS NOT NULL)::INTEGER                AS treated,
    (f.week_start >= DATE '2026-03-02')::INTEGER        AS post,
    ((e.customer_id IS NOT NULL)
       AND f.week_start >= e.sent_date)::INTEGER        AS treated_post,
    -- outcomes
    f.n_orders,
    ROUND(f.revenue, 2)                                 AS revenue,
    -- pre-treatment covariates (fixed per customer)
    c.region,
    c.age,
    c.is_loyalty::INTEGER                               AS is_loyalty,
    SUM(f.n_orders) FILTER (WHERE f.week_no <= 8)
        OVER (PARTITION BY f.customer_id)               AS pre_orders,
    ROUND(SUM(f.revenue) FILTER (WHERE f.week_no <= 8)
        OVER (PARTITION BY f.customer_id), 2)           AS pre_revenue
FROM filled AS f
JOIN clean.customers AS c USING (customer_id)
LEFT JOIN clean.email_campaign AS e USING (customer_id)
ORDER BY f.customer_id, f.week_start;
```

Read it top to bottom: the `WITH` clauses are steps, just like a Polars pipeline. Then create `day03/build_panel.py`, which runs the SQL, **checks** the result and exports it:

```python
# day03/build_panel.py
"""Mini-project: build, check and export the customer x week panel."""
from pathlib import Path

import duckdb

con = duckdb.connect("data/northmart.duckdb")
con.sql("CREATE SCHEMA IF NOT EXISTS marts")
con.sql(Path("sql/panel_customer_week.sql").read_text())


def scalar(sql: str):
    return con.sql(sql).fetchall()[0][0]


# ---- Checks: each one is a claim about the data that must be true ----
checks = {
    "one row per customer-week":
        scalar("SELECT COUNT(*) FROM marts.panel_customer_week")
        == scalar("SELECT COUNT(*) FROM clean.customers") * 16,
    "no duplicate keys":
        scalar("SELECT COUNT(*) - COUNT(DISTINCT (customer_id, week_start)) FROM marts.panel_customer_week") == 0,
    "no missing outcomes":
        scalar("SELECT COUNT(*) FROM marts.panel_customer_week WHERE n_orders IS NULL OR revenue IS NULL") == 0,
    "revenue matches clean orders":
        abs(scalar("SELECT SUM(revenue) FROM marts.panel_customer_week")
            - scalar("SELECT SUM(revenue) FROM clean.orders")) < 1,
    "treated_post only after the email":
        scalar("SELECT COUNT(*) FROM marts.panel_customer_week WHERE treated_post = 1 AND post = 0") == 0,
    "pre_orders constant within customer":
        scalar("""SELECT COUNT(*) FROM (SELECT customer_id FROM marts.panel_customer_week
                  GROUP BY customer_id HAVING COUNT(DISTINCT pre_orders) > 1)""") == 0,
}
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
assert all(checks.values()), "Fix the failing checks before using the panel."

# ---- A first look at the panel ----
print(con.sql("""
    SELECT treated, post,
           COUNT(DISTINCT customer_id)  AS customers,
           ROUND(AVG(n_orders), 3)      AS orders_per_week,
           ROUND(AVG(revenue), 2)       AS revenue_per_week
    FROM marts.panel_customer_week
    GROUP BY treated, post
    ORDER BY treated, post
"""))

Path("data/marts").mkdir(parents=True, exist_ok=True)
con.sql("COPY marts.panel_customer_week TO 'data/marts/panel_customer_week.parquet' (FORMAT parquet)")
print("Exported data/marts/panel_customer_week.parquet")
con.close()
```

```bash
python day03/build_panel.py
```

```
[PASS] one row per customer-week
[PASS] no duplicate keys
[PASS] no missing outcomes
[PASS] revenue matches clean orders
[PASS] treated_post only after the email
[PASS] pre_orders constant within customer
┌─────────┬───────┬───────────┬─────────────────┬──────────────────┐
│ treated │ post  │ customers │ orders_per_week │ revenue_per_week │
├─────────┼───────┼───────────┼─────────────────┼──────────────────┤
│       0 │     0 │      2721 │           0.231 │            35.99 │
│       0 │     1 │      2721 │           0.203 │            32.64 │
│       1 │     0 │      1279 │           0.404 │            63.22 │
│       1 │     1 │      1279 │           0.416 │            65.44 │
└─────────┴───────┴───────────┴─────────────────┴──────────────────┘
Exported data/marts/panel_customer_week.parquet
```

Each check is a claim about the data written as code. The last two protect causal validity directly: no "treated" rows before the email existed, and pre-period covariates that cannot change after treatment. On Day 8 these checks become formal **data contracts**.

The 2 × 2 table above is the raw material of difference-in-differences: (0.416 − 0.404) − (0.203 − 0.231) ≈ +0.04 orders per customer-week, about +11% on what treated customers would otherwise have ordered (0.404 − 0.028 = 0.376). Day 2's growth-rate version gave +15.5%. Which scale the "parallel trends" assumption should hold on, levels or growth rates, is a Module 5 question, along with standard errors.

**Try it.** Load the exported Parquet file in **Polars** and build an event-study table: average weekly orders by `event_week` for treated and control customers, side by side. (Hint: `group_by` then `pivot`.) Describe what you see before and after week 0.

<details>
<summary>Solution</summary>

```python
import polars as pl
panel = pl.read_parquet("data/marts/panel_customer_week.parquet")
print(panel.shape)
print(panel.group_by("event_week", "treated")
      .agg(orders=pl.col("n_orders").mean().round(3))
      .pivot(on="treated", index="event_week", values="orders")
      .sort("event_week"))
```

```
│ -8         ┆ 0.229 ┆ 0.334 │
│ -7         ┆ 0.238 ┆ 0.379 │
...
│ 6          ┆ 0.205 ┆ 0.431 │
│ 7          ┆ 0.218 ┆ 0.435 │
```

Treated customers order more in **every** week, including all pre-period weeks: the level gap is selection, not the email. What changes after week 0 is the gap's *size* relative to the control group's path. Plotting these two lines is the first figure of the Module 5 capstone.

</details>

---

## 3.6 Save with Git and look ahead (15 min)

Prove the whole pipeline is reproducible: delete every generated file and rebuild from scratch.

```bash
rm -rf data/raw data/clean data/marts data/northmart.duckdb
python data/make_northmart.py
python day02/clean.py
python day03/build_warehouse.py
python day03/calendar_spine.py
python day03/build_panel.py
```

All six checks should pass again with identical numbers. Then commit:

```bash
git add .
git commit -m "Day 3: DuckDB warehouse and tested customer x week panel"
git push
```

**Coming next (Days 4–10).** Today the pipeline is five commands you must run in the right order. Next we make it professional: Days 4–5 ingest data from an API with **dlt**; Days 6–7 move the SQL into **dbt** models with tests; Day 8 adds **data contracts** that catch leakage and treatment-timing errors; Day 9 schedules everything with **Prefect**; Day 10 is the Module 1 capstone, the **Causal Data Mart**.

### Day 3 checklist

- [ ] I can query CSV and Parquet files with DuckDB and pass results to Polars
- [ ] I can build a persistent DuckDB database with schemas
- [ ] I can explain why a panel needs a calendar spine and zero-filled outcomes
- [ ] I can use `LAG`, running sums and `FILTER` inside window functions
- [ ] I have a tested `panel_customer_week` table and can rebuild it from scratch
