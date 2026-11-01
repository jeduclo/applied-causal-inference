# Module 1 · Causal-Ready Data Engineering
## Days 4–7: APIs, Ingestion with dlt and Models with dbt

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 1–3
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 7:** Students pull NorthMart data from a live REST API, load it incrementally into DuckDB with dlt, and rebuild the customer × week panel as a tested dbt project that reproduces the Day 3 panel exactly.

---

### Where we are

After Day 3 you have:

| Piece | File |
|---|---|
| Four raw CSV files | `data/raw/` |
| A cleaning script with assertions | `day02/clean.py` |
| A local DuckDB warehouse | `data/northmart.duckdb` |
| A tested customer × week panel | `sql/panel_customer_week.sql`, `day03/build_panel.py` |

What's missing is how real data arrives. Companies don't email you CSV files: data sits behind **APIs**, arrives **every day**, and changes. And our panel is one SQL file run by hand, with checks in a Python script. Days 4–7 fix both:

1. **Day 4:** fetch data from an API with `requests`.
2. **Day 5:** load it into DuckDB with **dlt**, only new rows each time.
3. **Days 6–7:** rebuild the panel as a **dbt** project: SQL models with tests and documentation.

### New folders (built up over four days)

```
causal-decision-lab/
├── .dlt/
│   ├── config.toml              # settings (committed)
│   └── secrets.toml             # API key (NOT committed)
├── api/
│   ├── northmart_api.py         # provided: the simulated company API
│   └── clock.py                 # provided: moves the API's "today"
├── day04/  first_request.py, json_basics.py, paginate.py, flatten.py, api_client.py
├── day05/  dlt_hello.py, orders_resource.py, incremental_orders.py, northmart_pipeline.py
├── day07/  compare_panels.py
└── transform/                   # the dbt project
    ├── dbt_project.yml
    ├── profiles.yml
    ├── macros/generate_schema_name.sql
    ├── models/
    │   ├── staging/       _sources.yml, _staging.yml, stg_*.sql
    │   ├── intermediate/  int_weeks.sql, int_customer_weekly_orders.sql
    │   └── marts/         _marts.yml, panel_customer_week.sql
    └── tests/
        ├── generic/unique_combination.sql
        └── assert_*.sql
```

> **Update your environment first.** Replace `requirements.txt` and `.gitignore` with the versions at the start of Day 1 (they now list dlt, dbt and Prefect), then run `pip install -r requirements.txt`. If you already used those versions, nothing changes.

---

# DAY 4 — APIs and JSON for Economists

| Time | Block |
|---|---|
| 0:00 – 0:30 | 4.1 Why APIs, and starting the NorthMart API |
| 0:30 – 1:15 | 4.2 Your first requests: status codes and API keys |
| 1:15 – 1:30 | Break |
| 1:30 – 2:15 | 4.3 JSON: dictionaries and lists all the way down |
| 2:15 – 3:05 | 4.4 Pagination: getting every page |
| 3:05 – 3:20 | Break |
| 3:20 – 3:55 | 4.5 Flattening nested JSON with Polars |
| 3:55 – 4:45 | 4.6 Day 4 mini-project: a reusable API client |
| 4:45 – 5:00 | 4.7 Save your work with Git |

---

## 4.1 Why APIs, and starting the NorthMart API (30 min)

**Concept.** An **API** (application programming interface) is a web address that returns data instead of a web page. Statistics agencies, central banks, payment processors and marketing platforms all publish data this way. For causal work, APIs matter because they are where **treatment logs** live: who got an email, when a price changed, which region a campaign launched in.

For this course, NorthMart has its own API. It serves the same data as `data/raw/`, but the way a real company system would: behind a key, in pages, with nested fields, and only up to "today".

**Step 1.** Create the folder `api/` and the file `api/northmart_api.py`:

<details>
<summary>api/northmart_api.py (provided; click to expand)</summary>

```python
"""
NorthMart REST API (simulated). Provided for the course: you don't need to edit it.

Start it in its own terminal from the repository root:
    python api/northmart_api.py

It serves the files in data/raw/ as a paginated JSON API at http://127.0.0.1:8000
(interactive docs at http://127.0.0.1:8000/docs).

Like a real company API it:
  - requires an API key in the X-API-Key header
  - returns results in pages (page, page_size, next_page)
  - nests related fields inside objects
  - only shows orders up to the simulated "today" stored in api/clock.txt
    (move the clock with: python api/clock.py 2026-04-26)
"""
from datetime import date
from pathlib import Path

import polars as pl
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLOCK = Path(__file__).resolve().parent / "clock.txt"
API_KEY = "northmart-demo-key"
MAX_PAGE_SIZE = 1000

app = FastAPI(title="NorthMart API", version="1.0")

customers = pl.read_csv(RAW / "customers.csv", try_parse_dates=True)
products = pl.read_csv(RAW / "products.csv")
orders = pl.read_csv(RAW / "orders.csv", try_parse_dates=True)
campaign = pl.read_csv(RAW / "email_campaign.csv", try_parse_dates=True)


def today() -> date:
    """The simulated 'today'. Orders after this date have not happened yet."""
    if CLOCK.exists():
        return date.fromisoformat(CLOCK.read_text().strip())
    return date(2026, 4, 26)


def check_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def paginate(rows: list[dict], page: int, page_size: int) -> dict:
    start = (page - 1) * page_size
    chunk = rows[start:start + page_size]
    has_more = start + page_size < len(rows)
    return {
        "page": page,
        "page_size": page_size,
        "total": len(rows),
        "next_page": page + 1 if has_more else None,
        "data": chunk,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "today": today().isoformat()}


@app.get("/customers", dependencies=[Depends(check_key)])
def get_customers(page: int = Query(1, ge=1), page_size: int = Query(500, ge=1, le=MAX_PAGE_SIZE)) -> dict:
    rows = [
        {
            "customer_id": r["customer_id"],
            "signup_date": r["signup_date"].isoformat(),
            "profile": {"region": r["region"], "age": r["age"]},
            "is_loyalty": r["is_loyalty"],
        }
        for r in customers.iter_rows(named=True)
    ]
    return paginate(rows, page, page_size)


@app.get("/products", dependencies=[Depends(check_key)])
def get_products() -> dict:
    return {"data": products.to_dicts()}


@app.get("/orders", dependencies=[Depends(check_key)])
def get_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=MAX_PAGE_SIZE),
    updated_since: str | None = Query(None, description="ISO timestamp; only orders updated after it"),
) -> dict:
    visible = orders.filter(pl.col("order_date") <= today())
    rows = []
    for r in visible.iter_rows(named=True):
        updated_at = f"{r['order_date'].isoformat()}T23:00:00Z"
        if updated_since and updated_at <= updated_since:
            continue
        rows.append({
            "order_id": r["order_id"],
            "customer_id": r["customer_id"],
            "order_date": r["order_date"].isoformat(),
            "status": r["status"],
            "updated_at": updated_at,
            "line": {
                "product_id": r["product_id"],
                "quantity": r["quantity"],
                "unit_price": r["unit_price"],
                "discount_pct": r["discount_pct"],
            },
        })
    return paginate(rows, page, page_size)


@app.get("/campaigns/{campaign_id}/recipients", dependencies=[Depends(check_key)])
def get_recipients(campaign_id: str) -> dict:
    sent = campaign.filter((pl.col("campaign_id") == campaign_id) & (pl.col("sent_date") <= today()))
    if campaign.filter(pl.col("campaign_id") == campaign_id).height == 0:
        raise HTTPException(status_code=404, detail=f"Unknown campaign {campaign_id}")
    return {
        "campaign_id": campaign_id,
        "data": [{"customer_id": r["customer_id"], "sent_date": r["sent_date"].isoformat()}
                 for r in sent.iter_rows(named=True)],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

</details>

**Step 2.** Create `api/clock.py`:

```python
"""
Move the NorthMart API's simulated 'today'.

    python api/clock.py 2026-02-28    # only the 8 pre-email weeks exist
    python api/clock.py 2026-04-26    # all 16 weeks exist (the default)

The API reads the clock on every request, so no restart is needed.
"""
import sys
from datetime import date
from pathlib import Path

CLOCK = Path(__file__).resolve().parent / "clock.txt"

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

new_day = date.fromisoformat(sys.argv[1])          # fails loudly on a bad date
CLOCK.write_text(new_day.isoformat())
print(f"NorthMart API 'today' is now {new_day}")
```

**Step 3. Start the API in its own terminal.** In VS Code, click **+** in the Terminal panel to open a second terminal, then:

```bash
source .venv/bin/activate
python data/make_northmart.py      # only if data/raw/ is missing
python api/northmart_api.py
```

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Leave this terminal running all day. Do your work in the **first** terminal. Codespaces will offer to open port 8000 in the browser: add `/docs` to the address to see interactive documentation of every endpoint.

> If you close the Codespace, the API stops. Restart it with `python api/northmart_api.py` at the start of Days 4 and 5.

---

## 4.2 Your first requests: status codes and API keys (45 min)

**Concept.** Every API call returns a **status code** and a **body**:

| Code | Meaning | What you do |
|---|---|---|
| 200 | OK | read the body |
| 401 | not authorised | check your API key |
| 404 | not found | check the address |
| 429 | too many requests | wait and retry |
| 500–504 | server error | wait and retry |

Create `day04/first_request.py`:

```python
# day04/first_request.py
import requests

BASE = "http://127.0.0.1:8000"

# 1. A public endpoint: no key needed
r = requests.get(f"{BASE}/health", timeout=10)
print(r.status_code)        # 200 = OK
print(r.json())             # the body, turned into a Python dict

# 2. A protected endpoint without a key
r = requests.get(f"{BASE}/products", timeout=10)
print(r.status_code, r.json())   # 401 = not authorised

# 3. The same request with the key in a header
r = requests.get(f"{BASE}/products", headers={"X-API-Key": "northmart-demo-key"}, timeout=10)
r.raise_for_status()        # stop with an error if the status is not 2xx
products = r.json()["data"]
print(len(products), "products; first one:", products[0])
```

```bash
python day04/first_request.py
```

```
200
{'status': 'ok', 'today': '2026-04-26'}
401 {'detail': 'Missing or invalid X-API-Key header'}
40 products; first one: {'product_id': 'P001', 'category': 'Grocery', 'base_price': 23.73}
```

Notes:

- `timeout=10` stops your script from hanging forever if the server never answers. Always set one.
- `raise_for_status()` turns an error code into a Python error. Without it, a 401 silently gives you an error message where you expected data.
- The key travels in a **header**, not in the URL, so it doesn't end up in logs and browser history.

**Try it.** Request the recipients of campaign `SPRING26` (endpoint `/campaigns/SPRING26/recipients`) and count them. Then request a campaign that doesn't exist, `WINTER25`, and print the status code and message.

<details>
<summary>Solution</summary>

```python
import requests

BASE = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "northmart-demo-key"}

r = requests.get(f"{BASE}/campaigns/SPRING26/recipients", headers=HEADERS, timeout=10)
r.raise_for_status()
print("Recipients:", len(r.json()["data"]))

r = requests.get(f"{BASE}/campaigns/WINTER25/recipients", headers=HEADERS, timeout=10)
print(r.status_code, r.json()["detail"])
```

```
Recipients: 1279
404 Unknown campaign WINTER25
```

</details>

---

## 4.3 JSON: dictionaries and lists all the way down (45 min)

**Concept.** APIs answer in **JSON**, which maps directly onto Python: JSON objects become **dictionaries**, JSON arrays become **lists**. Objects can contain objects, so data is often **nested**.

Create `day04/json_basics.py`:

```python
# day04/json_basics.py
import json

import requests

BASE = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "northmart-demo-key"}

r = requests.get(f"{BASE}/orders", headers=HEADERS, params={"page_size": 2}, timeout=10)
body = r.json()

# 1. The top level is a dict: some metadata plus a list of records
print(list(body.keys()))
print("total orders:", body["total"], "| next page:", body["next_page"])

# 2. Pretty-print one record to see its shape
first = body["data"][0]
print(json.dumps(first, indent=2))

# 3. Nested fields: a dict inside a dict
print("quantity:", first["line"]["quantity"])
revenue = first["line"]["quantity"] * first["line"]["unit_price"]
print(f"revenue: {revenue:.2f}")

# 4. params= builds the query string for you
print(r.url)
```

```
['page', 'page_size', 'total', 'next_page', 'data']
total orders: 18542 | next page: 2
{
  "order_id": "O000001",
  "customer_id": "C00003",
  "order_date": "2026-01-08",
  "status": "completed",
  "updated_at": "2026-01-08T23:00:00Z",
  "line": {
    "product_id": "P011",
    "quantity": 1,
    "unit_price": 61.62,
    "discount_pct": 20
  }
}
quantity: 1
revenue: 61.62
http://127.0.0.1:8000/orders?page_size=2
```

Two new fields compared with the CSV:

- **`line`** groups the product fields in a nested object.
- **`updated_at`** records when the API last changed the record. Tomorrow, dlt uses it to load only new data.

**Try it.** Fetch 5 orders and use a **list comprehension** to compute each order's revenue; print the list and the total.

<details>
<summary>Solution</summary>

```python
import requests

r = requests.get("http://127.0.0.1:8000/orders", headers={"X-API-Key": "northmart-demo-key"},
                 params={"page_size": 5}, timeout=10)
orders = r.json()["data"]
revenues = [o["line"]["quantity"] * o["line"]["unit_price"] for o in orders]
print([round(x, 2) for x in revenues])
print("Total:", round(sum(revenues), 2))
```

```
[61.62, 75.36, 183.47, 103.62, 72.6]
Total: 496.67
```

`[expression for item in list]` builds a new list in one line: Python's version of applying a formula down a column.

</details>

---

## 4.4 Pagination: getting every page (50 min)

**Concept.** APIs rarely return everything at once. NorthMart returns at most 1,000 records per **page** and tells you the `next_page`, or `null` (Python `None`) on the last page. Forgetting pagination is one of the most common silent errors in applied work: your sample is the first 500 customers and nobody notices.

Create `day04/paginate.py`:

```python
# day04/paginate.py
import requests

BASE = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "northmart-demo-key"}


def fetch_all(endpoint: str, page_size: int = 1000) -> list[dict]:
    """Follow next_page until the API says there is no more."""
    rows: list[dict] = []
    page = 1
    while page is not None:
        r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS,
                         params={"page": page, "page_size": page_size}, timeout=30)
        r.raise_for_status()
        body = r.json()
        rows.extend(body["data"])
        print(f"  {endpoint}: page {page} -> {len(body['data'])} rows")
        page = body["next_page"]
    return rows


if __name__ == "__main__":
    orders = fetch_all("orders")
    print("Orders fetched:", len(orders))
```

```bash
python day04/paginate.py
```

```
  orders: page 1 -> 1000 rows
  ...
  orders: page 19 -> 542 rows
Orders fetched: 18542
```

`while page is not None` keeps asking until the API says it's done. The function is **generic**: it works for any paginated endpoint, which we use straight away.

**Try it.** Use `fetch_all` to download all customers, turn them into a Polars DataFrame, and count loyalty members and customers per region. (Hint: put your script in `day04/` and use `from paginate import fetch_all`.)

<details>
<summary>Solution</summary>

```python
import polars as pl

from paginate import fetch_all

customers = pl.DataFrame(fetch_all("customers")).unnest("profile")
print(customers.head(3))
print("Loyalty members:", customers["is_loyalty"].sum())
print(customers.group_by("region").len().sort("len", descending=True).head(6))
```

```
Loyalty members: 1194
│ ON     ┆ 1401 │
│ QC     ┆ 854  │
...
│ null   ┆ 74   │
```

The API returns regions exactly as messy as the CSV did. Cleaning remains our job, and on Day 6 dbt does it.

</details>

---

## 4.5 Flattening nested JSON with Polars (35 min)

**Concept.** Analysis needs flat tables. Polars stores a nested object as a **Struct** column, and `.unnest()` spreads its fields into ordinary columns.

Create `day04/flatten.py`:

```python
# day04/flatten.py
import polars as pl

from paginate import fetch_all

orders_json = fetch_all("orders")

# 1. Polars turns a list of dicts into a DataFrame; nested dicts become a Struct column
orders = pl.DataFrame(orders_json)
print(orders.schema)

# 2. unnest() spreads the struct's fields into ordinary columns
flat = orders.unnest("line").with_columns(pl.col("order_date").str.to_date())
print(flat.head(3))

# 3. Reconcile with the CSV we used on Days 1-3
csv = pl.read_csv("data/raw/orders.csv", try_parse_dates=True)
print("API rows:", flat.height, "| CSV rows:", csv.height)
same = flat.select(csv.columns).sort("order_id").equals(csv.sort("order_id"))
print("Identical to the CSV:", same)
```

```
Schema({'order_id': String, ..., 'line': Struct({'product_id': String, 'quantity': Int64, 'unit_price': Float64, 'discount_pct': Int64})})
...
API rows: 18542 | CSV rows: 18542
Identical to the CSV: True
```

That last line is a **reconciliation**: the same data obtained two ways must match. Whenever you change a data source, reconcile before you estimate anything.

---

## 4.6 Day 4 mini-project: a reusable API client (50 min)

**Goal.** Write `day04/api_client.py`, a small client that:

1. Reads the API key from an **environment variable**, never from the code.
2. Uses one `requests.Session` with automatic **retries** on 429 and 5xx errors.
3. Yields pages with a **generator** (`yield`), so large endpoints don't have to fit in memory at once.
4. Saves all four endpoints as Parquet in `data/api/`.

```python
# day04/api_client.py
"""Mini-project: a small, reusable client for the NorthMart API."""
import os
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = os.environ.get("NORTHMART_API_URL", "http://127.0.0.1:8000")
OUT = Path("data/api")


def make_session() -> requests.Session:
    """One session for all calls: shared headers, automatic retries on server errors."""
    key = os.environ.get("NORTHMART_API_KEY")
    if not key:
        raise RuntimeError("Set NORTHMART_API_KEY first, e.g.  export NORTHMART_API_KEY=northmart-demo-key")
    session = requests.Session()
    session.headers["X-API-Key"] = key
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_pages(session: requests.Session, endpoint: str, **params):
    """Yield one page of records at a time (a generator)."""
    page = 1
    while page is not None:
        r = session.get(f"{BASE}/{endpoint}", params={"page": page, "page_size": 1000, **params}, timeout=30)
        r.raise_for_status()
        body = r.json()
        yield body["data"]
        page = body.get("next_page")


def fetch_all(session: requests.Session, endpoint: str, **params) -> list[dict]:
    return [row for page in fetch_pages(session, endpoint, **params) for row in page]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = make_session()

    customers = pl.DataFrame(fetch_all(s, "customers")).unnest("profile")
    orders = pl.DataFrame(fetch_all(s, "orders")).unnest("line")
    products = pl.DataFrame(s.get(f"{BASE}/products", timeout=30).json()["data"])
    recipients = pl.DataFrame(s.get(f"{BASE}/campaigns/SPRING26/recipients", timeout=30).json()["data"])

    for name, df in [("customers", customers), ("orders", orders),
                     ("products", products), ("recipients", recipients)]:
        df.write_parquet(OUT / f"{name}.parquet")
        print(f"{name:<11} {df.height:>7,} rows  -> {OUT / name}.parquet")


if __name__ == "__main__":
    main()
```

Set the key for this terminal session, then run:

```bash
export NORTHMART_API_KEY=northmart-demo-key
python day04/api_client.py
```

```
customers     4,000 rows  -> data/api/customers.parquet
orders       18,542 rows  -> data/api/orders.parquet
products         40 rows  -> data/api/products.parquet
recipients    1,279 rows  -> data/api/recipients.parquet
```

Without the variable you get a clear message instead of a confusing 401:

```
RuntimeError: Set NORTHMART_API_KEY first, e.g.  export NORTHMART_API_KEY=northmart-demo-key
```

> **Why environment variables?** Code is shared on GitHub; keys must not be. A key in a script is leaked the moment you push. The demo key is harmless, but build the habit now.

A generator function (`yield` instead of `return`) hands back one page, pauses, and continues when asked for the next. Tomorrow you'll see that dlt is built around exactly this idea.

---

## 4.7 Save your work with Git (15 min)

```bash
git add .
git status          # data/api/ and api/clock.txt should NOT appear
git commit -m "Day 4: NorthMart API, pagination and a reusable client"
git push
```

### Day 4 checklist

- [ ] I can start the NorthMart API and read its `/docs` page
- [ ] I can make requests with headers, parameters and a timeout, and read status codes
- [ ] I can navigate nested JSON and flatten it with Polars `unnest`
- [ ] I can fetch every page of a paginated endpoint
- [ ] I keep API keys in environment variables, not in code

---

# DAY 5 — Ingestion Pipelines with dlt

| Time | Block |
|---|---|
| 0:00 – 0:40 | 5.1 Why a loading tool? Your first dlt pipeline |
| 0:40 – 1:30 | 5.2 Resources: from API pages to tables |
| 1:30 – 1:45 | Break |
| 1:45 – 2:35 | 5.3 Write dispositions: replace, append, merge |
| 2:35 – 3:20 | 5.4 Incremental loading as time passes |
| 3:20 – 3:35 | Break |
| 3:35 – 4:45 | 5.5 Day 5 mini-project: the NorthMart source |
| 4:45 – 5:00 | 5.6 Save your work with Git |

---

## 5.1 Why a loading tool? Your first dlt pipeline (40 min)

**Concept.** Yesterday's client works, but a production loader needs more: create tables with the right types, handle new columns, remember what was already loaded, never load an order twice, and record every load. **dlt** (data load tool) does all of this. You write a Python function that yields records; dlt turns them into tables.

Make sure the API is running in your second terminal, then create `day05/dlt_hello.py`:

```python
# day05/dlt_hello.py
from pathlib import Path

import dlt

# 1. Any list of dicts (or a generator of them) can be loaded
rates = [
    {"week": "2026-03-02", "segment": "loyalty", "orders_per_customer": 0.52},
    {"week": "2026-03-02", "segment": "regular", "orders_per_customer": 0.21},
]

# 2. A pipeline = a name + a destination + a dataset (schema) to write into
pipeline = dlt.pipeline(
    pipeline_name="hello",
    destination=dlt.destinations.duckdb(str(Path("data/northmart.duckdb").resolve())),
    pipelines_dir=".dlt_pipelines",   # keep pipeline state inside the project
    dataset_name="practice",
)

# 3. run() infers the schema, creates the table and loads the rows
info = pipeline.run(rates, table_name="weekly_rates")
print(info)

# 4. Read it back
with pipeline.sql_client() as client:
    print(client.execute_sql("SELECT * FROM weekly_rates"))
```

```bash
python day05/dlt_hello.py
```

```
Pipeline hello load step finished in 0.23 seconds
1 load package(s) were loaded to destination duckdb and into dataset practice
The duckdb destination used duckdb:////workspaces/causal-decision-lab/data/northmart.duckdb location to store data
Load package 1793486741.6855483 is LOADED and contains no failed jobs
[('2026-03-02', 'loyalty', 0.52, '1793486741.6855483', 'cKfwif9Og0Edog'), ('2026-03-02', 'regular', 0.21, ...)]
```

dlt created the schema `practice` and the table `weekly_rates`, and added two columns of its own: `_dlt_load_id` (which load wrote the row) and `_dlt_id` (a unique row ID). Every row can be traced back to the load that wrote it.

Two settings in the pipeline deserve a sentence each:

- **An absolute database path** (`Path(...).resolve()`). dlt remembers each pipeline between runs, including the folder it was first run from. A relative path can then point somewhere unexpected if you run the script from a different folder.
- **`pipelines_dir=".dlt_pipelines"`** keeps that memory inside your project (and out of Git), instead of a hidden system folder shared with other projects.

**Now run the same script a second time.** The final line shows **four** rows. By default dlt *appends*: running a load twice duplicates the data. Section 5.3 explains how to choose.

---

## 5.2 Resources: from API pages to tables (50 min)

**Concept.** A dlt **resource** is a generator decorated with `@dlt.resource`. Yesterday's pagination loop becomes a resource almost unchanged: instead of collecting rows into a list, it `yield`s each page.

Create `day05/orders_resource.py`:

```python
# day05/orders_resource.py
from pathlib import Path

import dlt
import requests

BASE = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "northmart-demo-key"}


@dlt.resource(name="orders", write_disposition="replace")
def orders():
    """A resource is a generator: it yields pages, dlt does the rest."""
    page = 1
    while page is not None:
        r = requests.get(f"{BASE}/orders", headers=HEADERS,
                         params={"page": page, "page_size": 1000}, timeout=30)
        r.raise_for_status()
        body = r.json()
        yield body["data"]
        page = body["next_page"]


pipeline = dlt.pipeline(
    pipeline_name="orders_demo",
    destination=dlt.destinations.duckdb(str(Path("data/northmart.duckdb").resolve())),
    pipelines_dir=".dlt_pipelines",   # keep pipeline state inside the project
    dataset_name="practice",
)
print(pipeline.run(orders()))

with pipeline.sql_client() as client:
    print(client.execute_sql("SELECT COUNT(*), COUNT(DISTINCT order_id) FROM orders"))
    cols = client.execute_sql("""
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'practice' AND table_name = 'orders' ORDER BY ordinal_position
    """)
    for name, dtype in cols:
        print(f"  {name:<22} {dtype}")
```

```
[(18542, 18359)]
  order_id               VARCHAR
  customer_id            VARCHAR
  order_date             VARCHAR
  status                 VARCHAR
  updated_at             TIMESTAMP WITH TIME ZONE
  line__product_id       VARCHAR
  line__quantity         BIGINT
  line__unit_price       DOUBLE
  line__discount_pct     BIGINT
  _dlt_load_id           VARCHAR
  _dlt_id                VARCHAR
```

Three things happened without you asking:

1. **Flattening.** The nested `line` object became `line__product_id`, `line__quantity`… (double underscore = one level of nesting).
2. **Type inference.** `updated_at` was recognised as a timestamp. `order_date` stayed text: dlt is cautious with dates, so we cast it ourselves in dbt.
3. **Nothing was deduplicated.** 18,542 rows but only 18,359 distinct `order_id`s: the API serves the same 183 duplicate rows as the CSV.

---

## 5.3 Write dispositions: replace, append, merge (50 min)

**Concept.** The **write disposition** decides what happens to existing rows when you load again:

| Disposition | What it does | Use it for |
|---|---|---|
| `replace` | drop the table, load fresh | small reference tables (products, customers) |
| `append` | add rows to the end | event logs where every row is new |
| `merge` | insert new keys, update existing ones | records that can change or repeat (orders) |

`merge` needs a **primary key**: the column(s) that identify a record. With `primary_key="order_id"`, loading the same order twice keeps one row. That fixes the duplicate problem at the source rather than downstream.

**Try it.** Load the same two rows twice with `append`, and twice with `replace`, into separate datasets. How many rows does each table hold?

<details>
<summary>Solution</summary>

```python
from pathlib import Path

import dlt

rows = [{"segment": "loyalty", "orders": 0.52}, {"segment": "regular", "orders": 0.21}]

for disposition in ["append", "replace"]:
    pipeline = dlt.pipeline(
        pipeline_name=f"demo_{disposition}",
        destination=dlt.destinations.duckdb(str(Path("data/northmart.duckdb").resolve())),
        pipelines_dir=".dlt_pipelines",
        dataset_name=f"practice_{disposition}",
    )
    for _ in range(2):                      # run the same load twice
        pipeline.run(rows, table_name="rates", write_disposition=disposition)
    with pipeline.sql_client() as client:
        n = client.execute_sql("SELECT COUNT(*) FROM rates")[0][0]
    print(f"{disposition:<8} after two runs: {n} rows")
```

```
append   after two runs: 4 rows
replace  after two runs: 2 rows
```

`write_disposition` can be set on the resource (as in 5.2) or overridden in `pipeline.run(...)`, as here.

</details>

---

## 5.4 Incremental loading as time passes (45 min)

**Concept.** NorthMart's orders grow every day. Reloading 16 weeks of history every morning wastes time and, for big companies, money. **Incremental loading** asks the API only for what changed since the last run. dlt stores the cursor (the largest `updated_at` seen) in the pipeline's state and hands it back next time.

The API's **clock** lets us replay time. Before the email: set the clock to Saturday 28 February 2026, the last day of week 8.

Create `day05/incremental_orders.py`:

```python
# day05/incremental_orders.py
"""Load only what is new, and never load the same order twice."""
from pathlib import Path

import dlt
import requests

BASE = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "northmart-demo-key"}


@dlt.resource(name="orders", primary_key="order_id", write_disposition="merge")
def orders(updated_at=dlt.sources.incremental("updated_at", initial_value="2026-01-01T00:00:00Z")):
    # dlt remembers the largest updated_at it has seen and gives it back to us next run.
    # Read it ONCE: last_value keeps moving while pages arrive during this run.
    since = updated_at.start_value
    print(f"  asking the API for orders updated after {since}")
    page = 1
    while page is not None:
        r = requests.get(f"{BASE}/orders", headers=HEADERS, timeout=30,
                         params={"page": page, "page_size": 1000, "updated_since": since})
        r.raise_for_status()
        body = r.json()
        yield body["data"]
        page = body["next_page"]


pipeline = dlt.pipeline(
    pipeline_name="incremental_demo",
    destination=dlt.destinations.duckdb(str(Path("data/northmart.duckdb").resolve())),
    pipelines_dir=".dlt_pipelines",
    dataset_name="practice_incremental",
)
pipeline.run(orders())

with pipeline.sql_client() as client:
    n, n_ids, last = client.execute_sql(
        "SELECT COUNT(*), COUNT(DISTINCT order_id), MAX(order_date) FROM orders")[0]
    print(f"orders table: {n:,} rows, {n_ids:,} distinct order_ids, latest order {last}")
```

Run it three times, moving the clock as a real company would experience it:

```bash
python api/clock.py 2026-02-28
python day05/incremental_orders.py      # first run: everything up to 28 Feb
python day05/incremental_orders.py      # same day again: nothing new
python api/clock.py 2026-04-26
python day05/incremental_orders.py      # eight weeks later: only the new orders
```

```
NorthMart API 'today' is now 2026-02-28
  asking the API for orders updated after 2026-01-01T00:00:00Z
orders table: 9,287 rows, 9,287 distinct order_ids, latest order 2026-02-28
  asking the API for orders updated after 2026-02-28T23:00:00Z
orders table: 9,287 rows, 9,287 distinct order_ids, latest order 2026-02-28
NorthMart API 'today' is now 2026-04-26
  asking the API for orders updated after 2026-02-28T23:00:00Z
orders table: 18,359 rows, 18,359 distinct order_ids, latest order 2026-04-26
```

The final table has exactly one row per order (18,359), and the duplicates are gone because of `merge`.

> **A bug worth knowing about.** The code reads `updated_at.start_value` **once**, before the loop. `last_value` changes *during* the run as rows arrive. If you pass `last_value` to every page request, later pages ask for "orders after the newest one seen so far" and silently skip older orders on those pages. In testing, that version loaded 2,978 orders instead of 9,287 and raised no error. Incremental code must be checked against a full reload at least once.

> **Late data.** Our API never changes an order after its date. Real systems do (refunds, corrections). A common safeguard is a *lookback*: ask for everything updated in the last few days, and let `merge` absorb the overlap.

---

## 5.5 Day 5 mini-project: the NorthMart source (70 min)

**Goal.** Group all four endpoints into one dlt **source** that loads into the schema `raw`, with settings in `.dlt/config.toml` and the key in `.dlt/secrets.toml`:

| Resource | Disposition | Primary key |
|---|---|---|
| `customers` | replace | `customer_id` |
| `products` | replace | `product_id` |
| `campaign_recipients` | replace | `customer_id` |
| `orders` | merge + incremental on `updated_at` | `order_id` |

**Step 1 — configuration and secrets.** Create the folder `.dlt/` and two files:

```toml
# .dlt/config.toml  (safe to commit: no secrets here)
[sources.northmart]
base_url = "http://127.0.0.1:8000"
```

```toml
# .dlt/secrets.toml  (NEVER commit this file: it is listed in .gitignore)
[sources.northmart]
api_key = "northmart-demo-key"
```

dlt fills any argument declared as `dlt.config.value` or `dlt.secrets.value` from these files, matched by name and section. In production you'd set the environment variable `SOURCES__NORTHMART__API_KEY` instead; dlt checks it automatically.

**Step 2 — the source.** Create `day05/northmart_pipeline.py`:

```python
# day05/northmart_pipeline.py
"""Mini-project: one dlt source for the whole NorthMart API, loaded into the `raw` schema."""
from pathlib import Path

import dlt
import requests

DB = str(Path("data/northmart.duckdb").resolve())


def get_pages(base_url: str, api_key: str, endpoint: str, **params):
    page = 1
    while page is not None:
        r = requests.get(f"{base_url}/{endpoint}", headers={"X-API-Key": api_key}, timeout=30,
                         params={"page": page, "page_size": 1000, **params})
        r.raise_for_status()
        body = r.json()
        yield body["data"]
        page = body.get("next_page")


def get_pages_once(base_url: str, api_key: str, endpoint: str) -> list[dict]:
    r = requests.get(f"{base_url}/{endpoint}", headers={"X-API-Key": api_key}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


@dlt.source(name="northmart")
def northmart_source(base_url: str = dlt.config.value, api_key: str = dlt.secrets.value):
    """base_url comes from .dlt/config.toml, api_key from .dlt/secrets.toml (or env vars)."""

    @dlt.resource(primary_key="customer_id", write_disposition="replace")
    def customers():
        yield from get_pages(base_url, api_key, "customers")

    @dlt.resource(primary_key="product_id", write_disposition="replace")
    def products():
        yield get_pages_once(base_url, api_key, "products")

    @dlt.resource(primary_key="customer_id", write_disposition="replace")
    def campaign_recipients():
        body = get_pages_once(base_url, api_key, "campaigns/SPRING26/recipients")
        yield [{**row, "campaign_id": "SPRING26"} for row in body]

    @dlt.resource(primary_key="order_id", write_disposition="merge")
    def orders(updated_at=dlt.sources.incremental("updated_at", initial_value="2026-01-01T00:00:00Z")):
        since = updated_at.start_value
        yield from get_pages(base_url, api_key, "orders", updated_since=since)

    return customers, products, campaign_recipients, orders


def main() -> None:
    pipeline = dlt.pipeline(
        pipeline_name="northmart",
        destination=dlt.destinations.duckdb(DB),
        pipelines_dir=".dlt_pipelines",
        dataset_name="raw",
    )
    pipeline.run(northmart_source())

    with pipeline.sql_client() as client:
        existing = {r[0] for r in client.execute_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw'")}
        for table in ["customers", "products", "campaign_recipients", "orders"]:
            if table not in existing:      # dlt only creates a table once it has received rows
                print(f"raw.{table:<20}  (no rows yet)")
                continue
            n = client.execute_sql(f"SELECT COUNT(*) FROM {table}")[0][0]
            print(f"raw.{table:<20} {n:>7,} rows")


if __name__ == "__main__":
    main()
```

**Step 3 — replay the timeline.**

```bash
python api/clock.py 2026-02-28
python day05/northmart_pipeline.py
python api/clock.py 2026-04-26
python day05/northmart_pipeline.py
```

```
NorthMart API 'today' is now 2026-02-28
raw.customers              4,000 rows
raw.products                  40 rows
raw.campaign_recipients   (no rows yet)
raw.orders                 9,287 rows
NorthMart API 'today' is now 2026-04-26
raw.customers              4,000 rows
raw.products                  40 rows
raw.campaign_recipients    1,279 rows
raw.orders                18,359 rows
```

On 28 February the email hasn't been sent, so there are no recipients, and dlt doesn't create a table until it receives rows. The script checks for that instead of crashing.

**Try it.** dlt records every load. Query `raw._dlt_loads`, then count orders by `_dlt_load_id`. Can you see the two loads?

<details>
<summary>Solution</summary>

```python
import duckdb

con = duckdb.connect("data/northmart.duckdb", read_only=True)
print(con.sql("""
    SELECT load_id, schema_name, status, inserted_at
    FROM raw._dlt_loads
    ORDER BY inserted_at
"""))
print(con.sql("""
    SELECT _dlt_load_id, COUNT(*) AS orders_loaded
    FROM raw.orders
    GROUP BY _dlt_load_id
    ORDER BY _dlt_load_id
"""))
```

```
│    _dlt_load_id    │ orders_loaded │
├────────────────────┼───────────────┤
│ 1793487005.3942654 │          9287 │
│ 1793487010.2876098 │          9072 │
```

Your load IDs differ (they are timestamps). 9,287 orders came in the first load and 9,072 in the second. This **lineage** answers questions like "which numbers changed after Tuesday's load?" months later.

</details>

---

## 5.6 Save your work with Git (15 min)

```bash
git add .
git status          # .dlt/secrets.toml and .dlt_pipelines/ must NOT appear
git commit -m "Day 5: dlt source with incremental, deduplicated orders"
git push
```

If `secrets.toml` appears in `git status`, stop and fix `.gitignore` before committing.

### Day 5 checklist

- [ ] I can write a dlt resource from a paginated API and load it into DuckDB
- [ ] I can choose between replace, append and merge, and explain why orders need merge
- [ ] I can load incrementally with a cursor and explain the `start_value` pitfall
- [ ] I keep settings in `config.toml` and keys in `secrets.toml`, never in Git
- [ ] I can trace any row back to the load that wrote it

---

# DAY 6 — dbt Foundations: Sources, Staging and Tests

| Time | Block |
|---|---|
| 0:00 – 0:45 | 6.1 Why dbt: SQL as a tested project |
| 0:45 – 1:30 | 6.2 Setting up the project and your first model |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 6.3 Sources and staging models |
| 2:45 – 3:30 | 6.4 Tests: unique, not_null, accepted_values, relationships |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 6.5 Day 6 mini-project: a fully tested staging layer |
| 4:45 – 5:00 | 6.6 Save your work with Git |

---

## 6.1 Why dbt: SQL as a tested project (45 min)

**Concept.** On Day 3 the panel was one SQL file plus a Python script of checks, run in the right order by hand. That works for one analyst and one table. It breaks when a team maintains dozens of tables that depend on each other. **dbt** (data build tool) turns SQL into a project:

| Without dbt | With dbt |
|---|---|
| `CREATE TABLE ... AS` written by hand | one `SELECT` per file; dbt creates the view or table |
| You remember the run order | dbt reads `ref()` and works out the order |
| Checks scattered in scripts | tests declared next to each model |
| Documentation in someone's head | generated website of every table and column |

dbt only does the **T** in ELT (extract, load, *transform*). dlt loaded the raw data; dbt transforms it inside DuckDB.

**The layers we'll build:**

| Layer | Schema | Purpose | Day |
|---|---|---|---|
| sources | `raw` | tables exactly as dlt loaded them | 5 |
| staging | `staging` | one cleaned model per source table: rename, cast, filter | 6 |
| intermediate | `intermediate` | building blocks: weekly outcomes, calendar | 7 |
| marts | `marts` | analysis-ready tables: the panel | 7 |

---

## 6.2 Setting up the project and your first model (45 min)

**Step 1 — the project folder.**

```bash
mkdir -p transform/models/staging transform/models/intermediate transform/models/marts
mkdir -p transform/tests/generic transform/macros
```

**Step 2 — `transform/dbt_project.yml`** names the project, sets defaults per layer, and holds **variables** we'll use tomorrow:

```yaml
name: northmart
version: "1.0.0"
profile: northmart

model-paths: ["models"]
test-paths: ["tests"]
macro-paths: ["macros"]

vars:
  first_week: "2026-01-05"      # Monday of week 1
  last_week: "2026-04-20"       # Monday of week 16
  campaign_id: "SPRING26"
  campaign_week: 9              # the email went out on Monday 2026-03-02

models:
  northmart:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: view
      +schema: intermediate
    marts:
      +materialized: table
      +schema: marts
```

**Step 3 — `transform/profiles.yml`** tells dbt which database to use. The path is relative to the `transform/` folder:

```yaml
northmart:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ../data/northmart.duckdb   # relative to the transform/ folder
      threads: 4
```

**Step 4 — `transform/macros/generate_schema_name.sql`.** By default dbt would name our schemas `main_staging` and `main_marts`. This small macro keeps the names exactly as written:

```sql
-- Use the schema names from dbt_project.yml as they are (staging, marts, ...),
-- instead of dbt's default of prefixing them with the target schema (main_staging).
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}
```

**Step 5 — check the connection.** All dbt commands run from inside `transform/`, with `--profiles-dir .` so dbt uses our `profiles.yml`:

```bash
cd transform
dbt debug --profiles-dir .
```

```
  Connection test: [OK connection ok]
All checks passed!
```

**Example — your first model.** Create `transform/models/first_model.sql` with one line:

```sql
select 'Did the email work?' as question, 42 as answer
```

```bash
dbt run --profiles-dir . --select first_model
dbt show --profiles-dir . --select first_model
```

```
1 of 1 OK created sql view model main.first_model .............................. [OK in 0.12s]

| question            | answer |
| ------------------- | ------ |
| Did the email work? |     42 |
```

A model is just a `SELECT`. dbt wrapped it in `CREATE VIEW main.first_model AS ...`. Delete `first_model.sql` now; it was only a test.

> Keep **two terminals** today too: one for the API (only needed if you re-run dlt) and one inside `transform/` for dbt. Run Python scripts from the repository root (`cd ..`).

---

## 6.3 Sources and staging models (60 min)

**Concept.** A **source** declares a table that dbt didn't create (here, dlt's `raw` tables). Models refer to it with `{{ source('raw', 'orders') }}`, never by hard-coded name, so dbt knows the dependency and can document it.

A **staging model** does the boring, essential work for one source table: rename columns, cast types, clean codes, remove rows that don't represent real events. Nothing else: no joins, no business logic.

**Example 1 — declare the sources.** `transform/models/staging/_sources.yml`:

```yaml
version: 2

sources:
  - name: raw
    description: "Tables loaded from the NorthMart API by dlt (day05/northmart_pipeline.py)."
    schema: raw
    tables:
      - name: customers
      - name: products
      - name: orders
      - name: campaign_recipients
```

**Example 2 — customers.** `transform/models/staging/stg_customers.sql`. This is Day 2's `clean_customers` function, now in SQL:

```sql
-- One row per customer. Region codes cleaned; missing region kept as UNKNOWN.
select
    customer_id,
    cast(signup_date as date)                                  as signup_date,
    coalesce(upper(trim(profile__region)), 'UNKNOWN')          as region,
    cast(profile__age as integer)                              as age,
    cast(is_loyalty as boolean)                                as is_loyalty
from {{ source('raw', 'customers') }}
```

**Example 3 — orders.** `transform/models/staging/stg_orders.sql`. It flattens dlt's column names, casts the date, computes revenue, and removes cancelled orders:

```sql
-- One row per completed order. Cancelled orders never happened, so they are removed here.
with ranked as (
    select
        *,
        row_number() over (partition by order_id order by _dlt_load_id desc) as rn
    from {{ source('raw', 'orders') }}
)
select
    order_id,
    customer_id,
    cast(order_date as date)                                   as order_date,
    line__product_id                                           as product_id,
    line__quantity                                             as quantity,
    line__unit_price                                           as unit_price,
    line__discount_pct                                         as discount_pct,
    round(line__quantity * line__unit_price, 2)                as revenue
from ranked
where rn = 1                       -- safety net: one row per order_id
  and status = 'completed'
```

`merge` already removed duplicate orders on Day 5, so why the `row_number()`? Because staging shouldn't trust that every future load is perfect. If it ever receives the same order twice, it keeps the latest load. That costs one line and protects every model downstream.

**Example 4 — the two small tables.** `stg_products.sql` and `stg_campaign_recipients.sql`:

```sql
select product_id, category, base_price
from {{ source('raw', 'products') }}
```

```sql
select
    customer_id,
    campaign_id,
    cast(sent_date as date) as sent_date
from {{ source('raw', 'campaign_recipients') }}
```

Run the staging layer and preview one model with an inline query:

```bash
dbt run --profiles-dir . --select staging
dbt show --profiles-dir . --inline "select region, count(*) n from {{ ref('stg_customers') }} group by 1 order by 2 desc"
```

```
| region |    n |
| ------ | ---- |
| ON     | 1480 |
| QC     |  912 |
| BC     |  610 |
| ATL    |  476 |
| AB     |  448 |
```

`dbt show` previews 5 rows by default; add `--limit 10` to see the `UNKNOWN` row too.

---

## 6.4 Tests: unique, not_null, accepted_values, relationships (45 min)

**Concept.** A dbt **test** is a query that looks for bad rows. If it finds any, the test fails. dbt ships four generic tests:

| Test | Claim | Causal reason to care |
|---|---|---|
| `unique` | no duplicate keys | duplicates inflate some units' outcomes |
| `not_null` | no missing values | missing outcomes silently drop units |
| `accepted_values` | only listed codes | stray codes split groups and break balance checks |
| `relationships` | every key exists in another table | orders from unknown customers can't be assigned to treatment |

Tests live in a YAML file next to the models. Create `transform/models/staging/_staging.yml`:

```yaml
version: 2

models:
  - name: stg_customers
    description: "One row per customer, regions cleaned."
    columns:
      - name: customer_id
        data_tests: [unique, not_null]
      - name: region
        data_tests:
          - accepted_values:
              arguments:
                values: ["ON", "QC", "BC", "AB", "ATL", "UNKNOWN"]
      - name: age
        data_tests: [not_null]

  - name: stg_orders
    description: "One row per completed order."
    columns:
      - name: order_id
        data_tests: [unique, not_null]
      - name: customer_id
        data_tests:
          - not_null
          - relationships:
              arguments:
                to: ref('stg_customers')
                field: customer_id
      - name: product_id
        data_tests:
          - relationships:
              arguments:
                to: ref('stg_products')
                field: product_id

  - name: stg_products
    columns:
      - name: product_id
        data_tests: [unique, not_null]

  - name: stg_campaign_recipients
    description: "Customers who received a campaign email, with the send date."
    columns:
      - name: customer_id
        data_tests:
          - unique
          - relationships:
              arguments:
                to: ref('stg_customers')
                field: customer_id
```

`dbt build` runs models **and** their tests, in dependency order:

```bash
dbt build --profiles-dir . --select staging
```

```
... OK created sql view model staging.stg_orders ......................... [OK in 0.30s]
... PASS unique_stg_orders_order_id ...................................... [PASS in 0.25s]
... PASS accepted_values_stg_customers_region__ON__QC__BC__AB__ATL__UNKNOWN  [PASS in 0.15s]
...
Done. PASS=17 WARN=0 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=17
```

> **dbt 1.12 syntax.** Test settings go under `arguments:` (as above). Older tutorials put `values:` directly under the test name; dbt still accepts that but warns that it's deprecated.

---

## 6.5 Day 6 mini-project: a fully tested staging layer (60 min)

**Part 1 — watch a test fail.** Tests only earn trust once you've seen them catch something. In `stg_customers.sql`, temporarily remove the cleaning so the line reads `coalesce(profile__region, 'UNKNOWN') as region`, then:

```bash
dbt build --profiles-dir . --select stg_customers
```

```
FAIL 10 accepted_values_stg_customers_region__ON__QC__BC__AB__ATL__UNKNOWN  [FAIL 10 in 0.18s]
  Got 10 results, configured to fail if != 0
Done. PASS=6 WARN=0 ERROR=1 SKIP=1 NO-OP=0 REUSED=0 TOTAL=8
```

dbt found the 10 messy codes (`on`, ` QC `…). Notice `SKIP=1`: a test downstream of the failure was skipped, not run on bad data. Restore the `upper(trim(...))` line and rebuild until everything passes.

**Part 2 — complete the tests.** Add tests to `stg_products` so that:

1. `category` only takes the five known values.
2. `base_price` is never missing.

Then run `dbt build --profiles-dir . --select staging` until it passes.

<details>
<summary>Solution</summary>

Replace the `stg_products` block in `_staging.yml`:

```yaml
  - name: stg_products
    columns:
      - name: product_id
        data_tests: [unique, not_null]
      - name: category
        data_tests:
          - accepted_values:
              arguments:
                values: ["Grocery", "Home", "Electronics", "Apparel", "Beauty"]
      - name: base_price
        data_tests: [not_null]
```

```
Done. PASS=19 WARN=0 ERROR=0 ...
```

</details>

**Part 3 — reconcile.** Use `dbt show --inline` to count completed orders in `stg_orders`. Does it match Day 2's clean file (17,839)?

<details>
<summary>Solution</summary>

```bash
dbt show --profiles-dir . --inline "select count(*) as n, round(sum(revenue)) as revenue from {{ ref('stg_orders') }}"
```

```
|     n |   revenue |
| ----- | --------- |
| 17839 | 2,810,574 |
```

The same 17,839 orders and $2,810,574 as Day 2, reached by a completely different route (API → dlt → dbt instead of CSV → Polars).

</details>

---

## 6.6 Save your work with Git (15 min)

```bash
cd ..                 # back to the repository root
git add .
git status            # transform/target/ and transform/logs/ must NOT appear
git commit -m "Day 6: dbt project with tested staging models"
git push
```

### Day 6 checklist

- [ ] I can explain what dbt adds to plain SQL files
- [ ] I can configure a dbt project and profile for DuckDB and run `dbt debug`
- [ ] I can declare sources and write staging models that rename, cast, clean and filter
- [ ] I can add generic tests and read `dbt build` output, including failures and skips
- [ ] I have reconciled the dbt staging layer against Day 2's clean data

---

# DAY 7 — dbt Marts: The Panel as a Model

| Time | Block |
|---|---|
| 0:00 – 0:40 | 7.1 Layers, `ref()` and dbt's own DAG |
| 0:40 – 1:30 | 7.2 Variables and the calendar spine model |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 7.3 The customer × week panel as a dbt model |
| 2:45 – 3:30 | 7.4 Custom tests: generic and singular |
| 3:30 – 3:45 | Break |
| 3:45 – 4:40 | 7.5 Day 7 mini-project: prove the two pipelines agree |
| 4:40 – 5:00 | 7.6 Documentation, a full rebuild, and Git |

---

## 7.1 Layers, `ref()` and dbt's own DAG (40 min)

**Concept.** Models refer to each other with `{{ ref('model_name') }}`. From these references dbt draws a graph of dependencies and runs models in the right order. dbt calls this graph a **DAG**, a directed acyclic graph: the same mathematical object as the causal DAGs of Module 2. Here arrows mean "is built from"; there they will mean "causes". In both, a cycle would be nonsense.

Today's graph:

```
raw.customers ─────────► stg_customers ──────────────┐
raw.orders ────────────► stg_orders ──► int_customer_weekly_orders ──┤
raw.campaign_recipients ► stg_campaign_recipients ───┼──► panel_customer_week
                          int_weeks (calendar) ──────┘
```

dbt's `--select` syntax follows the graph: `+panel_customer_week` means "the panel and everything upstream", `stg_orders+` means "stg_orders and everything downstream".

```bash
dbt ls --profiles-dir . --select +panel_customer_week --resource-type model --resource-type source
```

```
northmart.intermediate.int_customer_weekly_orders
northmart.intermediate.int_weeks
northmart.marts.panel_customer_week
northmart.staging.stg_campaign_recipients
northmart.staging.stg_customers
northmart.staging.stg_orders
source:northmart.raw.campaign_recipients
source:northmart.raw.customers
source:northmart.raw.orders
```

`stg_products` isn't listed: the panel doesn't use it. Without `--resource-type`, the list also includes every test attached to these models.

---

## 7.2 Variables and the calendar spine model (50 min)

**Concept.** The email date, the first week and the campaign ID are **study design choices**. Hard-coding them inside SQL hides them. dbt **variables** (`vars:` in `dbt_project.yml`) put them in one visible place, and `{{ var("campaign_week") }}` reads them. Changing the analysis window becomes a one-line, reviewable change.

**Example 1 — the calendar.** `transform/models/intermediate/int_weeks.sql`:

```sql
-- The calendar spine: one row per Monday, numbered 1..16.
select
    cast(week_start as date)                          as week_start,
    row_number() over (order by week_start)           as week_no
from generate_series(
    date '{{ var("first_week") }}',
    date '{{ var("last_week") }}',
    interval 7 day
) as g(week_start)
```

**Example 2 — weekly outcomes where orders exist.** `transform/models/intermediate/int_customer_weekly_orders.sql`:

```sql
-- Outcomes per customer-week, only where at least one order exists.
select
    customer_id,
    cast(date_trunc('week', order_date) as date)      as week_start,
    count(distinct order_id)                          as n_orders,
    sum(revenue)                                      as revenue
from {{ ref('stg_orders') }}
group by all
```

`group by all` groups by every column that isn't aggregated: a DuckDB convenience that saves retyping.

```bash
dbt build --profiles-dir . --select intermediate
dbt show --profiles-dir . --select int_weeks --limit 16
```

Variables can be overridden at run time without editing files, for example to rebuild with a shorter window:

```bash
dbt show --profiles-dir . --select int_weeks --vars '{"last_week": "2026-02-23"}'
```

---

## 7.3 The customer × week panel as a dbt model (60 min)

**Concept.** The panel is Day 3's SQL, reorganised: each `WITH` step became either its own model (`int_weeks`, `int_customer_weekly_orders`) or a short CTE, every table name became a `ref()`, and every design choice became a `var()`. dbt writes the `CREATE TABLE`; the file contains only the `SELECT`.

Create `transform/models/marts/panel_customer_week.sql`:

```sql
-- The Module 1 deliverable: one row per customer per week, ready for causal analysis.
-- Grain: customer_id x week_start. Treatment: the {{ var("campaign_id") }} email.

with spine as (
    select c.customer_id, w.week_start, w.week_no
    from {{ ref('stg_customers') }} as c
    cross join {{ ref('int_weeks') }} as w
),

filled as (
    select
        s.*,
        coalesce(o.n_orders, 0)  as n_orders,
        coalesce(o.revenue, 0)   as revenue
    from spine as s
    left join {{ ref('int_customer_weekly_orders') }} as o using (customer_id, week_start)
),

recipients as (
    select customer_id, sent_date
    from {{ ref('stg_campaign_recipients') }}
    where campaign_id = '{{ var("campaign_id") }}'
)

select
    f.customer_id,
    f.week_start,
    f.week_no,
    f.week_no - {{ var("campaign_week") }}                         as event_week,
    cast(r.customer_id is not null as integer)                     as treated,
    cast(f.week_no >= {{ var("campaign_week") }} as integer)       as post,
    cast(r.customer_id is not null
         and f.week_start >= r.sent_date as integer)               as treated_post,
    f.n_orders,
    round(f.revenue, 2)                                            as revenue,
    c.region,
    c.age,
    cast(c.is_loyalty as integer)                                  as is_loyalty,
    sum(f.n_orders) filter (where f.week_no < {{ var("campaign_week") }})
        over (partition by f.customer_id)                          as pre_orders,
    round(sum(f.revenue) filter (where f.week_no < {{ var("campaign_week") }})
        over (partition by f.customer_id), 2)                      as pre_revenue
from filled as f
join {{ ref('stg_customers') }} as c using (customer_id)
left join recipients as r using (customer_id)
```

Note that `pre_orders` uses `week_no < campaign_week`, so if the campaign date variable changes, the pre-period follows it automatically. On Day 3 the "8" was typed by hand.

Because `marts` is configured as `+materialized: table`, the panel is stored, not recomputed on every query: important for a table read by every estimation in Modules 2–8.

---

## 7.4 Custom tests: generic and singular (45 min)

**Concept.** The four built-in tests can't express everything. dbt offers two ways to write your own:

- A **generic test** is a reusable macro with parameters, applied in YAML like `unique`.
- A **singular test** is a one-off `.sql` file in `tests/` that returns the bad rows.

**Example 1 — a generic test.** dbt's `unique` checks one column; a panel's key is two columns. Create `transform/tests/generic/unique_combination.sql`:

```sql
-- A reusable test: no two rows share the same combination of columns.
{% test unique_combination(model, columns) %}
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}
```

**Example 2 — the panel's YAML.** `transform/models/marts/_marts.yml` applies it, plus column tests:

```yaml
version: 2

models:
  - name: panel_customer_week
    description: "One row per customer per week. Outcomes are 0 (not NULL) in weeks without orders."
    data_tests:
      - unique_combination:
          arguments:
            columns: ["customer_id", "week_start"]
    columns:
      - name: customer_id
        data_tests: [not_null]
      - name: treated
        data_tests:
          - accepted_values:
              arguments:
                values: [0, 1]
                quote: false
      - name: post
        data_tests:
          - accepted_values:
              arguments:
                values: [0, 1]
                quote: false
      - name: n_orders
        data_tests: [not_null]
      - name: revenue
        data_tests: [not_null]
      - name: pre_orders
        description: "Orders in weeks 1-8. Fixed per customer: a pre-treatment covariate."
        data_tests: [not_null]
```

**Example 3 — singular tests.** These are Day 3's Python checks, now part of the build. Create three files in `transform/tests/`:

```sql
-- Every customer must appear in every week. Returns rows (= failure) if the count is off.
select count(*) as panel_rows
from {{ ref('panel_customer_week') }}
having count(*) != (select count(*) from {{ ref('stg_customers') }})
                  * (select count(*) from {{ ref('int_weeks') }})
```

```sql
-- Panel revenue must equal order revenue (to the cent, allowing for rounding).
select
    (select sum(revenue) from {{ ref('panel_customer_week') }}) as panel_revenue,
    (select sum(revenue) from {{ ref('stg_orders') }})          as order_revenue
where abs(panel_revenue - order_revenue) >= 1
```

```sql
-- Causal validity: nobody can be "treated" before the email was sent.
select customer_id, week_start
from {{ ref('panel_customer_week') }}
where treated_post = 1 and post = 0
```

Build everything:

```bash
dbt build --profiles-dir .
```

```
Done. PASS=30 WARN=0 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=30
```

7 models and 23 tests. If a staging test fails, dbt **skips** the panel rather than building it on bad data.

**Try it.** Day 3 checked that `pre_orders` is constant within each customer. Write that as a singular test, `tests/assert_pre_orders_fixed.sql`.

<details>
<summary>Solution</summary>

```sql
-- tests/assert_pre_orders_fixed.sql
-- A pre-treatment covariate cannot change over time within a customer.
select customer_id
from {{ ref('panel_customer_week') }}
group by customer_id
having count(distinct pre_orders) > 1
```

`dbt build` now reports `PASS=31`.

</details>

---

## 7.5 Day 7 mini-project: prove the two pipelines agree (55 min)

**Goal.** We now have two independent routes to the panel:

| Route | Steps |
|---|---|
| Days 1–3 | CSV → Polars cleaning → hand-written DuckDB SQL → Parquet |
| Days 4–7 | API → dlt (incremental, merged) → dbt staging → intermediate → marts |

If both are right, they must produce the same 64,000 rows to the cent. Write `day07/compare_panels.py` to prove it.

> Day 3's `build_panel.py` and dbt both write a table called `marts.panel_customer_week`, and dbt has just replaced Day 3's version. That's why we compare against the Parquet file Day 3 exported.

```python
# day07/compare_panels.py
"""Mini-project check: the dbt panel must match the Day 3 panel exactly."""
import duckdb
import polars as pl

KEY = ["customer_id", "week_start"]

con = duckdb.connect("data/northmart.duckdb", read_only=True)
dbt_panel = con.sql("SELECT * FROM marts.panel_customer_week").pl().sort(KEY)
con.close()
day3_panel = pl.read_parquet("data/marts/panel_customer_week.parquet").sort(KEY)

print("rows    dbt:", dbt_panel.height, "| Day 3:", day3_panel.height)
print("columns match:", dbt_panel.columns == day3_panel.columns)

# Compare column by column, ignoring integer width (INT32 vs INT64)
problems = []
for col in day3_panel.columns:
    a, b = dbt_panel[col], day3_panel[col]
    if a.dtype.is_numeric():
        same = ((a.cast(pl.Float64) - b.cast(pl.Float64)).abs() < 1e-6).all()
    else:
        same = (a == b).all()
    if not same:
        problems.append(col)

print("columns that differ:", problems or "none")
assert not problems, "The dbt panel does not reproduce Day 3. Investigate before moving on."
print("The dbt model reproduces the Day 3 panel exactly.")
```

```bash
cd ..
python day07/compare_panels.py
```

```
rows    dbt: 64000 | Day 3: 64000
columns match: True
columns that differ: none
The dbt model reproduces the Day 3 panel exactly.
```

This is the strongest test in the module. Two pipelines written in different tools, from differently shaped data, agree on every cell. From now on, the dbt version is the official panel.

**Extension — a balance table.** Add a mart `models/marts/campaign_balance.sql` that compares treated and control customers **before** the email: number of customers, average pre-period orders and revenue, loyalty share and age. (Hint: filter to `week_no = 1` to get one row per customer.)

<details>
<summary>Solution</summary>

```sql
-- models/marts/campaign_balance.sql
-- Were emailed and non-emailed customers comparable BEFORE the email?
select
    treated,
    count(*)                         as customers,
    round(avg(pre_orders), 2)        as pre_orders,
    round(avg(pre_revenue), 0)       as pre_revenue,
    round(avg(is_loyalty), 3)        as loyalty_share,
    round(avg(age), 1)               as age
from {{ ref('panel_customer_week') }}
where week_no = 1                    -- one row per customer
group by treated
order by treated
```

```bash
dbt build --profiles-dir . --select campaign_balance
dbt show --profiles-dir . --select campaign_balance
```

```
| treated | customers | pre_orders | pre_revenue | loyalty_share |  age |
| ------- | --------- | ---------- | ----------- | ------------- | ---- |
|       0 |      2721 |       1.85 |         288 |         0.208 | 46.7 |
|       1 |      1279 |       3.23 |         506 |         0.491 | 47.3 |
```

Emailed customers spent 76% more before the email and were more than twice as likely to be loyalty members; age is balanced. This "Table 1" is the first thing a reviewer asks for in any causal study, and it now rebuilds automatically with the data. Module 4 shows what to do about the imbalance.

</details>

---

## 7.6 Documentation, a full rebuild, and Git (20 min)

**Documentation.** dbt builds a website from your YAML descriptions and the graph:

```bash
cd transform
dbt docs generate --profiles-dir .
dbt docs serve --profiles-dir . --port 8080
```

Open port 8080 when Codespaces offers it. Click **panel_customer_week**, then the graph icon (bottom right) to see the lineage from `raw` to the panel. Stop the server with `Ctrl+C`.

**Full rebuild from scratch.** With the API running in the other terminal:

```bash
cd ..
rm -rf data/northmart.duckdb .dlt_pipelines
python api/clock.py 2026-04-26
python day05/northmart_pipeline.py
cd transform && dbt build --profiles-dir . && cd ..
python day07/compare_panels.py
```

Four commands in the right order still need a human who remembers them. Day 9 hands that job to an orchestrator.

```bash
git add .
git commit -m "Day 7: dbt marts, custom tests, panel reproduces Day 3"
git push
```

> **Rebuilding the Day 3 comparison file.** `compare_panels.py` needs `data/marts/panel_customer_week.parquet`. If you deleted it, rerun Day 2's `clean.py` and Day 3's `build_warehouse.py`, `calendar_spine.py` and `build_panel.py` first.

**Coming next (Days 8–10).** Day 8 turns the causal checks into **data contracts** that fail loudly on leakage and treatment-timing errors. Day 9 orchestrates the whole pipeline with **Prefect**. Day 10 is the Module 1 capstone: the **Causal Data Mart**.

### Day 7 checklist

- [ ] I can explain how `ref()` builds dbt's dependency graph and use `+model` selection
- [ ] I keep study design choices (dates, campaign) in dbt variables
- [ ] I can build the customer × week panel as a layered dbt project
- [ ] I can write generic and singular tests, including causal-validity checks
- [ ] I have proved that the dbt panel reproduces the Day 3 panel exactly
