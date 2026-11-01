# Module 1 · Causal-Ready Data Engineering
## Days 8–10: Data Contracts, Orchestration and the Causal Data Mart

**Course:** Applied Causal Inference & Decision Science
**Audience:** Economists and analysts who completed Days 1–7
**Environment:** GitHub Codespaces (VS Code in the browser)
**Time per day:** ~5 hours (including breaks and exercises)
**Goal by end of Day 10:** Students run a scheduled, self-checking pipeline that publishes a documented Causal Data Mart, refuses to publish when the data would mislead a causal analysis, and has been replayed week by week through the whole study period.

---

### Where we are

After Day 7 you have:

| Piece | File |
|---|---|
| The NorthMart API and its clock | `api/` |
| An incremental, deduplicating dlt loader | `day05/northmart_pipeline.py` |
| A dbt project: staging → intermediate → marts, 30 checks | `transform/` |
| Proof that the dbt panel equals the Day 3 panel | `day07/compare_panels.py` |

Three gaps remain:

1. **Our tests check tidiness, not causal validity.** Nothing stops a feature from quietly using post-email data, or a treatment date from being wrong.
2. **A human still runs four commands in order.** Forget one and the panel is stale.
3. **Nobody outside the team knows what the data means**, or whether today's version is safe to use.

Day 8 writes **data contracts** for causal validity. Day 9 hands the pipeline to an **orchestrator** that publishes only when every contract holds. Day 10 is the capstone: replay all 16 weeks, then accept and document the **Causal Data Mart**.

### New and changed files (Days 8–10)

```
causal-decision-lab/
├── day08/break_things.py                  # fault injection drill
├── day09/hello_flow.py, retries_demo.py
├── day10/replay_weeks.py, acceptance.py, data_card.py
├── docs/
│   ├── CONTRACTS.md                       # the contract register
│   └── DATA_CARD.md                       # generated on Day 10
├── pipeline/
│   ├── northmart_flow.py                  # the whole pipeline as one Prefect flow
│   └── serve.py                           # the daily schedule
└── transform/
    ├── dbt_project.yml                    # + campaign_date variable
    ├── macros/customer_features.sql       # new
    ├── models/marts/customer_features.sql # new
    ├── models/marts/_marts.yml            # now with enforced contracts
    └── tests/assert_*.sql, warn_*.sql     # five new contract tests
```

---

# DAY 8 — Data Contracts: Leakage and Treatment Timing

| Time | Block |
|---|---|
| 0:00 – 0:40 | 8.1 From tests to contracts |
| 0:40 – 1:30 | 8.2 Schema contracts: names and types that cannot drift |
| 1:30 – 1:45 | Break |
| 1:45 – 2:50 | 8.3 The leakage contract: a time-travel test |
| 2:50 – 3:30 | 8.4 Treatment, completeness and overlap contracts |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 8.5 Day 8 mini-project: the fire drill |
| 4:45 – 5:00 | 8.6 The contract register and Git |

---

## 8.1 From tests to contracts (40 min)

**Concept.** A **data contract** is a promise about data that is written down, owned by someone, and enforced automatically. Tests are how we enforce them. The difference is intent: a contract states *why* the rule matters and *what happens* when it breaks.

For causal work, the most important contracts are not about tidiness. They protect the assumptions every estimate relies on:

| Assumption | What breaks it in data | Contract |
|---|---|---|
| Covariates are pre-treatment | a "control" computed with post-email orders | **leakage** test |
| Treatment timing is correct | backdated or late send dates | **timing** test |
| Units stay in their group | a customer switching between treated and control | **fixed treatment** test |
| Outcomes are complete | a missed load that looks like a demand collapse | **completeness** test |
| Comparable units exist (overlap) | a region with no untreated customers | **overlap** warning |

Each contract has a **severity**:

- **error:** stop. Nothing downstream is rebuilt or published.
- **warn:** report, keep going. A human decides.

Overlap problems are warnings because the right response (pool regions, drop a group, change the estimand) is an analytical choice, not a data bug.

---

## 8.2 Schema contracts: names and types that cannot drift (50 min)

**Concept.** Estimation code in Modules 2–8 reads `panel_customer_week` by column name and type. If someone renames `n_orders`, or a type changes from integer to text, every later analysis breaks or, worse, silently casts. A dbt **model contract** fixes the columns' names and types: the build fails if the SQL produces anything else.

**A surprise worth seeing first.** Ask DuckDB what type `pre_orders` has today:

```bash
cd transform
dbt show --profiles-dir . --inline "select typeof(pre_orders) as t from {{ ref('panel_customer_week') }} limit 1"
```

```
| t       |
| ------- |
| HUGEINT |
```

`SUM` over integers returns a 128-bit integer in DuckDB. Harmless in SQL, but Polars and pandas handle it differently, and in Module 6 machine-learning libraries may refuse it. A contract makes us choose types deliberately.

**Step 1 — make the type explicit.** In `models/marts/panel_customer_week.sql`, wrap `pre_orders` in a cast:

```sql
    cast(sum(f.n_orders) filter (where f.week_no < {{ var("campaign_week") }})
        over (partition by f.customer_id) as bigint)               as pre_orders,
```

**Step 2 — add a variable** for the email date in `dbt_project.yml` (under `vars:`), used by today's contracts:

```yaml
  campaign_date: "2026-03-02"   # features may only use data from BEFORE this date
```

**Step 3 — enforce the contract.** Replace `models/marts/_marts.yml` with this version. Every column now has a `data_type`, `contract: enforced: true` switches checking on, and `constraints` are written into the table definition itself. (The second model, `customer_features`, is created in 8.3.)

```yaml
version: 2

models:
  - name: panel_customer_week
    description: "One row per customer per week. Outcomes are 0 (not NULL) in weeks without orders."
    config:
      contract:
        enforced: true          # the build fails if a column's name or type changes
    data_tests:
      - unique_combination:
          arguments:
            columns: ["customer_id", "week_start"]
    columns:
      - name: customer_id
        data_type: varchar
        constraints: [{type: not_null}]
      - name: week_start
        data_type: date
        constraints: [{type: not_null}]
      - name: week_no
        data_type: bigint
      - name: event_week
        data_type: bigint
        description: "Weeks relative to the email: 0 = the email week, negative = before."
      - name: treated
        data_type: integer
        description: "1 if the customer ever received the SPRING26 email."
        data_tests:
          - accepted_values:
              arguments:
                values: [0, 1]
                quote: false
      - name: post
        data_type: integer
        data_tests:
          - accepted_values:
              arguments:
                values: [0, 1]
                quote: false
      - name: treated_post
        data_type: integer
      - name: n_orders
        data_type: bigint
        constraints: [{type: not_null}]
      - name: revenue
        data_type: double
        constraints: [{type: not_null}]
      - name: region
        data_type: varchar
      - name: age
        data_type: integer
      - name: is_loyalty
        data_type: integer
      - name: pre_orders
        data_type: bigint
        description: "Orders in weeks 1-8. Fixed per customer: a pre-treatment covariate."
        constraints: [{type: not_null}]
      - name: pre_revenue
        data_type: double

  - name: customer_features
    description: "One row per customer: treatment plus covariates computed ONLY from data before the email."
    config:
      contract:
        enforced: true
    columns:
      - name: customer_id
        data_type: varchar
        constraints: [{type: not_null}, {type: unique}]
      - name: treated
        data_type: integer
      - name: region
        data_type: varchar
      - name: age
        data_type: integer
      - name: is_loyalty
        data_type: integer
      - name: tenure_days
        data_type: bigint
        description: "Days from signup to the email date."
      - name: pre_orders
        data_type: bigint
      - name: pre_revenue
        data_type: double
      - name: pre_active_weeks
        data_type: bigint
      - name: pre_discount_share
        data_type: double
        description: "Share of pre-period orders that carried a discount (NULL if no orders)."
      - name: days_since_last_order
        data_type: bigint
        description: "Days from the last pre-period order to the email date (NULL if no orders)."
```

**See it fail.** Temporarily remove the `cast(... as bigint)` you just added and run:

```bash
dbt run --profiles-dir . --select panel_customer_week
```

```
ERROR creating sql table model marts.panel_customer_week
  This model has an enforced contract that failed.
  | column_name | definition_type | contract_type | mismatch_reason    |
  | ----------- | --------------- | ------------- | ------------------ |
  | pre_orders  | HUGEINT         | BIGINT        | data type mismatch |
```

The model is not built, so the previous version of the table stays in place. Put the cast back.

---

## 8.3 The leakage contract: a time-travel test (65 min)

**Concept.** **Leakage** means a variable that is supposed to describe units *before* treatment secretly contains information from *after* it. It's the data version of a bad control (Day 1, 1.4). Leakage is hard to spot by reading SQL because the mistake is usually one missing `WHERE`.

The **time-travel test** catches it mechanically:

1. Build the features from the full data, as the pipeline normally does.
2. Build them again from orders that physically **end** before the email date.
3. Compare. If any value differs, some feature looked into the future.

To run the same SQL twice on different inputs, we write it once as a **macro** with the orders table as an argument.

**Step 1 — the macro.** Create `transform/macros/customer_features.sql`:

```sql
{#
  Customer features "as of" the email date.
  The orders relation is an argument so a test can call this macro on
  orders that physically stop before the email date, and compare.
#}
{% macro customer_features_from(orders_relation) %}
with pre_orders as (
    select *
    from {{ orders_relation }}
    where order_date < date '{{ var("campaign_date") }}'
),
per_customer as (
    select
        customer_id,
        count(distinct order_id)                                   as pre_orders,
        sum(revenue)                                               as pre_revenue,
        count(distinct date_trunc('week', order_date))             as pre_active_weeks,
        avg(case when discount_pct > 0 then 1.0 else 0.0 end)      as pre_discount_share,
        max(order_date)                                            as last_order
    from pre_orders
    group by customer_id
)
select
    c.customer_id,
    cast(r.customer_id is not null as integer)                                     as treated,
    c.region,
    c.age,
    cast(c.is_loyalty as integer)                                                  as is_loyalty,
    cast(date_diff('day', c.signup_date, date '{{ var("campaign_date") }}') as bigint) as tenure_days,
    cast(coalesce(p.pre_orders, 0) as bigint)                                      as pre_orders,
    round(coalesce(p.pre_revenue, 0), 2)                                           as pre_revenue,
    cast(coalesce(p.pre_active_weeks, 0) as bigint)                                as pre_active_weeks,
    round(p.pre_discount_share, 4)                                                 as pre_discount_share,
    cast(date_diff('day', p.last_order, date '{{ var("campaign_date") }}') as bigint) as days_since_last_order
from {{ ref('stg_customers') }} as c
left join per_customer as p using (customer_id)
left join {{ ref('stg_campaign_recipients') }} as r
    on r.customer_id = c.customer_id and r.campaign_id = '{{ var("campaign_id") }}'
{% endmacro %}
```

**Step 2 — the model.** `transform/models/marts/customer_features.sql` is one line:

```sql
-- One row per customer, for the estimation modules (matching, weighting, causal ML).
{{ customer_features_from(ref('stg_orders')) }}
```

**Step 3 — the contract.** Create `transform/tests/assert_features_do_not_leak.sql`:

```sql
-- LEAKAGE CONTRACT ("time travel" test).
-- Rebuild the features from orders that physically end before the email date.
-- If any feature changes, it was using post-treatment information.
{% set truncated_orders %}
    (select * from {{ ref('stg_orders') }} where order_date < date '{{ var("campaign_date") }}')
{% endset %}

with as_built as (
    select * from {{ ref('customer_features') }}
),
time_travel as (
    {{ customer_features_from(truncated_orders) }}
),
differences as (
    (select 'only in as_built' as side, * from as_built
     except select 'only in as_built', * from time_travel)
    union all
    (select 'only in time_travel' as side, * from time_travel
     except select 'only in time_travel', * from as_built)
)
select * from differences
```

`EXCEPT` returns rows in the first set that are missing from the second. Doing it both ways and stacking the results lists every difference.

```bash
dbt build --profiles-dir . --select customer_features+
```

The test passes: every feature uses only `pre_orders`.

**See it catch a leak.** In the macro, change `from pre_orders` (inside `per_customer`) to `from {{ orders_relation }}`, as if a colleague forgot the date filter. Then:

```bash
dbt build --profiles-dir . --select customer_features+
```

```
FAIL 6246 assert_features_do_not_leak ................................... [FAIL 6246 in 0.14s]
```

Look at the model with `dbt show --select customer_features`: nothing looks wrong. Every number is plausible. Only the time-travel test reveals that `pre_orders`, `pre_revenue` and the rest now count post-email orders, which would make every estimate in Module 4 absorb part of the email's effect. Undo the change and rebuild.

---

## 8.4 Treatment, completeness and overlap contracts (40 min)

Create four more files in `transform/tests/`. Each header comment states the contract and its reason.

**Timing:** all recipients were emailed on the official date.

```sql
-- TREATMENT-TIMING CONTRACT: every recipient of the campaign got it on the official date.
-- A backdated or late send would silently mix treatment timing.
select customer_id, sent_date
from {{ ref('stg_campaign_recipients') }}
where campaign_id = '{{ var("campaign_id") }}'
  and sent_date != date '{{ var("campaign_date") }}'
```

**Fixed treatment:** a customer is treated in every week or in none.

```sql
-- TREATMENT CONTRACT: the treatment group is a property of the customer, not of the week.
select customer_id
from {{ ref('panel_customer_week') }}
group by customer_id
having count(distinct treated) > 1
```

**Completeness:** every week has orders.

```sql
-- COMPLETENESS CONTRACT: NorthMart sells every week. A week with zero orders
-- almost always means a load was missed (e.g. a broken incremental cursor).
select week_start, sum(n_orders) as orders
from {{ ref('panel_customer_week') }}
group by week_start
having sum(n_orders) = 0
```

**Overlap (warning):** every region has at least 30 emailed and 30 non-emailed customers.

```sql
-- OVERLAP (POSITIVITY) WARNING: every region needs both emailed and non-emailed customers,
-- otherwise there is nobody to compare with. A warning, not an error: the analyst decides.
{{ config(severity='warn') }}

select
    region,
    sum(treated)            as treated_customers,
    sum(1 - treated)        as control_customers
from {{ ref('customer_features') }}
group by region
having sum(treated) < 30 or sum(1 - treated) < 30
```

```bash
dbt build --profiles-dir .
```

```
WARN 1 warn_overlap_by_region ......................................... [WARN 1 in 0.08s]
Done. PASS=31 WARN=1 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32
```

**Read the warning.** Which region fails? Run the test's query yourself:

```bash
dbt show --profiles-dir . --inline "select region, sum(treated) t, sum(1-treated) c from {{ ref('customer_features') }} group by 1 order by 1"
```

The five provinces are fine. The `UNKNOWN` group (74 customers with no recorded region) has only 25 emailed customers. That isn't a bug to fix: it's a fact any analysis by region must handle, and Day 10's data card records it.

**Try it.** Write a contract that no customer signed up **after** the email date (they could not have been targeted by it). Use `customer_features.tenure_days`.

<details>
<summary>Solution</summary>

```sql
-- tests/assert_signup_before_email.sql
-- A customer who signed up after the email could not have been targeted by it.
select customer_id, tenure_days
from {{ ref('customer_features') }}
where tenure_days < 0
```

```
Done. PASS=1 WARN=0 ERROR=0 ...
```

It passes for NorthMart (all signups are before 2026). In real data, accounts created *because of* a campaign are common, and they must be excluded or analysed separately.

</details>

---

## 8.5 Day 8 mini-project: the fire drill (60 min)

**Goal.** Contracts you've never seen fail are contracts you can't trust. Damage the raw data on purpose, four different ways, and record which contract catches each fault.

Create `day08/break_things.py`:

```python
# day08/break_things.py
"""
Fire drill: damage the raw data on purpose, then check which contract catches it.

    python day08/break_things.py backdate     # 50 emails recorded two weeks early
    python day08/break_things.py drop_week    # a missed load: week 5 has no orders
    python day08/break_things.py duplicate    # 100 orders loaded twice
    python day08/break_things.py orphan       # 5 orders from a customer who doesn't exist
    python day08/break_things.py restore      # reload clean data from the API

After each fault, run `dbt build --profiles-dir .` in transform/ and read the results.
The NorthMart API must be running for `restore`.
"""
import shutil
import subprocess
import sys

import duckdb

DB = "data/northmart.duckdb"

FAULTS = {
    "backdate": """
        UPDATE raw.campaign_recipients SET sent_date = '2026-02-16'
        WHERE customer_id IN (SELECT customer_id FROM raw.campaign_recipients ORDER BY customer_id LIMIT 50)
    """,
    "drop_week": """
        DELETE FROM raw.orders WHERE order_date BETWEEN '2026-02-02' AND '2026-02-08'
    """,
    "duplicate": """
        INSERT INTO raw.orders
        SELECT * REPLACE ('9999999999.0' AS _dlt_load_id)
        FROM raw.orders ORDER BY order_id LIMIT 100
    """,
    "orphan": """
        UPDATE raw.orders SET customer_id = 'C99999'
        WHERE order_id IN (SELECT order_id FROM raw.orders WHERE status = 'completed' ORDER BY order_id LIMIT 5)
    """,
}


def restore() -> None:
    """Throw away the raw data and the pipeline's memory, then reload everything."""
    con = duckdb.connect(DB)
    con.sql("DROP SCHEMA IF EXISTS raw CASCADE")
    con.close()
    shutil.rmtree(".dlt_pipelines/northmart", ignore_errors=True)
    subprocess.run([sys.executable, "api/clock.py", "2026-04-26"], check=True)
    subprocess.run([sys.executable, "day05/northmart_pipeline.py"], check=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in [*FAULTS, "restore"]:
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    if name == "restore":
        restore()
        return
    con = duckdb.connect(DB)
    con.sql(FAULTS[name])
    con.close()
    print(f"Fault '{name}' injected into the raw schema. Now run dbt build.")


if __name__ == "__main__":
    main()
```

With the NorthMart API running, run each fault, rebuild, read the result, and restore before the next one:

```bash
python day08/break_things.py backdate
cd transform && dbt build --profiles-dir . ; cd ..
python day08/break_things.py restore
```

Repeat with `drop_week`, `duplicate` and `orphan`. The results:

```
backdate:   FAIL 50 assert_single_send_date        ... Done. PASS=19 WARN=0 ERROR=1 SKIP=12
drop_week:  FAIL 1  assert_no_empty_weeks          ... Done. PASS=30 WARN=1 ERROR=1 SKIP=0
duplicate:  (all contracts pass)                   ... Done. PASS=31 WARN=1 ERROR=0 SKIP=0
orphan:     FAIL 5  relationships_stg_orders_customer_id... Done. PASS=19 WARN=0 ERROR=1 SKIP=12
```

What to notice:

- **`backdate` and `orphan` fail in staging**, so dbt **skips** the 12 downstream nodes: the panel is not rebuilt on bad data.
- **`drop_week` passes every staging test.** The missing orders leave no orphan, no duplicate, no wrong type; revenue still reconciles (both sides lost the same week). Only the completeness contract notices. Internal consistency can't detect missing data; you need a rule about what *should* exist.
- **`duplicate` causes no damage.** Day 6's `row_number()` safety net in `stg_orders` absorbs it. A passing build is the correct result here.

> **dbt builds before it tests.** In the `drop_week` case the panel model itself ran, *then* its test failed. The table in DuckDB now holds the damaged version. That is why Day 9 adds a separate **publish** step that runs only after every contract passes.

---

## 8.6 The contract register and Git (15 min)

Contracts belong in a document people can read without opening SQL. Create `docs/CONTRACTS.md`:

```markdown
# NorthMart data contracts

Each contract is a promise the data must keep before anyone estimates a causal effect from it.
"Error" stops the build and nothing downstream is rebuilt; "warn" reports and lets the analyst decide.

| ID | Contract | Where | Severity | Causal reason |
|---|---|---|---|---|
| C1 | Panel columns keep their names and types | `_marts.yml` (`contract: enforced`) | error | Estimation code downstream breaks or silently mis-casts |
| C2 | One row per customer per week | `unique_combination`, `assert_panel_is_complete` | error | Missing or doubled unit-weeks bias averages |
| C3 | Outcomes are 0, never NULL | `not_null` constraints | error | NULL outcomes silently drop units |
| C4 | Panel revenue equals order revenue | `assert_revenue_reconciles` | error | Outcomes lost between layers |
| C5 | Every order belongs to a known customer | `relationships` test on `stg_orders` | error | Orders can't be assigned to treatment or control |
| C6 | All emails sent on the official date | `assert_single_send_date` | error | Mixed timing breaks the pre/post split |
| C7 | No treatment before the email | `assert_no_treatment_before_email` | error | "Effects" before treatment are impossible |
| C8 | Treatment group fixed per customer | `assert_treatment_is_fixed` | error | Units switching groups break the comparison |
| C9 | Features use only pre-email data | `assert_features_do_not_leak` (time travel) | error | Post-treatment controls absorb the effect |
| C10 | Every week has orders | `assert_no_empty_weeks` | error | A missed load looks like a collapse in demand |
| C11 | Each region has 30+ treated and 30+ control customers | `warn_overlap_by_region` | warn | No overlap means no valid comparison in that group |

## Known warnings

- **C11, region UNKNOWN:** only 25 emailed customers have no recorded region. Analyses by region should
  pool or drop this group and say so.

## Fire-drill results (day08/break_things.py)

| Fault injected | Caught by | Outcome |
|---|---|---|
| 50 emails backdated two weeks | C6 | Build stopped; panel not rebuilt |
| Week 5 orders missing | C10 | Build stopped at the panel test |
| 100 orders loaded twice | (staging deduplication) | No damage: all contracts pass |
| 5 orders from an unknown customer | C5 | Build stopped; panel not rebuilt |
```

```bash
git add .
git commit -m "Day 8: causal data contracts, time-travel leakage test, fire drill"
git push
```

### Day 8 checklist

- [ ] I can explain the difference between a test and a contract, and between error and warn
- [ ] I can enforce a dbt model contract and read a type-mismatch failure
- [ ] I can detect feature leakage with a time-travel test built on a macro
- [ ] I can write timing, fixed-treatment, completeness and overlap contracts
- [ ] I have run the fire drill and know which contract catches which fault

---

# DAY 9 — Orchestration with Prefect

| Time | Block |
|---|---|
| 0:00 – 0:40 | 9.1 Why orchestration? Flows and tasks |
| 0:40 – 1:20 | 9.2 Retries, logging and failure |
| 1:20 – 1:35 | Break |
| 1:35 – 3:00 | 9.3 The NorthMart flow: load, build, gate, publish |
| 3:00 – 3:15 | Break |
| 3:15 – 4:00 | 9.4 Schedules and the Prefect dashboard |
| 4:00 – 4:45 | 9.5 Day 9 mini-project: a failed run that protects analysts |
| 4:45 – 5:00 | 9.6 Save your work with Git |

---

## 9.1 Why orchestration? Flows and tasks (40 min)

**Concept.** An **orchestrator** runs the pipeline's steps in order, on a schedule, retries what fails for temporary reasons, stops what fails for real reasons, and keeps a record of every run. **Prefect** does this with two decorators:

- `@task`: one step (call the API, run dbt…). Prefect tracks its state: running, completed, failed, retrying.
- `@flow`: a function that calls tasks. Running a flow creates a **flow run** with its own log.

Your code stays ordinary Python; the decorators add the bookkeeping.

**Example 1.** Make sure the dbt panel exists (`dbt build` on Day 8), then create `day09/hello_flow.py`:

```python
# day09/hello_flow.py
import duckdb
from prefect import flow, task


@task
def count_orders(week_start: str) -> int:
    con = duckdb.connect("data/northmart.duckdb", read_only=True)
    n = con.sql(f"""
        SELECT COALESCE(SUM(n_orders), 0) FROM marts.panel_customer_week
        WHERE week_start = DATE '{week_start}'
    """).fetchall()[0][0]
    con.close()
    return int(n)


@task
def describe(week_start: str, n: int) -> str:
    return f"Week of {week_start}: {n:,} orders"


@flow(log_prints=True)
def weekly_summary(week_start: str = "2026-03-02"):
    n = count_orders(week_start)
    message = describe(week_start, n)
    print(message)
    return message


if __name__ == "__main__":
    weekly_summary()
    weekly_summary("2026-02-23")
```

```bash
python day09/hello_flow.py
```

```
INFO | prefect - Starting temporary server on http://127.0.0.1:8013
INFO | Flow run 'uptight-ocelot' - Beginning flow run 'uptight-ocelot' for flow 'weekly-summary'
INFO | Task run 'count_orders-09f' - Finished in state Completed()
INFO | Task run 'describe-91a' - Finished in state Completed()
INFO | Flow run 'uptight-ocelot' - Week of 2026-03-02: 1,130 orders
INFO | Flow run 'uptight-ocelot' - Finished in state Completed()
...
INFO | Flow run 'remarkable-butterfly' - Week of 2026-02-23: 1,070 orders
```

Notes:

- Every run gets a random name (`uptight-ocelot`) and a log. Flow arguments (`week_start`) are **parameters**, recorded with the run.
- `log_prints=True` sends `print()` to the Prefect log.
- With no Prefect server configured, Prefect starts a **temporary** one for the run and stops it afterwards. That's fine for today's first sections; in 9.4 we start a permanent one.

> **Optional:** `prefect config set PREFECT_SERVER_ANALYTICS_ENABLED=false` turns off Prefect's anonymous usage statistics.

**Try it.** Write `day09/weeks_flow.py`: a flow that takes a **list** of weeks, and a task that returns both orders and revenue for one week. Print the four weeks around the email (16 Feb to 9 Mar).

<details>
<summary>Solution</summary>

```python
# day09/weeks_flow.py
import duckdb
from prefect import flow, task


@task
def week_totals(week_start: str) -> dict:
    con = duckdb.connect("data/northmart.duckdb", read_only=True)
    n, rev = con.sql(f"""
        SELECT SUM(n_orders), SUM(revenue) FROM marts.panel_customer_week
        WHERE week_start = DATE '{week_start}'
    """).fetchall()[0]
    con.close()
    return {"week": week_start, "orders": int(n), "revenue": round(rev)}


@flow(log_prints=True)
def around_the_email(weeks: list[str]):
    for w in weeks:
        r = week_totals(w)
        print(f"{r['week']}: {r['orders']:,} orders, ${r['revenue']:,}")


if __name__ == "__main__":
    around_the_email(["2026-02-16", "2026-02-23", "2026-03-02", "2026-03-09"])
```

```
2026-02-16: 1,178 orders, $179,326
2026-02-23: 1,070 orders, $167,578
2026-03-02: 1,130 orders, $180,172
2026-03-09: 1,086 orders, $173,571
```

Week-to-week totals move by 5–10% with no intervention at all. Remember that before reading a single post-email week as "the effect".

</details>

---

## 9.2 Retries, logging and failure (40 min)

**Concept.** Some failures are **temporary**: a busy API, a dropped connection. Retrying after a pause usually fixes them. Others are **real**: a broken contract, a wrong password. Retrying those wastes time and hides the problem. Prefect lets you choose per task: `@task(retries=3, retry_delay_seconds=2)`.

Create `day09/retries_demo.py`. The task fails twice on purpose, like an API under load:

```python
# day09/retries_demo.py
from prefect import flow, task
from prefect.runtime import task_run


@task(retries=3, retry_delay_seconds=2)
def call_flaky_service() -> str:
    attempt = task_run.run_count
    print(f"attempt {attempt}")
    if attempt < 3:                       # fails twice, then works: like a busy API
        raise ConnectionError("service temporarily unavailable")
    return "data received"


@flow(log_prints=True)
def resilient():
    print(call_flaky_service())


if __name__ == "__main__":
    resilient()
```

```
Task run 'call_flaky_service-4e6' - attempt 1
Task run 'call_flaky_service-4e6' - Task run failed with exception: ConnectionError('service temporarily unavailable') - Retry 1/3 will start 2 second(s) from now
Task run 'call_flaky_service-4e6' - attempt 2
... Retry 2/3 will start 2 second(s) from now
Task run 'call_flaky_service-4e6' - attempt 3
Task run 'call_flaky_service-4e6' - Finished in state Completed()
Flow run 'mauve-hornet' - data received
```

**Try it.** Change the condition to `if attempt < 9:` so the service never recovers within the retries. What is the final state of the task and the flow?

<details>
<summary>Solution</summary>

```
Task run 'call_flaky_service-ed2' - ... Retry 3/3 will start 2 second(s) from now
Task run 'call_flaky_service-ed2' - attempt 4
Task run 'call_flaky_service-ed2' - Finished in state Failed('Task run encountered an exception ConnectionError: service temporarily unavailable')
Flow run 'speedy-crane' - Finished in state Failed(...)
```

One original attempt plus three retries, then the task fails and takes the flow with it. The script exits with an error code, which is how schedulers and colleagues learn something is wrong.

</details>

---

## 9.3 The NorthMart flow: load, build, gate, publish (85 min)

**Concept.** The whole Module 1 pipeline becomes one flow with four tasks:

```
check_api ──► load_raw (dlt) ──► build_models (dbt build = models + contracts) ──► publish
  retries 3      retries 2              no retries: a failed contract is real               Parquet + manifest
```

The key design choice is the **gate**. dbt writes tables into DuckDB as it goes, so even a failed build can leave a damaged table there (Day 8, 8.5). Analysts therefore never read the warehouse directly. They read `data/published/`, and the `publish` task only runs if `build_models` succeeded. A failed run leaves the last good release untouched.

The tasks call our existing scripts with `subprocess.run`, the same commands you typed by hand. An orchestrator doesn't replace your code; it runs it reliably.

Create the folder `pipeline/` and `pipeline/northmart_flow.py`:

```python
# pipeline/northmart_flow.py
"""
The whole Module 1 pipeline as one Prefect flow:

    API health -> dlt load (raw) -> dbt build (models + contracts) -> publish marts

If any dbt contract fails, the flow stops BEFORE publishing, so analysts keep
the last good version of the data.

Run from the repository root (with the NorthMart API running):
    python pipeline/northmart_flow.py
    python pipeline/northmart_flow.py --as-of 2026-03-08     # replay a past week
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests
from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "northmart.duckdb"
PUBLISHED = ROOT / "data" / "published"
API = "http://127.0.0.1:8000"
MARTS = ["panel_customer_week", "customer_features"]


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    """Run a command; raise with its output if it fails."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout


@task(retries=3, retry_delay_seconds=5)
def check_api() -> str:
    r = requests.get(f"{API}/health", timeout=10)
    r.raise_for_status()
    today = r.json()["today"]
    get_run_logger().info(f"API is up; its 'today' is {today}")
    return today


@task
def set_clock(as_of: str) -> None:
    run([sys.executable, "api/clock.py", as_of])


@task(retries=2, retry_delay_seconds=10)
def load_raw() -> str:
    out = run([sys.executable, "day05/northmart_pipeline.py"])
    lines = [l for l in out.splitlines() if l.startswith("raw.")]
    for line in lines:
        get_run_logger().info(line)
    return "\n".join(lines)


@task
def build_models() -> str:
    """dbt build = models + tests + contracts. Any error raises and stops the flow."""
    logger = get_run_logger()
    result = subprocess.run(["dbt", "--no-use-colors", "build", "--profiles-dir", "."],
                            cwd=ROOT / "transform", capture_output=True, text=True)
    lines = [l.split("  ", 1)[-1].strip() for l in result.stdout.splitlines()]   # drop timestamps
    summary = next((l for l in lines if l.startswith("Done. PASS=")), "no summary")
    for line in lines:
        if " FAIL " in line or " ERROR " in line or " WARN " in line:
            logger.warning(line)
    logger.info(summary)
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed, data NOT published. {summary}")
    return summary


@task
def publish(as_of: str, dbt_summary: str) -> dict:
    """Export the marts to Parquet with a small manifest describing this release."""
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    manifest = {
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": as_of,
        "dbt": dbt_summary,
        "tables": {},
    }
    for name in MARTS:
        path = PUBLISHED / f"{name}.parquet"
        con.sql(f"COPY marts.{name} TO '{path}' (FORMAT parquet)")
        n = con.sql(f"SELECT COUNT(*) FROM marts.{name}").fetchall()[0][0]
        manifest["tables"][name] = {"rows": n, "file": str(path.relative_to(ROOT))}
    con.close()
    (PUBLISHED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    get_run_logger().info(f"Published {len(MARTS)} tables to {PUBLISHED.relative_to(ROOT)}")
    return manifest


@flow(name="northmart-causal-data-mart", log_prints=True)
def northmart_pipeline(as_of: str | None = None) -> dict:
    if as_of:
        set_clock(as_of)
    today = check_api()
    load_raw()
    summary = build_models()
    manifest = publish(today, summary)
    print(f"Data as of {today}: " + ", ".join(f"{k} {v['rows']:,} rows" for k, v in manifest["tables"].items()))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", help="move the API clock to this date first (YYYY-MM-DD)")
    args = parser.parse_args()
    northmart_pipeline(args.as_of)
```

With the API running in the other terminal:

```bash
python pipeline/northmart_flow.py
```

```
Task run 'check_api-3c4' - API is up; its 'today' is 2026-04-26
Task run 'load_raw-1ea' - raw.customers              4,000 rows
Task run 'load_raw-1ea' - raw.products                  40 rows
Task run 'load_raw-1ea' - raw.campaign_recipients    1,279 rows
Task run 'load_raw-1ea' - raw.orders                18,359 rows
Task run 'build_models-e4d' - 24 of 32 WARN 1 warn_overlap_by_region ...
Task run 'build_models-e4d' - Done. PASS=31 WARN=1 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32
Task run 'publish-638' - Published 2 tables to data/published
Flow run 'cuddly-hoatzin' - Data as of 2026-04-26: panel_customer_week 64,000 rows, customer_features 4,000 rows
Flow run 'cuddly-hoatzin' - Finished in state Completed()
```

```bash
cat data/published/manifest.json
```

```json
{
  "published_at": "2026-10-31T23:06:03+00:00",
  "data_as_of": "2026-04-26",
  "dbt": "Done. PASS=31 WARN=1 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32",
  "tables": {
    "panel_customer_week": {
      "rows": 64000,
      "file": "data/published/panel_customer_week.parquet"
    },
    "customer_features": {
      "rows": 4000,
      "file": "data/published/customer_features.parquet"
    }
  }
}
```

The **manifest** travels with the data: anyone who opens the Parquet files can see what date the data runs to and that every contract passed. Add `data/published/` to `.gitignore` (it's already there if you copied the Day 1 version).

> `dbt --no-use-colors` removes the colour codes dbt prints for terminals, which would otherwise clutter the Prefect log and the manifest.

---

## 9.4 Schedules and the Prefect dashboard (45 min)

**Concept.** A pipeline that only runs when someone remembers is not a pipeline. Prefect's `serve()` registers the flow as a **deployment** with a schedule and waits for scheduled times. Schedules need a permanent Prefect server (the temporary one can't schedule), so today you use **three terminals**.

**Terminal 2 (already running):** the NorthMart API.

**Terminal 3 — the Prefect server:**

```bash
source .venv/bin/activate
prefect server start
```

Open port **4200** when Codespaces offers it: that's the Prefect dashboard.

> **Codespaces tip.** If the dashboard loads but says it can't reach the API, stop the server and start it like this instead:
> `PREFECT_UI_API_URL="https://${CODESPACE_NAME}-4200.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}/api" prefect server start`

**Terminal 1 — point Prefect at the server, then serve the flow:**

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Create `pipeline/serve.py`:

```python
# pipeline/serve.py
"""Run the pipeline on a schedule. Leave this running; stop with Ctrl+C."""
from northmart_flow import northmart_pipeline

if __name__ == "__main__":
    northmart_pipeline.serve(
        name="daily-6am",
        cron="0 6 * * *",                 # every day at 06:00
        tags=["northmart", "module-1"],
    )
```

```bash
python pipeline/serve.py
```

```
Your flow 'northmart-causal-data-mart' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'northmart-causal-data-mart/daily-6am'
```

`cron="0 6 * * *"` means minute 0, hour 6, every day. You don't have to wait until 6 a.m.: open a **fourth** terminal (or use the dashboard's **Run** button) and trigger a run now:

```bash
prefect deployment run 'northmart-causal-data-mart/daily-6am'
```

```
Created flow run 'funny-tarsier'.
└── URL: http://127.0.0.1:4200/runs/flow-run/8e4096fd-...
```

In the dashboard, open **Runs**: you'll see the run, each task's state and duration, and the full log. Click **Deployments → daily-6am** to see the next scheduled runs.

> **Cleaning up.** `Ctrl+C` stops `serve.py` and the server. To go back to temporary servers for single runs: `prefect config unset PREFECT_API_URL`. In production the server runs permanently on a shared machine or in Prefect Cloud; the flow code doesn't change.

---

## 9.5 Day 9 mini-project: a failed run that protects analysts (45 min)

**Goal.** Prove the gate works: make the data wrong in a way the loader can't fix, run the flow, and confirm nothing is published.

**Step 1.** Save the current manifest to compare later:

```bash
cp data/published/manifest.json /tmp/manifest_before.json
```

**Step 2 — inject a missed load.** Yesterday's `backdate` fault would be repaired by the next load (recipients use `replace`). A missing week is different: the incremental loader only asks for *new* orders, so the gap stays.

```bash
python day08/break_things.py drop_week
python pipeline/northmart_flow.py
echo "exit code: $?"
```

```
Task run 'load_raw-8e7' - raw.orders                17,121 rows
Task run 'build_models-a47' - 27 of 32 FAIL 1 assert_no_empty_weeks ....
Task run 'build_models-a47' - Done. PASS=30 WARN=1 ERROR=1 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32
Task run 'build_models-a47' - Task run failed with exception: RuntimeError('dbt build failed, data NOT published. ...')
Flow run 'mega-rat' - Finished in state Failed(...)
exit code: 1
```

**Step 3 — confirm the gate held.**

```bash
cmp /tmp/manifest_before.json data/published/manifest.json && echo "nothing published"
```

```
nothing published
```

Analysts still have the last good release. The failed run is visible in the dashboard and returns exit code 1, which any alerting system can pick up.

**Step 4 — recover.** Running the flow again won't help: the loader thinks those orders are already loaded. A missed incremental load needs a **full refresh**:

```bash
python day08/break_things.py restore
python pipeline/northmart_flow.py
```

The run completes and publishes again. Write down this recovery procedure in `docs/CONTRACTS.md` under a heading **Runbook**: in a real team, whoever is on call at 6 a.m. needs it.

---

## 9.6 Save your work with Git (15 min)

```bash
git add .
git commit -m "Day 9: Prefect flow with contract gate, schedule and runbook"
git push
```

### Day 9 checklist

- [ ] I can write Prefect flows and tasks with parameters and logging
- [ ] I can choose which tasks retry and explain why contract failures must not
- [ ] I can run the whole pipeline as one flow that publishes only after every contract passes
- [ ] I can serve a flow on a schedule, trigger it, and read runs in the dashboard
- [ ] I can recover from a missed incremental load with a full refresh

---

# DAY 10 — Capstone: The Causal Data Mart

| Time | Block |
|---|---|
| 0:00 – 0:30 | 10.1 The brief and how it is assessed |
| 0:30 – 1:15 | 10.2 Two fixes for a pipeline that runs all year |
| 1:15 – 1:30 | Break |
| 1:30 – 2:30 | 10.3 Replay the 16 weeks |
| 2:30 – 3:15 | 10.4 Acceptance tests |
| 3:15 – 3:30 | Break |
| 3:30 – 4:15 | 10.5 The data card |
| 4:15 – 4:45 | 10.6 The handover memo |
| 4:45 – 5:00 | 10.7 Final commit and Module 1 retrospective |

---

## 10.1 The brief and how it is assessed (30 min)

**The brief.** NorthMart's analytics lead writes:

> *"Next month the team will estimate the effect of the SPRING26 email, first with a DAG, then with difference-in-differences and matching. Before anyone estimates anything, I need a data mart I can trust: rebuilt automatically every week, checked for the problems that ruin causal studies, and documented well enough that a new analyst can use it without asking you. Show me it would have worked all spring, week by week."*

**Deliverables.**

| # | Deliverable | Where |
|---|---|---|
| D1 | A pipeline that runs every week of the study without manual steps | `pipeline/northmart_flow.py` |
| D2 | A week-by-week replay from an empty warehouse | `day10/reset_warehouse.py`, `day10/replay_weeks.py` |
| D3 | Acceptance tests on the published release | `day10/acceptance.py` |
| D4 | A generated data card | `day10/data_card.py`, `docs/DATA_CARD.md` |
| D5 | A contract register with fire-drill results and runbook | `docs/CONTRACTS.md` |
| D6 | A one-page handover memo | `docs/HANDOVER_MEMO.md` |

**Assessment rubric** (used again, in extended form, for the final course capstone):

| Criterion | Weight | Excellent |
|---|---|---|
| Reproducibility | 20% | Replay runs end to end from an empty warehouse with one command |
| Causal validity | 25% | Leakage, timing, fixed-treatment and completeness contracts enforced; fire drill documented |
| Correctness | 20% | Acceptance checks pass, including agreement with the independent Day 3 pipeline |
| Documentation | 15% | Data card generated from data, contract register, runbook |
| Communication | 20% | Memo states what the data can and cannot support, in plain language |

---

## 10.2 Two fixes for a pipeline that runs all year (45 min)

**Concept.** So far we always ran the pipeline with all 16 weeks available. A real pipeline starts on its first Sunday with an empty warehouse. Simulate that. Create `day10/reset_warehouse.py`:

```python
# day10/reset_warehouse.py
"""Empty the warehouse and the loader's memory, as on the first day of a new pipeline."""
import shutil
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def reset_warehouse() -> None:
    con = duckdb.connect(str(ROOT / "data" / "northmart.duckdb"))
    for schema in ["raw", "staging", "intermediate", "marts"]:
        con.sql(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    con.close()
    shutil.rmtree(ROOT / ".dlt_pipelines" / "northmart", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "published", ignore_errors=True)
    print("Warehouse emptied: raw, staging, intermediate and marts dropped; loader state cleared.")


if __name__ == "__main__":
    reset_warehouse()
```

Then empty the warehouse and run the flow as of Sunday 18 January:

```bash
python day10/reset_warehouse.py
python pipeline/northmart_flow.py --as-of 2026-01-18
```

```
build_models - 2 of 32 ERROR creating sql view model staging.stg_campaign_recipients .......... [ERROR in 0.29s]
build_models - Task run failed with exception: RuntimeError('dbt build failed, data NOT published.
               Done. PASS=16 WARN=0 ERROR=1 SKIP=15 NO-OP=0 REUSED=0 TOTAL=32')
```

**Problem 1: no recipient table.** Before 2 March nobody has been emailed, and dlt only creates tables that receive rows (Day 5). `stg_campaign_recipients` points at a table that doesn't exist yet.

Fix that (Fix 1 below) and run again, and a second problem appears:

```
build_models - 27 of 32 FAIL 14 assert_no_empty_weeks ......................................... [FAIL 14 in 0.08s]
build_models - Done. PASS=30 WARN=1 ERROR=1 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32
```

**Problem 2: empty future weeks.** `int_weeks` always generates 16 weeks. On 18 January, weeks 3–16 have no orders yet, and the completeness contract (rightly) rejects all 14 of them.

Both are design errors in our code, not in the data, and the contracts caught both before anything was published.

**Fix 1 — a treatment log that may not exist yet.** Replace `transform/models/staging/stg_campaign_recipients.sql`:

```sql
-- Customers who received a campaign email, with the send date.
-- Before the first email is sent, dlt has not created raw.campaign_recipients yet
-- (it only creates tables that received rows). In that case: an empty table, nobody treated.
{% set recipients = adapter.get_relation(
       database=source('raw', 'campaign_recipients').database,
       schema=source('raw', 'campaign_recipients').schema,
       identifier=source('raw', 'campaign_recipients').identifier) %}

{% if recipients is not none %}
select
    customer_id,
    campaign_id,
    cast(sent_date as date) as sent_date
from {{ source('raw', 'campaign_recipients') }}
{% else %}
select
    cast(null as varchar) as customer_id,
    cast(null as varchar) as campaign_id,
    cast(null as date)    as sent_date
where false
{% endif %}
```

`adapter.get_relation` asks the database whether the table exists when the model runs. If it doesn't, the model returns an empty table with the right columns, which means exactly what the business situation means: nobody is treated yet.

**Fix 2 — a calendar that ends with the data.** Replace `transform/models/intermediate/int_weeks.sql`:

```sql
-- The calendar spine: one row per Monday, numbered 1, 2, 3...
-- It stops at the last week that has data, so a weekly pipeline run in, say, February
-- doesn't create empty "future" weeks (which the completeness contract would reject).
with bounds as (
    select least(
        date '{{ var("last_week") }}',
        cast(date_trunc('week', max(order_date)) as date)
    ) as last_monday
    from {{ ref('stg_orders') }}
)
select
    cast(g.week_start as date)                        as week_start,
    row_number() over (order by g.week_start)         as week_no
from bounds,
     generate_series(date '{{ var("first_week") }}', bounds.last_monday, interval 7 day) as g(week_start)
```

The spine now ends at the earlier of the study's last week and the last week with orders. Our clock always stops on a Sunday, so every week included is complete. (A pipeline that ran mid-week would also need to drop the current, partial week; otherwise it would look like a sales drop.)

Run the flow for 18 January again:

```bash
python pipeline/northmart_flow.py --as-of 2026-01-18
```

```
build_models - Done. PASS=31 WARN=1 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32
Flow run 'origami-caracara' - Data as of 2026-01-18: panel_customer_week 8,000 rows, customer_features 4,000 rows
```

Two weeks × 4,000 customers, nobody treated yet, every contract passing. The overlap warning now fires for every region (no emailed customers exist yet), which is correct and, being a warning, doesn't block the release.

---

## 10.3 Replay the 16 weeks (60 min)

**Goal.** Start from an **empty** warehouse (the script calls `reset_warehouse()` itself) and run the full flow once per week, each Sunday from 11 January to 26 April, exactly as the scheduled pipeline would have. After each run, record what was published.

Create `day10/replay_weeks.py`:

```python
# day10/replay_weeks.py
"""
Capstone step 1: live through the 16 weeks as NorthMart's data team did.

Starts from an empty warehouse, then runs the full pipeline once per week,
every Sunday from 11 January to 26 April 2026. Each run loads only that week's
new orders, rebuilds the models, checks every contract, and publishes.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "day10"))
from northmart_flow import northmart_pipeline  # noqa: E402
from reset_warehouse import reset_warehouse  # noqa: E402

FIRST_SUNDAY = date(2026, 1, 11)
N_WEEKS = 16


def snapshot(as_of: date) -> dict:
    panel = pl.read_parquet(ROOT / "data" / "published" / "panel_customer_week.parquet")
    return {
        "as_of": as_of,
        "weeks": panel["week_start"].n_unique(),
        "panel_rows": panel.height,
        "orders_so_far": int(panel["n_orders"].sum()),
        "treated_customers": int(panel.filter(pl.col("week_no") == 1)["treated"].sum()),
    }


def main() -> None:
    reset_warehouse()
    history = []
    for k in range(N_WEEKS):
        as_of = FIRST_SUNDAY + timedelta(weeks=k)
        northmart_pipeline(as_of.isoformat())
        history.append(snapshot(as_of))

    table = pl.DataFrame(history)
    with pl.Config(tbl_rows=-1):
        print(table)
    table.write_csv(ROOT / "data" / "published" / "replay_history.csv")


if __name__ == "__main__":
    main()
```

With the API running (and `prefect config unset PREFECT_API_URL` if you set it yesterday), run it. It takes about three minutes:

```bash
python day10/replay_weeks.py
```

```
┌────────────┬───────┬────────────┬───────────────┬───────────────────┐
│ as_of      ┆ weeks ┆ panel_rows ┆ orders_so_far ┆ treated_customers │
╞════════════╪═══════╪════════════╪═══════════════╪═══════════════════╡
│ 2026-01-11 ┆ 1     ┆ 4000       ┆ 1050          ┆ 0                 │
│ 2026-01-18 ┆ 2     ┆ 8000       ┆ 2183          ┆ 0                 │
│ 2026-01-25 ┆ 3     ┆ 12000      ┆ 3350          ┆ 0                 │
│ 2026-02-01 ┆ 4     ┆ 16000      ┆ 4567          ┆ 0                 │
│ 2026-02-08 ┆ 5     ┆ 20000      ┆ 5770          ┆ 0                 │
│ 2026-02-15 ┆ 6     ┆ 24000      ┆ 6920          ┆ 0                 │
│ 2026-02-22 ┆ 7     ┆ 28000      ┆ 8098          ┆ 0                 │
│ 2026-03-01 ┆ 8     ┆ 32000      ┆ 9168          ┆ 0                 │
│ 2026-03-08 ┆ 9     ┆ 36000      ┆ 10298         ┆ 1279              │
│ 2026-03-15 ┆ 10    ┆ 40000      ┆ 11384         ┆ 1279              │
│ 2026-03-22 ┆ 11    ┆ 44000      ┆ 12417         ┆ 1279              │
│ 2026-03-29 ┆ 12    ┆ 48000      ┆ 13511         ┆ 1279              │
│ 2026-04-05 ┆ 13    ┆ 52000      ┆ 14509         ┆ 1279              │
│ 2026-04-12 ┆ 14    ┆ 56000      ┆ 15582         ┆ 1279              │
│ 2026-04-19 ┆ 15    ┆ 60000      ┆ 16690         ┆ 1279              │
│ 2026-04-26 ┆ 16    ┆ 64000      ┆ 17839         ┆ 1279              │
└────────────┴───────┴────────────┴───────────────┴───────────────────┘
```

Read it row by row:

- All 16 runs passed every contract and published.
- The panel grows by exactly 4,000 rows (one week of customers) per run.
- The 1,279 treated customers appear in the week the email was sent, not before.
- The final run has **17,839** orders: the same number as Day 2's cleaning of the CSV, reached here through 16 incremental API loads.

---

## 10.4 Acceptance tests (45 min)

**Concept.** dbt contracts check the data **inside** the warehouse. Acceptance tests check the **published release** from the user's side: is it complete, does it agree with an independent source, and does it support the analysis it was built for?

Create `day10/acceptance.py`:

```python
# day10/acceptance.py
"""Capstone step 2: acceptance tests for the published Causal Data Mart."""
import json
from pathlib import Path

import polars as pl

PUB = Path("data/published")
KEY = ["customer_id", "week_start"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


manifest = json.loads((PUB / "manifest.json").read_text())
panel = pl.read_parquet(PUB / "panel_customer_week.parquet").sort(KEY)
features = pl.read_parquet(PUB / "customer_features.parquet")

# A. The release is complete and clean
check("A1 data covers all 16 weeks", panel["week_start"].n_unique() == 16, str(manifest["data_as_of"]))
check("A2 dbt build had no errors", "ERROR=0" in manifest["dbt"], manifest["dbt"])
check("A3 one feature row per customer", features.height == features["customer_id"].n_unique() == 4000)

# B. It agrees with the independent Day 3 pipeline
day3_file = Path("data/marts/panel_customer_week.parquet")
if day3_file.exists():
    day3 = pl.read_parquet(day3_file).sort(KEY)
    same = all(
        ((panel[c].cast(pl.Float64) - day3[c].cast(pl.Float64)).abs() < 1e-6).all()
        if panel[c].dtype.is_numeric() else (panel[c] == day3[c]).all()
        for c in day3.columns
    )
    check("B1 identical to the Day 3 panel", same and panel.height == day3.height)
else:
    check("B1 identical to the Day 3 panel", False, "run Days 2-3 scripts to create the comparison file")

# C. It supports the causal analysis of Modules 2-5
cust = panel.group_by("customer_id", "treated").agg(
    pre=pl.col("n_orders").filter(pl.col("post") == 0).sum(),
    post=pl.col("n_orders").filter(pl.col("post") == 1).sum(),
)
m = cust.group_by("treated").agg(pl.col("pre").mean(), pl.col("post").mean()).sort("treated")
c, t = m.row(0, named=True), m.row(1, named=True)
naive = t["post"] / c["post"] - 1
growth_gap = (t["post"] / t["pre"] - 1) - (c["post"] / c["pre"] - 1)
check("C1 naive post-period gap reproduces Day 2", abs(naive - 1.06) < 0.01, f"{naive:+.1%}")
check("C2 difference in growth reproduces Day 2", abs(growth_gap - 0.155) < 0.005, f"{growth_gap:+.1%}")
check("C3 features balance: treated spent more pre-email",
      features.filter(pl.col("treated") == 1)["pre_revenue"].mean()
      > features.filter(pl.col("treated") == 0)["pre_revenue"].mean())

for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
passed = sum(ok for _, ok, _ in results)
print(f"\n{passed}/{len(results)} acceptance checks passed")
raise SystemExit(0 if passed == len(results) else 1)
```

```bash
python day10/acceptance.py
```

```
[PASS] A1 data covers all 16 weeks  (2026-04-26)
[PASS] A2 dbt build had no errors  (Done. PASS=31 WARN=1 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=32)
[PASS] A3 one feature row per customer
[PASS] B1 identical to the Day 3 panel
[PASS] C1 naive post-period gap reproduces Day 2  (+105.6%)
[PASS] C2 difference in growth reproduces Day 2  (+15.5%)
[PASS] C3 features balance: treated spent more pre-email

7/7 acceptance checks passed
```

Group C is the bridge to the rest of the course. The published mart reproduces both Day 2 numbers: the misleading naive gap (+106%) and the growth comparison close to the true +15%. Module 2 explains, with a DAG, why the first is biased; Module 5 turns the second into a proper difference-in-differences estimate with standard errors.

---

## 10.5 The data card (45 min)

**Concept.** A **data card** tells a new analyst what a dataset is for, how it was built, what each variable means, and what its known weaknesses are. Written by hand, it goes out of date with the next load. So we **generate** it from the published data and manifest.

Create `day10/data_card.py`:

```python
# day10/data_card.py
"""Capstone step 3: write docs/DATA_CARD.md from the published data, so it is never out of date."""
import json
from pathlib import Path

import polars as pl

PUB = Path("data/published")
manifest = json.loads((PUB / "manifest.json").read_text())
panel = pl.read_parquet(PUB / "panel_customer_week.parquet")
features = pl.read_parquet(PUB / "customer_features.parquet")

balance = (features.group_by("treated")
           .agg(customers=pl.len(),
                pre_orders=pl.col("pre_orders").mean().round(2),
                pre_revenue=pl.col("pre_revenue").mean().round(0),
                loyalty_share=pl.col("is_loyalty").mean().round(3),
                age=pl.col("age").mean().round(1),
                tenure_days=pl.col("tenure_days").mean().round(0))
           .sort("treated"))


def md_table(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.iter_rows())


card = f"""# Data card: NorthMart Causal Data Mart

*Generated by `day10/data_card.py` from the release published {manifest['published_at']}.
Do not edit by hand: rerun the script.*

## What question this data supports

Did the SPRING26 email (sent Monday 2 March 2026) change how much customers buy?

## Release

| Item | Value |
|---|---|
| Data as of | {manifest['data_as_of']} |
| dbt build | {manifest['dbt']} |
| Weeks | {panel['week_start'].min()} to {panel['week_start'].max()} ({panel['week_start'].n_unique()} weeks) |
| Customers | {panel['customer_id'].n_unique():,} |
| Emailed customers | {int(features['treated'].sum()):,} ({features['treated'].mean():.1%}) |

## Tables

| Table | Grain | Rows | Use it for |
|---|---|---|---|
| `panel_customer_week` | customer × week | {manifest['tables']['panel_customer_week']['rows']:,} | difference-in-differences, event studies (Module 5) |
| `customer_features` | customer | {manifest['tables']['customer_features']['rows']:,} | matching, weighting, causal ML (Modules 4 and 6) |

## Definitions

- **Treatment:** `treated = 1` if the customer appears in the SPRING26 recipient log. Fixed per customer.
- **Timing:** `post = 1` from week 9 (week of 2 March 2026); `event_week = 0` is the email week.
- **Outcomes:** `n_orders` (completed orders) and `revenue` (quantity × unit price after discount), 0 in weeks without orders. Cancelled orders are excluded.
- **Covariates:** region, age, loyalty, tenure, and pre-period behaviour, all measured **before** 2 March 2026 (contract C9, tested by time travel).

## Pre-treatment balance

{md_table(balance)}

Emailed customers were heavier shoppers and more often loyalty members **before** the email: the
assignment was targeted, not random. A raw comparison of outcomes is biased; see Modules 4-5.

## Known limitations

- Region is missing for {int((features['region'] == 'UNKNOWN').sum())} customers (`UNKNOWN`); this group has fewer than 30 emailed customers (warning C11).
- The pre-period covers 8 weeks only; longer histories would strengthen pre-trend checks.
- Data are simulated: the true effect is a 15% increase in order frequency, so methods can be graded.

## Rebuild

```bash
python api/northmart_api.py          # in a separate terminal
python pipeline/northmart_flow.py    # load, build, test, publish
python day10/acceptance.py
python day10/data_card.py
```

Contracts are listed in `docs/CONTRACTS.md`.
"""
Path("docs").mkdir(exist_ok=True)
Path("docs/DATA_CARD.md").write_text(card)
print(card[:1200])
print("...\nSaved docs/DATA_CARD.md")
```

```bash
python day10/data_card.py
```

Open `docs/DATA_CARD.md` in VS Code (right-click → **Open Preview**). The pre-treatment balance section reads:

```
| treated | customers | pre_orders | pre_revenue | loyalty_share | age | tenure_days |
|---|---|---|---|---|---|---|
| 0 | 2721 | 1.85 | 288.0 | 0.208 | 46.7 | 607.0 |
| 1 | 1279 | 3.23 | 506.0 | 0.491 | 47.3 | 594.0 |
```

Age and tenure are balanced; shopping intensity and loyalty are not. That single table tells a reader the email was targeted, and that any estimate must deal with it.

**Make it run every week.** Add the acceptance tests and the data card to the flow so each release documents itself. In `pipeline/northmart_flow.py`, add a task:

```python
@task
def document() -> None:
    run([sys.executable, "day10/acceptance.py"])
    run([sys.executable, "day10/data_card.py"])
```

and call `document()` at the end of `northmart_pipeline`, after `publish`. (Acceptance check A1 expects all 16 weeks, so add it only once the replay is done, or relax A1 to "weeks published = weeks loaded".)

---

## 10.6 The handover memo (30 min)

Write `docs/HANDOVER_MEMO.md`, one page, for the analytics lead. Use these headings and answer in plain language; a non-technical manager should follow it.

1. **The question and the data.** What decision does this data inform? What are the unit, the treatment, the outcome and the period?
2. **What the mart guarantees.** Which contracts run on every release, and what happens when one fails? (Refer to `CONTRACTS.md`; don't repeat it.)
3. **What the raw numbers say, and why not to trust them.** The naive gap is +106%. Explain in two sentences why that can't be the effect, using the balance table.
4. **What the data can support.** Name one comparison that is more credible than the naive one and the assumption it relies on. (Hint: Day 2's growth comparison. What must be true of the two groups' trends?)
5. **Known limitations and next steps.** At least three, including the UNKNOWN region and the 8-week pre-period.

<details>
<summary>Model answer for section 3</summary>

Emailed customers placed twice as many orders as other customers after 2 March, but they were already placing 75% more **before** the email was sent: marketing targeted its best customers. Most of the post-email gap is therefore a difference between the kinds of customers, not an effect of the email, and the raw comparison overstates the effect several times over.

</details>

<details>
<summary>Model answer for section 4</summary>

Comparing how each group's orders **changed** from before to after the email removes the stable difference between the groups. Emailed customers' orders grew by 3% while other customers' fell by 12%, a gap of about 15 percentage points. This is credible only if, without the email, both groups' orders would have followed the same trend (**parallel trends**). Eight weeks of pre-period data let us check that partially; Module 5 does so formally.

</details>

---

## 10.7 Final commit and Module 1 retrospective (15 min)

```bash
git add .
git status        # data/, .dlt/secrets.toml, .dlt_pipelines/ and transform/target/ must NOT appear
git commit -m "Day 10: Causal Data Mart capstone: replay, acceptance, data card, memo"
git push
```

**What you built in Module 1.**

| Day | Skill | Causal reason |
|---|---|---|
| 1 | Environment, causal vocabulary | Reproducible estimates; roles of variables |
| 2 | Polars cleaning, naive comparison | Selection bias, seen in the data |
| 3 | DuckDB panel with a calendar spine | Zeros are outcomes, not missing data |
| 4–5 | APIs and incremental dlt loads | Treatment logs live in company systems |
| 6–7 | dbt models and tests | Every transformation reviewed and tested |
| 8 | Causal data contracts | Leakage and timing errors stopped automatically |
| 9 | Prefect orchestration with a gate | Analysts never see data that failed a contract |
| 10 | Replay, acceptance, data card | A mart a team can trust and understand |

**Coming next: Module 2, Causal Thinking, Estimands & DAGs.** You'll state precisely what "the effect of the email" means (the ATT, not the ATE, and why), draw the DAG behind the targeting you documented today, and use it to decide which variables in `customer_features` to adjust for, and which to leave alone.

### Day 10 checklist

- [ ] My pipeline runs correctly on any Sunday of the study, before and after the email
- [ ] I have replayed all 16 weeks from an empty warehouse with every contract passing
- [ ] My published release passes all acceptance checks, including agreement with Day 3
- [ ] My data card is generated from the data and explains the treatment imbalance
- [ ] My handover memo explains, in plain language, what the data can and cannot support
