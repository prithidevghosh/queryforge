---
title: QueryForge Environment Server
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sql
  - reinforcement-learning
---

# QueryForge — SQL Debugging & Optimisation Environment

**Live environment:** https://prithvigg-queryforge.hf.space
**Interactive demo:** https://prithvigg-queryforge.hf.space/demo/

> Try it directly below (HF Space viewer only):
>
> <iframe src="https://prithvigg-queryforge.hf.space/demo/" width="100%" height="700px" frameborder="0"></iframe>

SQL is the language that runs the world's data infrastructure. Yet SQL bugs are silent killers — a missing JOIN condition inflates totals by 3×, a correlated subquery scans a million rows once per row, a typo in a keyword stops production cold. These bugs are rarely caught by linters, rarely surfaced by error messages, and routinely shipped to production.

QueryForge is an **OpenEnv-compatible reinforcement learning environment** where an agent learns to debug and optimise SQL queries. The agent receives a broken or slow query, submits fixes, and receives graded feedback from a deterministic DuckDB engine combined with an Anthropic AI quality judge — a smooth, informative reward signal across the full 0.0 → 1.0 range.

---

## Why SQL Debugging as an RL Environment?

LLMs can write SQL. What they struggle with is the **iterative, feedback-driven debugging loop** that real engineers do:

- Read the error message
- Form a hypothesis about the root cause
- Patch the query
- Check if the output is now correct
- Refine until it's both correct *and* efficient

This is precisely the loop that RL is built for. QueryForge provides the environment that closes this loop with a graded, multi-stage reward signal — not just "correct / incorrect" but partial credit for syntax validity, execution success, row correctness, and code quality.

---

## Environment Overview

| Property | Value |
|---|---|
| Task type | SQL debugging & optimisation |
| Action space | Single SQL query string |
| Observation space | Task description + graded feedback |
| Reward range | 0.0 – 1.0 (continuous) |
| Episode termination | Score ≥ 0.90, no improvement for 2 steps, or max steps |
| Grading engine | DuckDB (deterministic) + Anthropic AI judge |
| Concurrent sessions | Supported |

---

## Reward Scale

The grading pipeline has four stages that produce a smooth partial-progress signal:

| Score | Meaning |
|---|---|
| **0.00** | Syntax error — query could not be parsed |
| **0.15** | Syntax valid but runtime error |
| **0.30** | Executes but returns 0 rows or wrong row count |
| **0.30 – 0.80** | Partial row correctness (deterministic, DuckDB) |
| **0.80 – 1.00** | Correct rows + AI quality assessment (Anthropic) |

The AI judge scores on three axes: **Correctness** (0–0.50), **Optimization** (0–0.30 — penalises cartesian products, correlated subqueries), **Code quality** (0–0.20 — readability, aliases, formatting).

> **Offline mode:** If `ANTHROPIC_API_KEY` is not set, the AI judge is skipped and scoring is fully deterministic (capped at 0.80). The done threshold self-adjusts to 0.80 in this case so episodes still terminate correctly.

---

## Action Space

```python
class SQLAction(Action):
    sql: str  # The SQL query to submit for grading
```

One field. The agent submits a SQL string. No multi-statement queries (`;` separated) are allowed — rejected with score 0.0.

---

## Observation Space

```python
class SQLObservation(Observation):
    # Task context (set on reset, constant within an episode)
    task_id: str            # e.g. "task_easy_syntax"
    task_level: str         # "easy" | "medium" | "hard" | "custom"
    task_title: str         # Human-readable title
    task_description: str   # Full context: schema, broken query, error, goal

    # Per-step grading signals
    syntax_valid: bool      # True if query parsed without error
    execution_success: bool # True if query ran to completion in DuckDB
    execution_error: str    # Runtime error message, if any
    rows_returned: int      # Number of rows returned

    # Feedback
    feedback: str           # Detailed grading feedback (DuckDB + AI judge)
    hint: str               # Actionable hint (suppressed once score >= 0.90)

    # Episode progress
    attempt: int            # Number of queries submitted this episode
    best_score: float       # Highest score achieved so far
    done: bool
    reward: float           # Score for this specific step (0.0 – 1.0)
```

---

## Built-in Tasks

| ID | Level | Title | Max Steps |
|---|---|---|---|
| `task_easy_syntax` | easy | Fix Syntax Errors | 5 |
| `task_medium_join` | medium | Fix the Cartesian JOIN | 5 |
| `task_hard_cte` | hard | Rewrite Correlated Subquery as CTE | 6 |
| `task_expert_rank` | expert | Fix the Tie-Breaking Window Function | 6 |
| `task_expert_recursive` | expert | Traverse Org Chart with Recursive CTE | 7 |
| `task_expert_window` | expert | Fix Two Broken Window Functions | 6 |

### Easy — Fix Syntax Errors
Three SQL keywords are misspelled (`SELEC`, `FORM`, `WEHRE`). The agent must identify and correct them.

**Schema:** `users(id, name, age, city)` — 6 rows
**Goal:** Return name and age of users older than 30 in New York, ordered by name

### Medium — Fix the Cartesian JOIN
A missing `JOIN` condition (`o.product_id = p.id`) causes a cartesian product, inflating every total by 3×. The agent must rewrite using explicit `INNER JOIN … ON` syntax.

**Schema:** `orders`, `users`, `products` — e-commerce dataset
**Goal:** Correct per-(user, product) total amount spent, ordered by total DESC

### Hard — Rewrite Correlated Subquery as CTE
A semantically correct but O(N²) query re-executes `AVG(salary)` for every employee row. The agent must rewrite using a `WITH` clause that computes department averages exactly once.

**Schema:** `departments`, `employees` — 9 employees across 3 departments
**Goal:** Employees who earn strictly above their department average, ordered by dept/salary

### Expert — Fix the Tie-Breaking Window Function (2 bugs)
Two layered bugs: `ROW_NUMBER()` drops tied reps AND `ORDER BY revenue ASC` picks the lowest earners instead of the highest. Agent must fix the sort order AND switch to `RANK()`/`DENSE_RANK()` — fixing only one still produces wrong results.

**Schema:** `sales_reps(id, name, region, revenue)` — 6 reps across 2 regions with ties
**Goal:** All reps whose revenue is the highest in their region

### Expert — Traverse Org Chart with Recursive CTE (2 bugs)
Two layered bugs: the anchor uses `WHERE id = 3` (includes VP Eng himself in results) AND the query is a hardcoded two-level CTE that misses deeper employees. Agent must fix the anchor to `WHERE manager_id = 3` AND convert to `WITH RECURSIVE`.

**Schema:** `employees(id, name, manager_id)` — 14 employees, 4 levels deep
**Goal:** All 8 subordinates of VP Eng at any depth (excluding VP Eng), ordered by id

### Expert — Fix Broken Window Functions (3 bugs)
Three layered bugs: both `SUM` and `RANK` window functions are missing `PARTITION BY`, they need different `ORDER BY` clauses, AND the data contains tied revenue values (West Q3=Q4=16000) that must be ranked correctly.

**Schema:** `quarterly_sales(region, quarter, revenue)` — 8 rows across 2 regions with ties
**Goal:** Per-region running total (`ORDER BY quarter`) and within-region revenue rank (`ORDER BY revenue DESC`) with correct tie handling

> **Structural penalties** are enforced per task level/id to prevent gaming:
> - `hard`: requires `WITH` clause (−0.30 if absent)
> - `medium`: requires explicit `JOIN` (−0.20 if absent)
> - `task_expert_recursive`: requires `WITH RECURSIVE` (−0.30) + correct anchor via `manager_id` (−0.15)
> - `task_expert_rank`: penalises `ROW_NUMBER()` (−0.20) + penalises `ASC` ordering without `DESC` (−0.15)
> - `task_expert_window`: requires `PARTITION BY` in both window functions (−0.20 if absent, −0.10 if only one)

---

## Custom Tasks

Register any SQL task at runtime — no code changes needed.

### Via Python
```python
from tasks import REGISTRY, task_from_dict

REGISTRY.register(task_from_dict({
    "id": "my_window_task",
    "level": "hard",
    "title": "Rank Employees by Salary",
    "schema_ddl": "CREATE TABLE emp (id INT, name VARCHAR, dept VARCHAR, salary DECIMAL); INSERT INTO emp VALUES ...",
    "broken_query": "SELECT name, salary FROM emp ORDER BY salary DESC",
    "expected_rows": [{"name": "Alice", "rank": 1}, ...],
    "hint": "Use ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)",
    "solution_query": "SELECT name, RANK() OVER (ORDER BY salary DESC) AS rank FROM emp",
}))
```

### Via REST API (when server is running)
```bash
# Register a custom task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"id": "my_task", "schema_ddl": "...", "expected_rows": [...]}'

# List all tasks
curl http://localhost:8000/tasks

# Remove a custom task
curl -X DELETE http://localhost:8000/tasks/my_task
```

### Via JSON file
```python
REGISTRY.load_from_json("my_tasks.json")
```

---

## Quickstart

### Install dependencies
```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Run the local playbook (no server needed)
Tests all three built-in tasks directly, with progressive SQL attempts:
```bash
ANTHROPIC_API_KEY=your_key .venv/bin/python playbook.py
```

### Run the inference script
Runs any OpenAI-compatible LLM as an agent against all 6 tasks and reports scores:
```bash
# Against HuggingFace router
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_hf_token
export ENV_URL=http://127.0.0.1:8000       # or the live HF Space URL
python inference.py

# Against the live HF Space (no local server needed)
export ENV_URL=https://prithvigg-queryforge.hf.space
python inference.py
```

### Run the HTTP server
```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Baseline Results

The following scores were produced by running `meta-llama/Llama-3.1-8B-Instruct` (via HuggingFace router) as the agent against all 6 tasks with the full AI judge active.

| Task | Level | Steps Used | Best Score |
|---|---|---|---|
| Fix the Syntax Errors | easy | 1 | **1.000** |
| Fix the Cartesian JOIN | medium | 1 | **0.900** |
| Rewrite Correlated Subquery as CTE | hard | 1 | **0.900** |
| Fix the Tie-Breaking Window Function | expert | 1 | **1.000** |
| Traverse Org Chart with Recursive CTE | expert | 2 | **0.900** |
| Fix Two Broken Window Functions | expert | 3 | **0.900** |
| **Average** | | | **0.933** |

The easy–hard tasks and the rank/recursive expert tasks were solved in 1–2 steps. The dual-window expert task required 3 steps, demonstrating the feedback loop produces training-relevant multi-step trajectories for harder tasks.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Pass `{"task_id": "..."}` to pin to a task |
| `POST` | `/step` | Submit a SQL query: `{"sql": "SELECT ..."}` |
| `GET` | `/state` | Current episode ID and step count |
| `GET` | `/schema` | Action and observation JSON schemas |
| `POST` | `/tasks` | Register a custom task |
| `GET` | `/tasks` | List all registered tasks |
| `DELETE` | `/tasks/{task_id}` | Remove a custom task (built-ins protected) |
| `WS` | `/ws` | WebSocket endpoint for persistent low-latency sessions |
| `GET` | `/health` | Container health check |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### Examples

```bash
BASE=https://prithvigg-queryforge.hf.space   # or http://localhost:8000 for local

# Start an episode pinned to the hard task
curl -X POST $BASE/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_hard_cte"}'

# Submit a query
curl -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{"sql": "WITH dept_avg AS (SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id) SELECT e.name, e.department_id, e.salary FROM employees e JOIN dept_avg d ON e.department_id = d.department_id WHERE e.salary > d.avg_salary ORDER BY e.department_id, e.salary DESC"}'

# List all available tasks
curl $BASE/tasks
```

---

## Python Client

```python
from queryforge import QueryforgeEnv, SQLAction

with QueryforgeEnv(base_url="http://localhost:8000") as env:
    # Pin to a specific task
    obs = env.reset(task_id="task_medium_join")
    print(obs.task_description)

    # Submit a fix
    result = env.step(SQLAction(sql="""
        SELECT u.name, p.title, SUM(o.amount) AS total_spent
        FROM orders o
        INNER JOIN users u ON o.user_id = u.id
        INNER JOIN products p ON o.product_id = p.id
        GROUP BY u.name, p.title
        ORDER BY total_spent DESC
    """))
    print(f"Score: {result.reward:.3f}")
    print(f"Feedback: {result.observation.feedback}")
    print(f"Done: {result.done}")

    # Register and use a custom task
    env.register_task(TaskSpec(
        id="my_task",
        schema_ddl="CREATE TABLE ...; INSERT INTO ...",
        expected_rows=[{"col": "val"}],
        title="My Custom Task",
    ))
    obs = env.reset(task_id="my_task")
```

---

## Project Structure

```
queryforge/
├── __init__.py                     # Public exports (SQLAction, SQLObservation, TaskSpec, REGISTRY)
├── models.py                       # SQLAction, SQLObservation, TaskSpec Pydantic models
├── tasks.py                        # Built-in tasks (easy→expert) + thread-safe TaskRegistry
├── judge.py                        # 4-stage grading pipeline (DuckDB + Anthropic)
├── client.py                       # QueryforgeEnv client with task management helpers
├── playbook.py                     # Local test runner (no server required)
├── inference.py                    # Baseline inference script (any OpenAI-compatible LLM)
├── demo.py                         # Gradio interactive demo (mounted at /demo)
├── Dockerfile                      # Container image
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Locked dependencies
└── server/
    ├── app.py                      # FastAPI app — core + /tasks REST endpoints + Gradio mount
    ├── queryforge_environment.py   # Environment class (reset, step, state)
    └── requirements.txt            # Server dependencies
```

---

## Deployment

### Hugging Face Spaces (recommended)

```bash
UV_CACHE_DIR=/tmp/uv-cache openenv push . --repo-id <hf-username>/queryforge
```

Add `ANTHROPIC_API_KEY` as a Space secret after deployment. Without it, the environment runs in deterministic-only mode (scores capped at 0.80, done threshold self-adjusts accordingly).

### Docker

```bash
docker build -t queryforge:latest .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY queryforge:latest
```

The deployed environment exposes:
- **`/web`** — Interactive UI for exploring the environment
- **`/docs`** — Full OpenAPI / Swagger interface
- **`/ws`** — WebSocket endpoint for persistent agent sessions
- **`/health`** — Container health monitoring

---

## Environment Design Notes

**Why DuckDB?** DuckDB runs fully in-memory with no external process or network dependency. Each `step()` call creates an isolated connection, seeds it with the task's schema, runs the agent's query, then closes — complete isolation with zero shared state between steps.

**Why a 4-stage reward?** Binary correct/incorrect rewards give an agent no gradient to climb when its query is simply broken. The 4-stage pipeline means every improvement — fixing a typo, avoiding a runtime error, returning the right row count, getting the right rows, writing clean SQL — is rewarded. This produces a smooth loss landscape for policy gradient methods.

**Why structural penalties?** Without them, an agent could achieve 0.80 on the hard CTE task by submitting the original correlated subquery verbatim (rows match, but the task was never solved). Structural penalties enforce that the agent actually learned *what* to change, not just that rows matched.
