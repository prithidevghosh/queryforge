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

### Easy — Fix Syntax Errors
Three SQL keywords are misspelled (`SELEC`, `FORM`, `WEHRE`). The agent must identify and correct them.

**Schema:** `users(id, name, age, city)` — 6 rows
**Goal:** Return name and age of users older than 30 in New York, ordered by name
**Max steps:** 5

### Medium — Fix the Cartesian JOIN
A missing `JOIN` condition (`o.product_id = p.id`) causes a cartesian product, inflating every total by 3×. The agent must rewrite using explicit `INNER JOIN … ON` syntax.

**Schema:** `orders`, `users`, `products` — e-commerce dataset
**Goal:** Correct per-(user, product) total amount spent, ordered by total DESC
**Max steps:** 5

### Hard — Rewrite Correlated Subquery as CTE
A semantically correct but O(N²) query re-executes `AVG(salary)` for every employee row. The agent must rewrite using a `WITH` clause that computes department averages exactly once.

**Schema:** `departments`, `employees` — 9 employees across 3 departments
**Goal:** Employees who earn strictly above their department average, ordered by dept/salary
**Max steps:** 6

> Tasks have **structural penalties**: the hard task requires a `WITH` clause (−0.30 if absent); the medium task requires explicit `JOIN` syntax (−0.20 if absent). This prevents an agent from gaming the score by submitting the broken query verbatim.

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

### Run the baseline inference script
Runs a Claude model as an agent against all tasks and reports scores:
```bash
# Default model (claude-haiku-4-5)
ANTHROPIC_API_KEY=your_key .venv/bin/python baseline.py

# Specific model
ANTHROPIC_API_KEY=your_key .venv/bin/python baseline.py --model claude-opus-4-6

# Single task with verbose output
ANTHROPIC_API_KEY=your_key .venv/bin/python baseline.py --task task_hard_cte --verbose
```

### Run the HTTP server
```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Baseline Results

The following scores were produced by running `claude-haiku-4-5` as the agent against all three tasks with the full AI judge active. These serve as the reproducible baseline for this environment.

| Task | Level | Steps Used | Best Score |
|---|---|---|---|
| Fix the Syntax Errors | easy | 1 | **1.000** |
| Fix the Cartesian JOIN | medium | 1 | **0.900** |
| Rewrite Correlated Subquery as CTE | hard | 1 | **0.950** |
| **Average** | | | **0.950** |

All three tasks were solved (or near-solved) on the first step, demonstrating that:
- The reward pipeline returns meaningful signal immediately
- The environment terminates cleanly when the done threshold (≥ 0.90) is met
- A stronger model or a harder task set would produce more training-relevant trajectories

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
# Start an episode pinned to the hard task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_hard_cte"}'

# Submit a query
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"sql": "WITH dept_avg AS (SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id) SELECT e.name, e.department_id, e.salary FROM employees e JOIN dept_avg d ON e.department_id = d.department_id WHERE e.salary > d.avg_salary ORDER BY e.department_id, e.salary DESC"}'

# List all available tasks
curl http://localhost:8000/tasks
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
├── tasks.py                        # Built-in tasks + thread-safe TaskRegistry
├── judge.py                        # 4-stage grading pipeline (DuckDB + Anthropic)
├── client.py                       # QueryforgeEnv client with task management helpers
├── playbook.py                     # Local test runner (no server required)
├── baseline.py                     # Baseline inference script (Claude as agent)
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Locked dependencies
└── server/
    ├── app.py                      # FastAPI app — core + /tasks REST endpoints
    ├── queryforge_environment.py   # Environment class (reset, step, state)
    ├── Dockerfile                  # Container image
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
docker build -t queryforge:latest -f server/Dockerfile .
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
