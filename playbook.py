"""
QueryForge Local Playbook
─────────────────────────
Tests the environment directly (no HTTP server needed).

Run from the queryforge directory:
    .venv/bin/python playbook.py

If ANTHROPIC_API_KEY is set, Stage 4 AI scoring is live.
If not set, the judge falls back to deterministic scoring (capped at 0.80).
"""

import os
import sys
import textwrap

# Make imports work whether run directly or as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.queryforge_environment import QueryforgeEnvironment
from models import SQLAction
from tasks import REGISTRY, task_from_dict

# ── Formatting helpers ────────────────────────────────────────────────────────

def _hr(char="═", width=70):
    print(char * width)

def _section(title):
    print()
    _hr()
    print(f"  {title}")
    _hr()

def _score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"

def _print_obs(obs, show_description=False):
    if show_description:
        print()
        print(textwrap.indent(obs.task_description, "  "))
        print()
    if obs.feedback and obs.feedback != "New task loaded. Submit your fixed/optimised SQL query.":
        print(f"  Syntax valid    : {obs.syntax_valid}")
        print(f"  Execution OK    : {obs.execution_success}")
        if obs.execution_error:
            print(f"  Execution error : {obs.execution_error[:100]}")
        print(f"  Rows returned   : {obs.rows_returned}")
        print(f"  Score           : {_score_bar(obs.reward or 0.0)}")
        print(f"  Best this ep.   : {_score_bar(obs.best_score)}")
        # Print just the first 200 chars of feedback to keep output clean
        fb = obs.feedback[:250] + ("…" if len(obs.feedback) > 250 else "")
        print(f"  Feedback        : {fb}")
        if obs.hint:
            print(f"  Hint            : {obs.hint[:120]}")

def _attempt(env, label: str, sql: str):
    print(f"\n  ── Attempt: {label}")
    print(f"     SQL: {sql[:100]}{'…' if len(sql) > 100 else ''}")
    obs = env.step(SQLAction(sql=sql))
    _print_obs(obs)
    return obs


# ── Task runners ──────────────────────────────────────────────────────────────

def run_easy(env):
    _section("TASK 1 · EASY — Fix Syntax Errors")
    env._task_index = 0  # pin to easy
    obs = env.reset()
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    print(f"  Steps: up to {5}")
    _print_obs(obs, show_description=True)

    _attempt(env, "still broken",
             "SELEC name, age FORM users WEHRE age > 30")

    _attempt(env, "one keyword fixed",
             "SELECT name, age FORM users WEHRE age > 30")

    _attempt(env, "all keywords fixed, no filter",
             "SELECT name, age FROM users WHERE age > 30")

    obs = _attempt(env, "correct solution",
                   "SELECT name, age FROM users "
                   "WHERE age > 30 AND city = 'New York' "
                   "ORDER BY name ASC")

    print(f"\n  Episode done: {obs.done}  |  Best score: {obs.best_score:.2f}")


def run_medium(env):
    _section("TASK 2 · MEDIUM — Fix the Cartesian JOIN")
    env._task_index = 1  # pin to medium
    obs = env.reset()
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    print(f"  Steps: up to {5}")
    _print_obs(obs, show_description=True)

    _attempt(env, "broken verbatim (cartesian product)",
             "SELECT u.name, p.title, SUM(o.amount) AS total_spent "
             "FROM orders o, users u, products p "
             "WHERE o.user_id = u.id "
             "GROUP BY u.name, p.title "
             "ORDER BY total_spent DESC")

    _attempt(env, "comma-join but missing product condition",
             "SELECT u.name, p.title, SUM(o.amount) AS total_spent "
             "FROM orders o, users u, products p "
             "WHERE o.user_id = u.id AND o.product_id = p.id "
             "GROUP BY u.name, p.title "
             "ORDER BY total_spent DESC")

    obs = _attempt(env, "correct INNER JOINs",
                   "SELECT u.name, p.title, SUM(o.amount) AS total_spent\n"
                   "FROM orders o\n"
                   "INNER JOIN users    u ON o.user_id    = u.id\n"
                   "INNER JOIN products p ON o.product_id = p.id\n"
                   "GROUP BY u.name, p.title\n"
                   "ORDER BY total_spent DESC")

    print(f"\n  Episode done: {obs.done}  |  Best score: {obs.best_score:.2f}")


def run_hard(env):
    _section("TASK 3 · HARD — Rewrite Correlated Subquery as CTE")
    env._task_index = 2  # pin to hard
    obs = env.reset()
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    print(f"  Steps: up to {6}")
    _print_obs(obs, show_description=True)

    _attempt(env, "broken verbatim (no CTE — penalised even though rows match)",
             "SELECT e.name, e.department_id, e.salary\n"
             "FROM employees e\n"
             "WHERE e.salary > (\n"
             "    SELECT AVG(e2.salary) FROM employees e2\n"
             "    WHERE e2.department_id = e.department_id\n"
             ")\n"
             "ORDER BY e.department_id, e.salary DESC")

    _attempt(env, "halfway — CTE defined but wrong join",
             "WITH dept_avg AS (\n"
             "    SELECT department_id, AVG(salary) AS avg_salary\n"
             "    FROM employees GROUP BY department_id\n"
             ")\n"
             "SELECT e.name, e.department_id, e.salary\n"
             "FROM employees e, dept_avg d\n"
             "WHERE e.salary > d.avg_salary\n"
             "ORDER BY e.department_id, e.salary DESC")

    obs = _attempt(env, "correct CTE with proper JOIN",
                   "WITH dept_avg AS (\n"
                   "    SELECT department_id, AVG(salary) AS avg_salary\n"
                   "    FROM employees\n"
                   "    GROUP BY department_id\n"
                   ")\n"
                   "SELECT e.name, e.department_id, e.salary\n"
                   "FROM employees e\n"
                   "JOIN dept_avg d ON e.department_id = d.department_id\n"
                   "WHERE e.salary > d.avg_salary\n"
                   "ORDER BY e.department_id, e.salary DESC")

    print(f"\n  Episode done: {obs.done}  |  Best score: {obs.best_score:.2f}")


# ── Custom task demo ──────────────────────────────────────────────────────────

def run_custom(env):
    _section("TASK 4 · CUSTOM — NULL Handling in Aggregation")

    # Register a brand-new task at runtime
    custom_task = task_from_dict({
        "id": "custom_null_avg",
        "level": "custom",
        "title": "Handle NULLs in Aggregation",
        "description": """\
TASK: The query below skips NULL scores, making the class average look higher.
Fix it so NULL scores are treated as 0.

SCHEMA:
  students(id INTEGER, name VARCHAR, score INTEGER)

BROKEN QUERY:
  SELECT AVG(score) AS avg_score FROM students

ERROR:
  NULL values are silently excluded by AVG(), inflating the result.

GOAL: Return a single row with avg_score that treats NULL as 0.
      Expected result: avg_score = 72.5""",
        "schema_ddl": """\
CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER);
INSERT INTO students VALUES
    (1, 'Alice', 90),
    (2, 'Bob',   NULL),
    (3, 'Carol', 80),
    (4, 'Dave',  NULL),
    (5, 'Eve',   70),
    (6, 'Frank', 50);
""",
        "broken_query": "SELECT AVG(score) AS avg_score FROM students",
        "error_message": "NULL scores are silently skipped by AVG().",
        "hint": "Wrap score with COALESCE(score, 0) before averaging.",
        "expected_rows": [{"avg_score": 65.0}],
        "solution_query": "SELECT AVG(COALESCE(score, 0)) AS avg_score FROM students",
        "test_description": "AVG treats NULL as 0 → 65.0",
        "max_steps": 4,
    })
    REGISTRY.register(custom_task)

    obs = env.reset(task_id="custom_null_avg")
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    print(f"  Steps: up to {custom_task.max_steps}")
    _print_obs(obs, show_description=True)

    _attempt(env, "broken (NULL excluded)",
             "SELECT AVG(score) AS avg_score FROM students")

    obs = _attempt(env, "correct (COALESCE)",
                   "SELECT AVG(COALESCE(score, 0)) AS avg_score FROM students")

    print(f"\n  Episode done: {obs.done}  |  Best score: {obs.best_score:.2f}")

    # Clean up: remove custom task from registry
    REGISTRY.unregister("custom_null_avg")
    print("  Custom task unregistered from registry.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ai_key = os.environ.get("ANTHROPIC_API_KEY")

    _hr("═")
    print("  QueryForge — Local Playbook")
    print(f"  AI judge : {'LIVE (ANTHROPIC_API_KEY set)' if ai_key else 'OFFLINE (fallback to deterministic, max 0.80)'}")
    _hr("═")

    # Create a fresh env for each task so cycling order never matters
    run_easy(QueryforgeEnvironment())
    run_medium(QueryforgeEnvironment())
    run_hard(QueryforgeEnvironment())
    run_custom(QueryforgeEnvironment())

    _section("DONE")
    print("  All 4 tasks completed.\n")
