"""
QueryForge Client Playbook
──────────────────────────
Tests the environment through the HTTP server using the QueryforgeEnv client.

Requires the server to be running first:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Then run:
    python playbook.py

If ANTHROPIC_API_KEY is set, Stage 4 AI scoring is live.
If not set, the judge falls back to deterministic scoring (capped at 0.80).
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import QueryforgeEnv
from models import SQLAction, TaskSpec
from tasks import REGISTRY, task_from_dict

BASE_URL = "https://prithvigg-queryforge.hf.space"

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

def _print_result(result, show_description=False):
    obs = result.observation
    if show_description and obs.task_description:
        print()
        print(textwrap.indent(obs.task_description, "  "))
        print()
    if obs.feedback and obs.feedback != "New task loaded. Submit your fixed/optimised SQL query.":
        print(f"  Syntax valid    : {obs.syntax_valid}")
        print(f"  Execution OK    : {obs.execution_success}")
        if obs.execution_error:
            print(f"  Execution error : {obs.execution_error[:100]}")
        print(f"  Rows returned   : {obs.rows_returned}")
        print(f"  Score           : {_score_bar(result.reward or 0.0)}")
        print(f"  Best this ep.   : {_score_bar(obs.best_score)}")
        fb = obs.feedback[:250] + ("…" if len(obs.feedback) > 250 else "")
        print(f"  Feedback        : {fb}")
        if obs.hint:
            print(f"  Hint            : {obs.hint[:120]}")

def _attempt(client, label: str, sql: str):
    print(f"\n  ── Attempt: {label}")
    print(f"     SQL: {sql[:100]}{'…' if len(sql) > 100 else ''}")
    result = client.step(SQLAction(sql=sql))
    _print_result(result)
    return result


# ── Task runners ──────────────────────────────────────────────────────────────

def run_easy(client):
    _section("TASK 1 · EASY — Fix Syntax Errors")
    result = client.reset(task_id="task_easy_syntax")
    obs = result.observation
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    _print_result(result, show_description=True)

    _attempt(client, "still broken",
             "SELEC name, age FORM users WEHRE age > 30")

    _attempt(client, "one keyword fixed",
             "SELECT name, age FORM users WEHRE age > 30")

    _attempt(client, "all keywords fixed, no filter",
             "SELECT name, age FROM users WHERE age > 30")

    result = _attempt(client, "correct solution",
                      "SELECT name, age FROM users "
                      "WHERE age > 30 AND city = 'New York' "
                      "ORDER BY name ASC")

    print(f"\n  Episode done: {result.done}  |  Best score: {result.observation.best_score:.2f}")


def run_medium(client):
    _section("TASK 2 · MEDIUM — Fix the Cartesian JOIN")
    result = client.reset(task_id="task_medium_join")
    obs = result.observation
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    _print_result(result, show_description=True)

    _attempt(client, "broken verbatim (cartesian product)",
             "SELECT u.name, p.title, SUM(o.amount) AS total_spent "
             "FROM orders o, users u, products p "
             "WHERE o.user_id = u.id "
             "GROUP BY u.name, p.title "
             "ORDER BY total_spent DESC")

    _attempt(client, "comma-join with product condition (no explicit JOIN)",
             "SELECT u.name, p.title, SUM(o.amount) AS total_spent "
             "FROM orders o, users u, products p "
             "WHERE o.user_id = u.id AND o.product_id = p.id "
             "GROUP BY u.name, p.title "
             "ORDER BY total_spent DESC")

    result = _attempt(client, "correct INNER JOINs",
                      "SELECT u.name, p.title, SUM(o.amount) AS total_spent\n"
                      "FROM orders o\n"
                      "INNER JOIN users    u ON o.user_id    = u.id\n"
                      "INNER JOIN products p ON o.product_id = p.id\n"
                      "GROUP BY u.name, p.title\n"
                      "ORDER BY total_spent DESC")

    print(f"\n  Episode done: {result.done}  |  Best score: {result.observation.best_score:.2f}")


def run_hard(client):
    _section("TASK 3 · HARD — Rewrite Correlated Subquery as CTE")
    result = client.reset(task_id="task_hard_cte")
    obs = result.observation
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    _print_result(result, show_description=True)

    _attempt(client, "broken verbatim (no CTE)",
             "SELECT e.name, e.department_id, e.salary\n"
             "FROM employees e\n"
             "WHERE e.salary > (\n"
             "    SELECT AVG(e2.salary) FROM employees e2\n"
             "    WHERE e2.department_id = e.department_id\n"
             ")\n"
             "ORDER BY e.department_id, e.salary DESC")

    _attempt(client, "halfway — CTE defined but wrong join",
             "WITH dept_avg AS (\n"
             "    SELECT department_id, AVG(salary) AS avg_salary\n"
             "    FROM employees GROUP BY department_id\n"
             ")\n"
             "SELECT e.name, e.department_id, e.salary\n"
             "FROM employees e, dept_avg d\n"
             "WHERE e.salary > d.avg_salary\n"
             "ORDER BY e.department_id, e.salary DESC")

    result = _attempt(client, "correct CTE with proper JOIN",
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

    print(f"\n  Episode done: {result.done}  |  Best score: {result.observation.best_score:.2f}")


def run_custom(client):
    _section("TASK 4 · CUSTOM — NULL Handling in Aggregation")

    # Register a brand-new task at runtime via the REST API
    client.register_task(TaskSpec(
        id="custom_null_avg",
        level="custom",
        title="Handle NULLs in Aggregation",
        description="""\
TASK: The query below skips NULL scores, making the class average look higher.
Fix it so NULL scores are treated as 0.

SCHEMA:
  students(id INTEGER, name VARCHAR, score INTEGER)

BROKEN QUERY:
  SELECT AVG(score) AS avg_score FROM students

ERROR:
  NULL values are silently excluded by AVG(), inflating the result.

GOAL: Return a single row with avg_score that treats NULL as 0.
      Expected result: avg_score = 65.0""",
        schema_ddl="""\
CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER);
INSERT INTO students VALUES
    (1, 'Alice', 90),
    (2, 'Bob',   NULL),
    (3, 'Carol', 80),
    (4, 'Dave',  NULL),
    (5, 'Eve',   70),
    (6, 'Frank', 50);
""",
        broken_query="SELECT AVG(score) AS avg_score FROM students",
        error_message="NULL scores are silently skipped by AVG().",
        hint="Wrap score with COALESCE(score, 0) before averaging.",
        expected_rows=[{"avg_score": 65.0}],
        solution_query="SELECT AVG(COALESCE(score, 0)) AS avg_score FROM students",
        test_description="AVG treats NULL as 0 → 65.0",
        max_steps=4,
    ))

    result = client.reset(task_id="custom_null_avg")
    obs = result.observation
    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]")
    _print_result(result, show_description=True)

    _attempt(client, "broken (NULL excluded)",
             "SELECT AVG(score) AS avg_score FROM students")

    result = _attempt(client, "correct (COALESCE)",
                      "SELECT AVG(COALESCE(score, 0)) AS avg_score FROM students")

    print(f"\n  Episode done: {result.done}  |  Best score: {result.observation.best_score:.2f}")

    # Clean up
    client.delete_task("custom_null_avg")
    print("  Custom task unregistered from registry.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ai_key = os.environ.get("ANTHROPIC_API_KEY")

    _hr("═")
    print("  QueryForge — Client Playbook")
    print(f"  Server : {BASE_URL}")
    print(f"  AI judge : {'LIVE (ANTHROPIC_API_KEY set)' if ai_key else 'OFFLINE (fallback to deterministic, max 0.80)'}")
    _hr("═")

    with QueryforgeEnv(base_url=BASE_URL).sync() as client:
        # run_easy(client)
        run_medium(client)
        run_hard(client)
        # run_custom(client)

    _section("DONE")
    print("  All 4 tasks completed.\n")
