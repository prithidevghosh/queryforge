"""
SQL task definitions and runtime task registry for the QueryForge environment.

Built-in tasks:
  easy   — fix three misspelled SQL keywords
  medium — fix a cartesian JOIN producing wrong results
  hard   — rewrite a correlated subquery as a CTE

Custom tasks can be added at runtime via REGISTRY.register() or
POST /tasks on the running server.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single test case: expected output rows for correctness grading."""

    description: str
    expected_rows: List[Dict[str, Any]]
    order_by: Optional[str] = None  # comma-separated columns to sort by


@dataclass
class SQLTask:
    """Full definition of one SQL challenge."""

    id: str
    level: str          # "easy" | "medium" | "hard" | "custom"
    title: str
    description: str
    schema_ddl: str     # DDL + seed INSERT statements for DuckDB
    broken_query: str   # broken/slow query the agent must fix
    error_message: str  # error or performance warning shown to agent
    hint: str
    test_cases: List[TestCase]
    solution_query: str # reference solution used by the AI judge
    max_steps: int = 5


# ── Built-in tasks ────────────────────────────────────────────────────────────

_TASK_EASY = SQLTask(
    id="task_easy_syntax",
    level="easy",
    title="Fix the Syntax Errors",
    description="""\
TASK: Fix the syntax errors in the query below so it runs correctly.

SCHEMA:
  users(id INTEGER, name VARCHAR, age INTEGER, city VARCHAR)

BROKEN QUERY:
  SELEC name, age FORM users WEHRE age > 30 AND city = 'New York'

ERROR:
  Parser Error: syntax error at or near "SELEC"

GOAL: Return a valid SQL query that retrieves `name` and `age`
of users who are older than 30 AND live in New York.
Order by name ASC.""",
    schema_ddl="""\
CREATE TABLE users (
    id   INTEGER,
    name VARCHAR,
    age  INTEGER,
    city VARCHAR
);
INSERT INTO users VALUES
    (1, 'Alice',  35, 'New York'),
    (2, 'Bob',    28, 'New York'),
    (3, 'Carol',  42, 'Chicago'),
    (4, 'Dave',   31, 'New York'),
    (5, 'Eve',    25, 'New York'),
    (6, 'Frank',  38, 'New York');
""",
    broken_query="SELEC name, age FORM users WEHRE age > 30 AND city = 'New York'",
    error_message='Parser Error: syntax error at or near "SELEC"',
    hint="Three SQL keywords are misspelled: SELEC → SELECT, FORM → FROM, WEHRE → WHERE.",
    test_cases=[
        TestCase(
            description="Users over 30 living in New York, ordered by name",
            expected_rows=[
                {"name": "Alice", "age": 35},
                {"name": "Dave",  "age": 31},
                {"name": "Frank", "age": 38},
            ],
            order_by="name",
        )
    ],
    solution_query=(
        "SELECT name, age FROM users "
        "WHERE age > 30 AND city = 'New York' "
        "ORDER BY name ASC"
    ),
)

_TASK_MEDIUM = SQLTask(
    id="task_medium_join",
    level="medium",
    title="Fix the Cartesian JOIN",
    description="""\
TASK: The query below produces wildly inflated totals because a JOIN condition
is missing, creating a cartesian product with the `products` table. Fix it.

SCHEMAS:
  users(id INTEGER, name VARCHAR, age INTEGER)
  products(id INTEGER, title VARCHAR, price DECIMAL)
  orders(id INTEGER, user_id INTEGER, product_id INTEGER, amount DECIMAL)

BROKEN QUERY:
  SELECT u.name, p.title, SUM(o.amount) AS total_spent
  FROM orders o, users u, products p
  WHERE o.user_id = u.id
  GROUP BY u.name, p.title
  ORDER BY total_spent DESC

PROBLEM:
  Missing join condition `o.product_id = p.id`.
  Every order row is multiplied by ALL products, inflating every total by 3×.

GOAL: Rewrite using explicit INNER JOIN … ON syntax with all correct join
conditions. Return user name, product title, and true total amount spent per
(user, product) pair, ordered by total_spent DESC.""",
    schema_ddl="""\
CREATE TABLE users    (id INTEGER, name VARCHAR, age INTEGER);
CREATE TABLE products (id INTEGER, title VARCHAR, price DECIMAL);
CREATE TABLE orders   (id INTEGER, user_id INTEGER, product_id INTEGER, amount DECIMAL);

INSERT INTO users    VALUES (1,'Alice',30),(2,'Bob',25),(3,'Carol',35);
INSERT INTO products VALUES (1,'Laptop',999.99),(2,'Phone',599.99),(3,'Tablet',399.99);
INSERT INTO orders   VALUES
    (1,1,1,999.99),(2,1,2,599.99),
    (3,2,1,999.99),(4,2,3,399.99),
    (5,3,2,599.99),(6,3,1,999.99);
""",
    broken_query="""\
SELECT u.name, p.title, SUM(o.amount) AS total_spent
FROM orders o, users u, products p
WHERE o.user_id = u.id
GROUP BY u.name, p.title
ORDER BY total_spent DESC""",
    error_message=(
        "Query runs but produces WRONG results: totals are 3× too high "
        "because every order is joined to every product (cartesian product)."
    ),
    hint=(
        "Use INNER JOIN … ON for every table. "
        "You need both: o.user_id = u.id  AND  o.product_id = p.id."
    ),
    test_cases=[
        TestCase(
            description="Correct per-(user, product) totals",
            expected_rows=[
                {"name": "Alice", "title": "Laptop", "total_spent": 999.99},
                {"name": "Alice", "title": "Phone",  "total_spent": 599.99},
                {"name": "Bob",   "title": "Laptop", "total_spent": 999.99},
                {"name": "Bob",   "title": "Tablet", "total_spent": 399.99},
                {"name": "Carol", "title": "Laptop", "total_spent": 999.99},
                {"name": "Carol", "title": "Phone",  "total_spent": 599.99},
            ],
            order_by="name,title",
        )
    ],
    solution_query="""\
SELECT u.name, p.title, SUM(o.amount) AS total_spent
FROM orders o
INNER JOIN users    u ON o.user_id    = u.id
INNER JOIN products p ON o.product_id = p.id
GROUP BY u.name, p.title
ORDER BY total_spent DESC""",
)

_TASK_HARD = SQLTask(
    id="task_hard_cte",
    level="hard",
    title="Rewrite Correlated Subquery as CTE",
    description="""\
TASK: The query below is semantically correct but executes the inner AVG(salary)
once per employee row — O(N) full scans. Rewrite it using a WITH (CTE) so the
department averages are computed exactly once.

SCHEMAS:
  departments(id INTEGER, dept_name VARCHAR)
  employees(id INTEGER, name VARCHAR, department_id INTEGER, salary DECIMAL)

SLOW QUERY:
  SELECT e.name, e.department_id, e.salary
  FROM employees e
  WHERE e.salary > (
      SELECT AVG(e2.salary)
      FROM employees e2
      WHERE e2.department_id = e.department_id
  )
  ORDER BY e.department_id, e.salary DESC

PERFORMANCE WARNING:
  For 1 M employees the inner subquery executes 1 M times.
  DuckDB's EXPLAIN shows: 'FILTER ... (subquery)' with nested loop.

GOAL: Rewrite using a CTE that computes per-department average salary once,
then join it to employees and filter. The result must be identical:
employees who earn strictly above their own department's average salary,
ordered by department_id ASC, salary DESC.""",
    schema_ddl="""\
CREATE TABLE departments (id INTEGER, dept_name VARCHAR);
CREATE TABLE employees   (id INTEGER, name VARCHAR, department_id INTEGER, salary DECIMAL);

INSERT INTO departments VALUES (1,'Engineering'),(2,'Marketing'),(3,'Sales');
INSERT INTO employees VALUES
    (1,'Alice', 1, 95000),(2,'Bob',   1, 75000),(3,'Carol', 1, 85000),
    (4,'Dave',  2, 65000),(5,'Eve',   2, 70000),(6,'Frank', 2, 60000),
    (7,'Grace', 3, 55000),(8,'Hank',  3, 72000),(9,'Iris',  3, 58000);
""",
    broken_query="""\
SELECT e.name, e.department_id, e.salary
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
)
ORDER BY e.department_id, e.salary DESC""",
    error_message=(
        "PERFORMANCE: Correlated subquery re-executes AVG() for every row. "
        "On large tables this is O(N²). Rewrite as a CTE for O(N) execution."
    ),
    hint=(
        "WITH dept_avg AS (SELECT department_id, AVG(salary) AS avg_salary "
        "FROM employees GROUP BY department_id) — then JOIN employees to dept_avg "
        "and filter WHERE e.salary > d.avg_salary."
    ),
    test_cases=[
        TestCase(
            description="Employees strictly above their department's average salary",
            expected_rows=[
                {"name": "Alice", "department_id": 1, "salary": 95000.0},
                {"name": "Eve",   "department_id": 2, "salary": 70000.0},
                {"name": "Hank",  "department_id": 3, "salary": 72000.0},
            ],
            order_by="department_id,name",
        )
    ],
    solution_query="""\
WITH dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT e.name, e.department_id, e.salary
FROM employees e
JOIN dept_avg d ON e.department_id = d.department_id
WHERE e.salary > d.avg_salary
ORDER BY e.department_id, e.salary DESC""",
    max_steps=6,
)


# ── Task Registry ─────────────────────────────────────────────────────────────

class TaskRegistry:
    """
    Thread-safe registry of SQL tasks, shared across all environment sessions.

    Built-in tasks (easy / medium / hard) are always present and cannot be removed.
    Custom tasks can be added via register(), load_from_json(), or POST /tasks.
    """

    _BUILTIN_IDS: frozenset = frozenset(
        ["task_easy_syntax", "task_medium_join", "task_hard_cte"]
    )

    def __init__(self, initial_tasks: List[SQLTask]) -> None:
        self._lock = Lock()
        # Insertion-ordered dict preserves cycling order
        self._tasks: Dict[str, SQLTask] = {t.id: t for t in initial_tasks}
        self._cycle_index: int = 0

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def register(self, task: SQLTask) -> None:
        """Add or replace a task. Replaces silently if the ID already exists."""
        with self._lock:
            self._tasks[task.id] = task

    def unregister(self, task_id: str) -> None:
        """
        Remove a custom task.
        Raises ValueError for built-in tasks, KeyError if not found.
        """
        if task_id in self._BUILTIN_IDS:
            raise ValueError(f"Built-in task '{task_id}' cannot be removed.")
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(task_id)
            del self._tasks[task_id]

    def get(self, task_id: str) -> SQLTask:
        """Return a task by ID. Raises KeyError with available IDs if not found."""
        with self._lock:
            if task_id not in self._tasks:
                available = ", ".join(self._tasks.keys())
                raise KeyError(
                    f"Task '{task_id}' not found. "
                    f"Available: {available}"
                )
            return self._tasks[task_id]

    def list_all(self) -> List[SQLTask]:
        """Return all registered tasks in insertion order."""
        with self._lock:
            return list(self._tasks.values())

    def ids(self) -> List[str]:
        """Return all task IDs in insertion order."""
        with self._lock:
            return list(self._tasks.keys())

    # ── Cycling ───────────────────────────────────────────────────────────────

    def cycle_next(self) -> SQLTask:
        """Return the next task in round-robin order (wraps at end)."""
        with self._lock:
            tasks = list(self._tasks.values())
            task = tasks[self._cycle_index % len(tasks)]
            self._cycle_index += 1
            return task

    # ── Bulk loading ──────────────────────────────────────────────────────────

    def load_from_json(self, path: str) -> int:
        """
        Load tasks from a JSON file (list of task spec objects).
        Returns the number of tasks loaded.

        Minimal required fields per task:
          id, schema_ddl, expected_rows

        Example file::

            [
              {
                "id": "my_null_task",
                "level": "medium",
                "title": "Handle NULLs in aggregation",
                "schema_ddl": "CREATE TABLE ...; INSERT ...",
                "broken_query": "SELECT AVG(score) FROM ...",
                "expected_rows": [{"avg_score": 72.5}],
                "hint": "Use COALESCE to handle NULL scores."
              }
            ]
        """
        raw = json.loads(Path(path).read_text())
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            self.register(task_from_dict(item))
        return len(raw)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._tasks


# ── Conversion helper ─────────────────────────────────────────────────────────

def task_from_dict(d: Dict[str, Any]) -> SQLTask:
    """
    Construct an SQLTask from a plain dict (JSON payload or loaded file).

    Required keys : id, schema_ddl, expected_rows
    Optional keys : level, title, description, broken_query, error_message,
                    hint, order_by, solution_query, test_description, max_steps
    """
    return SQLTask(
        id=d["id"],
        level=d.get("level", "custom"),
        title=d.get("title", d["id"]),
        description=d.get("description", ""),
        schema_ddl=d["schema_ddl"],
        broken_query=d.get("broken_query", ""),
        error_message=d.get("error_message", ""),
        hint=d.get("hint", ""),
        test_cases=[
            TestCase(
                description=d.get("test_description", "Custom test case"),
                expected_rows=d["expected_rows"],
                order_by=d.get("order_by"),
            )
        ],
        solution_query=d.get("solution_query", ""),
        max_steps=d.get("max_steps", 5),
    )


# ── Global singleton ──────────────────────────────────────────────────────────

REGISTRY = TaskRegistry([_TASK_EASY, _TASK_MEDIUM, _TASK_HARD])

# Backwards-compat: snapshot of the three built-in tasks at import time
TASKS: List[SQLTask] = [_TASK_EASY, _TASK_MEDIUM, _TASK_HARD]
TASK_BY_ID: Dict[str, SQLTask] = {t.id: t for t in TASKS}
