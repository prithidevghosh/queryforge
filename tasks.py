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


# ── Expert tasks ──────────────────────────────────────────────────────────────

_TASK_EXPERT_RANK = SQLTask(
    id="task_expert_rank",
    level="expert",
    title="Fix the Tie-Breaking Window Function",
    description="""\
TASK: The query below finds the top-earning sales rep per region, but it
silently drops reps who are tied for first place. Fix it so ALL reps
tied at rank 1 are returned.

SCHEMA:
  sales_reps(id INTEGER, name VARCHAR, region VARCHAR, revenue DECIMAL)

BROKEN QUERY:
  SELECT name, region, revenue
  FROM (
      SELECT name, region, revenue,
             ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) AS rn
      FROM sales_reps
  ) ranked
  WHERE rn = 1
  ORDER BY region, name

PROBLEM:
  ROW_NUMBER() assigns unique sequential numbers even for tied revenue values.
  When two reps share the top revenue in a region, ROW_NUMBER arbitrarily
  picks one and discards the other.

GOAL: Return ALL reps whose revenue is the highest in their region.
     Use RANK() or DENSE_RANK() instead of ROW_NUMBER().
     Order by region ASC, name ASC.""",
    schema_ddl="""\
CREATE TABLE sales_reps (id INTEGER, name VARCHAR, region VARCHAR, revenue DECIMAL);
INSERT INTO sales_reps VALUES
    (1, 'Alice', 'North', 95000),
    (2, 'Bob',   'North', 87000),
    (3, 'Carol', 'North', 95000),
    (4, 'Dave',  'South', 88000),
    (5, 'Eve',   'South', 88000),
    (6, 'Frank', 'South', 75000);
""",
    broken_query="""\
SELECT name, region, revenue
FROM (
    SELECT name, region, revenue,
           ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) AS rn
    FROM sales_reps
) ranked
WHERE rn = 1
ORDER BY region, name""",
    error_message=(
        "Query runs but returns only 2 rows — one per region. "
        "Tied reps at the top are silently dropped by ROW_NUMBER()."
    ),
    hint="Replace ROW_NUMBER() with RANK() or DENSE_RANK(). Both include all tied rows.",
    test_cases=[
        TestCase(
            description="All reps tied at rank 1 per region",
            expected_rows=[
                {"name": "Alice", "region": "North", "revenue": 95000.0},
                {"name": "Carol", "region": "North", "revenue": 95000.0},
                {"name": "Dave",  "region": "South", "revenue": 88000.0},
                {"name": "Eve",   "region": "South", "revenue": 88000.0},
            ],
            order_by="region,name",
        )
    ],
    solution_query="""\
SELECT name, region, revenue
FROM (
    SELECT name, region, revenue,
           RANK() OVER (PARTITION BY region ORDER BY revenue DESC) AS rk
    FROM sales_reps
) ranked
WHERE rk = 1
ORDER BY region, name""",
    max_steps=6,
)


_TASK_EXPERT_RECURSIVE = SQLTask(
    id="task_expert_recursive",
    level="expert",
    title="Traverse Org Chart with Recursive CTE",
    description="""\
TASK: The query below attempts to find all subordinates of the VP of Engineering
(id=3) using a two-level CTE expansion. It misses employees more than two levels
deep. Rewrite it using a recursive CTE that traverses all levels.

SCHEMA:
  employees(id INTEGER, name VARCHAR, manager_id INTEGER)

DATA (partial):
  VP Eng (id=3) → Lead A (id=5), Lead B (id=6)
  Lead A (id=5) → Dev 1 (id=8), Dev 2 (id=9)
  Lead B (id=6) → Dev 3 (id=10), Dev 4 (id=11)
  Dev 1 (id=8)  → Junior 1 (id=13), Junior 2 (id=14)

BROKEN QUERY:
  WITH direct AS (
      SELECT id, name, manager_id FROM employees WHERE manager_id = 3
  ),
  level2 AS (
      SELECT e.id, e.name, e.manager_id
      FROM employees e
      INNER JOIN direct d ON e.manager_id = d.id
  )
  SELECT id, name, manager_id FROM direct
  UNION ALL
  SELECT id, name, manager_id FROM level2
  ORDER BY id

PROBLEM:
  This hardcoded two-level expansion returns 6 rows but misses Junior 1 (id=13)
  and Junior 2 (id=14), who report to Dev 1 — three levels below VP Eng.
  Adding a level3 CTE would help for now but still break if the tree grows deeper.

GOAL: Use WITH RECURSIVE to return ALL 8 subordinates of VP Eng (id=3)
     at any depth. Return id, name, manager_id columns, ordered by id ASC.""",
    schema_ddl="""\
CREATE TABLE employees (id INTEGER, name VARCHAR, manager_id INTEGER);
INSERT INTO employees VALUES
    (1,  'CEO',      NULL),
    (2,  'CFO',      1),
    (3,  'VP Eng',   1),
    (4,  'VP Sales', 1),
    (5,  'Lead A',   3),
    (6,  'Lead B',   3),
    (7,  'Sales Mgr',4),
    (8,  'Dev 1',    5),
    (9,  'Dev 2',    5),
    (10, 'Dev 3',    6),
    (11, 'Dev 4',    6),
    (12, 'Sales Rep',7),
    (13, 'Junior 1', 8),
    (14, 'Junior 2', 8);
""",
    broken_query="""\
WITH direct AS (
    SELECT id, name, manager_id FROM employees WHERE manager_id = 3
),
level2 AS (
    SELECT e.id, e.name, e.manager_id
    FROM employees e
    INNER JOIN direct d ON e.manager_id = d.id
)
SELECT id, name, manager_id FROM direct
UNION ALL
SELECT id, name, manager_id FROM level2
ORDER BY id""",
    error_message=(
        "Query returns only 6 rows — two levels under VP Eng. "
        "Junior 1 (id=13) and Junior 2 (id=14) who report to Dev 1 are missing. "
        "A hardcoded level3 CTE would fix this instance but not scale to deeper trees."
    ),
    hint="Use WITH RECURSIVE. Start from manager_id = 3, then JOIN employees to the CTE itself on manager_id = cte.id.",
    test_cases=[
        TestCase(
            description="All 8 subordinates of VP Eng at any depth",
            expected_rows=[
                {"id": 5,  "name": "Lead A",   "manager_id": 3},
                {"id": 6,  "name": "Lead B",   "manager_id": 3},
                {"id": 8,  "name": "Dev 1",    "manager_id": 5},
                {"id": 9,  "name": "Dev 2",    "manager_id": 5},
                {"id": 10, "name": "Dev 3",    "manager_id": 6},
                {"id": 11, "name": "Dev 4",    "manager_id": 6},
                {"id": 13, "name": "Junior 1", "manager_id": 8},
                {"id": 14, "name": "Junior 2", "manager_id": 8},
            ],
            order_by="id",
        )
    ],
    solution_query="""\
WITH RECURSIVE subordinates AS (
    SELECT id, name, manager_id
    FROM employees
    WHERE manager_id = 3
    UNION ALL
    SELECT e.id, e.name, e.manager_id
    FROM employees e
    INNER JOIN subordinates s ON e.manager_id = s.id
)
SELECT id, name, manager_id
FROM subordinates
ORDER BY id""",
    max_steps=7,
)


_TASK_EXPERT_WINDOW = SQLTask(
    id="task_expert_window",
    level="expert",
    title="Fix Two Broken Window Functions: Running Total and Revenue Rank",
    description="""\
TASK: The query below computes a cumulative running total and a
within-region revenue rank for each quarter, but BOTH window functions
are broken — neither has a PARTITION BY, so they treat all rows as one
giant partition instead of computing independently per region.

SCHEMA:
  quarterly_sales(region VARCHAR, quarter INTEGER, revenue DECIMAL)

BROKEN QUERY:
  SELECT region, quarter, revenue,
         SUM(revenue) OVER (ORDER BY region, quarter)        AS running_total,
         RANK()       OVER (ORDER BY revenue DESC)            AS revenue_rank
  FROM quarterly_sales
  ORDER BY region, quarter

PROBLEM:
  - running_total accumulates across both regions: West's Q1 shows 65000
    (continuing from East's Q4) instead of resetting to 11000.
  - revenue_rank ranks revenue across ALL regions globally, so East Q4 (20000)
    and West Q3 (16000) compete directly instead of being ranked within their
    own region.

GOAL: Fix BOTH window functions so they operate independently per region.
     - running_total must reset to 0 at the start of each region (ORDER BY quarter).
     - revenue_rank must rank revenue within each region (ORDER BY revenue DESC).
     Both OVER clauses need PARTITION BY region, but with different ORDER BY columns.
     Final output: ORDER BY region ASC, quarter ASC.""",
    schema_ddl="""\
CREATE TABLE quarterly_sales (region VARCHAR, quarter INTEGER, revenue DECIMAL);
INSERT INTO quarterly_sales VALUES
    ('East', 1, 15000),
    ('East', 2, 18000),
    ('East', 3, 12000),
    ('East', 4, 20000),
    ('West', 1, 11000),
    ('West', 2, 14000),
    ('West', 3, 16000),
    ('West', 4, 13000);
""",
    broken_query="""\
SELECT region, quarter, revenue,
       SUM(revenue) OVER (ORDER BY region, quarter) AS running_total,
       RANK()       OVER (ORDER BY revenue DESC)    AS revenue_rank
FROM quarterly_sales
ORDER BY region, quarter""",
    error_message=(
        "Query runs but both window functions are wrong. "
        "West Q1 running_total shows 76000 (continuing from East) instead of 11000. "
        "revenue_rank is a global ranking across all 8 rows instead of per-region. "
        "Both SUM and RANK are missing PARTITION BY region."
    ),
    hint=(
        "Add PARTITION BY region to BOTH window functions, but with different ORDER BY: "
        "SUM(revenue) OVER (PARTITION BY region ORDER BY quarter) for running total, "
        "RANK() OVER (PARTITION BY region ORDER BY revenue DESC) for within-region rank."
    ),
    test_cases=[
        TestCase(
            description="Per-region running total and within-region revenue rank",
            expected_rows=[
                {"region": "East", "quarter": 1, "revenue": 15000.0, "running_total": 15000.0, "revenue_rank": 3},
                {"region": "East", "quarter": 2, "revenue": 18000.0, "running_total": 33000.0, "revenue_rank": 2},
                {"region": "East", "quarter": 3, "revenue": 12000.0, "running_total": 45000.0, "revenue_rank": 4},
                {"region": "East", "quarter": 4, "revenue": 20000.0, "running_total": 65000.0, "revenue_rank": 1},
                {"region": "West", "quarter": 1, "revenue": 11000.0, "running_total": 11000.0, "revenue_rank": 4},
                {"region": "West", "quarter": 2, "revenue": 14000.0, "running_total": 25000.0, "revenue_rank": 3},
                {"region": "West", "quarter": 3, "revenue": 16000.0, "running_total": 41000.0, "revenue_rank": 1},
                {"region": "West", "quarter": 4, "revenue": 13000.0, "running_total": 54000.0, "revenue_rank": 2},
            ],
            order_by="region,quarter",
        )
    ],
    solution_query="""\
SELECT region, quarter, revenue,
       SUM(revenue) OVER (PARTITION BY region ORDER BY quarter)        AS running_total,
       RANK()       OVER (PARTITION BY region ORDER BY revenue DESC)   AS revenue_rank
FROM quarterly_sales
ORDER BY region, quarter""",
    max_steps=6,
)


# ── Task Registry ─────────────────────────────────────────────────────────────

class TaskRegistry:
    """
    Thread-safe registry of SQL tasks, shared across all environment sessions.

    Built-in tasks (easy / medium / hard) are always present and cannot be removed.
    Custom tasks can be added via register(), load_from_json(), or POST /tasks.
    """

    _BUILTIN_IDS: frozenset = frozenset([
        "task_easy_syntax", "task_medium_join", "task_hard_cte",
        "task_expert_rank", "task_expert_recursive", "task_expert_window",
    ])

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

REGISTRY = TaskRegistry([
    _TASK_EASY, _TASK_MEDIUM, _TASK_HARD,
    _TASK_EXPERT_RANK, _TASK_EXPERT_RECURSIVE, _TASK_EXPERT_WINDOW,
])

# Backwards-compat: snapshot of all built-in tasks at import time
TASKS: List[SQLTask] = [
    _TASK_EASY, _TASK_MEDIUM, _TASK_HARD,
    _TASK_EXPERT_RANK, _TASK_EXPERT_RECURSIVE, _TASK_EXPERT_WINDOW,
]
TASK_BY_ID: Dict[str, SQLTask] = {t.id: t for t in TASKS}
