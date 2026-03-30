"""
QueryForge Judge — deterministic DuckDB grading + Anthropic AI quality scoring.

Grading pipeline for each submitted SQL query:

  Stage 1 — Syntax (0.0 → 0.15)
    DuckDB EXPLAIN parses the query.  Fail → score = 0.0.

  Stage 2 — Execution (→ 0.30)
    Run the full query against in-memory DuckDB seeded with task data.
    Fail → score = 0.15 (syntax was fine, runtime error).

  Stage 3 — Correctness (→ 0.80)
    Compare returned rows against expected rows.
    Perfect match → deterministic score reaches 0.80.
    Partial credit for correct row count or partial row matches.

  Stage 4 — AI Quality (→ 1.0)
    Anthropic claude-haiku-4-5 evaluates optimization, code style, and
    semantic correctness vs. the reference solution.
    The AI score can move the final score up to 1.0 when rows are correct,
    or provide nuanced feedback even when rows are partially wrong.

Environment variable required:
  ANTHROPIC_API_KEY — standard Anthropic SDK key.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import duckdb

try:
    from .tasks import SQLTask, TestCase
except ImportError:
    from tasks import SQLTask, TestCase

JUDGE_MODEL = "claude-haiku-4-5-20251001"
# ---------------------------------------------------------------------------
# Stage 1 — Syntax check
# ---------------------------------------------------------------------------

def _reject_multi_statement(query: str) -> Optional[str]:
    """Return an error message if the query contains multiple statements."""
    # Strip string literals and comments before checking for semicolons
    stripped = re.sub(r"'[^']*'", "", query)   # remove string literals
    stripped = re.sub(r"--[^\n]*", "", stripped)  # remove line comments
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)  # block comments
    stripped = stripped.strip().rstrip(";")  # allow a single trailing semicolon
    if ";" in stripped:
        return "Multi-statement queries are not allowed."
    return None


def check_syntax(query: str) -> Tuple[bool, Optional[str]]:
    """
    Return (is_valid, error_message).

    Strategy: run EXPLAIN against an empty in-memory DuckDB.
    - "Parser Error" in the exception → genuine syntax error → invalid.
    - "Catalog Error" / "Binder Error" → tables unknown but syntax is fine → valid.
    - Any other exception → treat as syntax error to be safe.
    """
    multi_err = _reject_multi_statement(query)
    if multi_err:
        return False, multi_err

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(f"EXPLAIN {query}")
        return True, None
    except Exception as exc:
        msg = str(exc)
        # Catalog/Binder errors mean the SQL parsed fine; tables just aren't seeded.
        if any(
            tag in msg
            for tag in ("Catalog Error", "Binder Error", "Table with name",
                        "Referenced column", "does not exist", "column")
        ):
            return True, None
        return False, msg
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stage 2 — Execution
# ---------------------------------------------------------------------------

def execute_query(
    schema_ddl: str, query: str
) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Seed a fresh DuckDB in-memory DB with *schema_ddl*, then run *query*.
    Returns (success, rows_as_list_of_dicts, error_message).
    """
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(schema_ddl)
        result = conn.execute(query).fetchdf()
        rows = result.to_dict(orient="records")
        # Convert numpy types to native Python
        clean: List[Dict[str, Any]] = []
        for row in rows:
            clean.append({k: _native(v) for k, v in row.items()})
        return True, clean, None
    except Exception as exc:
        return False, None, str(exc)
    finally:
        conn.close()


def _native(value: Any) -> Any:
    """Convert numpy scalars → native Python types for JSON-safe comparison."""
    try:
        import numpy as np  # duckdb fetchdf() returns numpy types
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    return value


# ---------------------------------------------------------------------------
# Stage 3 — Row correctness
# ---------------------------------------------------------------------------

def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Round floats to 2 dp so 999.99000000001 == 999.99."""
    return {
        k: (round(float(v), 2) if isinstance(v, float) else v)
        for k, v in row.items()
    }


def _sort_key(row: Dict[str, Any], order_by: Optional[str]) -> tuple:
    if order_by:
        cols = [c.strip() for c in order_by.split(",")]
        return tuple(str(row.get(c, "")) for c in cols)
    return tuple(str(v) for v in row.values())


def rows_match(
    actual: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
    order_by: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Compare *actual* vs *expected* rows.

    Scoring:
      1.0  — exact match
      0.5–0.9 — row count matches, some rows differ
      0.3  — row count wrong but partial overlap
      0.0  — empty when non-empty expected
    """
    if not expected:
        return (1.0, "No expected rows — query accepted.") if not actual else (
            0.8, f"Expected empty result but got {len(actual)} row(s)."
        )

    if not actual:
        return 0.0, f"Query returned 0 rows; expected {len(expected)}."

    # Project actual rows to only the expected columns (agent may SELECT extra).
    # Use case-insensitive matching: build a map from lower(actual_col) → actual_col.
    expected_cols = list(expected[0].keys())
    lower_map = {k.lower(): k for k in actual[0].keys()} if actual else {}

    def _project(row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ec in expected_cols:
            actual_key = lower_map.get(ec.lower())
            if actual_key is not None:
                out[ec] = row[actual_key]
        return out

    projected = [_project(row) for row in actual]

    actual_norm = [_normalize(r) for r in projected]
    expected_norm = [_normalize(r) for r in expected]

    if len(projected) != len(expected):
        # Count how many returned rows are actually in the expected set
        expected_set = [tuple(sorted(r.items())) for r in expected_norm]
        correct_rows = sum(1 for r in actual_norm if tuple(sorted(r.items())) in expected_set)
        # Score based on fraction of expected rows correctly returned
        coverage = correct_rows / len(expected)
        # Base 0.10 for count mismatch, up to 0.45 for high coverage of correct rows
        score = 0.10 + 0.35 * coverage
        return score, (
            f"Row count mismatch: got {len(projected)}, expected {len(expected)}. "
            f"{correct_rows}/{len(expected)} expected rows present."
        )

    actual_sorted = sorted(actual_norm, key=lambda r: _sort_key(r, order_by))
    expected_sorted = sorted(expected_norm, key=lambda r: _sort_key(r, order_by))

    matches = sum(1 for a, e in zip(actual_sorted, expected_sorted) if a == e)
    row_accuracy = matches / len(expected)

    if row_accuracy == 1.0:
        return 1.0, "All rows match perfectly."

    score = 0.5 + 0.4 * row_accuracy
    return score, f"{matches}/{len(expected)} rows match correctly."


# ---------------------------------------------------------------------------
# Stage 4 — Anthropic AI judge
# ---------------------------------------------------------------------------

def call_anthropic_judge(
    task: SQLTask,
    agent_query: str,
    execution_success: bool,
    execution_error: Optional[str],
    actual_rows: Optional[List[Dict[str, Any]]],
    deterministic_score: float,
) -> Tuple[float, str, str]:
    """
    Call claude-sonnet-4-6 to evaluate query quality across three axes:
      - Correctness  (0–0.50)
      - Optimization (0–0.30)  — avoids inefficiencies, uses best SQL patterns
      - Code quality (0–0.20)  — readable, well-aliased, idiomatic SQL

    Returns (final_score, feedback, improvement_hint).
    Falls back to deterministic_score if the API call fails.
    """
    client = anthropic.Anthropic()

    sample_actual = json.dumps(actual_rows[:5] if actual_rows else [], indent=2)
    sample_expected = json.dumps(
        task.test_cases[0].expected_rows if task.test_cases else [], indent=2
    )

    prompt = f"""\
You are a strict SQL expert judge scoring an agent's query for the task below.

## Task  ({task.level})
{task.description}

## Agent Query
```sql
{agent_query}
```

## Execution
- Success: {execution_success}
- Error: {execution_error or "None"}
- Rows returned (first 5): {sample_actual}
- Expected rows: {sample_expected}

## Reference Solution
```sql
{task.solution_query}
```

## Deterministic row-match score (0.0–1.0): {deterministic_score:.3f}

Score the agent query on THREE axes and sum them for the final score:

| Axis         | Max  | Criteria |
|--------------|------|----------|
| Correctness  | 0.50 | Produces the right rows for the stated goal |
| Optimization | 0.30 | Avoids cartesian products / correlated subqueries; uses efficient patterns (CTEs, explicit JOINs, proper GROUP BY) |
| Code quality | 0.20 | Readable aliases, clean formatting, no redundant clauses |

IMPORTANT rules:
- If execution failed with a runtime error, Correctness ≤ 0.10.
- If rows are fully correct per deterministic score ≥ 0.95, Correctness ≥ 0.40.
- For the medium task: a query that still uses comma-join syntax scores Optimization ≤ 0.05.
- For the hard task: a query without a CTE scores Optimization ≤ 0.10.

Respond with ONLY valid JSON (no markdown fences):
{{
  "correctness":  <float 0.0–0.50>,
  "optimization": <float 0.0–0.30>,
  "code_quality": <float 0.0–0.20>,
  "score":        <sum of above, float 0.0–1.0>,
  "feedback":     "<2–3 sentences summarising what the agent did right/wrong>",
  "hint":         "<one concrete actionable improvement, or 'Excellent!' if score >= 0.95>"
}}"""

    try:
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            messages=[
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": "{"},   # prefill forces JSON-only reply
            ],
        )
        # Prepend the prefilled "{" back before parsing
        raw = "{" + message.content[0].text.strip()

        # Belt-and-suspenders: extract the first {...} block in case of any preamble
        brace_start = raw.find("{")
        brace_end   = raw.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            raw = raw[brace_start:brace_end]

        data = json.loads(raw)
        score = float(data["score"])
        score = max(0.0, min(1.0, score))
        feedback = str(data.get("feedback", ""))
        hint = str(data.get("hint", ""))
        return score, feedback, hint

    except Exception as exc:
        # Graceful fallback — no API key, network error, or parse failure
        msg = str(exc).lower()
        if "api_key" in msg or "auth" in msg or "authentication" in msg:
            reason = "ANTHROPIC_API_KEY not set — deterministic scoring only (max 0.80)"
        else:
            reason = f"AI judge call failed ({type(exc).__name__}) — fell back to deterministic score"
        return (
            deterministic_score,
            f"[AI Judge unavailable] {reason}.",
            task.hint,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(
    task: SQLTask, agent_query: str
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Full grading pipeline.  Returns (score 0.0–1.0, feedback, details_dict).

    Partial progress scoring:
      0.00  — syntax error (unparseable)
      0.15  — syntax valid, runtime error
      0.30  — executes, but 0 rows returned
      0.30–0.80 — partial row matches (deterministic)
      0.80–1.00 — correct rows + AI quality assessment
    """
    details: Dict[str, Any] = {}

    # ── Stage 1: syntax ──────────────────────────────────────────────────────
    syntax_ok, syntax_error = check_syntax(agent_query)
    details["syntax_valid"] = syntax_ok
    details["syntax_error"] = syntax_error

    if not syntax_ok:
        return 0.0, f"Syntax error: {syntax_error}", details

    # ── Stage 2: execution ───────────────────────────────────────────────────
    exec_ok, rows, exec_error = execute_query(task.schema_ddl, agent_query)
    details["execution_success"] = exec_ok
    details["execution_error"] = exec_error
    details["rows_returned"] = len(rows) if rows else 0

    if not exec_ok:
        # Syntax valid but runtime error — call AI for nuanced feedback
        ai_score, ai_feedback, ai_hint = call_anthropic_judge(
            task, agent_query, False, exec_error, None, 0.15
        )
        details["ai_score"] = ai_score
        details["ai_feedback"] = ai_feedback
        final = max(0.15, ai_score * 0.3)  # cap at 0.3 when execution fails
        return final, f"Runtime error: {exec_error} | AI: {ai_feedback}", details

    # ── Stage 3: row correctness ─────────────────────────────────────────────
    test_case = task.test_cases[0]
    row_score, row_feedback = rows_match(rows, test_case.expected_rows, test_case.order_by)
    details["row_match_score"] = row_score
    details["row_match_feedback"] = row_feedback

    # ── Stage 3b: structural checks (task-specific) ─────────────────────────
    # These prevent high scores when the agent submits the broken query verbatim
    # or ignores the task's structural requirement.
    structural_penalty = 0.0
    query_upper = agent_query.upper()

    if task.level == "hard" and "WITH " not in query_upper:
        structural_penalty = 0.30  # hard task demands a CTE
        row_feedback += " (Penalty: no CTE detected — task requires WITH clause.)"
    elif task.level == "medium" and "JOIN " not in query_upper:
        structural_penalty = 0.20  # medium task demands explicit JOINs
        row_feedback += " (Penalty: no explicit JOIN — task requires JOIN … ON syntax.)"
    elif task.id == "task_expert_recursive":
        # Two bugs: anchor uses WHERE id=3 (includes VP Eng) + non-recursive CTE (misses deep levels)
        if "RECURSIVE" not in query_upper:
            structural_penalty += 0.30
            row_feedback += " (Penalty: WITH RECURSIVE required — hardcoded levels won't scale.)"
        if "MANAGER_ID = 3" not in query_upper and "MANAGER_ID=3" not in query_upper:
            structural_penalty += 0.15
            row_feedback += " (Penalty: anchor should select subordinates via manager_id, not the VP themselves.)"
        structural_penalty = min(structural_penalty, 0.40)
    elif task.id == "task_expert_rank":
        # Two bugs: ROW_NUMBER (drops ties) + ASC ordering (picks lowest instead of highest)
        if "ROW_NUMBER" in query_upper:
            structural_penalty += 0.20
            row_feedback += " (Penalty: ROW_NUMBER() drops tied rows — use RANK() or DENSE_RANK().)"
        if "ASC" in query_upper and "DESC" not in query_upper:
            structural_penalty += 0.15
            row_feedback += " (Penalty: ordering by revenue ASC picks lowest earners, not highest.)"
        structural_penalty = min(structural_penalty, 0.35)
    elif task.id == "task_expert_window":
        # Three bugs: missing PARTITION BY on both windows + tied revenues need correct ranking
        if "PARTITION BY" not in query_upper:
            structural_penalty += 0.20
            row_feedback += " (Penalty: missing PARTITION BY — both SUM and RANK must be partitioned per region.)"
        # Count PARTITION BY occurrences — need at least 2 (one per window function)
        partition_count = query_upper.count("PARTITION BY")
        if 0 < partition_count < 2:
            structural_penalty += 0.10
            row_feedback += " (Penalty: only one window function has PARTITION BY — both need it.)"
        structural_penalty = min(structural_penalty, 0.30)

    details["structural_penalty"] = structural_penalty

    # Deterministic score: 0.30 base for executing + up to 0.50 for rows − penalty
    deterministic_score = max(0.30, 0.30 + 0.50 * row_score - structural_penalty)

    # ── Stage 4: AI quality ──────────────────────────────────────────────────
    ai_score, ai_feedback, ai_hint = call_anthropic_judge(
        task, agent_query, True, None, rows, deterministic_score
    )
    details["ai_score"] = ai_score
    details["ai_feedback"] = ai_feedback
    details["ai_hint"] = ai_hint

    # Final blending:
    #   AI judge offline (fallback) → use deterministic score directly
    #   rows fully correct  → trust AI score (can reach 1.0)
    #   rows partially wrong → clamp AI score to not exceed deterministic
    ai_is_fallback = abs(ai_score - deterministic_score) < 0.001
    if ai_is_fallback:
        # AI judge was unavailable — use deterministic score as-is
        final_score = deterministic_score
    elif row_score >= 0.95:
        final_score = ai_score
    elif row_score >= 0.5:
        # Blend: AI provides nuance but can't exceed deterministic ceiling
        final_score = min(deterministic_score, ai_score + 0.05)
    else:
        # Low row accuracy — stay near deterministic
        final_score = min(deterministic_score, ai_score * 0.6)

    final_score = max(0.0, min(1.0, final_score))

    feedback = (
        f"[Rows] {row_feedback}  "
        f"[AI Judge] {ai_feedback}  "
        f"[Hint] {ai_hint}"
    )
    return final_score, feedback, details
