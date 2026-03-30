"""
Data models for the QueryForge SQL environment.

SQLAction    — the agent's submitted SQL query.
SQLObservation — task description + grading feedback returned after each step.
TaskSpec     — payload for registering a custom task via POST /tasks.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class SQLAction(Action):
    """Action: submit a SQL query for evaluation."""

    sql: str = Field(..., description="The SQL query to submit for grading")


class SQLObservation(Observation):
    """Observation returned after reset() or step()."""

    # ── Task context ─────────────────────────────────────────────────────────
    task_id: str = Field(default="", description="Active task identifier")
    task_level: str = Field(
        default="", description="Difficulty: easy | medium | hard | expert"
    )
    task_title: str = Field(default="", description="Human-readable task title")
    task_description: str = Field(
        default="",
        description=(
            "Full task description: schema, broken query, error message, and goal"
        ),
    )

    # ── Per-step grading signals ──────────────────────────────────────────────
    syntax_valid: bool = Field(
        default=False, description="True if the submitted query parsed without error"
    )
    execution_success: bool = Field(
        default=False, description="True if the query ran to completion in DuckDB"
    )
    execution_error: Optional[str] = Field(
        default=None, description="Runtime error message, if any"
    )
    rows_returned: int = Field(
        default=0, description="Number of rows the query returned"
    )
    feedback: str = Field(
        default="",
        description="Detailed grading feedback from DuckDB + AI judge",
    )
    hint: str = Field(
        default="", description="Actionable hint for the next attempt"
    )

    # ── Episode progress ──────────────────────────────────────────────────────
    attempt: int = Field(
        default=0, description="Number of queries submitted this episode"
    )
    best_score: float = Field(
        default=0.0, description="Highest score achieved so far this episode"
    )


class TaskSpec(BaseModel):
    """
    Payload for registering a custom SQL task via POST /tasks
    or directly via REGISTRY.register(task_from_dict(spec.model_dump())).

    Required: id, schema_ddl, expected_rows
    Everything else has sensible defaults.
    """

    id: str = Field(
        ..., description="Unique task identifier, e.g. 'null_handling_task'"
    )
    level: str = Field(
        default="custom",
        description="Difficulty label: easy | medium | hard | custom",
    )
    title: str = Field(..., description="Human-readable task title")
    description: str = Field(
        default="",
        description="Full task description shown to the agent (schema, goal, etc.)",
    )
    schema_ddl: str = Field(
        ...,
        description="CREATE TABLE + INSERT statements to seed the DuckDB test DB",
    )
    broken_query: str = Field(
        default="",
        description="The broken or slow query the agent must fix",
    )
    error_message: str = Field(
        default="",
        description="Error or performance warning shown to the agent alongside the task",
    )
    hint: str = Field(
        default="",
        description="Actionable hint surfaced in the observation after each wrong attempt",
    )
    expected_rows: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Exact rows the correct query must return. "
            "Used for deterministic row-match scoring."
        ),
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Comma-separated column names used to sort rows before comparison",
    )
    solution_query: str = Field(
        default="",
        description="Reference solution shown to the AI judge for quality scoring",
    )
    test_description: str = Field(
        default="Custom test case",
        description="One-line description of what the test case checks",
    )
    max_steps: int = Field(
        default=5, ge=1, le=20,
        description="Maximum number of step() calls allowed per episode",
    )
