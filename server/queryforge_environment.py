"""
QueryForge SQL Environment — server-side implementation.

The agent interacts with a SQL debugging and optimisation challenge:
  reset()              → next task in round-robin rotation
  reset(task_id="x")  → pin to a specific task by ID (built-in or custom)
  step()               → grade the submitted query, return scored observation
  state                → episode_id + step count

Reward scale:
  0.00        syntax error
  0.15        syntax valid, runtime error
  0.30        executes, wrong / empty results
  0.30–0.80   partial row correctness (deterministic, DuckDB)
  0.80–1.00   correct results + AI quality assessment (Anthropic)

Episode ends when:
  - score >= 0.90  (correct + high-quality solution)
  - best_score has not improved for 2 consecutive steps (early stopping)
  - max_steps for the task is exhausted
"""

import logging
import os
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SQLAction, SQLObservation
    from ..tasks import REGISTRY, SQLTask
    from ..judge import grade
except ImportError:
    from models import SQLAction, SQLObservation
    from tasks import REGISTRY, SQLTask
    from judge import grade

logger = logging.getLogger(__name__)
_AI_JUDGE_ACTIVE = bool(os.environ.get("ANTHROPIC_API_KEY"))

print("here", os.environ.get("ANTHROPIC_API_KEY"))
logger.info(
    "QueryForge environment loaded | AI judge: %s | done_threshold: %s",
    "ACTIVE (scores up to 1.0)" if _AI_JUDGE_ACTIVE else "OFFLINE — deterministic only (max score 0.80)",
    "0.90" if _AI_JUDGE_ACTIVE else "0.80",
)


class QueryforgeEnvironment(Environment):
    """
    SQL Query Debugger & Optimiser environment.

    Built-in tasks (cycled in order by default):
      1. easy   — fix three misspelled SQL keywords
      2. medium — fix a missing JOIN condition causing a cartesian product
      3. hard   — rewrite a correlated subquery as a CTE

    Custom tasks can be registered at runtime via POST /tasks and then
    requested by passing task_id to reset():
      env.reset(task_id="my_custom_task")

    Each episode ends when:
      - The agent achieves score ≥ 0.90 (correct + high-quality solution), or
      - best_score has not improved for 2 consecutive steps (early stopping), or
      - The maximum steps for the current task is exhausted.

    Supports concurrent WebSocket sessions (each client gets its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Episode ends when score >= this threshold.
    # Falls back to 0.80 when ANTHROPIC_API_KEY is unset (AI judge offline,
    # deterministic scoring caps at 0.80).
    DONE_THRESHOLD: float = 0.80 if not __import__("os").environ.get("ANTHROPIC_API_KEY") else 0.90
    # Episode ends when best_score has not improved for this many consecutive steps
    EARLY_STOP_STEPS: int = 2

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[SQLTask] = None
        self._best_score: float = 0.0
        self._attempt: int = 0
        self._stale_steps: int = 0  # consecutive steps with no best_score improvement

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SQLObservation:
        """
        Start a new episode.

        Args:
            task_id:    Pin to a specific task by ID.  If None, the registry
                        cycles round-robin through all registered tasks.
            seed:       Ignored (reserved for future use).
            episode_id: Optional custom episode identifier.
        """
        ep_id = episode_id or str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)
        self._best_score = 0.0
        self._attempt = 0
        self._stale_steps = 0

        logger.info(
            "reset() | task_id=%s | AI judge: %s",
            task_id or "round-robin",
            "ACTIVE" if _AI_JUDGE_ACTIVE else "OFFLINE",
        )

        if task_id is not None:
            try:
                self._current_task = REGISTRY.get(task_id)
            except KeyError as exc:
                # Unknown task_id — return an error observation so the caller
                # gets clear feedback instead of a silent 500.
                return SQLObservation(
                    feedback=str(exc),
                    hint=f"Available task IDs: {', '.join(REGISTRY.ids())}",
                    done=True,
                    reward=0.0,
                )
        else:
            self._current_task = REGISTRY.cycle_next()

        return SQLObservation(
            task_id=self._current_task.id,
            task_level=self._current_task.level,
            task_title=self._current_task.title,
            task_description=self._current_task.description,
            syntax_valid=False,
            execution_success=False,
            execution_error=None,
            rows_returned=0,
            feedback="New task loaded. Submit your fixed/optimised SQL query.",
            hint=self._current_task.hint,
            attempt=0,
            best_score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: SQLAction) -> SQLObservation:  # type: ignore[override]
        """Grade the submitted SQL query and return a scored observation."""
        self._state.step_count += 1
        self._attempt += 1

        if self._current_task is None:
            return SQLObservation(
                feedback="No task active. Call reset() first.",
                hint="Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        logger.info(
            "step() | task=%s | attempt=%d | AI judge: %s",
            self._current_task.id,
            self._attempt,
            "ACTIVE" if _AI_JUDGE_ACTIVE else "OFFLINE",
        )
        score, feedback, details = grade(self._current_task, action.sql)

        # Fix 1 — early stopping: track consecutive steps with no improvement
        if score > self._best_score:
            self._stale_steps = 0
        else:
            self._stale_steps += 1
        self._best_score = max(self._best_score, score)

        # Fix 3 — lower done threshold + early stopping condition
        done = (
            score >= self.DONE_THRESHOLD
            or self._stale_steps >= self.EARLY_STOP_STEPS
            or self._state.step_count >= self._current_task.max_steps
        )

        return SQLObservation(
            task_id=self._current_task.id,
            task_level=self._current_task.level,
            task_title=self._current_task.title,
            task_description=self._current_task.description,
            syntax_valid=bool(details.get("syntax_valid", False)),
            execution_success=bool(details.get("execution_success", False)),
            execution_error=details.get("execution_error"),
            rows_returned=int(details.get("rows_returned", 0)),
            feedback=feedback,
            hint="" if score >= 0.9 else self._current_task.hint,
            attempt=self._attempt,
            best_score=self._best_score,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state
