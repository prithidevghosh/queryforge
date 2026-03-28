"""QueryForge Environment Client."""

from typing import Any, Dict, List, Optional

import httpx
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SQLAction, SQLObservation, TaskSpec
except ImportError:
    from models import SQLAction, SQLObservation, TaskSpec


class QueryforgeEnv(EnvClient[SQLAction, SQLObservation, State]):
    """
    Client for the QueryForge SQL Debugger & Optimiser environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session (isolated task state).

    Example:
        >>> with QueryforgeEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     print(obs.task_title)
        ...     print(obs.task_description)
        ...
        ...     result = env.step(SQLAction(sql="SELECT name, age FROM users WHERE age > 30"))
        ...     print(result.reward, result.observation.feedback)

    Example with Docker:
        >>> env = QueryforgeEnv.from_docker_image("queryforge-env:latest")
        >>> try:
        ...     obs = env.reset()
        ...     result = env.step(SQLAction(sql="SELECT ..."))
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: SQLAction) -> Dict:
        return {"sql": action.sql}

    def _parse_result(self, payload: Dict) -> StepResult[SQLObservation]:
        obs_data = payload.get("observation", {})
        observation = SQLObservation(
            task_id=obs_data.get("task_id", ""),
            task_level=obs_data.get("task_level", ""),
            task_title=obs_data.get("task_title", ""),
            task_description=obs_data.get("task_description", ""),
            syntax_valid=obs_data.get("syntax_valid", False),
            execution_success=obs_data.get("execution_success", False),
            execution_error=obs_data.get("execution_error"),
            rows_returned=obs_data.get("rows_returned", 0),
            feedback=obs_data.get("feedback", ""),
            hint=obs_data.get("hint", ""),
            attempt=obs_data.get("attempt", 0),
            best_score=obs_data.get("best_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # ── Task Registry helpers ─────────────────────────────────────────────────

    def register_task(self, spec: TaskSpec) -> Dict[str, Any]:
        """Register a custom task on the server. Returns the server response dict."""
        resp = httpx.post(
            f"{self.base_url}/tasks",
            json=spec.model_dump(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return all registered tasks (built-in + custom) as a list of dicts."""
        resp = httpx.get(f"{self.base_url}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """Remove a custom task by ID. Raises httpx.HTTPStatusError on 403/404."""
        resp = httpx.delete(f"{self.base_url}/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
