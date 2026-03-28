# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Queryforge Environment.

This module creates an HTTP server that exposes the QueryforgeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SQLAction, SQLObservation, TaskSpec
    from ..tasks import REGISTRY, task_from_dict
    from .queryforge_environment import QueryforgeEnvironment
except ImportError:
    from models import SQLAction, SQLObservation, TaskSpec
    from tasks import REGISTRY, task_from_dict
    from server.queryforge_environment import QueryforgeEnvironment

import gradio as gr
from fastapi import HTTPException


# Create the app with web interface and README integration
app = create_app(
    QueryforgeEnvironment,
    SQLAction,
    SQLObservation,
    env_name="queryforge",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# ── Task Registry REST endpoints ──────────────────────────────────────────────

@app.post("/tasks", tags=["Task Registry"], status_code=201)
async def register_task(spec: TaskSpec):
    """Register a custom SQL task. Replaces silently if the ID already exists."""
    task = task_from_dict(spec.model_dump())
    REGISTRY.register(task)
    return {"ok": True, "task_id": task.id, "total_tasks": len(REGISTRY)}


@app.get("/tasks", tags=["Task Registry"])
async def list_tasks():
    """List all registered tasks (built-in + custom)."""
    return [
        {"id": t.id, "level": t.level, "title": t.title}
        for t in REGISTRY.list_all()
    ]


@app.delete("/tasks/{task_id}", tags=["Task Registry"])
async def delete_task(task_id: str):
    """Remove a custom task. Returns 403 for built-in tasks, 404 if not found."""
    try:
        REGISTRY.unregister(task_id)
        return {"ok": True, "task_id": task_id}
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")


# ── Gradio demo — mounted at /demo so the HF Space App tab shows it ───────────

try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from demo import demo as gradio_demo
    app = gr.mount_gradio_app(app, gradio_demo, path="/demo")
except Exception as _e:
    import warnings
    warnings.warn(f"Gradio demo not mounted: {_e}")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m queryforge.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn queryforge.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
