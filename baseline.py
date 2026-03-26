"""
QueryForge Baseline Inference Script
─────────────────────────────────────
Runs a Claude model as an agent against all 3 built-in tasks and reports
a reproducible baseline score.

Usage:
    # All tasks, default model (claude-haiku-4-5):
    python baseline.py

    # Specific model:
    python baseline.py --model claude-opus-4-6

    # Single task:
    python baseline.py --task task_easy_syntax

    # More verbose output:
    python baseline.py --verbose

Requirements:
    ANTHROPIC_API_KEY must be set in the environment.
"""

import argparse
import os
import re
import sys

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SQLAction
from server.queryforge_environment import QueryforgeEnvironment
from tasks import REGISTRY

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
You are an expert SQL engineer. You will be given a SQL debugging or \
optimisation challenge. Your job is to submit a corrected or improved SQL query.

Rules:
- Respond with ONLY a single SQL query inside a ```sql ... ``` code block.
- Do not explain your reasoning outside the code block.
- Do not include multiple statements (no semicolons except at the very end).
- If you receive feedback on a previous attempt, use it to improve your query.
"""

# ── SQL extraction ─────────────────────────────────────────────────────────────

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """Pull the first SQL code block out of Claude's response."""
    match = _SQL_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole response stripped — better than crashing
    return text.strip()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _hr(char="═", width=70):
    print(char * width)

def _score_bar(score: float, width: int = 25) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.3f}"


# ── Per-task agent loop ────────────────────────────────────────────────────────

def run_task(
    task_id: str,
    model: str,
    client: anthropic.Anthropic,
    verbose: bool = False,
) -> dict:
    """
    Run one episode of a single task.

    Returns a dict with keys:
        task_id, task_title, task_level,
        best_score, attempts, done
    """
    env = QueryforgeEnvironment()
    obs = env.reset(task_id=task_id)

    if obs.done:
        # reset() returned an error (unknown task_id)
        print(f"  ERROR: {obs.feedback}")
        return {"task_id": task_id, "best_score": 0.0, "attempts": 0, "done": False}

    print(f"\n  Task : {obs.task_title}  [{obs.task_level}]  (max {env._current_task.max_steps} steps)")
    if verbose:
        print(f"  ID   : {obs.task_id}")

    # ── Build initial conversation ────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": (
                f"Here is your SQL challenge:\n\n{obs.task_description}\n\n"
                "Provide your fixed SQL query."
            ),
        }
    ]

    step = 0
    while not obs.done:
        step += 1

        # ── Call Claude ───────────────────────────────────────────────────────
        with client.messages.stream(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text

        sql = _extract_sql(response_text)

        if verbose:
            print(f"\n  ── Step {step}")
            short_sql = sql[:120] + ("…" if len(sql) > 120 else "")
            print(f"     SQL: {short_sql}")

        # ── Submit to environment ─────────────────────────────────────────────
        obs = env.step(SQLAction(sql=sql))

        score_bar = _score_bar(obs.reward or 0.0)
        status = "✓ DONE" if obs.done else f"step {step}/{env._current_task.max_steps}"
        print(f"  [{status}]  Score: {score_bar}")

        if verbose and obs.feedback:
            fb = obs.feedback[:200] + ("…" if len(obs.feedback) > 200 else "")
            print(f"     Feedback: {fb}")

        if obs.done:
            break

        # ── Append exchange to conversation for next attempt ──────────────────
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your query scored {obs.reward:.3f}. Here is the feedback:\n\n"
                f"{obs.feedback}\n\n"
                f"Hint: {obs.hint}\n\n"
                "Please try again with an improved SQL query."
            ),
        })

    return {
        "task_id": task_id,
        "task_title": obs.task_title,
        "task_level": obs.task_level,
        "best_score": obs.best_score,
        "attempts": obs.attempt,
        "done": obs.done,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QueryForge Baseline Inference")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Anthropic model ID to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--task", default=None,
        help="Run a single task by ID instead of all built-in tasks"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print SQL queries and full feedback for each step"
    )
    args = parser.parse_args()

    # ── Validate API key ──────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Determine tasks to run ────────────────────────────────────────────────
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = ["task_easy_syntax", "task_medium_join", "task_hard_cte"]

    # ── Header ────────────────────────────────────────────────────────────────
    _hr()
    print("  QueryForge — Baseline Inference")
    print(f"  Model  : {args.model}")
    print(f"  Tasks  : {', '.join(task_ids)}")
    _hr()

    # ── Run each task ─────────────────────────────────────────────────────────
    results = []
    for task_id in task_ids:
        print(f"\n{'─' * 70}")
        result = run_task(task_id, args.model, client, verbose=args.verbose)
        results.append(result)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  BASELINE RESULTS")
    print(f"  Model: {args.model}")
    print(f"{'═' * 70}")
    print(f"  {'Task':<28} {'Level':<8} {'Steps':>5}  {'Best Score'}")
    print(f"  {'─' * 28} {'─' * 8} {'─' * 5}  {'─' * 30}")

    total_score = 0.0
    for r in results:
        title = r.get("task_title", r["task_id"])[:27]
        level = r.get("task_level", "?")
        attempts = r.get("attempts", "?")
        score = r["best_score"]
        total_score += score
        bar = _score_bar(score)
        print(f"  {title:<28} {level:<8} {attempts:>5}  {bar}")

    avg = total_score / len(results) if results else 0.0
    print(f"{'─' * 70}")
    print(f"  {'AVERAGE':<28} {'':8} {'':5}  {_score_bar(avg)}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
