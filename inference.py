"""
QueryForge Inference Script
===================================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier to use for inference
  HF_TOKEN       Your Hugging Face / API key

Optional env vars:
  ENV_URL          QueryForge environment server URL (default: http://localhost:8000)
  ANTHROPIC_API_KEY  Enables AI judge for scores up to 1.0 (default: deterministic mode)
"""

import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import QueryforgeEnv
from models import SQLAction

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_URL      = os.getenv("ENV_URL", "https://prithvigg-queryforge.hf.space")

MAX_STEPS   = 5      # max attempts per task (overridden by task's own max_steps)
TEMPERATURE = 0.2
MAX_TOKENS  = 512

TASK_IDS = [
    "task_easy_syntax",
    "task_medium_join",
    "task_hard_cte",
    "task_expert_rank",
    "task_expert_recursive",
    "task_expert_window",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL engineer tasked with debugging and optimising SQL queries.
    You will be given a SQL challenge that includes a schema, a broken or slow query,
    and a description of what the correct output should be.

    Rules:
    - Respond with ONLY a single SQL query inside a ```sql ... ``` code block.
    - Do not explain your reasoning outside the code block.
    - Do not include multiple statements separated by semicolons.
    - If you receive grading feedback on a previous attempt, use it to improve.
""").strip()

# ── SQL extraction ─────────────────────────────────────────────────────────────

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the first SQL code block from the model response."""
    match = _SQL_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ── Formatting ────────────────────────────────────────────────────────────────

def score_bar(score: float, width: int = 25) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.3f}"


def hr(char="═", width=70):
    print(char * width)


# ── Per-task agent loop ────────────────────────────────────────────────────────

def run_task(task_id: str, llm: OpenAI, env_client) -> dict:
    """
    Run one episode for a single task.
    Returns dict with task_id, task_title, task_level, best_score, attempts, done.
    """
    result = env_client.reset(task_id=task_id)
    obs = result.observation

    if result.done:
        print(f"  ERROR loading task: {obs.feedback}")
        return {"task_id": task_id, "best_score": 0.0, "attempts": 0, "done": False}

    print(f"\n  Task  : {obs.task_title}  [{obs.task_level}]")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here is your SQL challenge:\n\n{obs.task_description}\n\n"
                "Provide your fixed SQL query."
            ),
        },
    ]

    step = 0
    while not result.done:
        step += 1

        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  LLM call failed at step {step}: {exc}")
            break

        sql = extract_sql(response_text)

        # ── Print generated SQL ───────────────────────────────────────────────
        print(f"\n  ┌─ Step {step} · SQL submitted {'─' * (50 - len(str(step)))}")
        for line in sql.splitlines():
            print(f"  │  {line}")
        print(f"  └{'─' * 56}")

        result = env_client.step(SQLAction(sql=sql))
        obs = result.observation

        score = result.reward or 0.0
        done_marker = "  ✓ DONE" if result.done else ""
        print(f"  Score : {score_bar(score)}{done_marker}")

        if not obs.syntax_valid:
            print(f"  ✗ Syntax error — query could not be parsed")
        elif not obs.execution_success:
            print(f"  ✗ Execution failed — {(obs.execution_error or '')[:80]}")
        else:
            print(f"  ✓ Executed · rows returned: {obs.rows_returned}")

        if result.done:
            break

        # ── Why are we going to the next step? ───────────────────────────────
        print(f"\n  ↻ Retrying — score {score:.3f} below threshold")
        if obs.feedback:
            # Split the feedback into its tagged sections for readable multi-line output
            for part in obs.feedback.split("  "):
                part = part.strip()
                if part:
                    print(f"  {part}")
        if obs.hint:
            print(f"  Hint     : {obs.hint[:120]}")

        # Feed grading result back to the model for the next attempt
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your query scored {result.reward:.3f}.\n\n"
                f"Feedback: {obs.feedback}\n\n"
                f"Hint: {obs.hint}\n\n"
                "Please submit an improved SQL query."
            ),
        })

    return {
        "task_id": task_id,
        "task_title": obs.task_title,
        "task_level": obs.task_level,
        "best_score": obs.best_score,
        "attempts": obs.attempt,
        "done": result.done,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Validate required config ──────────────────────────────────────────────
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME env var is not set.")
        sys.exit(1)

    if not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY) is not set.")
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    hr()
    print("  QueryForge — Inference")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Env    : {ENV_URL}")
    print(f"  Tasks  : {', '.join(TASK_IDS)}")
    hr()

    results = []

    with QueryforgeEnv(base_url=ENV_URL).sync() as env_client:
        for task_id in TASK_IDS:
            print(f"\n{'─' * 70}")
            result = run_task(task_id, llm, env_client)
            results.append(result)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  RESULTS")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'═' * 70}")
    print(f"  {'Task':<28} {'Level':<8} {'Steps':>5}  {'Best Score'}")
    print(f"  {'─' * 28} {'─' * 8} {'─' * 5}  {'─' * 30}")

    total = 0.0
    for r in results:
        title   = r.get("task_title", r["task_id"])[:27]
        level   = r.get("task_level", "?")
        steps   = r.get("attempts", "?")
        score   = r["best_score"]
        total  += score
        print(f"  {title:<28} {level:<8} {steps:>5}  {score_bar(score)}")

    avg = total / len(results) if results else 0.0
    print(f"{'─' * 70}")
    print(f"  {'AVERAGE':<28} {'':8} {'':5}  {score_bar(avg)}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
