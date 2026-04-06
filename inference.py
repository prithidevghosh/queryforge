"""
QueryForge Inference Script
===================================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier to use for inference
  HF_TOKEN       Your Hugging Face / API key

Optional env vars:
  ENV_URL          QueryForge environment server URL (default: live HF Space)
  ANTHROPIC_API_KEY  Enables AI judge for scores up to 1.0 (default: deterministic mode)

STDOUT FORMAT (required by evaluator):
  [START] task=<task_id> env=queryforge model=<model_name>
  [STEP]  step=<n> action=<sql_oneline> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
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
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://prithvigg-queryforge.hf.space")

MAX_STEPS              = 5
TEMPERATURE            = 0.2
MAX_TOKENS             = 512
SUCCESS_SCORE_THRESHOLD = 0.9

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

# ── Structured log helpers (evaluator-required format) ────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=queryforge model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # SQL may contain newlines — collapse to single line (spec: no newlines within a line)
    action_oneline = " ".join(action.split())
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_oneline} reward={reward:.2f}"
        f" done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── SQL extraction ─────────────────────────────────────────────────────────────

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    match = _SQL_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ── Formatting helpers (human-readable output) ────────────────────────────────

def score_bar(score: float, width: int = 25) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.3f}"


def hr(char="═", width=70):
    print(char * width)


# ── Per-task agent loop ────────────────────────────────────────────────────────

def run_task(task_id: str, llm: OpenAI, env_client) -> dict:
    # Initialise before anything that can throw — guarantees [END] is always emitted.
    step       = 0
    rewards: List[float] = []
    success    = False
    best_score = 0.0
    task_title = task_id
    task_level = "unknown"
    attempts   = 0
    done       = False

    log_start(task=task_id, model=MODEL_NAME)

    try:
        result = env_client.reset(task_id=task_id)
        obs    = result.observation

        if result.done:
            print(f"  ERROR loading task: {obs.feedback}")
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return {"task_id": task_id, "best_score": 0.0, "attempts": 0, "done": False}

        task_title = obs.task_title
        task_level = obs.task_level
        print(f"\n  Task  : {task_title}  [{task_level}]")

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

            print(f"\n  ┌─ Step {step} · SQL submitted {'─' * (50 - len(str(step)))}")
            for line in sql.splitlines():
                print(f"  │  {line}")
            print(f"  └{'─' * 56}")

            result = env_client.step(SQLAction(sql=sql))
            obs    = result.observation

            reward = result.reward or 0.0
            rewards.append(reward)
            done = result.done

            if not obs.syntax_valid:
                step_error: Optional[str] = "syntax_error"
                print(f"  ✗ Syntax error — query could not be parsed")
            elif not obs.execution_success:
                step_error = (obs.execution_error or "execution_error")[:120]
                print(f"  ✗ Execution failed — {step_error[:80]}")
            else:
                step_error = None
                print(f"  ✓ Executed · rows returned: {obs.rows_returned}")

            done_marker = "  ✓ DONE" if done else ""
            print(f"  Score : {score_bar(reward)}{done_marker}")

            log_step(step=step, action=sql, reward=reward, done=done, error=step_error)

            if done:
                break

            print(f"\n  ↻ Retrying — score {reward:.3f} below threshold")
            if obs.feedback:
                for part in obs.feedback.split("  "):
                    part = part.strip()
                    if part:
                        print(f"  {part}")
            if obs.hint:
                print(f"  Hint     : {obs.hint[:120]}")

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Your query scored {reward:.3f}.\n\n"
                    f"Feedback: {obs.feedback}\n\n"
                    f"Hint: {obs.hint}\n\n"
                    "Please submit an improved SQL query."
                ),
            })

        best_score = obs.best_score
        attempts   = obs.attempt
        success    = best_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"  FATAL error in task {task_id}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=step, score=best_score, rewards=rewards)

    return {
        "task_id":    task_id,
        "task_title": task_title,
        "task_level": task_level,
        "best_score": best_score,
        "attempts":   attempts,
        "done":       done,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
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
    try:
        env_ctx = QueryforgeEnv(base_url=ENV_URL).sync()
        with env_ctx as env_client:
            for task_id in TASK_IDS:
                print(f"\n{'─' * 70}")
                results.append(run_task(task_id, llm, env_client))
    except Exception as exc:
        print(f"FATAL: could not connect to environment at {ENV_URL}: {exc}", flush=True)
        sys.exit(1)

    # ── Results summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  RESULTS")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'═' * 70}")
    print(f"  {'Task':<28} {'Level':<8} {'Steps':>5}  {'Best Score'}")
    print(f"  {'─' * 28} {'─' * 8} {'─' * 5}  {'─' * 30}")

    total = 0.0
    for r in results:
        title  = r.get("task_title", r["task_id"])[:27]
        level  = r.get("task_level", "?")
        steps  = r.get("attempts", "?")
        score  = r["best_score"]
        total += score
        print(f"  {title:<28} {level:<8} {steps:>5}  {score_bar(score)}")

    avg = total / len(results) if results else 0.0
    print(f"{'─' * 70}")
    print(f"  {'AVERAGE':<28} {'':8} {'':5}  {score_bar(avg)}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
