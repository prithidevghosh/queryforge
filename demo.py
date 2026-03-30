"""
QueryForge — Gradio Demo
========================
Interactive SQL debugger UI that runs the environment in-process (no server needed).

Run locally:
    python demo.py
    # opens http://localhost:7860

On HF Spaces:
    Set ANTHROPIC_API_KEY secret in Space settings for AI judging (optional).
    The demo auto-detects it.
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SQLAction
from server.queryforge_environment import QueryforgeEnvironment
from tasks import REGISTRY

# ── Helpers ───────────────────────────────────────────────────────────────────

AI_JUDGE_ACTIVE = bool(os.environ.get("ANTHROPIC_API_KEY"))

TASK_CHOICES = [
    (f"[{t.level.upper()}] {t.title}", t.id)
    for t in REGISTRY.list_all()
]


def _score_html(score: float, done: bool) -> str:
    filled  = int(score * 20)
    bar     = "█" * filled + "░" * (20 - filled)
    color   = "#22c55e" if score >= 0.9 else ("#f59e0b" if score >= 0.5 else "#ef4444")
    suffix  = "  ✓ Solved!" if done and score >= 0.9 else ("  ⏹ Ended" if done else "")
    return (
        f'<div style="font-family:monospace;font-size:1.1rem;">'
        f'<span style="color:{color}">[{bar}]</span> '
        f'<b>{score:.3f}</b>{suffix}</div>'
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def load_task(task_id: str):
    """Reset environment and populate UI with the chosen task."""
    env  = QueryforgeEnvironment()
    obs  = env.reset(task_id=task_id)
    task = REGISTRY.get(task_id)
    state = {"env": env, "history": [], "done": False}
    return (
        obs.task_description,       # task description box
        task.broken_query,          # pre-fill SQL editor with broken query
        "<div style='color:#6b7280;font-family:monospace'>Submit a query to see your score.</div>",
        "",                         # clear feedback
        [],                         # clear history table
        state,
        gr.update(interactive=True),  # enable submit button
    )


def submit_query(sql: str, state: dict):
    """Grade the submitted SQL and update all output components."""
    if state is None or "env" not in state:
        return (
            "<div style='color:red'>Load a task first.</div>",
            "", [], state,
        )
    if state.get("done"):
        return (
            "<div style='color:#6b7280'>Episode already ended. Load a new task.</div>",
            "", state["history"], state,
        )

    env   = state["env"]
    obs   = env.step(SQLAction(sql=sql.strip()))
    score = obs.reward or 0.0

    # ── Score HTML ────────────────────────────────────────────────────────────
    score_html = _score_html(score, obs.done)

    # ── Feedback (split into labelled sections) ───────────────────────────────
    sections = [p.strip() for p in obs.feedback.split("  ") if p.strip()]
    feedback_md = "\n\n".join(f"**{s.split(']')[0].lstrip('[').strip()}**{s.split(']',1)[1] if ']' in s else s}"
                               for s in sections)
    if obs.hint and not obs.done:
        feedback_md += f"\n\n> 💡 **Hint:** {obs.hint}"

    # ── History table ─────────────────────────────────────────────────────────
    status = "✓ Solved" if (obs.done and score >= 0.9) else ("⏹ Ended" if obs.done else "↻ Retry")
    state["history"].append([obs.attempt, f"{score:.3f}", obs.rows_returned, status])

    state["done"] = obs.done

    return score_html, feedback_md, state["history"], state


# ── UI layout ─────────────────────────────────────────────────────────────────

HEADER = """
# 🔧 QueryForge — SQL Debugger & Optimiser

Fix broken or slow SQL queries and get instant graded feedback.
{ai_status}
""".format(
    ai_status=(
        "🟢 **AI Judge active** — scores up to 1.0 (Anthropic)"
        if AI_JUDGE_ACTIVE else
        "🟡 **Deterministic mode** — max score 0.80 (set `ANTHROPIC_API_KEY` to enable AI judge)"
    )
)

with gr.Blocks(title="QueryForge") as demo:

    state = gr.State(None)

    gr.Markdown(HEADER)

    # ── Task selection row ────────────────────────────────────────────────────
    with gr.Row():
        task_dd  = gr.Dropdown(
            choices=TASK_CHOICES,
            value=TASK_CHOICES[0][1],
            label="Select Task",
            scale=4,
        )
        load_btn = gr.Button("Load Task ▶", variant="primary", scale=1)

    # ── Main two-column layout ────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            task_desc = gr.Textbox(
                label="📋 Task Description",
                lines=18,
                interactive=False,
                placeholder="Load a task to see the description and broken query…",
            )

        with gr.Column(scale=1):
            sql_input = gr.Code(
                label="✏️  Your SQL Query",
                language="sql",
                lines=12,
            )
            submit_btn = gr.Button(
                "Submit Query ⚡",
                variant="primary",
                interactive=False,
            )

    # ── Score + feedback ──────────────────────────────────────────────────────
    score_html    = gr.HTML(
        value="<div style='color:#6b7280;font-family:monospace'>Submit a query to see your score.</div>",
        label="Score",
    )
    feedback_display = gr.Markdown(label="Feedback")

    # ── Attempt history ───────────────────────────────────────────────────────
    history_table = gr.Dataframe(
        headers=["Step", "Score", "Rows Returned", "Status"],
        datatype=["number", "str", "number", "str"],
        label="📊 Attempt History",
        interactive=False,
        wrap=True,
    )

    # ── Wire up events ────────────────────────────────────────────────────────
    load_btn.click(
        load_task,
        inputs=[task_dd],
        outputs=[task_desc, sql_input, score_html, feedback_display, history_table, state, submit_btn],
    )

    submit_btn.click(
        submit_query,
        inputs=[sql_input, state],
        outputs=[score_html, feedback_display, history_table, state],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
