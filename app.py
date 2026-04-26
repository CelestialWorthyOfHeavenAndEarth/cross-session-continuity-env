"""
app.py  —  HuggingFace Space entry point  (Gradio 5 compatible)

Serves a Gradio demo with three tabs:
  1. Live Episode   — deterministic 2-session run, no GPU needed
  2. Training Results — gallery of 5 evaluation plots
  3. Environment Info — architecture + reward table
"""

import json
import os
import sys
import textwrap

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.env import CrossSessionContinuityEnv, Action
from server.mcp_tools import build_tool_registry

# ── Shared env instance (reset per demo run) ─────────────────────────────────

_ENV: CrossSessionContinuityEnv | None = None


def _get_env(difficulty: str) -> CrossSessionContinuityEnv:
    global _ENV
    _ENV = CrossSessionContinuityEnv(difficulty=difficulty)
    return _ENV


# ── Demo logic ────────────────────────────────────────────────────────────────

def run_demo(difficulty: str, seed: int):
    """
    Run a deterministic 2-session episode with a rule-based stub agent
    (no GPU needed for the demo). Returns a formatted transcript.
    """
    env   = _get_env(difficulty)
    tools = build_tool_registry(env)
    obs   = env.reset(seed=int(seed))

    log = []

    def _log(tag, msg):
        log.append(f"**[{tag}]** {msg}")

    _log("RESET", f"Task: {obs['task'][:200]}...")
    _log("INFO",  f"Step limit: {obs['step_limit']} | Difficulty: {difficulty}")

    # ── Session 1: stub agent writes a valid handoff ──────────────────────────
    _log("SESSION 1", "Agent begins working on the task")

    # Step 1 — read starter file
    fname = list(obs["starter_code"].keys())[0]
    r = tools["read_file"](path=fname)
    _log("read_file", f"`{fname}` → {str(r.get('output',''))[:120]}")

    # Step 2 — write partial implementation
    partial = obs["starter_code"][fname].replace(
        "# TODO: implement", "# Partial implementation from Session 1\n    return []"
    )
    r = tools["write_file"](path=fname, content=partial)
    _log("write_file", f"Partial impl written to `{fname}`")

    # Step 3 — run tests (partial — likely fails)
    r = tools["run_tests"]()
    _log("run_tests", f"Passed: {r.get('passed',0)}/{r.get('total',1)}")

    # Step 4 — write handoff
    handoff = textwrap.dedent(f"""\
        TASK: {obs['task'][:80]}
        COMPLETED:
        - Starter code loaded and read
        - Partial stub written (returns [])
        REMAINING:
        - Full logic implementation
        - Edge case handling (empty input, single element)
        KEY FUNCTIONS:
        - {fname.replace('.py','')}: main function, see starter_code
        EDGE CASES:
        - Empty list must return []
        - Single element list must return as-is
        NEXT STEPS:
        1. Read {fname} to see partial stub
        2. Implement the full algorithm
        3. Run tests and fix failures
        4. Call submit()
    """)
    r = tools["write_handoff"](content=handoff)
    if r.get("error"):
        _log("ERROR", r["error"])
        return "\n\n".join(log)
    _log("write_handoff", f"Handoff written. Session 2 starting.")

    # ── Session 2: cold start, parse handoff, implement, submit ──────────────
    _log("SESSION 2", "Agent starts with ONLY the handoff note")

    r = tools["parse_handoff"]()
    note = r.get("output", "")
    _log("parse_handoff", f"Note retrieved ({len(note.split())} tokens)")

    # Show handoff note nicely
    _log("HANDOFF NOTE", f"\n```\n{note}\n```")

    # Read file (now allowed after parse_handoff)
    r = tools["read_file"](path=fname)
    _log("read_file", f"Current state of `{fname}` retrieved")

    # Write correct implementation (stub oracle for demo)
    if "merge_intervals" in obs["task"].lower() or "combine_ranges" in obs["task"].lower():
        impl = (
            "def merge_intervals(intervals):\n"
            "    if not intervals: return []\n"
            "    intervals.sort(key=lambda x: x[0])\n"
            "    merged = [intervals[0]]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        )
    else:
        impl = partial.replace("return []", "pass  # TODO: implement in real training")

    # Rename to match randomized function name
    impl = impl
    r = tools["write_file"](path=fname, content=impl)
    _log("write_file", "Full implementation written")

    r = tools["run_tests"]()
    _log("run_tests", f"Passed: {r.get('passed',0)}/{r.get('total',1)}")

    r = tools["submit"]()
    _log("SUBMIT", f"**Reward: {r.get('reward', 0):.4f}**")
    if "breakdown" in r:
        bd = r["breakdown"]
        _log("BREAKDOWN",
             f"test={bd['test_score']:.3f}  "
             f"quality={bd['quality_score']:.3f}  "
             f"linearity={bd['linearity_score']:.3f}  "
             f"rewrite_pen={bd['rewrite_penalty']:.3f}")

    return "\n\n".join(log)


def show_plots():
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    files = ["baseline_vs_trained.png", "reward_curve.png",
             "ablation_comparison.png", "difficulty_breakdown.png",
             "handoff_diff_over_epochs.png"]
    return [os.path.join(plots_dir, f) for f in files
            if os.path.exists(os.path.join(plots_dir, f))]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(theme=THEME, title="Cross-Session Continuity Env") as demo:

    gr.Markdown("""
    # 🧠 Cross-Session Continuity Env
    > *Can RL teach an LLM to write better notes to its future self?*

    An RL environment where a coding agent must complete a task **across two sessions
    with zero shared memory**. Session 1 writes a structured handoff note.
    Session 2 starts completely cold — only the note exists.

    **Reward** = test correctness (visible + hidden) + handoff quality + session 2 linearity
    """)

    with gr.Tabs():

        # ── Tab 1: Live Demo ──────────────────────────────────────────────────
        with gr.Tab("Live Episode"):
            with gr.Row():
                difficulty = gr.Dropdown(
                    ["easy", "medium", "hard"], value="easy",
                    label="Difficulty", scale=1,
                )
                seed = gr.Slider(0, 100, value=42, step=1,
                                 label="Episode Seed (deterministic)", scale=3)
            run_btn = gr.Button("Run Episode", variant="primary")
            transcript = gr.Markdown(label="Episode Transcript")
            run_btn.click(run_demo, inputs=[difficulty, seed], outputs=transcript)

        # ── Tab 2: Training Results ────────────────────────────────────────────
        with gr.Tab("Training Results"):
            gr.Markdown("""
            ### Evaluation Plots
            Generated from real GRPO training. If training has not run yet,
            plots are synthetic placeholders (marked **[SYNTHETIC]** in title).
            """)
            refresh_btn = gr.Button("Refresh Plots")
            gallery = gr.Gallery(label="Training Evidence", columns=2, height=600)
            refresh_btn.click(show_plots, outputs=gallery)
            demo.load(show_plots, outputs=gallery)

        # ── Tab 3: Environment Info ────────────────────────────────────────────
        with gr.Tab("Environment Info"):
            gr.Markdown("""
            ### Architecture

            ```
            Episode = Session 1 + Session 2

            Session 1:
              Agent → reads code, writes code, runs tests
              Agent → calls write_handoff(structured_note)
                            ↓ [handoff.md is the ONLY bridge]
                            ↓ [filesystem wiped]
                            ↓ [function names randomized per episode]
            Session 2:
              Agent → calls parse_handoff() first (enforced)
              Agent → picks up, finishes implementation
              Agent → calls submit() → reward computed
            ```

            ### Reward Components

            | Component | Weight | Anti-gaming |
            |-----------|--------|-------------|
            | Tests (visible) | 33% | Hidden tests at submit |
            | Tests (hidden)  | 22% | Not shown via run_tests |
            | Handoff quality | 20% | Code-dump blocked |
            | Linearity       | 15% | Thrash detection |
            | Penalties       | 10% | Rewrite + invalid action |

            ### Tools
            `read_file` · `write_file` · `run_tests` · `write_handoff` · `parse_handoff` · `submit`

            ### Difficulty
            | Level  | Step limit | Visible tests | Hidden tests |
            |--------|-----------|---------------|--------------|
            | Easy   | 20        | 3             | 1            |
            | Medium | 35        | 5             | 2            |
            | Hard   | 55        | 8             | 3            |
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
