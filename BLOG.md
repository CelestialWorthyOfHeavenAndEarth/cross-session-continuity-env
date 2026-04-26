---
title: "Teaching LLMs to Write Better Notes to Their Future Self"
thumbnail: /blog/assets/cross-session-continuity/baseline_vs_trained.png
authors:
  - user: Aswini-Kumar
---

# Teaching LLMs to Write Better Notes to Their Future Self

*Can reinforcement learning teach a coding agent to communicate better across sessions with zero shared memory?*

---

## The Problem

Every time you start a new chat with an LLM, it forgets everything from the last session.

For short tasks this is fine. For long ones — a multi-hour coding project, a research investigation, a debugging marathon — this is catastrophic. The model re-reads the same files, re-discovers the same bugs, and wastes your time.

Humans solve this with notes. Good ones. A developer leaving for the night writes: *"fixed the import error in utils.py, still need to handle the empty-list edge case in merge_intervals, run the tests first when you're back."*

Can we train an LLM to do the same?

## The Environment

**Cross-Session Continuity Env** is an RL environment built on OpenEnv where a coding agent must complete a task **across two separate sessions with zero shared memory**.

```
Session 1                              Session 2
─────────────────────────────────────────────────────────
Agent receives task + starter code     Agent receives ONLY handoff note
Agent works: read → write → test  ─>  Agent calls parse_handoff()
Agent ends: writes handoff note        Agent completes task → submit
                ↓
    [filesystem wiped]
    [function names randomized]
    [no code persists]
```

The agent has 6 tools: `read_file`, `write_file`, `run_tests`, `write_handoff`, `parse_handoff`, `submit`.

The handoff note has a strict structure (enforced by the validator):

```
TASK:          what the overall task is
COMPLETED:     what was implemented + verified
REMAINING:     what Session 2 must implement
KEY FUNCTIONS: function names, signatures
EDGE CASES:    constraints or tricky logic
NEXT STEPS:    ordered action list for Session 2
```

**Max 400 tokens. Max 5 lines of code. All 6 sections required.**

If the note doesn't meet these constraints, the validator rejects it (no penalty — retry is allowed). This forces the agent to develop *information-dense, structured communication* rather than just copy-pasting code.

## Reward Design

The reward is composable and anti-gaming:

| Component | Weight | Anti-gaming |
|-----------|--------|-------------|
| Tests (visible, Session 2) | 33% | Hidden tests at submit time |
| Tests (hidden) | 22% | Not accessible via `run_tests` |
| Handoff quality | 20% | Code-dump blocked by validator |
| Linearity | 15% | Thrash/rewrite detection |
| Penalties | 10% | Invalid actions, reconstruction |

40% of the test score comes from hidden test cases that are never revealed to the agent. This ensures the agent can't memorize specific test patterns.

## Training

We train **Qwen2.5-Coder-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) via Hugging Face TRL and Unsloth, on a Colab T4 GPU.

The training uses a 3-phase curriculum:
- **Epochs 1–2**: Easy tasks (step limit 20, 3 visible tests)  
- **Epochs 3–4**: Medium tasks (step limit 35, 5 visible tests)
- **Epochs 5–6**: Hard tasks (step limit 55, 8 visible tests)

This bootstraps the agent on simpler tasks before exposing it to harder generalization challenges.

## Results

The core question: does training actually improve the agent's ability to use its handoff note?

![Baseline vs Trained](plots/baseline_vs_trained.png)
*Session 2 test pass rate across 4 conditions. Error bars = ± std, 3 seeds.*

The trained agent achieves **~63% Session 2 test pass rate** vs **~8% for no handoff** and **~11% for random handoff**. This is a **+55 percentage point improvement** over the lower bound.

The reward curve shows clear learning:

![Reward Curve](plots/reward_curve.png)
*Total reward across training episodes. All 4 conditions on same axes. Band = ±1 std.*

And the training loss descends cleanly:

![Loss Curve](plots/loss_curve.png)
*Policy loss (top) + KL divergence (bottom) across training steps. Curriculum phases shown as shaded regions.*

## What the Agent Actually Learned

The most interesting result is *how the handoff notes changed*.

![Handoff Evolution](plots/handoff_diff_over_epochs.png)
*Token count per handoff section across 6 training epochs.*

- **Epoch 1**: ~700 tokens. Rambling. Code blocks everywhere. Repeats the task description verbatim. The NEXT STEPS section is almost empty.
- **Epoch 6**: ~175 tokens. Surgical. Zero code. COMPLETED shrinks (less over-documentation). NEXT STEPS grows to dominate — the most actionable information for Session 2.

The agent learned that Session 2 doesn't need to know *what was done*, it needs to know *exactly what to do next*. That's a genuine insight about communication.

## Ablation Study

![Ablation Study](plots/ablation_comparison.png)
*Removing any reward component degrades performance. All configs on same axes.*

- **No compression reward**: -16 pp. Agent produces bloated notes. Session 2 spends steps parsing instead of coding.  
- **No linearity reward**: -11 pp. Session 2 thrashes — rewrites code instead of building on it.  
- **No auxiliary reward**: -8 pp. Slower convergence; the shaped S1 rewards help bootstrap early.

## Why This Matters

The capability gap we're targeting — **structured cross-session state transfer** — is genuinely unsolved. Every production deployment of a coding agent hits this wall when tasks span multiple conversations.

The environment is designed to force the agent to develop a real skill, not to game a metric:
- Function names are randomized per episode (can't memorize by name)
- Hidden tests at submit time (can't overfit to visible tests)
- Validator blocks code dumps (must communicate structurally)

An agent that scores well here has actually learned to write better notes. That's the bet.

## Links

- **HF Space (live demo)**: https://huggingface.co/spaces/Aswini-Kumar/cross-session-continuity-env
- **Training Notebook**: https://colab.research.google.com/github/CelestialWorthyOfHeavenAndEarth/cross-session-continuity-env/blob/main/training/train_grpo.ipynb
- **GitHub**: https://github.com/CelestialWorthyOfHeavenAndEarth/cross-session-continuity-env
