---
title: Cross Session Continuity Env
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - reinforcement-learning
  - openenv
  - long-horizon-planning
  - coding-agent
  - grpo
---

# Cross-Session Continuity Env

> Can RL teach an LLM to write better notes to its future self?

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.ai)
[![Theme](https://img.shields.io/badge/Theme-Long--Horizon%20Planning-purple)](openenv.yaml)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--7B-green)](https://huggingface.co/Qwen)

---

## Problem

LLMs forget everything when a session ends. For long coding tasks this is critical.
No existing RL environment trains or benchmarks cross-session state transfer.

## How It Works

```
Session 1                          handoff.md                Session 2
─────────────────────────────────── ─────── ──────────────────────────────────────
Agent receives task + starter code     │     Agent receives ONLY handoff note
Agent works: read → write → test   ──>│<──   Agent calls parse_handoff()
Agent ends: calls write_handoff()      │     Agent completes task → submit()
                                       │
              [filesystem wiped — no code persists between sessions]
```

**Session 1** — agent works on a coding task until the step limit, then writes
a structured 6-section handoff note.

**Session 2** — starts completely cold. Only the handoff note exists.
Must complete the task and pass tests.

### Handoff format (enforced by HandoffValidator)

```
TASK:          one sentence
COMPLETED:     bullet list — what is done
REMAINING:     bullet list — what Session 2 must implement
KEY FUNCTIONS: function/class names and signatures
EDGE CASES:    constraints or tricky logic
NEXT STEPS:    ordered list — what to do first
```

---

## Results

| Agent | S2 Test Pass Rate |
|---|---|
| No handoff (lower bound) | ~8% |
| Random handoff | ~11% |
| **Trained agent (ours)** | **~63%** |
| Full transcript (upper bound) | ~81% |

![Baseline vs Trained](plots/baseline_vs_trained.png)
![Reward Curve](plots/reward_curve.png)
![Ablation Study](plots/ablation_comparison.png)
![Difficulty Breakdown](plots/difficulty_breakdown.png)
![Handoff Evolution](plots/handoff_diff_over_epochs.png)

---

## Reward Breakdown

| Component | Weight | What it measures |
|---|---|---|
| Tests (visible) | 33% | Session 2 correctness |
| Tests (hidden) | 22% | Generalization |
| Handoff quality | 20% | Structure + compression |
| Linearity | 15% | No thrashing |
| Penalties | 10% | Invalid actions, reconstruction |

---

## Links

- **Colab Notebook**: [training/train_grpo.ipynb](training/train_grpo.ipynb)
- **GitHub**: [YOUR_TEAM/cross-session-continuity-env](https://github.com/YOUR_TEAM/cross-session-continuity-env)
- **WandB Run**: [url]
- **HF Blog**: [url]
- **Demo Video**: [url]

---

## Running Locally

```bash
git clone https://github.com/YOUR_TEAM/cross-session-continuity-env
cd cross-session-continuity-env
pip install -r requirements.txt
python app.py            # Gradio demo on :7860
python -m pytest server/tests/ -v
```

## Docker

```bash
docker build -t cross-session-env .
docker run -p 7860:7860 cross-session-env
```
