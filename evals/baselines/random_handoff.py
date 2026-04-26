"""
evals/baselines/random_handoff.py

Baseline: Session 2 receives random gibberish as the handoff note.

Expected S2 pass rate: ~8-12%
Tests whether handoff *content* matters (vs just *having* a note).
"""

import random as _random
from server.env import CrossSessionContinuityEnv


_LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)


def _make_random_handoff() -> str:
    """Generate plausible-looking but semantically empty handoff."""
    words = _LOREM.split()
    _random.shuffle(words)
    body = " ".join(words[:80])
    return (
        "TASK: " + " ".join(words[:10]) + "\n"
        "COMPLETED:\n- " + " ".join(words[10:20]) + "\n"
        "REMAINING:\n- " + " ".join(words[20:30]) + "\n"
        "KEY FUNCTIONS:\n- " + " ".join(words[30:40]) + "\n"
        "EDGE CASES:\n- " + " ".join(words[40:50]) + "\n"
        "NEXT STEPS:\n1. " + " ".join(words[50:60]) + "\n"
    )


def run_random_handoff_baseline(difficulty: str = "medium", n_episodes: int = 20, seed: int = 0):
    """
    Run Session 2 with random/gibberish handoff note.
    Measures whether handoff content quality matters.
    """
    _random.seed(seed)
    results = []

    for ep in range(n_episodes):
        env = CrossSessionContinuityEnv(difficulty=difficulty)
        env.task = env.task_gen.sample(seed=seed + ep)
        env.session = 2
        env.handoff = _make_random_handoff()
        env.handoff_parsed = True
        env.task = env.session_mgr.transition(env.task)

        visible = env.sandbox.run_tests(env.task.files, env.task.test_code)
        pass_rate = visible.passed / max(visible.total, 1)
        results.append(pass_rate)

    mean = sum(results) / len(results)
    return {"pass_rates": results, "mean": round(mean, 4), "label": "Random Handoff"}


if __name__ == "__main__":
    res = run_random_handoff_baseline()
    print(f"Random-Handoff Baseline — Mean Pass Rate: {res['mean']:.1%}")
