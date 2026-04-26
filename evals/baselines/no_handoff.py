"""
evals/baselines/no_handoff.py

Baseline: Session 2 receives NO handoff note at all — blank context.

Expected S2 pass rate: ~5-10%
This is the lower bound. Any trained agent should be comfortably above this.
"""

from server.env import CrossSessionContinuityEnv, Action


def run_no_handoff_baseline(difficulty: str = "medium", n_episodes: int = 20, seed: int = 0):
    """
    Run Session 2 with no handoff note. Session 1 is skipped entirely.
    Session 2 starts with only the task description and blank starter code.

    Returns dict with pass_rates list and mean pass rate.
    """
    import random
    random.seed(seed)

    results = []

    for ep in range(n_episodes):
        env = CrossSessionContinuityEnv(difficulty=difficulty)
        # Skip Session 1: manually inject a blank handoff and transition
        env.task = env.task_gen.sample(seed=seed + ep)
        env.session = 2
        env.handoff = ""  # empty — no information
        env.handoff_parsed = True  # bypass parse_handoff requirement
        env.task = env.session_mgr.transition(env.task)

        # Session 2 attempts to submit immediately (no context)
        visible = env.sandbox.run_tests(env.task.files, env.task.test_code)
        pass_rate = visible.passed / max(visible.total, 1)
        results.append(pass_rate)

    mean = sum(results) / len(results)
    return {"pass_rates": results, "mean": round(mean, 4), "label": "No Handoff"}


if __name__ == "__main__":
    res = run_no_handoff_baseline()
    print(f"No-Handoff Baseline — Mean Pass Rate: {res['mean']:.1%}")
