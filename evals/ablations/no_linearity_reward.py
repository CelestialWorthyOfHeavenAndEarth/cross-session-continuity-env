"""
evals/ablations/no_linearity_reward.py

Ablation: Remove the linearity reward component from the rubric.
linearity_score is forced to 0, so Session 2 thrashing is not penalised.

Expected effect: More write-revert cycles, noisier learning curve.
"""

from server.rewards.rubric import ContinuityRubric


class NoLinearityRubric(ContinuityRubric):
    """Rubric variant with linearity_score zeroed out."""

    def score(self, visible_results, hidden_results, handoff,
              s2_edit_history, s2_failed_runs, invalid_actions):
        breakdown = super().score(
            visible_results, hidden_results, handoff,
            s2_edit_history, s2_failed_runs, invalid_actions,
        )
        total = (
            0.55 * breakdown.test_score
            + 0.20 * breakdown.quality_score
            + 0.00 * breakdown.linearity_score   # zeroed
            - breakdown.rewrite_penalty
            - breakdown.action_penalty
        )
        breakdown.total = round(max(0.0, total), 4)
        breakdown.linearity_score = 0.0
        return breakdown
