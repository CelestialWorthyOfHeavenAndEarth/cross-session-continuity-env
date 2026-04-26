"""
evals/ablations/no_compression_reward.py

Ablation: Remove the compression reward component from the rubric.
quality_score is forced to 0, so the agent gets no signal to keep handoffs short.

Expected effect: Handoffs become bloated (>400 tokens), Session 2 overwhelmed.
"""

from server.rewards.rubric import ContinuityRubric


class NoCompressionRubric(ContinuityRubric):
    """Rubric variant with quality_score (compression) zeroed out."""

    def score(self, visible_results, hidden_results, handoff,
              s2_edit_history, s2_failed_runs, invalid_actions):
        breakdown = super().score(
            visible_results, hidden_results, handoff,
            s2_edit_history, s2_failed_runs, invalid_actions,
        )
        # Recalculate total without quality
        total = (
            0.55 * breakdown.test_score
            + 0.00 * breakdown.quality_score   # zeroed
            + 0.15 * breakdown.linearity_score
            - breakdown.rewrite_penalty
            - breakdown.action_penalty
        )
        breakdown.total = round(max(0.0, total), 4)
        breakdown.quality_score = 0.0
        return breakdown
