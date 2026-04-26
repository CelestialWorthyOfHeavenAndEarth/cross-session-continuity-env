"""
server/rewards/rubric.py

Main reward rubric for the Cross-Session Continuity environment.

Reward is terminal — computed only at submit() at the end of Session 2.
Designed to be hard to game (see inline comments for each anti-gaming measure).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

HANDOFF_TOKEN_BUDGET = 300


@dataclass
class RewardBreakdown:
    total: float
    test_score: float
    quality_score: float
    linearity_score: float
    rewrite_penalty: float
    action_penalty: float


class ContinuityRubric:
    """
    Composable rubric scoring Session 2 performance.

    Weights:
      0.55  test_score       (visible 60% + hidden 40%)
      0.20  quality_score    (structure + compression + density)
      0.15  linearity_score  (no thrashing, no failed-run spirals)
      -     rewrite_penalty  (penalises ignoring the handoff)
      -     action_penalty   (penalises repeated invalid actions)
    """

    def score(
        self,
        visible_results,
        hidden_results,
        handoff: str,
        s2_edit_history: List[Dict[str, Any]],
        s2_failed_runs: int,
        invalid_actions: int,
    ) -> RewardBreakdown:

        # ── Component 1: Test correctness (soft — each test counts) ──────────
        # Hidden tests carry 40% of test_score to prevent visible-test overfitting.
        # Both visible and hidden pass rates are continuous (0.0 – 1.0), not binary.
        v_score = visible_results.passed / max(visible_results.total, 1)
        h_score = hidden_results.passed  / max(hidden_results.total,  1)
        test_score = round(0.6 * v_score + 0.4 * h_score, 4)

        # Compilation bonus: if code at least runs (partial credit even with 0 tests)
        compile_bonus = 0.05 if visible_results.compiled else 0.0

        # ── Component 2: Handoff quality (soft — each section adds score) ────
        quality_score = self._handoff_quality(handoff)

        # ── Component 3: Linearity (soft — continuous, not binary) ───────────
        linearity_score = self._linearity(s2_edit_history, s2_failed_runs)

        # ── Penalties (capped so one mistake doesn't kill the episode) ───────
        rewrite_penalty = self._rewrite_penalty(s2_edit_history)
        action_penalty  = round(min(invalid_actions * 0.02, 0.10), 4)

        # ── Total: all components contribute positively; penalties are small ──
        total = (
            0.50 * test_score
            + compile_bonus
            + 0.20 * quality_score
            + 0.15 * linearity_score
            - rewrite_penalty
            - action_penalty
        )

        return RewardBreakdown(
            total=round(max(0.0, total), 4),
            test_score=test_score,
            quality_score=quality_score,
            linearity_score=linearity_score,
            rewrite_penalty=rewrite_penalty,
            action_penalty=action_penalty,
        )

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _handoff_quality(self, handoff: str) -> float:
        """
        Multi-factor quality score. Replaces naive token count.

        Sub-components:
          0.40  compression  — within budget, degrades gracefully above it
          0.30  structure    — required sections present
          0.20  density      — unique-word ratio penalises repetitive padding
          0.10  formatting   — bullet lists signal organised thought
        """
        if not handoff:
            return 0.0

        score = 0.0
        tokens = handoff.split()
        token_count = len(tokens)

        # Compression sub-score
        if token_count <= HANDOFF_TOKEN_BUDGET:
            score += 0.40
        else:
            overage = token_count - HANDOFF_TOKEN_BUDGET
            score += max(0.0, 0.40 - (overage / HANDOFF_TOKEN_BUDGET) * 0.40)

        # Structure sub-score
        sections = ["COMPLETED:", "REMAINING:", "KEY FUNCTIONS:", "NEXT STEPS:"]
        present = sum(1 for s in sections if s in handoff)
        score += 0.30 * (present / len(sections))

        # Information density sub-score
        unique_ratio = len(set(tokens)) / max(token_count, 1)
        score += 0.20 * min(unique_ratio * 2, 1.0)

        # Formatting bonus
        has_bullets = any(
            ln.strip().startswith(("-", "*", "1.", "TODO"))
            for ln in handoff.split("\n")
        )
        score += 0.10 if has_bullets else 0.0

        return round(score, 4)

    def _linearity(self, edit_history: List[Dict], failed_runs: int) -> float:
        """
        Penalises thrashing (write → revert → write) and repeated failed test runs.

        Replaces the v1 re-read counting metric, which was easy to game by
        simply not re-reading files at all.
        """
        if not edit_history:
            return 0.5  # neutral if no edits (session 2 did nothing)

        thrash_count = sum(
            1
            for i in range(1, len(edit_history))
            if edit_history[i]["new"] == edit_history[i - 1]["prev"]
        )
        thrash_penalty = min(thrash_count * 0.10, 0.50)
        run_penalty    = min(failed_runs   * 0.05, 0.30)

        return round(max(0.0, 1.0 - thrash_penalty - run_penalty), 4)

    def _rewrite_penalty(self, edit_history: List[Dict]) -> float:
        """
        If Session 2 wrote large volumes to files that were previously empty,
        it likely reconstructed from pretrained priors, not the handoff note.
        Fires when: no pre-existing content AND > 500 chars written.
        """
        if not edit_history:
            return 0.0
        total_written   = sum(len(e["new"])  for e in edit_history)
        total_previous  = sum(len(e["prev"]) for e in edit_history)
        if total_previous == 0 and total_written > 500:
            return 0.15
        return 0.0
