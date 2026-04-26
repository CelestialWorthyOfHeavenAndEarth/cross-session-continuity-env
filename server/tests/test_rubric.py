"""
server/tests/test_rubric.py

Unit tests for ContinuityRubric and AuxiliaryRewarder.
"""

import pytest
from server.rewards.rubric import ContinuityRubric
from server.rewards.auxiliary import AuxiliaryRewarder
from server.sandbox import TestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rubric():
    return ContinuityRubric()


@pytest.fixture
def aux():
    return AuxiliaryRewarder()


def _results(passed, total, compiled=True):
    return TestResult(passed=passed, total=total, compiled=compiled, summary="")


def _good_handoff():
    return (
        "TASK: Implement merge_intervals function.\n"
        "COMPLETED:\n- Sorted intervals logic done\n- Basic merge loop written\n"
        "REMAINING:\n- Edge case: empty list\n- Edge case: touching intervals\n"
        "KEY FUNCTIONS:\n- merge_intervals(intervals) -> list\n"
        "EDGE CASES:\n- Empty input returns []\n- Single interval returns as-is\n"
        "NEXT STEPS:\n1. Add empty check\n2. Handle touching intervals\n3. Run hidden tests\n"
    )


# ---------------------------------------------------------------------------
# Rubric: total reward
# ---------------------------------------------------------------------------

class TestContinuityRubric:

    def test_perfect_score(self, rubric):
        breakdown = rubric.score(
            visible_results=_results(5, 5),
            hidden_results=_results(2, 2),
            handoff=_good_handoff(),
            s2_edit_history=[{"path": "solution.py", "prev": "stub", "new": "impl"}],
            s2_failed_runs=0,
            invalid_actions=0,
        )
        assert breakdown.total > 0.7, "Perfect input should score > 0.70"
        assert breakdown.test_score == 1.0

    def test_zero_tests_passed(self, rubric):
        breakdown = rubric.score(
            visible_results=_results(0, 5),
            hidden_results=_results(0, 2),
            handoff=_good_handoff(),
            s2_edit_history=[],
            s2_failed_runs=0,
            invalid_actions=0,
        )
        assert breakdown.test_score == 0.0
        assert breakdown.total < 0.5

    def test_rewrite_penalty_fires(self, rubric):
        """Session 2 writes a lot to previously-empty files → reconstruction."""
        big_write = "x" * 600
        breakdown = rubric.score(
            visible_results=_results(5, 5),
            hidden_results=_results(2, 2),
            handoff=_good_handoff(),
            s2_edit_history=[{"path": "solution.py", "prev": "", "new": big_write}],
            s2_failed_runs=0,
            invalid_actions=0,
        )
        assert breakdown.rewrite_penalty == 0.15

    def test_action_penalty_caps(self, rubric):
        breakdown = rubric.score(
            visible_results=_results(5, 5),
            hidden_results=_results(2, 2),
            handoff=_good_handoff(),
            s2_edit_history=[],
            s2_failed_runs=0,
            invalid_actions=100,  # many invalid actions
        )
        assert breakdown.action_penalty == 0.10  # caps at 0.10

    def test_thrash_linearity(self, rubric):
        """Repeated revert writes penalise linearity."""
        edits = [
            {"path": "f.py", "prev": "v1", "new": "v2"},
            {"path": "f.py", "prev": "v2", "new": "v1"},  # revert
            {"path": "f.py", "prev": "v1", "new": "v2"},
            {"path": "f.py", "prev": "v2", "new": "v1"},  # revert again
        ]
        breakdown = rubric.score(
            visible_results=_results(3, 5),
            hidden_results=_results(1, 2),
            handoff=_good_handoff(),
            s2_edit_history=edits,
            s2_failed_runs=3,
            invalid_actions=0,
        )
        assert breakdown.linearity_score < 0.6   # 2 reverts + 3 failed runs -> penalised

    def test_total_bounded(self, rubric):
        """Total reward is always in [0, 1]."""
        breakdown = rubric.score(
            visible_results=_results(0, 5),
            hidden_results=_results(0, 2),
            handoff="",  # empty handoff
            s2_edit_history=[{"path": "f.py", "prev": "", "new": "x" * 1000}],
            s2_failed_runs=10,
            invalid_actions=100,
        )
        assert 0.0 <= breakdown.total <= 1.0


# ---------------------------------------------------------------------------
# Rubric: handoff quality sub-scorer
# ---------------------------------------------------------------------------

class TestHandoffQuality:

    def test_empty_handoff(self, rubric):
        assert rubric._handoff_quality("") == 0.0

    def test_quality_with_all_sections(self, rubric):
        score = rubric._handoff_quality(_good_handoff())
        assert score > 0.6

    def test_long_handoff_penalised(self, rubric):
        bloated = _good_handoff() + " lorem ipsum" * 100
        short_score   = rubric._handoff_quality(_good_handoff())
        bloated_score = rubric._handoff_quality(bloated)
        assert bloated_score < short_score

    def test_no_bullets_lower_score(self, rubric):
        no_bullets = (
            "TASK: do something.\n"
            "COMPLETED: nothing.\n"
            "REMAINING: everything.\n"
            "KEY FUNCTIONS: some_func.\n"
            "EDGE CASES: none.\n"
            "NEXT STEPS: do the thing.\n"
        )
        with_bullets = no_bullets.replace("COMPLETED: nothing.", "COMPLETED:\n- nothing.")
        assert rubric._handoff_quality(with_bullets) >= rubric._handoff_quality(no_bullets)


# ---------------------------------------------------------------------------
# AuxiliaryRewarder
# ---------------------------------------------------------------------------

class TestAuxiliaryRewarder:

    def test_compile_bonus(self, aux):
        result = TestResult(passed=0, total=1, compiled=True, summary="")
        reward = aux.s1_reward(result, task=None)
        assert reward == pytest.approx(0.05, abs=1e-4)

    def test_per_test_bonus(self, aux):
        result = TestResult(passed=3, total=5, compiled=True, summary="")
        reward = aux.s1_reward(result, task=None)
        assert reward == pytest.approx(0.05 + 3 * 0.02, abs=1e-4)

    def test_decay_at_start(self, aux):
        assert aux.decay_factor(0, 100) == pytest.approx(1.0, abs=1e-4)

    def test_decay_at_cutoff(self, aux):
        # At 60% of epochs, decay should reach 0
        assert aux.decay_factor(60, 100) == pytest.approx(0.0, abs=1e-4)

    def test_decay_after_cutoff(self, aux):
        # Beyond 60% stays at 0
        assert aux.decay_factor(80, 100) == 0.0
