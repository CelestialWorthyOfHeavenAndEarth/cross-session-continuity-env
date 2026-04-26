"""
server/rewards/auxiliary.py

Session 1 auxiliary shaped rewards.

Problem: Pure GRPO on a delayed 2-session reward causes early training plateau
because Session 1 actions get no direct signal.

Fix: Small auxiliary rewards during Session 1 for compilation and test passage,
decayed over training epochs so late training relies purely on the real reward.
"""


class AuxiliaryRewarder:
    """
    Provides small shaped rewards during Session 1 to bootstrap credit assignment.

    These rewards are multiplied by decay_factor() so they fade out at 60% of
    training — the agent transitions to optimising the real terminal reward.
    """

    COMPILE_BONUS: float = 0.05   # reward for code that at least compiles
    PER_TEST_BONUS: float = 0.02  # reward per visible test passed in Session 1
    DECAY_CUTOFF: float = 0.60    # fade-out completes at 60% of total epochs

    def s1_reward(self, test_result, task) -> float:  # noqa: ARG002
        """
        Compute auxiliary reward after a run_tests action in Session 1.

        Args:
            test_result: object with .compiled (bool) and .passed (int).
            task: current Task object (unused here, available for extensions).

        Returns:
            Float reward — intentionally small to avoid dominating terminal reward.
        """
        reward = 0.0
        if test_result.compiled:
            reward += self.COMPILE_BONUS
        reward += self.PER_TEST_BONUS * test_result.passed
        return round(reward, 4)

    def decay_factor(self, epoch: int, total_epochs: int) -> float:
        """
        Linear decay from 1.0 → 0.0, reaching zero at DECAY_CUTOFF * total_epochs.

        Args:
            epoch: current training epoch (0-indexed).
            total_epochs: total planned training epochs.

        Returns:
            Float multiplier in [0.0, 1.0].
        """
        cutoff = total_epochs * self.DECAY_CUTOFF
        return max(0.0, 1.0 - (epoch / cutoff))
