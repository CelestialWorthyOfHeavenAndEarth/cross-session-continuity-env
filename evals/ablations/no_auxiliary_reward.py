"""
evals/ablations/no_auxiliary_reward.py

Ablation: Disable the Session 1 auxiliary shaped rewards (AuxiliaryRewarder).
The agent only receives signal from the terminal reward at submit().

Expected effect: Slower convergence — agent requires more episodes to learn
basic handoff structure because early episodes have zero reward signal.
"""

from server.rewards.auxiliary import AuxiliaryRewarder


class NoAuxiliaryRewarder(AuxiliaryRewarder):
    """AuxiliaryRewarder variant that always returns 0."""

    def s1_reward(self, test_result, task) -> float:
        return 0.0

    def decay_factor(self, epoch: int, total_epochs: int) -> float:
        return 0.0
