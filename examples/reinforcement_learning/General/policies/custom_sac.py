from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import RolloutReturn

from examples.reinforcement_learning.General.policies.common import CustomPolicy


class CustomSAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # rollout is one step in each environment
    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        result = super().collect_rollouts(*args, **kwargs)
        policy: CustomPolicy = self.policy
        policy.after_rollout(self.num_timesteps, *args, **kwargs)
        return result
