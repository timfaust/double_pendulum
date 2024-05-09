from typing import List

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import RolloutReturn

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.policies.common import CustomPolicy


class CustomSAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # rollout is one step in each environment
    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        result = super().collect_rollouts(*args, **kwargs)
        policy: CustomPolicy = self.policy
        envs: List[GeneralEnv] = [monitor.env for monitor in args[0].envs]
        progress = self.num_timesteps/envs[0].training_steps
        policy.after_rollout(envs, progress, *args, **kwargs)
        return result
