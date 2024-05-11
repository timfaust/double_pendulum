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
        envs: List[GeneralEnv] = [monitor.env for monitor in args[0].envs]
        progress = self.num_timesteps/envs[0].training_steps
        self.policy_class.progress = progress
        self.policy.after_rollout(envs, *args, **kwargs)
        return result

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        self.policy.after_train()
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                self.logger.record(f"actor/grads/{name}", param.grad.norm().item())
            self.logger.record(f"actor/weights/{name}", param.data.norm().item())

        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                self.logger.record(f"critic/grads/{name}", param.grad.norm().item())
            self.logger.record(f"critic/weights/{name}", param.data.norm().item())

