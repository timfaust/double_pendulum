import numpy as np

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.clean_action_memory = None
        self.past_action_number = 200
        self.reset()
        super().__init__(4 + self.past_action_number)

    def build_state(self, env: GeneralEnv, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        self.clean_action_memory = np.append(self.clean_action_memory, clean_action)
        state = np.append(dirty_observation.copy(), self.clean_action_memory[-self.past_action_number:])
        return state

    def reset(self):
        self.clean_action_memory = np.zeros(self.past_action_number)


class PastActionsSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        self.additional_actor_kwargs['net_arch'] = [256, 128, 64, 32]
        self.additional_critic_kwargs['net_arch'] = self.additional_actor_kwargs['net_arch']
        super().__init__(*args, **kwargs)
