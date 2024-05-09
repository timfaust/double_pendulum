import numpy as np
from examples.reinforcement_learning.General.policies.common import DefaultTranslator, DefaultActor, CustomPolicy, \
    DefaultCritic


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.action_memory = None
        self.past_action_number = 2
        self.reset()
        super().__init__(4 + self.past_action_number)

    def extract_observation(self, state: np.ndarray) -> np.ndarray:
        return state[:4]

    def build_state(self, clean_observation: np.ndarray, dirty_observation: np.ndarray, clean_action: float, dirty_action: float) -> np.ndarray:
        self.action_memory = np.append(self.action_memory, clean_action)
        return np.append(clean_observation, self.action_memory[-self.past_action_number:])

    def reset(self):
        self.action_memory = np.zeros(self.past_action_number)


class PastActionsActor(DefaultActor):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PastActionsSACPolicy(CustomPolicy):
    actor_class = PastActionsActor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
