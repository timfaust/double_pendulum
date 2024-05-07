import numpy as np
from examples.reinforcement_learning.General.policies.common import Translator, CustomActor, CustomPolicy


class DefaultTranslator(Translator):
    def __init__(self):
        super().__init__(4)

    def extract_observation(self, state: np.ndarray) -> np.ndarray:
        return state

    def build_state(self, observation: np.ndarray, action: float) -> np.ndarray:
        return observation

    def reset(self):
        pass


class DefaultActor(CustomActor):

    @classmethod
    def get_translator(cls) -> DefaultTranslator:
        return DefaultTranslator()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DefaultSACPolicy(CustomPolicy):
    actor_class = DefaultActor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
