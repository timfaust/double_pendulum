import numpy as np

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.misc_helper import find_index_and_dict
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.past_action_number = 10
        self.reset()
        super().__init__(4 + self.past_action_number)

    def build_state(self, observation, env: GeneralEnv) -> np.ndarray:
        index, observation_dict = find_index_and_dict(observation, env)
        dirty_observation = observation_dict['X_meas'][index]

        u_con = observation_dict['U_con']
        action_memory = np.zeros(self.past_action_number)
        actions_to_copy = min(index, len(u_con), self.past_action_number)

        if actions_to_copy > 0:
            action_memory[-actions_to_copy:] = u_con[:actions_to_copy]

        state = np.append(dirty_observation.copy(), action_memory)
        return state


class PastActionsSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        self.additional_actor_kwargs['net_arch'] = [256, 128, 64, 32]
        self.additional_critic_kwargs['net_arch'] = self.additional_actor_kwargs['net_arch']
        super().__init__(*args, **kwargs)
