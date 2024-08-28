import numpy as np

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.misc_helper import find_index_and_dict, get_state_values
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.past_action_number = 0
        self.reset()
        super().__init__(8 + self.past_action_number)

    def build_state(self, observation, env: GeneralEnv) -> np.ndarray:
        index, observation_dict = find_index_and_dict(observation, env)
        dirty_observation = observation_dict['X_meas'][index]

        u_con = observation_dict['U_con']
        action_memory = np.zeros(self.past_action_number)
        actions_to_copy = min(index, len(u_con), self.past_action_number)

        if actions_to_copy > 0:
            action_memory[-actions_to_copy:] = u_con[-actions_to_copy:]

        state_values = get_state_values(observation_dict, offset=index + 1 - len(observation_dict['T']))
        l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([
            state_values['x2'][1] / l_ges,
            state_values['v2'][0] / env.dynamics_func.max_velocity,
            state_values['c1'],
            state_values['c2']
        ])

        state = np.append(additional, dirty_observation.copy())
        state = np.append(state, action_memory)

        return state


class PastActionsSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        self.additional_actor_kwargs['net_arch'] = [256,512,256]
        self.additional_critic_kwargs['net_arch'] = self.additional_actor_kwargs['net_arch']
        super().__init__(*args, **kwargs)
