import copy
import json
import os
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import numpy as np
from dynamics_functions import load_param
class GeneralEnv(CustomEnv):

    def __init__(
        self,
        dynamic_function,
        reward_function,
        termination_function,
        reset_function,
        torque_limit
    ):
        self.mpar = load_param(torque_limit=torque_limit)
        self.observation_dict = {"T": [], 'X_meas': [], 'X_real': [], 'U_con': [], 'U_real': [], "push": [],
                                 "mpar": self.mpar}
        self.observation_dict_old = None    # Updated after reset is called to store the old values
        self.reset_function = reset_function
        super().__init__(
            dynamic_function,
            reward_function,
            termination_function,
            self.custom_reset,
        )
        self.dynamics_func.simulator.plant.observation_dict = self.observation_dict

    def custom_reset(self):

        self.observation_dict_old = copy.deepcopy(self.observation_dict)
        if 'dynamics_func' not in self.observation_dict:
            self.observation_dict['dynamics_func'] = self.dynamics_func
        for key in self.observation_dict:
            if key != 'dynamics_func' and key != 'max_episode_steps' and key != 'mpar':
                self.observation_dict[key].clear()

        clean_observation = np.array(self.reset_function())
        self.append_observation_dict(clean_observation, clean_observation, 0.0)
        self.observation_dict['U_con'].append(0.0)

        return clean_observation
    def append_observation_dict(self, clean_observation, dirty_observation, dirty_action: float):
        time = 0
        if len(self.observation_dict['T']) > 0:
            time = self.dynamics_func.dt + self.observation_dict['T'][-1]
        self.observation_dict['T'].append(np.round(time, decimals=5))
        self.observation_dict['U_real'].append(dirty_action)
        self.observation_dict['X_meas'].append(dirty_observation)
        self.observation_dict['X_real'].append(clean_observation)
