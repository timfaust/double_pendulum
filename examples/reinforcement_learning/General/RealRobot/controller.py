from typing import Type
from sbx import SAC
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
import numpy as np

class GeneralController(AbstractController):
    def __init__(self, dynamics_func: Type[double_pendulum_dynamics_func], model_path, actions_in_state, mode="run"):
        super().__init__()

        self.model = SAC.load(model_path, print_system_info=True)
        self.dynamics_func = dynamics_func
        self.dt = dynamics_func.dt
        self.scaling = dynamics_func.scaling
        self.integrator = dynamics_func.integrator
        self.actions_in_state = actions_in_state
        self.last_actions = [0, 0]
        self.mode = mode

    def get_control_output_(self, x, t=None):

        if self.mode == "connect":
            u = self.dynamics_func.unscale_action(0)
            return u

        if self.actions_in_state:
            x = np.append(x, self.last_actions)

        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            action = self.model.predict(observation=obs, deterministic=True)
            u = self.dynamics_func.unscale_action(action)
        else:
            action = self.model.predict(observation=x, deterministic=True)
            u = self.dynamics_func.unscale_action(action)

        if self.actions_in_state:
            self.last_actions[-1] = self.last_actions[-2]
            if not np.all(u == 0):
                max_abs_value_index = np.abs(u).argmax()
                self.last_actions[-2] = u[max_abs_value_index]
            else:
                self.last_actions[-2] = 0
        return u
