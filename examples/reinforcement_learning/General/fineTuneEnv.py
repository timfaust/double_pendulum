import json

from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.misc_helper import *
from examples.reinforcement_learning.General.reward_functions import get_state_values
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import *
from param_helper import load_env_attributes
from double_pendulum.simulation.simulation import Simulator


class FineTuneEnv(CustomEnv):
    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(
            self,
            robot,
            seed,
            env_params,
            data,
            dyn_function=None,
    ):

        self.seed = seed
        self.robot = robot
        self.env_params = env_params
        self.data = data

        self.actions_in_state = data["actions_in_state"] == 1
        self.reward_name = env_params["reward_function"]
        dynamic_class_name = data["dynamic_class"]

        if dyn_function is None:
            dynamics_function, reset_function, reward_function, number_of_envs, use_same_env = load_env_attributes(
                env_params)

            dynamics_function, _, _ = dynamics_function(robot, data["dt"], data["max_torque"],
                                                        globals()[dynamic_class_name])
        else:
            _, reset_function, reward_function, number_of_envs, use_same_env = load_env_attributes(
                env_params)
            dynamics_function = dyn_function

        low_pos = [-0.5, 0, 0, 0]
        if dynamic_class_name == "custom_dynamics_func_PI":
            low_pos = [-1.0, 0, 0, 0]

        self.same_env = True
        self.n_envs = number_of_envs
        self.virtual_sensor_state_tracking = [0.0, 0.0]
        self.reset_function = lambda: reset_function(low_pos)
        self.reward_function = lambda obs, act, state_dict: reward_function(obs, act, robot, self.dynamics_func,
                                                                            state_dict,
                                                                            self.virtual_sensor_state_tracking)

        if not self.actions_in_state:
            obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
        else:
            obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

        super().__init__(
            dynamics_function,
            self.reward_function,
            kill_switch,
            self.custom_reset,
            obs_space,
            act_space,
            True
        )

        self.state_dict = {"T": [], "X_meas": [], "U_con": [], "push": [], "plant": self.dynamics_func.simulator.plant,
                           "max_episode_steps": self.max_episode_steps, "current_force": []}
        self.dynamics_func.simulator.plant.state_dict = self.state_dict

    def custom_reset(self):
        observation = self.reset_function()
        if self.actions_in_state:
            observation = np.append(observation, np.zeros(2))
        return observation

    def get_envs(self, log_dir):
        if self.same_env:
            dynamics_function = self.dynamics_func
        else:
            dynamics_function = None

        envs = make_vec_env(
            env_id=FineTuneEnv,
            n_envs=self.n_envs,
            env_kwargs={
                "robot": self.robot,
                "seed": self.seed,
                "env_params": self.env_params,
                "data": self.data,
                "dyn_function": dynamics_function,
            },
            monitor_dir=log_dir,
            seed=self.seed
        )
        return envs

    def update_observation(self, obs):
        self.observation = obs

    def step(self, action):
        last_actions = self.observation[-2:]

        if self.reward_name == "saturated_distance_from_target":
            reward = self.reward_func(self.observation, action, self.state_dict)
        else:
            reward = self.reward_func(self.observation, action, self.state_dict) / self.max_episode_steps

        info = {}
        truncated = False
        terminated = False
        if self.actions_in_state:
            self.observation[-1] = last_actions[0]
            self.observation[-2] = action[0]

        return self.observation, reward, terminated, truncated, info

