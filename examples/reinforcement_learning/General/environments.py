import json

from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.misc_helper import updown_reset, balanced_reset, no_termination, \
    noisy_reset, low_reset, high_reset, random_reset, semi_random_reset, debug_reset, kill_switch
from examples.reinforcement_learning.General.reward_functions import get_state_values
from examples.reinforcement_learning.General.visualizer import Visualizer
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import default_dynamics, random_dynamics, \
    random_push_dynamics, push_dynamics, load_param, custom_dynamics_func_PI, custom_dynamics_func_4PI, real_robot
from examples.reinforcement_learning.General.reward_functions import future_pos_reward, pos_reward, quadratic_rew, saturated_distance_from_target, score_reward
from double_pendulum.simulation.simulation import Simulator


class GeneralEnv(CustomEnv):

    def __init__(
        self,
        env_type,
        param_name,
        policy,
        seed,
        path="parameters.json",
        is_evaluation_environment=False,
        existing_dynamics_function=None,
        existing_plant=None
    ):

        self.policy = policy
        self.translator = policy.actor_class.get_translator() if any("SAC" in attr or "sac" in attr for attr in
                                                                     dir(self.policy) if isinstance(attr, str)) else None
        # Translator is only for SAC Algorithms to translate State of model for Networks
        self.seed = seed
        self.is_evaluation_environment = is_evaluation_environment
        self.param_name = param_name
        self.env_type = env_type
        self.param_data = json.load(open(path))[param_name]

        self.type = None
        self.render_every_steps = None
        self.render_every_envs = None
        self.same_environment = None
        self.n_envs = None
        self.training_steps = None
        self.initialize_from_params()

        self.plant = None
        self.simulation = None
        self.reward_function = None
        self.reward_name = None
        self.reset_function = None
        dynamics_function = self.initialize_functions(existing_dynamics_function, existing_plant, env_type)

        self.clean_action_history = None
        self.velocity_noise = None
        self.velocity_bias = None
        self.position_noise = None
        self.position_bias = None
        self.action_noise = None
        self.action_bias = None
        self.start_delay = None
        self.delay = None
        self.initialize_disturbances()

        self.observation_dict = {"T": [], 'X_meas': [], 'X_real': [], 'U_con': [], 'U_meas': [], "push": [], "max_episode_steps": self.max_episode_steps, "current_force": []}
        self.render_mode = "None"
        self.visualizer = Visualizer(self.env_type, self.observation_dict)

        super().__init__(
            dynamics_function,
            self.reward_function,
            kill_switch,
            self.custom_reset,
            self.translator.obs_space,
            self.translator.act_space, #if any("SAC" in attr or "sac" in attr for attr in
                                      #                               dir(self.policy) if isinstance(attr, str)) else None,
            self.max_episode_steps,
            True
        )

        self.observation_dict['dynamics_func'] = self.dynamics_func
        self.dynamics_func.simulator.plant.observation_dict = self.observation_dict

    def initialize_disturbances(self):
        self.clean_action_history = np.array([0.0])
        self.velocity_noise = 0.0
        self.velocity_bias = 0.0
        self.position_noise = 0.0
        self.position_bias = 0.0
        self.action_noise = 0.0
        self.action_bias = 0.0
        self.start_delay = 0.0
        self.delay = 0.0

    def initialize_from_params(self):
        self.type = "train_env"
        if self.is_evaluation_environment:
            self.type = "eval_env"
        self.n_envs = self.param_data[self.type]["n_envs"]
        self.same_environment = self.param_data[self.type]["same_environment"]
        self.max_episode_steps = self.param_data["max_episode_steps"]
        self.render_every_steps = self.param_data["render_every_steps"]
        self.render_every_envs = self.param_data["render_every_envs"]
        self.training_steps = self.param_data["training_steps"]

    def initialize_functions(self, existing_dynamics_function, existing_plant, env_type):
        dynamics_function_class = None
        if existing_dynamics_function is None:
            dynamics_function_class = globals()[self.param_data[self.type]["dynamics_function"]]

        if existing_plant is not None:
            self.plant = existing_plant
            self.simulation = Simulator(plant=existing_plant)

        normalization_class_name = self.param_data["normalization"]
        low_pos = [-0.5, 0, 0, 0]
        if normalization_class_name == "custom_dynamics_func_PI":
            low_pos = [-1.0, 0, 0, 0]

        if dynamics_function_class is not None and hasattr(dynamics_function_class, '__code__'):
            existing_dynamics_function, self.simulation, self.plant = dynamics_function_class(env_type, self.param_data["dt"], self.param_data["max_torque"], globals()[normalization_class_name])

        reset_function = globals()[self.param_data[self.type]["reset_function"]]
        self.reset_function = lambda: reset_function(low_pos)

        reward_function = globals()[self.param_data[self.type]["reward_function"]]
        self.reward_name = self.param_data[self.type]["reward_function"]
        self.reward_function = lambda obs, act, observation_dict: reward_function(obs, act, env_type, existing_dynamics_function, observation_dict)

        return existing_dynamics_function

    def custom_reset(self):
        if self.simulation is not None:
            self.simulation.reset()
        for key in self.observation_dict:
            if key != 'dynamics_func' and key != 'max_episode_steps':
                self.observation_dict[key].clear()
        self.clean_action_history = np.array([0.0])
        self.visualizer.reset()
        self.policy.after_reset(self)
        if any("SAC" in attr or "sac" in attr for attr in dir(self.policy) if isinstance(attr, str)):
            self.translator.reset()
        clean_observation = np.array(self.reset_function())
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, 0.0, 0.0)
        state = self.translator.build_state(dirty_observation, 0.0) if any("SAC" in attr or "sac" in attr for attr in dir(self.policy) if isinstance(attr, str)) else None

        return state

    def get_envs(self, log_dir):
        existing_dynamics_function = None
        existing_plant = None
        if self.same_environment:
            existing_dynamics_function = self.dynamics_func
            existing_plant = self.plant

        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=self.n_envs,
            env_kwargs={
                "env_type": self.env_type,
                "param_name": self.param_name,
                "existing_dynamics_function": existing_dynamics_function,
                "is_evaluation_environment": self.is_evaluation_environment,
                "existing_plant": existing_plant,
                "seed": self.seed,
                "policy": self.policy
            },
            monitor_dir=log_dir,
            seed=self.seed
        )
        return envs

    # TODO: currently normalized noise
    def apply_observation_disturbances(self, clean_observation):
        dirty_observation = clean_observation.copy()
        dirty_observation[:2] += np.random.normal(self.position_bias, self.position_noise, size=2)
        dirty_observation[-2:] += np.random.normal(self.velocity_bias, self.velocity_noise, size=2)

        return dirty_observation

    # TODO: make more efficient with direct calculation of indizes
    def find_delay_action(self):
        list = self.observation_dict['T']
        timestep = list[-1]
        delay = self.delay
        if timestep < self.start_delay:
            delay = self.start_delay
        target = np.around(timestep - delay, decimals=5)
        index = 0
        # if target >= 0:
        #     index = np.round(target / self.dynamics_func.dt, decimals=0).astype(int) + 1
        for i in reversed(range(len(list))):
            value = list[i]
            if value <= target:
                index = i + 1
                break
        delayed_action = self.clean_action_history[index]
        return delayed_action

    # TODO: currently normalized noise
    def get_dirty_action(self, clean_action):
        self.clean_action_history = np.append(self.clean_action_history, clean_action)
        dirty_action = self.find_delay_action()
        dirty_action += np.random.normal(0, self.action_noise)
        return dirty_action

    def get_last_clean_observation(self):
        return self.observation_dict['X_real'][-1].copy()

    def get_new_observation(self, dirty_action, internal_dt=0.0):
        if internal_dt == 0.0:
            internal_dt = self.dynamics_func.dt

        dt = self.dynamics_func.dt
        self.dynamics_func.dt = internal_dt

        new_observation = self.get_last_clean_observation()
        for i in range(0, np.round(dt/internal_dt).astype(int)):
            new_observation = self.dynamics_func(new_observation, np.array([dirty_action]), scaling=self.scaling)

        self.dynamics_func.dt = dt
        return new_observation

    def step(self, clean_action):
        clean_action = clean_action[0].astype(np.float64)
        dirty_action = self.get_dirty_action(clean_action)

        clean_observation = self.get_new_observation(dirty_action)
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, clean_action, dirty_action)
        new_state = self.translator.build_state(dirty_observation, clean_action, **self.observation_dict)
        self.observation = new_state

        reward = self.get_reward(dirty_observation, clean_action)
        terminated = self.terminated_func(self.observation_dict['dynamics_func'].unscale_state(self.observation_dict['X_meas'][-1]))
        truncated = self.check_episode_end()

        self.update_visualizer(reward, clean_action)

        info = {}
        return self.observation, reward, terminated, truncated, info

    def check_episode_end(self):
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return truncated

    def get_reward(self, new_observation, action):
        if self.reward_name == "saturated_distance_from_target":
            return self.reward_func(new_observation, action, self.observation_dict)
        else:
            return self.reward_func(new_observation, action, self.observation_dict) / self.max_episode_steps

    def append_observation_dict(self, clean_observation, dirty_observation, clean_action: float, dirty_action: float):
        time = 0
        if len(self.observation_dict["T"]) > 0:
            time = self.dynamics_func.dt + self.observation_dict["T"][-1]
        self.observation_dict["T"].append(np.around(time, decimals=5))
        self.observation_dict['U_con'].append(clean_action)
        self.observation_dict['U_meas'].append(dirty_action)
        self.observation_dict['X_meas'].append(dirty_observation)
        self.observation_dict['X_real'].append(clean_observation)

    def render(self, mode="human"):
        if self.render_mode == "human" and self.step_counter % self.render_every_steps == 0 and len(self.observation_dict['X_meas']) > 1:
            self.visualizer.render()

    def update_visualizer(self, reward, action):
        if self.render_mode == "human":
            self.visualizer.reward_visualization = reward
            self.visualizer.acc_reward_visualization += reward
            self.visualizer.action_visualization = action




class GeneralEnv_MPC(CustomEnv):

    def __init__(
        self,
        env_type,
        param_name,
        policy,
        seed,
        path="parameters.json",
        is_evaluation_environment=False,
        existing_dynamics_function=None,
        existing_plant=None
    ):

        self.policy = policy
        self.translator = policy.actor_class.get_translator()
        self.seed = seed
        self.is_evaluation_environment = is_evaluation_environment
        self.param_name = param_name
        self.env_type = env_type
        self.param_data = json.load(open(path))[param_name]

        self.type = None
        self.render_every_steps = None
        self.render_every_envs = None
        self.same_environment = None
        self.n_envs = None
        self.training_steps = None
        self.initialize_from_params()

        self.plant = None
        self.simulation = None
        self.reward_function = None
        self.reward_name = None
        self.reset_function = None
        dynamics_function = self.initialize_functions(existing_dynamics_function, existing_plant, env_type)

        self.clean_action_history = None
        self.velocity_noise = None
        self.velocity_bias = None
        self.position_noise = None
        self.position_bias = None
        self.action_noise = None
        self.action_bias = None
        self.start_delay = None
        self.delay = None
        self.initialize_disturbances()

        self.observation_dict = {"T": [], 'X_meas': [], 'X_real': [], 'U_con': [], 'U_meas': [], "push": [], "max_episode_steps": self.max_episode_steps, "current_force": []}
        self.render_mode = "None"
        self.visualizer = Visualizer(self.env_type, self.observation_dict)

        super().__init__(
            dynamics_function,
            self.reward_function,
            kill_switch,
            self.custom_reset,
            self.translator.obs_space,
            self.translator.act_space,
            self.max_episode_steps,
            True
        )

        self.observation_dict['dynamics_func'] = self.dynamics_func
        self.dynamics_func.simulator.plant.observation_dict = self.observation_dict

    def initialize_disturbances(self):
        self.clean_action_history = np.array([0.0])
        self.velocity_noise = 0.0
        self.velocity_bias = 0.0
        self.position_noise = 0.0
        self.position_bias = 0.0
        self.action_noise = 0.0
        self.action_bias = 0.0
        self.start_delay = 0.0
        self.delay = 0.0

    def initialize_from_params(self):
        self.type = "train_env"
        if self.is_evaluation_environment:
            self.type = "eval_env"
        self.n_envs = self.param_data[self.type]["n_envs"]
        self.same_environment = self.param_data[self.type]["same_environment"]
        self.max_episode_steps = self.param_data["max_episode_steps"]
        self.render_every_steps = self.param_data["render_every_steps"]
        self.render_every_envs = self.param_data["render_every_envs"]
        self.training_steps = self.param_data["training_steps"]

    def initialize_functions(self, existing_dynamics_function, existing_plant, env_type):
        dynamics_function_class = None
        if existing_dynamics_function is None:
            dynamics_function_class = globals()[self.param_data[self.type]["dynamics_function"]]

        if existing_plant is not None:
            self.plant = existing_plant
            self.simulation = Simulator(plant=existing_plant)

        normalization_class_name = self.param_data["normalization"]
        low_pos = [-0.5, 0, 0, 0]
        if normalization_class_name == "custom_dynamics_func_PI":
            low_pos = [-1.0, 0, 0, 0]

        if dynamics_function_class is not None and hasattr(dynamics_function_class, '__code__'):
            existing_dynamics_function, self.simulation, self.plant = dynamics_function_class(env_type, self.param_data["dt"], self.param_data["max_torque"], globals()[normalization_class_name])

        reset_function = globals()[self.param_data[self.type]["reset_function"]]
        self.reset_function = lambda: reset_function(low_pos)

        reward_function = globals()[self.param_data[self.type]["reward_function"]]
        self.reward_name = self.param_data[self.type]["reward_function"]
        self.reward_function = lambda obs, act, observation_dict: reward_function(obs, act, env_type, existing_dynamics_function, observation_dict)

        return existing_dynamics_function

    def custom_reset(self):
        if self.simulation is not None:
            self.simulation.reset()
        for key in self.observation_dict:
            if key != 'dynamics_func' and key != 'max_episode_steps':
                self.observation_dict[key].clear()
        self.clean_action_history = np.array([0.0])
        self.visualizer.reset()
        self.policy.after_reset(self)
        self.translator.reset()

        clean_observation = np.array(self.reset_function())
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, 0.0, 0.0)
        state = self.translator.build_state(dirty_observation, 0.0)

        return state

    def get_envs(self, log_dir):
        existing_dynamics_function = None
        existing_plant = None
        if self.same_environment:
            existing_dynamics_function = self.dynamics_func
            existing_plant = self.plant

        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=self.n_envs,
            env_kwargs={
                "env_type": self.env_type,
                "param_name": self.param_name,
                "existing_dynamics_function": existing_dynamics_function,
                "is_evaluation_environment": self.is_evaluation_environment,
                "existing_plant": existing_plant,
                "seed": self.seed,
                "policy": self.policy
            },
            monitor_dir=log_dir,
            seed=self.seed
        )
        return envs

    # TODO: currently normalized noise
    def apply_observation_disturbances(self, clean_observation):
        dirty_observation = clean_observation.copy()
        dirty_observation[:2] += np.random.normal(self.position_bias, self.position_noise, size=2)
        dirty_observation[-2:] += np.random.normal(self.velocity_bias, self.velocity_noise, size=2)

        return dirty_observation

    # TODO: make more efficient with direct calculation of indizes
    def find_delay_action(self):
        list = self.observation_dict['T']
        timestep = list[-1]
        delay = self.delay
        if timestep < self.start_delay:
            delay = self.start_delay
        target = np.around(timestep - delay, decimals=5)
        index = 0
        # if target >= 0:
        #     index = np.round(target / self.dynamics_func.dt, decimals=0).astype(int) + 1
        for i in reversed(range(len(list))):
            value = list[i]
            if value <= target:
                index = i + 1
                break
        delayed_action = self.clean_action_history[index]
        return delayed_action

    # TODO: currently normalized noise
    def get_dirty_action(self, clean_action):
        self.clean_action_history = np.append(self.clean_action_history, clean_action)
        dirty_action = self.find_delay_action()
        dirty_action += np.random.normal(0, self.action_noise)
        return dirty_action

    def get_last_clean_observation(self):
        return self.observation_dict['X_real'][-1].copy()

    def get_new_observation(self, dirty_action, internal_dt=0.0):
        if internal_dt == 0.0:
            internal_dt = self.dynamics_func.dt

        dt = self.dynamics_func.dt
        self.dynamics_func.dt = internal_dt

        new_observation = self.get_last_clean_observation()
        for i in range(0, np.round(dt/internal_dt).astype(int)):
            new_observation = self.dynamics_func(new_observation, np.array([dirty_action]), scaling=self.scaling)

        self.dynamics_func.dt = dt
        return new_observation

    def step(self, clean_action):
        clean_action = clean_action[0].astype(np.float64)
        dirty_action = self.get_dirty_action(clean_action)

        clean_observation = self.get_new_observation(dirty_action)
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, clean_action, dirty_action)
        new_state = self.translator.build_state(dirty_observation, clean_action, **self.observation_dict)
        self.observation = new_state

        reward = self.get_reward(dirty_observation, clean_action)
        terminated = self.terminated_func(self.observation_dict['dynamics_func'].unscale_state(self.observation_dict['X_meas'][-1]))
        truncated = self.check_episode_end()

        self.update_visualizer(reward, clean_action)

        info = {}
        return self.observation, reward, terminated, truncated, info

    def check_episode_end(self):
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return truncated

    def get_reward(self, new_observation, action):
        if self.reward_name == "saturated_distance_from_target":
            return self.reward_func(new_observation, action, self.observation_dict)
        else:
            return self.reward_func(new_observation, action, self.observation_dict) / self.max_episode_steps

    def append_observation_dict(self, clean_observation, dirty_observation, clean_action: float, dirty_action: float):
        time = 0
        if len(self.observation_dict["T"]) > 0:
            time = self.dynamics_func.dt + self.observation_dict["T"][-1]
        self.observation_dict["T"].append(np.around(time, decimals=5))
        self.observation_dict['U_con'].append(clean_action)
        self.observation_dict['U_meas'].append(dirty_action)
        self.observation_dict['X_meas'].append(dirty_observation)
        self.observation_dict['X_real'].append(clean_observation)

    def render(self, mode="human"):
        if self.render_mode == "human" and self.step_counter % self.render_every_steps == 0 and len(self.observation_dict['X_meas']) > 1:
            self.visualizer.render()

    def update_visualizer(self, reward, action):
        if self.render_mode == "human":
            self.visualizer.reward_visualization = reward
            self.visualizer.acc_reward_visualization += reward
            self.visualizer.action_visualization = action
