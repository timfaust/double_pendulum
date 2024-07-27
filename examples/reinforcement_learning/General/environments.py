import copy
import json
from typing import Dict, Any

from sympy import lambdify

from examples.reinforcement_learning.General.misc_helper import updown_reset, balanced_reset, no_termination, \
    noisy_reset, low_reset, high_reset, random_reset, semi_random_reset, debug_reset, kill_switch, get_unscaled_action, \
    get_stabilized, disturbed_parameters
from examples.reinforcement_learning.General.override_sb3.utils import CustomDummyVecEnv, make_vec_env
from examples.reinforcement_learning.General.reward_functions import get_state_values
from examples.reinforcement_learning.General.visualizer import Visualizer
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import default_dynamics, load_param, custom_dynamics_func_4PI
from examples.reinforcement_learning.General.reward_functions import score_reward, pos_reward, quadratic_rew, saturated_distance_from_target, future_pos_reward
from double_pendulum.simulation.simulation import Simulator

from src.python.double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array


# overwritten
class GeneralEnv(CustomEnv):

    def __init__(
        self,
        env_type,
        param_name,
        seed,
        path="parameters.json",
        is_evaluation_environment=False,
        existing_dynamics_function=None
    ):

        self.sac = None
        self.seed = seed
        self.is_evaluation_environment = is_evaluation_environment
        self.param_name = param_name
        self.env_type = env_type
        self.killed_because = 0
        self.stabilized = False
        self.param_data = json.load(open(path))[param_name]

        self.type = None
        self.configuration = None
        self.render_every_steps = None
        self.render_every_envs = None
        self.same_environment = None
        self.n_envs = None
        self.training_steps = None
        self.initialize_from_params()

        self.reward_function = None
        self.reward_name = None
        self.reset_function = None
        dynamics_function = self.initialize_functions(existing_dynamics_function)

        self.velocity_noise = None
        self.velocity_bias = None
        self.position_noise = None
        self.position_bias = None
        self.action_noise = None
        self.action_bias = None
        self.start_delay = None
        self.delay = None
        self.responsiveness = None
        self.use_perturbations = None
        self.perturbations = None
        self.initialize_disturbances()

        self.mpar = load_param(self.param_data["max_torque"])
        self.observation_dict = {"T": [], 'X_meas': [], 'X_real': [], 'U_con': [], 'U_real': [], "push": [], "max_episode_steps": self.max_episode_steps, "mpar": self.mpar}
        self.observation_dict_old = None
        self.render_mode = "None"
        self.visualizer = Visualizer(self)

        self.episode_id = 0

        super().__init__(
            dynamics_function,
            self.reward_function,
            lambda observation, action: kill_switch(observation, action, dynamics_function),
            self.custom_reset,
            max_episode_steps=self.max_episode_steps
        )

        self.dynamics_func.simulator.plant.observation_dict = self.observation_dict

    def initialize_disturbances(self):
        self.velocity_noise = 0.00
        self.velocity_bias = 0.0
        self.position_noise = 0.00
        self.position_bias = 0.0
        self.action_noise = 0.00
        self.action_bias = 0.0
        self.start_delay = 0.0
        self.delay = 0.0
        self.responsiveness = 1
        self.perturbations = []

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

    def initialize_functions(self, existing_dynamics_function):
        dynamics_function_class = None
        if existing_dynamics_function is None:
            dynamics_function_class = globals()[self.param_data[self.type]["dynamics_function"]]
        elif not self.same_environment:
            existing_dynamics_function = copy.deepcopy(existing_dynamics_function)

        normalization_class_name = self.param_data["normalization"]
        low_pos = [0, 0, 0, 0]

        if dynamics_function_class is not None and hasattr(dynamics_function_class, '__code__'):
            existing_dynamics_function, _, _ = dynamics_function_class(self.env_type, self.param_data["dt"], self.param_data["max_torque"], globals()[normalization_class_name])

        reset_function = globals()[self.param_data[self.type]["reset_function"]]
        self.reset_function = lambda: reset_function(low_pos)

        reward_function = globals()[self.param_data[self.type]["reward_function"]]
        self.reward_name = self.param_data[self.type]["reward_function"]
        self.reward_function = lambda obs, act, observation_dict: reward_function(obs, act, self.env_type, existing_dynamics_function, observation_dict)

        return existing_dynamics_function

    def custom_reset(self):
        self.visualizer.reset()
        if self.dynamics_func.simulator is not None:
            self.dynamics_func.simulator.reset()
        self.observation_dict_old = copy.deepcopy(self.observation_dict)
        self.observation_dict_old['killed_because'] = self.killed_because
        if 'dynamics_func' not in self.observation_dict:
            self.observation_dict['dynamics_func'] = self.dynamics_func
        for key in self.observation_dict:
            if key != 'dynamics_func' and key != 'max_episode_steps' and key != 'mpar':
                self.observation_dict[key].clear()

        if self.sac and (self.configuration[1] == -1 or self.use_perturbations):
            self.change_dynamics()

        clean_observation = np.array(self.reset_function())
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, 0.0)
        self.observation_dict['U_con'].append(0.0)
        self.episode_id += 1
        self.killed_because = 0
        self.stabilized = False

        return dirty_observation

    def get_envs(self, log_dir):
        existing_dynamics_function = self.dynamics_func

        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=self.n_envs,
            env_kwargs={
                "env_type": self.env_type,
                "param_name": self.param_name,
                "existing_dynamics_function": existing_dynamics_function,
                "is_evaluation_environment": self.is_evaluation_environment,
                "seed": self.seed
            },
            monitor_dir=log_dir,
            seed=self.seed,
            vec_env_cls=CustomDummyVecEnv
        )
        return envs

    def apply_observation_disturbances(self, clean_observation):
        dirty_observation = clean_observation.copy()
        dirty_observation[:2] += np.random.normal(self.position_bias, self.position_noise, size=2)
        dirty_observation[-2:] += np.random.normal(self.velocity_bias, self.velocity_noise, size=2)

        return dirty_observation

    def find_delay_action(self):
        T = self.observation_dict['T']
        U_con = self.observation_dict['U_con']

        current_time = T[-1]

        offset = np.round(self.dynamics_func.dt / 2, decimals=5)
        if current_time < self.start_delay - offset:
            return 0.0

        adjusted_delay = max(0, self.delay - offset)
        delay_time = current_time - adjusted_delay

        if adjusted_delay <= 0.0:
            index = len(U_con) - 1
        else:
            index = np.searchsorted(T, delay_time) - 1

        delayed_action = U_con[max(0, index)]
        last_action = U_con[max(0, index - 1)]

        return last_action + self.responsiveness * (delayed_action - last_action)

    # normalized noise
    def get_dirty_action(self, clean_action):
        self.observation_dict['U_con'].append(clean_action)
        dirty_action = self.find_delay_action()
        dirty_action += np.random.normal(self.action_bias, self.action_noise)
        dirty_action = np.clip(dirty_action, -1, 1)
        return dirty_action

    def get_last_clean_observation(self):
        return self.observation_dict['X_real'][-1].copy()

    def get_new_observation(self, dirty_action, internal_dt=0.0):
        if internal_dt == 0.0:
            internal_dt = self.dynamics_func.dt

        dt = self.dynamics_func.dt
        self.dynamics_func.dt = internal_dt

        new_observation = self.get_last_clean_observation()

        torque = self.dynamics_func.unscale_action(dirty_action)

        if self.use_perturbations:
            timestep = len(self.observation_dict['T']) - 1
            torque[0] += self.perturbations[0][timestep]
            torque[1] += self.perturbations[1][timestep]

        for i in range(0, np.round(dt/internal_dt).astype(int)):
            new_observation = self.dynamics_func(new_observation, torque, scaling=self.scaling)

        self.dynamics_func.dt = dt
        return new_observation

    # overwritten
    def step(self, clean_action):
        clean_action = clean_action[0].astype(np.float64)
        dirty_action = self.get_dirty_action(clean_action)

        clean_observation = self.get_new_observation(dirty_action)
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, dirty_action)

        terminated = self.terminated_func(dirty_observation, clean_action)
        self.killed_because = (np.argmax(terminated) + 1) if np.any(terminated) else 0
        done = self.killed_because != 0
        if not done:
            self.stabilized = get_stabilized(self.observation_dict) >= 1
            done = self.stabilized

        reward_list = self.get_reward(clean_observation, clean_action)
        for i in range(len(reward_list)):
            key = 'reward_' + str(i)
            if key not in self.observation_dict:
                self.observation_dict[key] = []
                self.observation_dict[key].append(0.0)
            if done and not self.stabilized:
                reward_list[i] -= 1.0
            if done and self.stabilized:
                reward_list[i] += 1.0
            self.observation_dict[key].append(reward_list[i])

        truncated = self.check_episode_end()

        info = {'episode_id': self.episode_id}
        return dirty_observation, reward_list, done, truncated, info

    def check_episode_end(self):
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return truncated

    def get_reward(self, new_observation, action):
        reward = None
        if self.reward_name == "saturated_distance_from_target":
            reward = self.reward_func(new_observation, action, self.observation_dict)
        else:
            reward = self.reward_func(new_observation, action, self.observation_dict)
        if isinstance(reward, list):
            return reward
        return [reward]

    def append_observation_dict(self, clean_observation, dirty_observation, dirty_action: float):
        time = 0
        if len(self.observation_dict['T']) > 0:
            time = self.dynamics_func.dt + self.observation_dict['T'][-1]
        self.observation_dict['T'].append(np.round(time, decimals=5))
        self.observation_dict['U_real'].append(dirty_action)
        self.observation_dict['X_meas'].append(dirty_observation)
        self.observation_dict['X_real'].append(clean_observation)

    def get_disturbance_values(self, progress: float):
        p_factor = 1
        n_factor = 1
        disturbances = {
            'l': 0.0,
            'm': 0.25 * p_factor,
            'b': 0.1 * p_factor,
            'coulomb_fric': 0.2 * p_factor,
            'com': 0.25 * p_factor,
            'I': 0.25 * p_factor,
            'Ir': 0.0001 * p_factor,
            'start_delay': 0.0,
            'delay': 0.04 * n_factor,
            'velocity_noise': 0.5 / self.dynamics_func.max_velocity * n_factor,
            'velocity_bias': 0.0,
            'position_noise': 0.0,
            'position_bias': 0.0,
            'action_noise': 1.1 / self.dynamics_func.torque_limit[0] * n_factor,
            'action_bias': 0.0,
            'n_pert_per_joint': 3,
            'min_t_dist': 1.0,
            'sigma_minmax': [0.05, 0.1],
            'amplitude_min_max': [0.5, 5.0],
            'responsiveness': [1 - 0.9 * n_factor, 1 + 1 * n_factor]
        }

        return disturbances

    def change_dynamics(self, disturbance=None, progress: float = 0.0, N: int = 21):
        if disturbance is None:
            disturbance = self.configuration

        disturbances = self.get_disturbance_values(progress)

        plant = self.dynamics_func.simulator.plant
        plant_parameters = {
            'l': self.mpar.l.copy(),
            'm': self.mpar.m.copy(),
            'b': self.mpar.b.copy(),
            'coulomb_fric': self.mpar.cf.copy(),
            'com': self.mpar.r.copy(),
            'I': self.mpar.I.copy(),
            'Ir': self.mpar.Ir,
        }

        for key, value in plant_parameters.items():
            setattr(plant, key, value)

        self.initialize_disturbances()

        parameter = disturbed_parameters[disturbance[0]]
        step_index = disturbance[1]
        parameter_index = -1
        if parameter[-1].isdigit():
            parameter_index = int(parameter[-1]) - 1
            parameter = parameter[:-1]

        self.use_perturbations = False
        if parameter == 'n_pert_per_joint':
            self.use_perturbations = True
            self.perturbations, *_ = get_random_gauss_perturbation_array(
                self.dynamics_func.dt * self.max_episode_steps,
                self.dynamics_func.dt,
                disturbances['n_pert_per_joint'],
                disturbances['min_t_dist'],
                disturbances['sigma_minmax'],
                disturbances['amplitude_min_max']
            )
        else:
            value = disturbances.get(parameter)
            if value is not None:
                if parameter in ['l', 'I', 'm', 'com']:
                    base_value = plant_parameters[parameter][parameter_index]
                    steps = np.linspace((1 - value) * base_value, (1 + value) * base_value, N)
                    new_value = getattr(plant, parameter)
                    new_value[parameter_index] = steps[step_index]
                    if step_index == -1:
                        new_value[parameter_index] = np.random.choice(steps)
                    setattr(plant, parameter, new_value)
                elif parameter in ['coulomb_fric', 'b']:
                    steps = np.linspace(-value, value, N)
                    new_value = getattr(plant, parameter)
                    new_value[parameter_index] = steps[step_index]
                    if step_index == -1:
                        new_value[parameter_index] = np.random.choice(steps)
                    setattr(plant, parameter, new_value)
                elif parameter == 'Ir':
                    steps = np.linspace(0, value, N)
                    new_value = steps[step_index]
                    if step_index == -1:
                        new_value = np.random.choice(steps)
                    setattr(plant, parameter, new_value)
                elif parameter == 'responsiveness':
                    steps = np.linspace(value[0], value[1], N)
                    new_value = steps[step_index]
                    if step_index == -1:
                        new_value = np.random.choice(steps)
                    self.responsiveness = new_value
                elif parameter in ["position_noise", "velocity_noise", "action_noise", "delay", "start_delay"]:
                    steps = np.linspace(0, value, N)
                    new_value = steps[step_index]
                    if step_index == -1:
                        new_value = np.random.choice(steps)
                    setattr(self, parameter, new_value)
                elif parameter in ["position_bias", "velocity_bias", "action_bias"]:
                    steps = np.linspace(-value, value, N)
                    new_value = steps[step_index]
                    if step_index == -1:
                        new_value = np.random.choice(steps)
                    setattr(self, parameter, new_value)

        if (self.configuration is None or self.configuration[1] == -1) and parameter not in ['nothing', 'delay', 'velocity_noise', 'action_noise', 'responsiveness', 'n_pert_per_joint']:
            self.update_plant()
        self.configuration = disturbance

    def update_plant(self):
        plant = self.dynamics_func.simulator.plant
        plant.lambdify_matrices()

    def render(self, mode="human"):
        show_env = self.step_counter % self.render_every_steps == 0
        if self.render_mode == "human" and show_env and len(self.observation_dict['X_meas']) > 1:
            self.visualizer.render()
