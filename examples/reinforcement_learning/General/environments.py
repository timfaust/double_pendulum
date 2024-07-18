import copy
import json

from sympy import lambdify

from examples.reinforcement_learning.General.misc_helper import updown_reset, balanced_reset, no_termination, \
    noisy_reset, low_reset, high_reset, random_reset, semi_random_reset, debug_reset, kill_switch, get_unscaled_action, \
    get_stabilized
from examples.reinforcement_learning.General.override_sb3.utils import CustomDummyVecEnv, make_vec_env
from examples.reinforcement_learning.General.reward_functions import get_state_values
from examples.reinforcement_learning.General.visualizer import Visualizer
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import default_dynamics, load_param, custom_dynamics_func_4PI
from examples.reinforcement_learning.General.reward_functions import score_reward, pos_reward, quadratic_rew, saturated_distance_from_target
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
        existing_dynamics_function=None,
        existing_plant=None
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
        self.velocity_noise = 0.0
        self.velocity_bias = 0.0
        self.position_noise = 0.0
        self.position_bias = 0.0
        self.action_noise = 0.0
        self.action_bias = 0.0
        self.start_delay = 0.0
        self.delay = 0.0
        self.responsiveness = 1
        self.use_perturbations = False
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

    def initialize_functions(self, existing_dynamics_function, existing_plant, env_type):
        dynamics_function_class = None
        if existing_dynamics_function is None:
            dynamics_function_class = globals()[self.param_data[self.type]["dynamics_function"]]

        if existing_plant is not None:
            self.plant = existing_plant
            self.simulation = Simulator(plant=existing_plant)

        normalization_class_name = self.param_data["normalization"]
        low_pos = [0, 0, 0, 0]

        if dynamics_function_class is not None and hasattr(dynamics_function_class, '__code__'):
            existing_dynamics_function, self.simulation, self.plant = dynamics_function_class(env_type, self.param_data["dt"], self.param_data["max_torque"], globals()[normalization_class_name])

        reset_function = globals()[self.param_data[self.type]["reset_function"]]
        self.reset_function = lambda: reset_function(low_pos)

        reward_function = globals()[self.param_data[self.type]["reward_function"]]
        self.reward_name = self.param_data[self.type]["reward_function"]
        self.reward_function = lambda obs, act, observation_dict: reward_function(obs, act, env_type, existing_dynamics_function, observation_dict)

        return existing_dynamics_function

    def custom_reset(self):
        self.visualizer.reset()
        if self.simulation is not None:
            self.simulation.reset()
        self.observation_dict_old = copy.deepcopy(self.observation_dict)
        if 'dynamics_func' not in self.observation_dict:
            self.observation_dict['dynamics_func'] = self.dynamics_func
        for key in self.observation_dict:
            if key != 'dynamics_func' and key != 'max_episode_steps' and key != 'mpar':
                self.observation_dict[key].clear()

        if self.sac and self.killed_because == 0:
            self.sac.after_environment_reset(self)

        clean_observation = np.array(self.reset_function())
        dirty_observation = self.apply_observation_disturbances(clean_observation)
        self.append_observation_dict(clean_observation, dirty_observation, 0.0)
        self.observation_dict['U_con'].append(0.0)
        self.episode_id += 1
        self.killed_because = 0
        self.stabilized = False

        return dirty_observation

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
                "seed": self.seed
            },
            monitor_dir=log_dir,
            seed=self.seed,
            vec_env_cls=CustomDummyVecEnv
        )
        return envs

    # normalized noise
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
            # if self.stabilized:
            #     reward_list[i] += 0.5
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

    def change_dynamics(self, changing_values, progress):

        if self.use_perturbations and 'n_pert_per_joint' in changing_values and changing_values['n_pert_per_joint'] > 0:
            perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
                self.dynamics_func.dt * self.max_episode_steps,
                self.dynamics_func.dt,
                changing_values['n_pert_per_joint'],
                changing_values['min_t_dist'],
                changing_values['sigma_minmax'],
                changing_values['amplitude_min_max']
            )
            self.perturbations = perturbation_array

        plant = self.dynamics_func.simulator.plant

        plant_parameters = {
            'l': self.mpar.l,
            'm': self.mpar.m,
            'b': self.mpar.b,
            'coulomb_fric': self.mpar.cf,
            'com': self.mpar.r,
            'I': self.mpar.I,
            'Ir': self.mpar.Ir,
        }

        for key, value in plant_parameters.items():
            if key in changing_values:
                sigma = changing_values[key]
                if key == 'l' or key == 'I' or key == 'm' or key == 'com':
                    new_value = np.array(value) + np.random.uniform(-value[0] * sigma, value[0] * sigma, 2)
                    new_value = new_value.tolist()
                elif key == 'coulomb_fric' or key == 'b':
                    new_value = np.array(value) + np.random.uniform(-sigma, sigma, 2)
                    new_value = new_value.tolist()
                elif key == 'Ir':
                    new_value = (np.array(value) + np.random.uniform(0, sigma, 1))[0]

                setattr(plant, key, new_value)

        environment_parameters = [
            'velocity_noise', 'velocity_bias', 'position_noise', 'position_bias',
            'action_noise', 'action_bias', 'start_delay', 'delay'
        ]

        for param in environment_parameters:
            if param in changing_values:
                value = np.random.uniform(-changing_values[param], changing_values[param])
                if 'bias' not in param:
                    value = np.abs(value)
                setattr(self, param, value)

        if 'responsiveness' in changing_values:
            self.responsiveness = changing_values['responsiveness']
        else:
            self.responsiveness = 1

        self.update_plant()

    def update_plant(self):
        plant = self.dynamics_func.simulator.plant
        plant.lambdify_matrices()

    def render(self, mode="human"):
        if self.render_mode == "human" and self.step_counter % self.render_every_steps == 0 and len(self.observation_dict['X_meas']) > 1:
            self.visualizer.render()
