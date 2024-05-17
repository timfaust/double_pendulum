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

        self.window_size = 800
        self.pendulum_length = 350
        self.render_mode = "None"
        self.window = None
        self.clock = None
        self.reward = 0
        self.acc_reward = 0
        self.step_counter = 0
        self.action = np.array([0, 0])

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

    def calculate_obs(self, action):
        if self.actions_in_state:
            last_actions = self.observation[-2:]

        self.observation = self.dynamics_func(self.observation, action, scaling=self.scaling)

        if self.actions_in_state:
            self.observation[-1] = last_actions[0]
            self.observation[-2] = action[0]

        if self.render_mode == "human":
            self.action = action
            self.step_counter += 1

        return self.observation

    def step(self, action):
        if self.actions_in_state:
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

    def getXY(self, point):
        transformed = (self.window_size // 2 + point[0] * self.pendulum_length * 2,
                       self.window_size // 2 + point[1] * self.pendulum_length * 2)
        return transformed
    def render(self, mode="human"):
        self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        y, x1, x2, v1, v2, action, goal, dt, threshold, u_p, u_pp = get_state_values(self.observation, self.action,
                                                                                     self.robot, self.dynamics_func)
        canvas.fill((255, 255, 255))

        pygame.draw.line(canvas, (0, 0, 0), self.getXY(np.array([0, 0])), self.getXY(x1), 5)
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(x1), self.getXY(x2), 5)

        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(np.array([0, 0])), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x1), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x2), 5)
        pygame.draw.circle(canvas, (255, 200, 200), self.getXY(goal), threshold * 4 * self.pendulum_length)
        pygame.draw.circle(canvas, (255, 50, 50), self.getXY(goal), threshold * 2 * self.pendulum_length)

        myFont = pygame.font.SysFont("Times New Roman", 36)

        canvas.blit(myFont.render(str(self.step_counter), 1, (0, 0, 0), ), (10, self.window_size - 80))
        canvas.blit(myFont.render(str(round(self.action[0], 4)), 1, (0, 0, 0), ), (10, self.window_size - 40))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
