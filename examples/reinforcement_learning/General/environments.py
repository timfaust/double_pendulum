from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.dynamics_functions import default_dynamics
from examples.reinforcement_learning.General.misc_helper import no_termination, noisy_reset
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from double_pendulum.utils.wrap_angles import wrap_angles_diff


class GeneralEnv(CustomEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
            self,
            robot,
            dynamics_function,
            reward_function,
            max_episode_steps=0
    ):

        self.robot = robot
        self.dynamics_function = dynamics_function
        self.reward_function = reward_function
        self.max_episode_steps = max_episode_steps

        if hasattr(dynamics_function, '__code__'):
            dynamics_function, self.simulation, self.plant = dynamics_function(robot)

        super().__init__(
            dynamics_function,
            reward_function,
            no_termination,
            noisy_reset,
            gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])),
            gym.spaces.Box(np.array([-1]), np.array([1])),
            max_episode_steps,
            True
        )

        self.window_size = 512
        render_mode = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def get_envs(self, n_envs, log_dir, same):
        if same:
            dynamics_function = self.dynamics_func
        else:
            dynamics_function = self.dynamics_function
        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=n_envs,
            env_kwargs={
                "robot": self.robot,
                "dynamics_function": dynamics_function,
                "reward_function": self.reward_function,
                "max_episode_steps": self.max_episode_steps,
            },
            monitor_dir=log_dir
        )
        return envs

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed, options)
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        s = np.array(
            [
                self.observation[0] * np.pi + np.pi,  # [0, 2pi]
                (self.observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            ]
        )

        y = wrap_angles_diff(s)
        length = 100

        start = np.array([self.window_size // 2, self.window_size // 2])
        end_1 = start + np.array([np.sin(y[0]), np.cos(y[0])]) * length
        end_2 = end_1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * length

        pygame.draw.line(canvas, (0, 0, 0), tuple(np.round(start)), tuple(np.round(end_1)))
        pygame.draw.line(canvas, (0, 0, 0), tuple(np.round(end_1)), tuple(np.round(end_2)))

        myFont = pygame.font.SysFont("Times New Roman", 18)
        step = myFont.render(str(self.step_counter), 1, (0, 0, 0), )
        canvas.blit(step, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])


class DefaultEnv(GeneralEnv):
    def __init__(self, robot, reward_function):
        super().__init__(robot, default_dynamics, reward_function)