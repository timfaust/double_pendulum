import gymnasium as gym
import numpy as np

from src.python.double_pendulum.simulation.gym_env import CustomEnv


class PPOEnv(CustomEnv):
    def __init__(
            self,
            dynamics_func,
            reward_func,
            terminated_func,
            reset_func,
            obs_space=gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])),
            act_space=gym.spaces.Box(np.array([-1]), np.array([1])),
            max_episode_steps=1000,
            scaling=True
    ):
        super().__init__(
            dynamics_func,
            reward_func,
            terminated_func,
            reset_func,
            obs_space,
            act_space,
            max_episode_steps,
            scaling
        )
