import gymnasium as gym
import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class PPOEnv(CustomEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            dynamics_func,
            reward_func,
            terminated_func,
            reset_func,
            obs_space=gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])),
            act_space=gym.spaces.Box(np.array([-1]), np.array([1])),
            max_episode_steps=1000,
            scaling=True,
            render_mode=None
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
        self.window_size = 512  # The size of the PyGame window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

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
                self.observation[2],
                self.observation[3],
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
        step = myFont.render(str(self.step_counter), 1, (0, 0, 0),)
        canvas.blit(step, (10, 10))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
