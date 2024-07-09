import json

from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.misc_helper import *
from examples.reinforcement_learning.General.reward_functions import get_state_values
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import *
from param_helper import load_env_attributes


class GeneralEnv(CustomEnv):
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

        self.max_episode_steps = data["max_episode_steps"]
        self.reward_name = env_params["reward_function"]
        self.render_every_steps = data["render_every_steps"]
        self.render_every_envs = data["render_every_envs"]
        self.actions_in_state = data["actions_in_state"] == 1
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

        self.same_env = use_same_env
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
            self.max_episode_steps,
            True
        )

        self.window_size = 800
        self.pendulum_length = 350
        self.render_mode = "None"
        self.window = None
        self.clock = None
        self.reward = 0
        self.acc_reward = 0
        self.action = np.array([0, 0])

        self.state_dict = {"T": [], "X_meas": [], "U_con": [], "push": [], "plant": self.dynamics_func.simulator.plant,
                           "max_episode_steps": self.max_episode_steps, "current_force": []}
        self.dynamics_func.simulator.plant.state_dict = self.state_dict

    def custom_reset(self):
        observation = self.reset_function()
        x = self.dynamics_func.unscale_state(observation[:4])
        self.dynamics_func.simulator.set_state(0, x)
        self.dynamics_func.simulator.reset()
        if self.actions_in_state:
            observation = np.append(observation, np.zeros(2))
        return observation

    def get_envs(self, log_dir):
        if self.same_env:
            dynamics_function = self.dynamics_func
        else:
            dynamics_function = None


        envs = make_vec_env(
            env_id=GeneralEnv,
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

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed, options)
        self.dynamics_func.virtual_sensor_state = [0.0, 0.0]
        self.virtual_sensor_state_tracking = [0.0, 0.0]
        for key in self.state_dict:
            if key != 'plant' and key != 'max_episode_steps':
                self.state_dict[key].clear()

        self.reward = 0
        self.acc_reward = 0
        self.action = np.array([0, 0])
        return observation, info

    def step(self, action):
        last_actions = self.observation[-2:]
        if self.actions_in_state:
            self.observation = self.observation[:-2]

        self.observation = self.dynamics_func(self.observation, action, scaling=self.scaling)

        time = self.dynamics_func.dt

        if len(self.state_dict["T"]) > 0:
            time = time + self.state_dict["T"][-1]
        self.state_dict["T"].append(time)
        self.state_dict["U_con"].append(self.dynamics_func.unscale_action(action))
        self.state_dict["X_meas"].append(self.dynamics_func.unscale_state(self.observation))

        if self.actions_in_state:
            self.observation = np.append(self.observation, last_actions)

        if self.reward_name == "saturated_distance_from_target":
            reward = self.reward_func(self.observation, action, self.state_dict)
        else:
            reward = self.reward_func(self.observation, action, self.state_dict) / self.max_episode_steps

        ignore_state = True
        if self.data["dynamic_class"] == "custom_dynamics_func_PI":
            self.virtual_sensor_state_tracking += self.dynamics_func.virtual_sensor_state
            if self.robot == "acrobot":
                ignore_state = True

        terminated = self.terminated_func(self.observation, self.virtual_sensor_state_tracking, ignore_state)

        info = {}
        truncated = False

        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0

        if self.actions_in_state:
            self.observation[-1] = last_actions[0]
            self.observation[-2] = action[0]
        if self.render_mode == "human":
            self.reward = reward
            self.acc_reward += reward
            self.action = action
        return self.observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.step_counter % self.render_every_steps == 0:
            self._render_frame()

    def getXY(self, point):
        transformed = (self.window_size // 2 + point[0] * self.pendulum_length * 2,
                       self.window_size // 2 + point[1] * self.pendulum_length * 2)
        return transformed

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
        x3 = x2 + dt * v2

        action = action[0]
        distance = np.linalg.norm(x2 - goal)
        distance_next = np.linalg.norm(x3 - goal)
        v1_total = np.linalg.norm(v1)
        v2_total = np.linalg.norm(v2)
        x_1 = self.observation[0]
        x_2 = self.observation[1]
        canvas.fill((255, 255, 255))

        if distance_next < threshold:
            canvas.fill((184, 255, 191))
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(np.array([0, 0])), self.getXY(x1), 5)
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(x1), self.getXY(x2), 5)

        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(np.array([0, 0])), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x1), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x2), 5)
        pygame.draw.circle(canvas, (255, 200, 200), self.getXY(goal), threshold * 4 * self.pendulum_length)
        pygame.draw.circle(canvas, (255, 50, 50), self.getXY(goal), threshold * 2 * self.pendulum_length)
        pygame.draw.circle(canvas, (95, 2, 99), self.getXY(x3), threshold * 2 * self.pendulum_length)

        myFont = pygame.font.SysFont("Times New Roman", 36)
        acc_reward = myFont.render(str(np.round(self.acc_reward, 5)), 1, (0, 0, 0), )
        reward = myFont.render(str(np.round(self.reward, 5)), 1, (0, 0, 0), )
        canvas.blit(acc_reward, (10, 10))
        canvas.blit(reward, (10, 60))

        canvas.blit(myFont.render(str(self.step_counter), 1, (0, 0, 0), ), (10, self.window_size - 320))
        canvas.blit(myFont.render(str(round(x_1, 4)), 1, (0, 0, 0), ), (10, self.window_size - 280))
        canvas.blit(myFont.render(str(round(x_2, 4)), 1, (0, 0, 0), ), (10, self.window_size - 240))
        canvas.blit(myFont.render(str(round(distance, 4)), 1, (0, 0, 0), ), (10, self.window_size - 200))
        canvas.blit(myFont.render(str(round(distance_next, 4)), 1, (0, 0, 0), ), (10, self.window_size - 160))
        canvas.blit(myFont.render(str(round(v1_total, 4)), 1, (0, 0, 0), ), (10, self.window_size - 120))
        canvas.blit(myFont.render(str(round(v2_total, 4)), 1, (0, 0, 0), ), (10, self.window_size - 80))
        canvas.blit(myFont.render(str(round(action, 4)), 1, (0, 0, 0), ), (10, self.window_size - 40))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
