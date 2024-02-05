import json

from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.misc_helper import updown_reset, balanced_reset, no_termination, noisy_reset, low_reset, high_reset, random_reset, semi_random_reset, debug_reset
from examples.reinforcement_learning.General.reward_functions import get_state_values
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import default_dynamics, random_dynamics, \
    random_push_dynamics, push_dynamics, load_param, custom_dynamics_func_PI, custom_dynamics_func_4PI
from examples.reinforcement_learning.General.reward_functions import future_pos_reward, pos_reward, unholy_reward_4, saturated_distance_from_target, score_reward
from double_pendulum.simulation.simulation import Simulator

class GeneralEnv(CustomEnv):
    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(
        self,
        robot,
        param,
        path="parameters.json",
        eval=False,
        dynamics_function=None,
        plant=None
    ):

        self.eval = eval
        self.param = param
        self.pendulum_length = 350
        self.reward = 0
        self.action = None
        self.acc_reward = 0
        self.robot = robot
        self.data = json.load(open(path))[param]

        type = "train_env"
        if self.eval:
            type = "eval_env"

        if dynamics_function is None:
            dynamics_function = globals()[self.data[type]["dynamics_function"]]
        reset_function = globals()[self.data[type]["reset_function"]]
        dynamic_class_name = self.data["dynamic_class"]
        low_pos = [-0.5, 0, 0, 0]
        if dynamic_class_name == "custom_dynamics_func_PI":
            low_pos = [1.0, 0, 0, 0]
        self.reset_function = lambda: reset_function(low_pos)
        reward_function = globals()[self.data[type]["reward_function"]]

        self.n_envs = self.data[type]["n_envs"]
        self.same_env = self.data[type]["same_env"]

        self.dynamics_function = dynamics_function
        self.reward_function = lambda obs, act, state_dict: reward_function(obs, act, robot, self.dynamics_func, state_dict)
        self.max_episode_steps = self.data["max_episode_steps"]
        self.render_every_steps = self.data["render_every_steps"]
        self.render_every_envs = self.data["render_every_envs"]
        self.actions_in_state = self.data["actions_in_state"] == 1
        self.use_sim = self.data["use_sim_values"] == 1
        self.simulation = None
        if not plant is None:
            self.plant = plant
            self.simulation = Simulator(plant=plant)

        if hasattr(dynamics_function, '__code__'):
            dynamics_function, self.simulation, self.plant = dynamics_function(robot, self.data["dt"], self.data["max_torque"], globals()[dynamic_class_name])

        if not self.actions_in_state:
            obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
        else:
            obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        act_space = gym.spaces.Box(np.array([-0.5]), np.array([0.5]))
        if robot == "acrobot":
            act_space = gym.spaces.Box(np.array([-0.6]), np.array([0.6]))
        super().__init__(
            dynamics_function,
            self.reward_function,
            no_termination,
            self.reset_function,
            obs_space,
            act_space,
            self.max_episode_steps,
            True
        )

        self.window_size = 800
        self.render_mode = "None"
        self.window = None
        self.clock = None

        self.mpar = load_param(robot, self.dynamics_func.torque_limit)
        self.state_dict = {"T": [], "X_meas": [], "U_con": [], "plant": self.dynamics_func.simulator.plant, "max_episode_steps": self.max_episode_steps}

    def get_envs(self, log_dir):
        if self.same_env:
            dynamics_function = self.dynamics_func
            plant = self.plant
        else:
            dynamics_function = self.dynamics_function
            plant = None

        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=self.n_envs,
            env_kwargs={
                "robot": self.robot,
                "param": self.param,
                "dynamics_function": dynamics_function,
                "eval": self.eval,
                "plant": plant
            },
            monitor_dir=log_dir
        )
        return envs

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed, options)

        if self.actions_in_state:
            observation = np.append(observation, np.zeros(2))

        if self.simulation is not None:
            self.simulation.reset()
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

        if not self.simulation is None and self.use_sim:
            self.simulation.step(self.dynamics_func.unscale_action(action), self.dynamics_func.dt)
            t, x = self.simulation.get_state()
            self.observation = self.dynamics_func.normalize_state(x)

        time = self.dynamics_func.dt
        if len(self.state_dict["T"]) > 0:
            time = time + self.state_dict["T"][-1]
        self.state_dict["T"].append(time)
        self.state_dict["U_con"].append(self.dynamics_func.unscale_action(action))
        self.state_dict["X_meas"].append(self.dynamics_func.unscale_state(self.observation))

        if self.actions_in_state:
            self.observation = np.append(self.observation, last_actions)
        reward = self.reward_func(self.observation, action, self.state_dict)
        terminated = self.terminated_func(self.observation)
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
        transformed = (self.window_size // 2 + point[0]*self.pendulum_length*2, self.window_size // 2 + point[1]*self.pendulum_length*2)
        return transformed

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        y, x1, x2, v1, v2, action, goal, dt, threshold, u_p, u_pp = get_state_values(self.observation, self.action, self.robot, self.dynamics_func)
        x3 = x2 + dt * v2
        """if self.robot == "pendubot":
            action = action[0]
        else:
            action = action[1]"""

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
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(np.array([0,0])), self.getXY(x1), 5)
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(x1), self.getXY(x2), 5)

        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(np.array([0,0])), 10)
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
