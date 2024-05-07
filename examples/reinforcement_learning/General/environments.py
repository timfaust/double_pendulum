import json

from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.General.custom_policies import CustomPolicy, Translator
from examples.reinforcement_learning.General.misc_helper import updown_reset, balanced_reset, no_termination, \
    noisy_reset, low_reset, high_reset, random_reset, semi_random_reset, debug_reset, kill_switch
from examples.reinforcement_learning.General.reward_functions import get_state_values
from src.python.double_pendulum.simulation.gym_env import CustomEnv
import pygame
import numpy as np
import gymnasium as gym
from examples.reinforcement_learning.General.dynamics_functions import default_dynamics, random_dynamics, \
    random_push_dynamics, push_dynamics, load_param, custom_dynamics_func_PI, custom_dynamics_func_4PI, real_robot
from examples.reinforcement_learning.General.reward_functions import future_pos_reward, pos_reward, quadratic_rew, saturated_distance_from_target, score_reward
from double_pendulum.simulation.simulation import Simulator

class GeneralEnv(CustomEnv):
    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(
        self,
        robot,
        param,
        policy: CustomPolicy,
        seed,
        path="parameters.json",
        eval=False,
        dynamics_function=None,
        plant=None
    ):

        self.policy = policy
        self.translator = policy.actor_class.get_translator()
        self.seed = seed
        self.eval = eval
        self.param = param
        self.pendulum_length = 350
        self.reward_visualization = 0
        self.action_visualization = None
        self.acc_reward_visualization = 0
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
            low_pos = [-1.0, 0, 0, 0]
        self.reset_function = lambda: reset_function(low_pos)
        reward_function = globals()[self.data[type]["reward_function"]]
        self.reward_name = self.data[type]["reward_function"]
        self.n_envs = self.data[type]["n_envs"]
        self.same_env = self.data[type]["same_env"]

        self.dynamics_function = dynamics_function
        self.reward_function = lambda obs, act, state_dict: reward_function(obs, act, robot, self.dynamics_func, state_dict, self.virtual_sensor_state_tracking)
        self.max_episode_steps = self.data["max_episode_steps"]
        self.render_every_steps = self.data["render_every_steps"]
        self.render_every_envs = self.data["render_every_envs"]

        self.virtual_sensor_state_tracking = [0.0, 0.0]
        self.simulation = None
        if not plant is None:
            self.plant = plant
            self.simulation = Simulator(plant=plant)

        if hasattr(dynamics_function, '__code__'):
            dynamics_function, self.simulation, self.plant = dynamics_function(robot, self.data["dt"], self.data["max_torque"], globals()[dynamic_class_name])

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

        self.window_size = 800
        self.render_mode = "None"
        self.window = None
        self.clock = None

        self.mpar = load_param(robot, self.dynamics_func.torque_limit)
        self.observation_dict = {"T": [], "X_meas": [], "U_con": [], "push": [], "plant": self.dynamics_func.simulator.plant, "max_episode_steps": self.max_episode_steps, "current_force": []}
        self.dynamics_func.simulator.plant.observation_dict = self.observation_dict

    def custom_reset(self):
        observation = self.reset_function()
        self.translator.reset()
        state = self.translator.build_state(observation, 0)
        return state

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
                "plant": plant,
                "seed": self.seed,
                "policy": self.policy
            },
            monitor_dir=log_dir,
            seed=self.seed
        )
        return envs

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed, options)

        if self.simulation is not None:
            self.simulation.reset()
        for key in self.observation_dict:
            if key != 'plant' and key != 'max_episode_steps':
                self.observation_dict[key].clear()

        self.dynamics_func.virtual_sensor_state = [0.0, 0.0]
        self.virtual_sensor_state_tracking = [0.0, 0.0]
        self.reward_visualization = 0
        self.acc_reward_visualization = 0
        self.action_visualization = np.array([0, 0])

        return observation, info

    def step(self, action):
        old_state = self.observation
        old_observation = self.translator.extract_observation(old_state)
        new_observation = self.dynamics_func(old_observation, action, scaling=self.scaling)
        new_state = self.translator.build_state(new_observation, action)
        self.observation = new_state

        self.append_observation_dict(new_observation, action)
        reward = self.get_reward(new_observation, action)
        ignore_state = self.add_virtual_sensor()
        terminated = self.terminated_func(new_observation, self.virtual_sensor_state_tracking, ignore_state)
        truncated = self.check_episode_end()

        if self.render_mode == "human":
            self.reward_visualization = reward
            self.acc_reward_visualization += reward
            self.action_visualization = action

        info = {}
        return self.observation, reward, terminated, truncated, info

    def check_episode_end(self):
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return truncated

    def add_virtual_sensor(self):
        ignore_state = True
        if self.data["dynamic_class"] == "custom_dynamics_func_PI":
            self.virtual_sensor_state_tracking += self.dynamics_func.virtual_sensor_state
            if self.robot == "acrobot":
                ignore_state = True
        return ignore_state

    def get_reward(self, new_observation, action):
        if self.reward_name == "saturated_distance_from_target":
            return self.reward_func(new_observation, action, self.observation_dict)
        else:
            return self.reward_func(new_observation, action, self.observation_dict) / self.max_episode_steps

    def append_observation_dict(self, new_observation, action):
        time = self.dynamics_func.dt
        if len(self.observation_dict["T"]) > 0:
            time = time + self.observation_dict["T"][-1]
        self.observation_dict["T"].append(time)
        self.observation_dict["U_con"].append(self.dynamics_func.unscale_action(action))
        self.observation_dict["X_meas"].append(self.dynamics_func.unscale_state(new_observation))

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
        y, x1, x2, v1, v2, action, goal, dt, threshold, u_p, u_pp = get_state_values(self.observation, self.action_visualization, self.robot, self.dynamics_func)
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
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(np.array([0,0])), self.getXY(x1), 5)
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(x1), self.getXY(x2), 5)

        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(np.array([0,0])), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x1), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x2), 5)
        pygame.draw.circle(canvas, (255, 200, 200), self.getXY(goal), threshold * 4 * self.pendulum_length)
        pygame.draw.circle(canvas, (255, 50, 50), self.getXY(goal), threshold * 2 * self.pendulum_length)
        pygame.draw.circle(canvas, (95, 2, 99), self.getXY(x3), threshold * 2 * self.pendulum_length)

        myFont = pygame.font.SysFont("Times New Roman", 36)
        acc_reward = myFont.render(str(np.round(self.acc_reward_visualization, 5)), 1, (0, 0, 0), )
        reward = myFont.render(str(np.round(self.reward_visualization, 5)), 1, (0, 0, 0), )
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
