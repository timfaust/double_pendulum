from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
import gymnasium as gym
import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_diff
import load_parameter

robot = "acrobot"
dt = 0.01
integrator = "runge_kutta"
mpar = load_parameter.load_param("acrobot", [0, 5.0])
termination = False

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

#define observation and action state for gym
#observation (min (q1, q2, qd1, qd2), max (q1, q2, qd1, qd2))
observation_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
action_space = gym.spaces.Box(np.array([-1]), np.array([1]))

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
)

def terminated_func(observation):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * 8.0,
            observation[3] * 8.0,
        ]
    )

    y = wrap_angles_diff(s)
    bonus, rad = check_if_state_in_roa(S, rho, y)
    if termination:
        if bonus:
            print("terminated")
            return bonus
    else:
        return False


class inv_double_pendelum_env(CustomEnv):
    def __init__(
        self,
        actions,
        dynamics_func,
        reward_func,
        terminated_func,
        reset_func,
        obs_space,
        act_space,
        max_episode_steps,
    ):
        super().__init__(
            lambda state, action: dynamics_func(state, np.array([actions[action]])),
            lambda state, action: reward_func(state, np.array([actions[action]])),
            terminated_func,
            reset_func,
            obs_space,
            act_space,
            max_episode_steps,
        )
        self.actions = actions
        self.n_actions = len(actions)

    def step(self, action):
        self.observation = self.dynamics_func(self.observation, action)
        reward = self.reward_func(self.observation, action)
        terminated = self.terminated_func(self.observation)

        self.step_counter += 1

        return self.observation, reward, terminated, {}

    def reset(self):
        self.step_counter = 0

        return super().reset()


