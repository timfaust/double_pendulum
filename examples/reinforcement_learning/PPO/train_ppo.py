import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold, BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from tqdm.auto import tqdm

from examples.reinforcement_learning.PPO.environment import PPOEnv


def reward_func(observation, action):
    # quadratic with roa attraction
    Q = np.zeros((4, 4))
    Q[0, 0] = 10
    Q[1, 1] = 10
    Q[2, 2] = 0.4
    Q[3, 3] = 0.3
    R = np.array([[0.0001]])

    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * 8.0,
            observation[3] * 8.0,
        ]
    )

    u = 5.0 * action

    goal = [np.pi, 0.0, 0.0, 0.0]

    y = wrap_angles_diff(s)

    flag = False
    bonus = False

    # openAI
    p1 = y[0]
    p2 = y[1]
    ee1_pos_y = -0.2 * np.cos(p1)

    ee2_pos_y = ee1_pos_y - 0.3 * np.cos(p1 + p2)

    control_line = 0.4

    # criteria 4
    if ee2_pos_y >= control_line:
        flag = True
    else:
        flag = False

    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r
    if flag:
        reward += 100
        if bonus:
            # roa method
            reward += 1e3
            # print("!!!bonus = True")
    else:
        reward = reward

    return reward


def terminated_func(observation):
    return False


def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation


def load_param(robot="acrobot", torque_limit=5.0):
    """
    param: robot: String robot name
            torque_limit = float
    """
    # model parameter
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"
        torque_array = [torque_limit, 0.0]

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"
        torque_array = [0.0, torque_limit]

    model_par_path = (
            "../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0., 0.])
    mpar.set_cfric([0., 0.])

    return mpar


def get_dynamics_function(mpar, robot, dt=0.01, integrator="runge_kutta"):
    plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant)
    dynamics_function = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=2,
    )
    return dynamics_function


log_dir = "./log_data/PPO_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

robot = "acrobot"
training_steps = 100000
learning_rate = 0.01
verbose = False

mpar = load_param(robot=robot)

dynamics_function = get_dynamics_function(mpar, robot)

envs = make_vec_env(
    env_id=PPOEnv,
    n_envs=1,
    env_kwargs={
        "dynamics_func": dynamics_function,
        "reward_func": reward_func,
        "terminated_func": terminated_func,
        "reset_func": noisy_reset_func,
    },
    monitor_dir=log_dir
)
eval_env = PPOEnv(
    dynamics_func=dynamics_function,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
)

agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    learning_rate=learning_rate,
)

callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=3e7, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(log_dir,
                                      "best_model"),
    log_path=log_dir,
    eval_freq=5000,
    verbose=verbose,
    n_eval_episodes=5,
)

agent.learn(total_timesteps=training_steps, callback=eval_callback)
