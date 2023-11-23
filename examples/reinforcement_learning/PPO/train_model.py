import os
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (EvalCallback, StopTrainingOnRewardThreshold, CallbackList)
from stable_baselines3.common.env_util import make_vec_env

from examples.reinforcement_learning.PPO.environment import PPOEnv, ProgressBarManager
from examples.reinforcement_learning.PPO import helpers


log_dir = "./log_data/PPO_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# hyperparam baseline

training_steps = 100000
learning_rate = 0.01
verbose = False
n_envs = 100
reward_threshold = 3e7
eval_freq = 5000
n_eval_episodes = 5
max_episode_steps = 200

# hyperparam simulation
robot = "acrobot"
mpar, _, _ = helpers.load_param(robot=robot)

dynamics_function, sim = helpers.get_dynamics_function(mpar, robot)
reward_func = helpers.reward_func
terminated_func = helpers.terminated_func
noisy_reset_func = helpers.noisy_reset_func


envs = make_vec_env(
    env_id=PPOEnv,
    n_envs=n_envs,
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
    render_mode='human',
    max_episode_steps=max_episode_steps
)

agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    learning_rate=learning_rate,
)

callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(log_dir, "best_model"),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
    render=True
)

with ProgressBarManager(training_steps) as callback:
    agent.learn(training_steps, callback=CallbackList([callback, eval_callback]))
