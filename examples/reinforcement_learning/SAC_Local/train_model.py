from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from examples.reinforcement_learning.SAC_Local.environment import SACEnv
from examples.reinforcement_learning.SAC_Local.cost import *

dynamics_function, sim = get_dynamics_function(mpar, robot, torque_limit=torque_limit, scaling=scaling)

reward_func = simple_reward_func
terminated_func = terminated_func
noisy_reset_func = noisy_reset_func

reward_func_saturated = saturated_reward_func
test_model = SAC.load(best_model_path)

# create the environment
envs = make_vec_env(
    env_id=SACEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_function,
        "reward_func": reward_func,
        "terminated_func": terminated_func,
        "reset_func": noisy_reset_func,
    },
    monitor_dir=log_dir
)

# create the agent
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    learning_rate=learning_rate
)

# we need to evaluate our trained policy
eval_env = SACEnv(
    dynamics_func=dynamics_function,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    render_mode='human',
    max_episode_steps=max_episode_steps
)
eval_env = Monitor(eval_env, log_dir)

# this will save the best policy
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(log_dir, "best_model", robot),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
    render=True
)

#agent.learn(training_steps, callback=eval_callback, progress_bar=Falase)
test_model.set_env(envs)
test_model.learn(training_steps, callback=eval_callback, progress_bar=False)

#agent.save(os.path.join(log_dir, "saved_model", robot))
test_model.save(os.path.join(log_dir, "saved_model", robot))
