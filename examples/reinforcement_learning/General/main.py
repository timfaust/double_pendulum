from examples.reinforcement_learning.General.environments import DefaultEnv
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3 import sac, ppo, td3, a2c, ddpg


def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        return progress * initial_value

    return func


if __name__ == '__main__':
    env_type = "pendubot"
    default_env = DefaultEnv(env_type, lambda obs, act: future_pos_reward(obs, act, env_type))
    sac = Trainer('future_pos', default_env, SAC, sac.MlpPolicy)
    print("training")
    sac.train(learning_rate=0.01, training_steps=1e6, max_episode_steps=500, eval_freq=1e4, n_envs=25, show_progress_bar=True, save_freq=1e4)
    print("training finished")
    sac.simulate(model_path="/saved_model/trained_model")
