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
    sac.train(0.01, 1e7, 500, 1e6, 25)
    sac.simulate()
