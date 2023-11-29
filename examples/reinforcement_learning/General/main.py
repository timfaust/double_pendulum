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
    default_env = DefaultEnv("acrobot", up_slow_reward_acrobot)
    sac = Trainer('future_slow', default_env, SAC, sac.MlpPolicy)
    sac.train(0.001, 5e6, 500, 5e5, 20)
    sac.simulate()
