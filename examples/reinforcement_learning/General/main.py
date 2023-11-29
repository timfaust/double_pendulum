from stable_baselines3.sac.policies import SACPolicy
from examples.reinforcement_learning.General.environments import DefaultEnv
from examples.reinforcement_learning.General.reward_functions import future_pos_reward_acrobot
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3 import SAC


def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        return progress * initial_value

    return func


if __name__ == '__main__':
    default_env = DefaultEnv("acrobot", future_pos_reward_acrobot)
    sac = Trainer('future_slow', default_env, SAC, SACPolicy)
    sac.train(0.001, 5e6, 500, 5e5, 20)
    sac.simulate()
