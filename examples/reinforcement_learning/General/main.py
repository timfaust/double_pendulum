from stable_baselines3.sac.policies import SACPolicy
from examples.reinforcement_learning.General.environments import DefaultEnv
from examples.reinforcement_learning.General.reward_functions import simple_reward_acrobot
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3 import SAC

if __name__ == '__main__':
    default_env = DefaultEnv("acrobot", simple_reward_acrobot)
    sac = Trainer('test', default_env, SAC, SACPolicy)
    sac.train(0.01, 1e6, 500, 10000, 20)
