from stable_baselines3.sac.policies import SACPolicy
from examples.reinforcement_learning.General.environments import DefaultEnv
from examples.reinforcement_learning.General.reward_functions import future_pos_reward_acrobot
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3 import SAC

if __name__ == '__main__':
    default_env = DefaultEnv("acrobot", future_pos_reward_acrobot)
    sac = Trainer('future_pos', default_env, SAC, SACPolicy)
    sac.train(0.005, 5e6, 500, 1e4, 20)
    sac.simulate()
