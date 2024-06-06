import random
import torch

from examples.reinforcement_learning.General.policies.common import CustomPolicy
from examples.reinforcement_learning.General.policies.lstm_policy import LSTMSACPolicy
from examples.reinforcement_learning.General.policies.CNN_policies import CNN_SAC_Policy
from examples.reinforcement_learning.General.policies.past_actions_policy import PastActionsSACPolicy
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import argparse
from tdmpc2.tdmpc2.tdmpc2_policy import TDMPC2_Policy
from examples.reinforcement_learning.General.tdmpc2.tdmpc2.config_loader import ConfigLoader
import os



seed = 42

if __name__ == '__main__':

    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # arguments for trainer
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="tdmpc2_tests")
    parser.add_argument('--mode', default="train", choices=["train", "retrain", "evaluate", "simulate"])
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--env_type', default="pendubot", choices=["pendubot", "acrobot"])
    parser.add_argument('--param', default="default")
    args = parser.parse_args()

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0]), sigma=0.1 * np.ones(1), theta=0.15, dt=1e-2)
    agent = Trainer(args.name, args.env_type, args.param, TDMPC2_Policy, seed, action_noise)
    print(f"Current working directory: {os.getcwd()}")

    cfg = ConfigLoader('tdmpc2/tdmpc2/config.yaml')

    if args.mode == "train":
        print("training new model")


        agent.train()
        print("training finished")

    if args.mode == "retrain":
        try:
            print("retraining last model")
            agent.retrain(model_path=args.model_path)
        except Exception as e:
            print(e)

    if args.mode == "evaluate":
        try:
            print("evaluate model")
            agent.evaluate(model_path=args.model_path)
        except Exception as e:
            print(e)

    if args.mode == "simulate":
        agent.simulate(model_path=args.model_path, tf=3.0)
