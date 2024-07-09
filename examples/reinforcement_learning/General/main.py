import random
import torch
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from param_helper import load_json_params
import argparse

seed = 42

if __name__ == '__main__':

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="test_1")
    parser.add_argument('--mode', default="fine_tune", choices=["train", "retrain", "evaluate", "simulate", "fine_tune"])
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--robot', default="pendubot", choices=["pendubot", "acrobot"])
    parser.add_argument('--param_name', default="real_robot_1")

    args = parser.parse_args()
    robot = args.robot
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0]), sigma=0.1 * np.ones(1), theta=0.15, dt=1e-2)

    data = load_json_params(args.param_name)
    sac = Trainer(args.name, robot, data, seed, action_noise)



    if args.mode == "train":
        print("training new model")
        sac.train()
        print("training finished")

    if args.mode == "retrain":
        try:
            print("retraining last model")
            sac.retrain(model_path=args.model_path)
        except Exception as e:
            print(e)

    if args.mode == "evaluate":
        try:
            print("evaluate model")
            sac.evaluate(model_path=args.model_path)
        except Exception as e:
            print(e)

    if args.mode == "simulate":
        sac.simulate(model_path=args.model_path, tf=10.0)

    if args.mode == "fine_tune":
        sac.simulate(model_path=args.model_path, tf=5.0, fine_tune=True, train_freq=10, training_steps=100)
