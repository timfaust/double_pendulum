from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="debug")
    parser.add_argument('--mode', default="train", choices=["train", "retrain", "evaluate", "simulate"])
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--env_type', default="acrobot", choices=["pendubot", "acrobot"])
    parser.add_argument('--param', default="test")

    args = parser.parse_args()
    env_type = args.env_type
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0]), sigma=0.1 * np.ones(1), theta=0.15, dt=1e-2)

    sac = Trainer(args.name, env_type, args.param, action_noise)

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
        sac.simulate(model_path=args.model_path)
