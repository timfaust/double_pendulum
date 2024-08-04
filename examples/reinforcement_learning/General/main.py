import cProfile
import pstats
import io
from pstats import SortKey
import random
import torch

from examples.reinforcement_learning.General.misc_helper import stabilize, default_decider, swing_up
from examples.reinforcement_learning.General.override_sb3.common import CustomPolicy, MultiplePoliciesReplayBuffer, \
    SplitReplayBuffer
from examples.reinforcement_learning.General.override_sb3.sequence_policy import SequenceSACPolicy
from examples.reinforcement_learning.General.override_sb3.past_actions_policy import PastActionsSACPolicy
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import argparse

seed = 42

"""
this is used only for remote debugging!
leave it out otherwise
"""
#import pydevd_pycharm
#pydevd_pycharm.settrace('localhost', port=55972, stdoutToServer=True, stderrToServer=True)

if __name__ == '__main__':

    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # arguments for trainer
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="default_6_3")
    parser.add_argument('--mode', default="train", choices=["train", "retrain", "evaluate", "simulate"])
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--env_type', default="acrobot", choices=["pendubot", "acrobot"])
    parser.add_argument('--param', default="test")
    args = parser.parse_args()

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0]), sigma=0.1 * np.ones(1), theta=0.15, dt=1e-2)
    sac = Trainer(args.name, args.env_type, args.param, [SequenceSACPolicy], [MultiplePoliciesReplayBuffer], [default_decider], seed, action_noise)

    profiler = cProfile.Profile()
    profiler.enable()

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

    profiler.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.dump_stats("sac_training_profile.prof")
