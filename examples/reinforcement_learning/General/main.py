from examples.reinforcement_learning.General.dynamics_functions import default_dynamics
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from sbx import SAC
from sbx.sac.policies import SACPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import argparse


def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        return progress * initial_value

    return func


def exponential_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        k = 5
        return initial_value * np.exp(-k * (1 - progress))

    return func


training_steps = 1e6
learning_rate = exponential_schedule(0.01)
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0]), sigma=0.1 * np.ones(1), theta=0.15, dt=1e-2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="SAC_MLP_1")
    parser.add_argument('--mode', default="train", choices=["train", "retrain", "simulate"])
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--env_type', default="acrobot", choices=["pendubot", "acrobot"])
    parser.add_argument('--use_custom_param', default='False', choices=["True", "False"])
    parser.add_argument('--custom_param_name', default="SAC_Custom")

    args = parser.parse_args()
    env_type = args.env_type

    Non_mdp_reward_ = Non_mdp_reward()

    default_env = GeneralEnv(env_type, default_dynamics,
                             lambda obs, act: Non_mdp_reward_.calc_non_mdp_reward(obs, act, env_type))
    sac = Trainer(args.name, default_env, SAC, SACPolicy, action_noise)


    if args.mode == "train":
        custom_param = None
        if args.use_custom_param == "True":
            custom_param = args.custom_param_name

        print("training new model")
        sac.train(learning_rate=1e-2,
                  training_steps=training_steps,
                  max_episode_steps=200,
                  eval_freq=1e5,
                  n_envs=10,
                  show_progress_bar=False,
                  save_freq=1e4,
                  verbose=True,
                  custom_param=custom_param
                  )
        print("training finished")
    if args.mode == "retrain":
        try:
            print("retraining last model")
            sac.retrain_model(model_path=args.model_path,
                              learning_rate=learning_rate,
                              training_steps=training_steps,
                              max_episode_steps=300,
                              eval_freq=1e5,
                              n_envs=100,
                              show_progress_bar=False,
                              verbose=True,
                              save_freq=1e4)
        except Exception as e:
            print(e)

    if args.mode == "simulate":
        sac.simulate(model_path=args.model_path, save_video=True)

