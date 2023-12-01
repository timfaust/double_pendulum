from examples.reinforcement_learning.General.dynamics_functions import default_dynamics
from examples.reinforcement_learning.General.environments import GeneralEnv
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
    default_env = GeneralEnv(env_type, default_dynamics, lambda obs, act: future_pos_reward(obs, act, env_type))
    sac = Trainer('future_pos_new3', default_env, SAC, sac.MlpPolicy)

    try:
        print("retraining last model")

        #   saved_model_path = "/saved_model/saved_model_...._steps"
        #   trained_model_path = "/saved_model/trained_model"
        sac.retrain_model(model_path="/saved_model/trained_model", training_steps=1e7, max_episode_steps=500, eval_freq=5e5, n_envs=10, show_progress_bar=False, save_freq=5e5, verbose=True)
    except Exception as e:
        print(e)
        print("training new model")
        sac.train(learning_rate=linear_schedule(0.01), training_steps=1e7, max_episode_steps=500, eval_freq=5e5, n_envs=10, show_progress_bar=False, save_freq=5e5, verbose=True)

    print("training finished")
    #sac.simulate(model_path="/saved_model/saved_model_1000000_steps")
