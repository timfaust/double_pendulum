from examples.reinforcement_learning.General.dynamics_functions import default_dynamics
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.reward_functions import *
from examples.reinforcement_learning.General.trainer import Trainer
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3 import sac, ppo, td3, a2c, ddpg
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


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


####################################### Hyperparameter ############################

############Hyperparams are on standard settings, checl sac.py for standard values and check
# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#for explanation


#Parallel Sim
n_envs = 10 #number of environments

#Adam Optimizer
learning_rate = exponential_schedule(0.01) #learning rate for Q-Networks


#Off Policy
buffer_size = int(10e6) #size of the replay buffer

#Gradient
batch_size = 256 #minibatch_size for each gradient update
gradient_steps = 3 #How many gradient steps to do after each rollout (see train_freq) Set to -1 means to do as many
# gradient steps as steps done in the environment during the rollout.



#SAC
tau = 0.005 #the soft update coefficient (“Polyak update”, between 0 and 1)
gamma = 0.99 #discount factor
ent_coef = "auto" #Entropy regularization coefficient

#Action Noise to simulate real Motor outputs
#this can help for hard exploration problem
mean = mean = np.array([0.0])
sigma = 0.1
theta=0.15 #rate at which noise will revert to mean
dt = 1e-2  # The time step of the noise

action_noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma * np.ones(1), theta=theta, dt=dt)

#State Dependent Exploration ---> For hard Exploration
use_sde = False
sde_sample_freq = -1 #Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning
# of the rollout)
use_sde_at_warmup = False #Whether to use gSDE instead of uniform sampling during the warm up phase
# (before learning starts)

training_steps = 0.1 *10e6
max_episode_steps = 500


#Eval and callback
eval_freq=1e5
save_freq=1e5
###################################################################################







if __name__ == '__main__':
    env_type = "pendubot"
    default_env = GeneralEnv(env_type, default_dynamics, lambda obs, act: unholy_reward_5(obs, act, env_type))
    sac = Trainer('unholy_reward_5', default_env, SAC, sac.MlpPolicy)

    try:
        print("retraining last model")

        sac.retrain_model(model_path="/saved_model/trained_model",
                          learning_rate=learning_rate,
                          training_steps=training_steps,
                          max_episode_steps=max_episode_steps,
                          eval_freq=eval_freq,
                          n_envs=n_envs,
                          show_progress_bar=True,
                          save_freq=save_freq)
    except Exception as e:
        print(e)
        print("training new model")
        sac.train(learning_rate=learning_rate,
                  training_steps=training_steps,
                  max_episode_steps=max_episode_steps,
                  eval_freq=eval_freq,
                  n_envs=n_envs,
                  show_progress_bar=True,
                  save_freq=save_freq,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  gradient_steps=gradient_steps,
                  tau=tau,
                  gamma=gamma,
                  ent_coef=ent_coef,
                  action_noise=action_noise,
                  use_sde=use_sde,
                  sde_sample_freq=sde_sample_freq,
                  use_sde_at_warmup=use_sde_at_warmup
                  )

    print("training finished")

    sac.simulate(model_path="/saved_model/trained_model")
