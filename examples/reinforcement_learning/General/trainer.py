import os
import numpy as np
from double_pendulum.utils.csv_trajectory import save_trajectory
from sbx.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback
)
from callbacks import *
from controller import *
from double_pendulum.utils.plotting import plot_timeseries

from examples.reinforcement_learning.General.environments import GeneralEnv
from fineTuneEnv import FineTuneEnv

class Trainer:
    def __init__(self, name, robot, data, seed, action_noise=None):
        self.log_dir = './log_data/' + name + '/' + robot

        self.name = name
        self.action_noise = action_noise
        self.data = data
        self.robot = robot
        self.seed = seed

        self.use_action_noise = self.data["use_action_noise"] == 1
        self.max_episode_steps = self.data["max_episode_steps"]
        self.training_steps = self.data["training_steps"]
        self.eval_freq = self.data["eval_freq"]
        self.save_freq = self.data["save_freq"]
        self.verbose = self.data["verbose"] == 1
        self.render_eval = self.data["render_eval"] == 1
        self.n_eval_episodes = self.data["n_eval_episodes"]
        self.show_progressBar = self.data["show_progress_bar"]
        self.environment = None
        self.eval_environment = None

        if not self.use_action_noise:
            self.action_noise = None

    def train(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callback_list = self.get_callback_list()

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)


        valid_keys = ['gradient_steps', 'ent_coef', 'learning_rate', 'qf_learning_rate']
        filtered_data = {key: value for key, value in self.data.items() if key in valid_keys}

        agent = SAC(
            SACPolicy,
            envs,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            action_noise=self.action_noise,
            seed=self.environment.seed,
            **filtered_data
        )

        self.load_custom_params(agent)

        agent.learn(self.training_steps, callback=callback_list)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def retrain(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        callback_list = self.get_callback_list()

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)

        agent = SAC.load(self.log_dir + model_path, print_system_info=True)
        self.load_custom_params(agent)
        agent.set_env(envs)

        agent.learn(self.training_steps, callback=callback_list, reset_num_timesteps=True)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def evaluate(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        _ = self.get_callback_list()

        agent = SAC.load(self.log_dir + model_path, print_system_info=True)
        self.load_custom_params(agent)

        eval_envs = self.eval_environment.get_envs(log_dir=self.log_dir)
        for i in range(len(eval_envs.envs)):
            monitor = eval_envs.envs[i]
            if self.render_eval and i % self.eval_environment.render_every_envs == 0:
                monitor.env.render_mode = 'human'

        episode_rewards = []
        episode_lengths = []
        for episode in range(self.n_eval_episodes):
            state = eval_envs.reset()
            done = False
            total_rewards = 0
            steps = 0
            while not np.all(done):
                action, _states = agent.predict(observation=state, deterministic=True)
                state, reward, done, info = eval_envs.step(action)
                if self.render_eval:
                    eval_envs.render()
                total_rewards += reward
                steps += 1

            episode_rewards.append(total_rewards)
            episode_lengths.append(steps)

        eval_envs.close()

        print(f"Average reward: {np.mean(episode_rewards)} +/- {np.std(episode_rewards)}")
        print(f"Average episode length: {np.mean(episode_lengths)}")
        return episode_rewards, episode_lengths

    def get_callback_list(self):

        self.environment = GeneralEnv(self.robot, self.seed, self.data["train_env"], self.data)
        self.eval_environment = GeneralEnv(self.robot, self.seed, self.data["eval_env"], self.data)

        eval_envs = self.eval_environment.get_envs(log_dir=self.log_dir)
        for i in range(len(eval_envs.envs)):
            monitor = eval_envs.envs[i]
            if self.render_eval and i % self.eval_environment.render_every_envs == 0:
                monitor.env.render_mode = 'human'

        eval_callback = EvalCallback(
            eval_envs,
            best_model_save_path=os.path.join(self.log_dir, 'best_model'),
            log_path=self.log_dir,
            eval_freq=int(self.eval_freq / self.environment.n_envs),
            verbose=self.verbose,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render_eval,
            deterministic=True
        )

        checkpoint_callback = CheckpointCallback(save_freq=int(self.save_freq / self.environment.n_envs),
                                                 save_path=os.path.join(self.log_dir, 'saved_model'),
                                                 name_prefix="saved_model")

        progress_bar_callback = ProgressBarCallback(self.training_steps, self.log_dir, self.environment.data, self.environment.n_envs)
        if self.show_progressBar:
            return CallbackList([eval_callback, checkpoint_callback, progress_bar_callback])
        else:
            return CallbackList([eval_callback, checkpoint_callback])

    def simulate(self, model_path="/best_model/best_model", tf=10.0, fine_tune=False):
        model_path = self.log_dir + model_path
        model = SAC.load(model_path, print_system_info=True)

        callbacks = self.get_callback_list()
        env = self.eval_environment

        if fine_tune:
            fine_tune_env = FineTuneEnv(self.robot, self.seed, self.data["train_env"], self.data)
            envs = fine_tune_env.get_envs(log_dir=self.log_dir)
            self.load_custom_params(model)
            model.set_env(envs)
            env = fine_tune_env

        controller = GeneralController(env, model, self.robot, callbacks=callbacks, fine_tune=fine_tune)
        controller.init()
        controller.simulation.set_state(0, [0, 0, 0, 0])

        T, X, U = controller.simulation.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=tf,
            dt=controller.dt * 0.1,
            controller=controller,
            integrator=controller.integrator,
            save_video=False,
            video_name=os.path.join(self.log_dir, "sim_video.gif"),
            scale=0.25
        )

        if fine_tune:
            model.save(os.path.join(self.log_dir, "saved_model", "fine_tuned_model"))

        save_trajectory(os.path.join(self.log_dir, "sim_swingup.csv"), T=T, X_meas=X, U_con=U)

        plot_timeseries(
            T,
            X,
            U,
            X_meas=controller.simulation.meas_x_values,
            pos_y_lines=[-np.pi, 0.0, np.pi],
            vel_y_lines=[0.0],
            tau_y_lines=[-5.0, 0.0, 5.0],
            save_to=os.path.join(self.log_dir, "timeseries"),
            show=True,
            scale=0.5,
        )

    def load_custom_params(self, agent):
        for key in self.data:
            if hasattr(agent, key):
                setattr(agent, key, self.data[key])
