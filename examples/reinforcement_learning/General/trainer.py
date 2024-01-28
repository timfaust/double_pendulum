import os
from typing import Type

import numpy as np
from double_pendulum.utils.csv_trajectory import save_trajectory
from sbx import SAC
from sbx.sac.policies import SACPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback
)

from examples.reinforcement_learning.General.environments import GeneralEnv
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries


class Trainer:
    def __init__(self, name, env_type, param, action_noise=None):
        self.environment = GeneralEnv(env_type, param)
        self.eval_environment = GeneralEnv(env_type, param, eval=True)
        self.log_dir = './log_data/' + name + '/' + env_type
        self.name = name
        self.action_noise = action_noise

        self.use_action_noise = self.environment.data["use_action_noise"] == 1
        self.max_episode_steps = self.environment.data["max_episode_steps"]
        self.training_steps = self.environment.data["training_steps"]
        self.eval_freq = self.environment.data["eval_freq"]
        self.save_freq = self.environment.data["save_freq"]
        self.verbose = self.environment.data["verbose"] == 1
        self.render_eval = self.environment.data["render_eval"] == 1
        self.n_eval_episodes = self.environment.data["n_eval_episodes"]

        if not self.use_action_noise:
            self.action_noise = None

    def train(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)
        callback_list = self.get_callback_list()

        agent = SAC(
            SACPolicy,
            envs,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            action_noise=self.action_noise,
        )
        self.load_custom_params(agent)

        agent.learn(self.training_steps, callback=callback_list)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def retrain(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)
        agent = SAC.load(self.log_dir + model_path)
        self.load_custom_params(agent)
        agent.set_env(envs)

        callback_list = self.get_callback_list()

        agent.learn(self.training_steps, callback=callback_list, reset_num_timesteps=True)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))


    def evaluate(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        agent = SAC.load(self.log_dir + model_path)
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
            while not done:
                action, _states = agent.predict(state)
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
            render=self.render_eval
        )

        checkpoint_callback = CheckpointCallback(save_freq=int(self.save_freq / self.environment.n_envs),
                                                 save_path=os.path.join(self.log_dir, 'saved_model'),
                                                 name_prefix="saved_model")

        return CallbackList([eval_callback, checkpoint_callback])

    def load_custom_params(self, agent):
        for key in self.environment.data:
            if hasattr(agent, key):
                setattr(agent, key, self.environment.data[key])

    class GeneralController(AbstractController):
        def __init__(self, model: Type[BaseAlgorithm], environment: Type[GeneralEnv], model_path):
            super().__init__()

            self.model = model.load(model_path)
            self.simulation = environment.simulation
            self.dynamics_func = environment.dynamics_func
            self.model.predict([0.0, 0.0, 0.0, 0.0])
            self.dt = environment.dynamics_func.dt
            self.scaling = environment.dynamics_func.scaling
            self.integrator = environment.dynamics_func.integrator

        def get_control_output_(self, x, t=None):
            if self.scaling:
                obs = self.dynamics_func.normalize_state(x)
                action = self.model.predict(obs)
                u = self.dynamics_func.unscale_action(action)
            else:
                action = self.model.predict(x)
                u = self.dynamics_func.unscale_action(action)

            return u

    def simulate(self, model_path="/best_model/best_model", tf=10.0):

        controller = self.get_controller(model_path)
        controller.init()

        T, X, U = controller.simulation.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=tf,
            dt=controller.dt,
            controller=controller,
            integrator=controller.integrator,
            save_video=True,
            video_name=os.path.join(self.log_dir, "sim_video.gif"),
            scale=0.25
        )

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
            show=False,
            scale=0.5,
        )

    def create_leader_board(self):
        leaderboard_config = {
        "csv_path": self.log_dir + "/sim_swingup.csv",
        "name": self.name,
        "simple_name": "SAC LQR",
        "short_description": "Swing-up with an RL Policy learned with SAC.",
        "readme_path": f"readmes/{self.name}.md",
        "username": "chiniklas",
        }

        return leaderboard_config

    def get_controller(self, model_path="/best_model/best_model"):
        model_path = self.log_dir + model_path
        controller = self.GeneralController(self.model, self.environment, model_path)
        return controller
