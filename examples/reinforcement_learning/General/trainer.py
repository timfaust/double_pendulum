import os
from typing import Type

import numpy as np
import json
from double_pendulum.utils.csv_trajectory import save_trajectory
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback, BaseCallback
)
from tqdm.auto import tqdm
from stable_baselines3.common.noise import ActionNoise

from examples.reinforcement_learning.General.environments import GeneralEnv
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries

from examples.reinforcement_learning.General.misc_helper import low_reset


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


class ProgressBarManager(object):
    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class Trainer:
    def __init__(self, name, environment: Type[GeneralEnv], model: Type[BaseAlgorithm], policy: Type[BasePolicy], action_noise: Type[ActionNoise] = None):
        self.environment = environment
        self.log_dir = './log_data/' + name + '/' + environment.robot
        self.model = model
        self.policy = policy
        self.action_noise = action_noise
        self.name = name

    def train(self, learning_rate, training_steps, max_episode_steps, eval_freq, n_envs=1, n_eval_episodes=1,
              save_freq=5000, show_progress_bar=True, same_environment=True, verbose=False, custom_param=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.environment.render_mode = None
        self.environment.reset()
        self.environment.max_episode_steps = max_episode_steps
        envs = self.environment.get_envs(n_envs=n_envs, log_dir=self.log_dir, same=same_environment)

        callback_list = self.get_callback_list(eval_freq, n_envs, n_eval_episodes, save_freq, verbose)

        agent = self.model(
            self.policy,
            envs,
            verbose=verbose,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            learning_rate=learning_rate,
            action_noise=self.action_noise,
            gradient_steps=3
        )

        if custom_param is not None:
            self.load_custom_params(agent, custom_param)

        if show_progress_bar:
            with ProgressBarManager(training_steps) as callback:
                agent.learn(training_steps, callback=CallbackList([callback_list, callback]))
        else:
            agent.learn(training_steps, callback=callback_list)

        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def retrain_model(self, model_path, training_steps, max_episode_steps, eval_freq, n_envs=1, n_eval_episodes=1,
                      save_freq=5000, show_progress_bar=True, same_environment=True, verbose=False, learning_rate=None):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        self.environment.render_mode = None
        self.environment.reset()
        self.environment.max_episode_steps = max_episode_steps
        envs = self.environment.get_envs(n_envs=n_envs, log_dir=self.log_dir, same=same_environment)

        agent = self.model.load(self.log_dir + model_path)
        agent.set_env(envs)
        if learning_rate is not None:
            agent.learning_rate = learning_rate

        callback_list = self.get_callback_list(eval_freq, n_envs, n_eval_episodes, save_freq, verbose)

        if show_progress_bar:
            with ProgressBarManager(training_steps) as callback:
                agent.learn(training_steps, callback=CallbackList([callback_list, callback]), reset_num_timesteps=True)
        else:
            agent.learn(training_steps, callback=callback_list, reset_num_timesteps=True)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def get_callback_list(self, eval_freq, n_envs=1, n_eval_episodes=1, save_freq=5000, verbose=False):

        eval_env = self.environment.clone()
        eval_env.render_mode = 'human'
        eval_env.reset_func = low_reset
        eval_env = Monitor(eval_env, self.log_dir)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.log_dir, 'best_model'),
            log_path=self.log_dir,
            eval_freq=int(eval_freq / n_envs),
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
            render=True
        )

        checkpoint_callback = CheckpointCallback(save_freq=int(save_freq / n_envs),
                                                 save_path=os.path.join(self.log_dir, 'saved_model'),
                                                 name_prefix="saved_model")

        return CallbackList([eval_callback, checkpoint_callback])

    def load_custom_params(self, agent, param_name):
        if not os.path.exists("parameters.json"):
            print("parameter.json doesn't exist!")
            return

        data = json.load(open("parameters.json"))
        if not param_name in data:
            print("couldn't find key: ", param_name)
            return

        for key in data[param_name]:
            if hasattr(agent, key):
                setattr(agent, key, data[param_name][key])

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

    def simulate(self, model_path="/best_model/best_model", tf=10.0, save_video=True):

        controller = self.get_controller(model_path)
        controller.init()

        T, X, U = controller.simulation.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=tf,
            dt=controller.dt * 0.1,
            controller=controller,
            integrator=controller.integrator,
            save_video=save_video,
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
