import os
from typing import Type

import numpy as np
from double_pendulum.utils.csv_trajectory import save_trajectory
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback
)

from examples.reinforcement_learning.General.environments import GeneralEnv
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries


class Trainer:
    def __init__(self, name, environment: Type[GeneralEnv], model: Type[BaseAlgorithm], policy: Type[BasePolicy]):
        self.environment = environment
        self.log_dir = './log_data/' + name + '/' + environment.robot
        self.model = model
        self.policy = policy

    def train(self, learning_rate, training_steps, max_episode_steps, eval_freq, n_envs=1, n_eval_episodes=1,
              save_freq=5000, show_progress_bar=True, same_environment=True, verbose=False):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.environment.render_mode = None
        self.environment.reset()
        self.environment.max_episode_steps = max_episode_steps
        envs = self.environment.get_envs(n_envs=n_envs, log_dir=self.log_dir, same=same_environment)

        eval_env = self.environment
        eval_env.render_mode = 'human'
        eval_env = Monitor(eval_env, self.log_dir)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.log_dir, 'best_model'),
            log_path=self.log_dir,
            eval_freq=eval_freq / n_envs,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
            render=True
        )

        checkpoint_callback = CheckpointCallback(save_freq=int(save_freq/n_envs),
                                                 save_path=os.path.join(self.log_dir, 'saved_model'),
                                                 name_prefix="saved_model")

        agent = self.model(
            self.policy,
            envs,
            verbose=verbose,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            learning_rate=learning_rate,
        )

        agent.learn(training_steps, callback=CallbackList([eval_callback, checkpoint_callback]),
                    progress_bar=show_progress_bar)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

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

        model_path = self.log_dir+model_path
        controller = self.GeneralController(self.model, self.environment, model_path)

        T, X, U = controller.simulation.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=tf,
            dt=controller.dt * 0.1,
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
