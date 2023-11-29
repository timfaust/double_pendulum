import os
from typing import Type

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.SAC_Local.environment import ProgressBarManager
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries

class Trainer:
    def __init__(self, name, environment: Type[GeneralEnv], model: Type[BaseAlgorithm], policy: Type[BasePolicy]):
        self.environment = environment
        self.log_dir = './log_data/' + name
        self.model = model
        self.policy = policy

    def train(self, learning_rate, training_steps, max_episode_steps, eval_freq, n_envs=1, n_eval_episodes=1, same_environment=True, verbose=False):
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
            eval_freq=eval_freq/n_envs,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
            render=True
        )

        agent = self.model(
            self.policy,
            envs,
            verbose=verbose,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            learning_rate=learning_rate,
        )

        with ProgressBarManager(training_steps) as callback:
            agent.learn(training_steps, callback=CallbackList([callback, eval_callback]))

    class GeneralController(AbstractController):
        def __init__(self, model: Type[BaseAlgorithm], environment: Type[GeneralEnv], log_dir):
            super().__init__()

            self.model = model.load(log_dir + "/best_model/best_model")
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

    def simulate(self, tf=10.0):
        controller = self.GeneralController(self.model, self.environment, self.log_dir)

        T, X, U = controller.simulation.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=tf,
            dt=controller.dt,
            controller=controller,
            integrator=controller.integrator,
            save_video=False,
        )
        plot_timeseries(
            T,
            X,
            U,
            X_meas=controller.simulation.meas_x_values,
            pos_y_lines=[np.pi],
            tau_y_lines=[-5.0, 5.0],
        )


