import os
from typing import Type

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)

from examples.reinforcement_learning.General.environments import AbstractEnv
from examples.reinforcement_learning.SAC_Local.environment import ProgressBarManager


class Trainer:
    def __init__(self, name, environment: Type[AbstractEnv], model: Type[BaseAlgorithm], policy: Type[BasePolicy]):
        self.environment = environment
        self.log_dir = './log_data/' + name
        self.model = model
        self.policy = policy

    def train(self, learning_rate, training_steps, max_episode_steps, eval_freq, n_envs=1, n_eval_episodes=1, same_environment=True, verbose=False):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.environment.max_episode_steps = max_episode_steps
        envs = self.environment.get_envs(n_envs=n_envs, log_dir=self.log_dir, same=same_environment)
        eval_env = self.environment
        eval_env.render_mode = 'human'
        eval_env = Monitor(eval_env, self.log_dir)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.log_dir, 'best_model'),
            log_path=self.log_dir,
            eval_freq=eval_freq,
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
