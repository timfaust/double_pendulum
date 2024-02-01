import json
import os
import re
from typing import Type
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
from double_pendulum.utils.csv_trajectory import save_trajectory
from sbx import SAC
from sbx.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback
)

from examples.reinforcement_learning.General.environments import GeneralEnv
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries


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


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_steps, log_dir, data, n_envs):
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_steps = total_steps
        self.log_dir = log_dir
        self.data = data
        self.n_envs = n_envs

    def find_next_log_dir(self):
        tb_log_dir = os.path.join(self.log_dir, "tb_logs")
        os.makedirs(tb_log_dir, exist_ok=True)

        sac_dirs = [d for d in os.listdir(tb_log_dir) if re.match(r'SAC_\d+', d)]
        highest_number = 0
        for d in sac_dirs:
            num = int(d.split('_')[-1])
            highest_number = max(highest_number, num)

        next_sac_dir = f"SAC_{highest_number}"
        return os.path.join(tb_log_dir, next_sac_dir)

    def _on_training_start(self):
        sac_log_dir = self.find_next_log_dir()
        self.pbar = tqdm(total=self.total_steps, desc='Training Progress')
        with SummaryWriter(sac_log_dir) as writer:
            config_str = json.dumps(self.data, indent=4)
            writer.add_text("Configuration", f"```json\n{config_str}\n```", 0)

    def _on_step(self):
        self.pbar.update(self.n_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()


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

        valid_keys = ['gradient_steps', 'ent_coef', 'learning_rate']
        filtered_data = {key: value for key, value in self.environment.data.items() if key in valid_keys}
        agent = SAC(
            SACPolicy,
            envs,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            action_noise=self.action_noise,
            **filtered_data
        )

        agent.learn(self.training_steps, callback=callback_list)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def retrain(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)
        agent = SAC.load(self.log_dir + model_path)
        agent.set_env(envs)

        callback_list = self.get_callback_list()

        agent.learn(self.training_steps, callback=callback_list, reset_num_timesteps=True)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def evaluate(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".zip"):
            raise Exception("model not found")

        agent = SAC.load(self.log_dir + model_path)

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

        progress_bar_callback = ProgressBarCallback(self.training_steps, self.log_dir, self.environment.data, self.environment.n_envs)
        return CallbackList([eval_callback, checkpoint_callback, progress_bar_callback])

    def simulate(self, model_path="/best_model/best_model", tf=10.0):

        controller = self.get_controller(model_path)
        controller.init()
        controller.simulation.set_state(0, [0, 0, 0, 0])

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

    def get_controller(self, model_path="/best_model/best_model"):
        model_path = self.log_dir + model_path
        controller = GeneralController(self.environment, model_path)
        return controller


class GeneralController(AbstractController):
    def __init__(self, environment: Type[GeneralEnv], model_path):
        super().__init__()

        self.model = SAC.load(model_path)
        self.simulation = environment.simulation
        self.dynamics_func = environment.dynamics_func
        self.dt = environment.dynamics_func.dt
        self.scaling = environment.dynamics_func.scaling
        self.integrator = environment.dynamics_func.integrator
        self.actions_in_state = environment.actions_in_state
        self.last_actions = [0, 0]

    def get_control_output_(self, x, t=None):

        if self.actions_in_state:
            x = np.append(x, self.last_actions)

        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            action = self.model.predict(observation=obs, deterministic=True)
            u = self.dynamics_func.unscale_action(action)
        else:
            action = self.model.predict(observation=x, deterministic=True)
            u = self.dynamics_func.unscale_action(action)

        if self.actions_in_state:
            self.last_actions[-1] = self.last_actions[-2]
            self.last_actions[-2] = u[u != 0][0]

        return u
