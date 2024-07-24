import ast
import json
import os
import re
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
from double_pendulum.utils.csv_trajectory import save_trajectory
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback
)

from examples.reinforcement_learning.General.environments import GeneralEnv
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries

from examples.reinforcement_learning.General.misc_helper import debug_reset
from examples.reinforcement_learning.General.override_sb3.callbacks import CustomEvalCallback
from examples.reinforcement_learning.General.override_sb3.custom_sac import CustomSAC


def get_filtered_data(environment):
    # keys which can be replaced from param
    valid_keys = ['actor_schedule', 'critic_schedule', 'entropy_schedule', 'gradient_steps', 'ent_coef',
                  'learning_rate', 'qf_learning_rate', 'batch_size', 'buffer_size', 'target_update_interval',
                  'train_freq', 'gamma']
    filtered_data = {key: value for key, value in environment.param_data.items() if key in valid_keys}
    if isinstance(filtered_data['train_freq'], str) and "'" in filtered_data['train_freq']:
        filtered_data['train_freq'] = ast.literal_eval(filtered_data['train_freq'])
    return filtered_data


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
    def __init__(self, name, env_type, param_name, policy_classes, replay_buffer_classes, decider, seed, action_noise=None):
        self.policy_classes = policy_classes
        self.replay_buffer_classes = replay_buffer_classes
        self.decider = decider
        self.environment = GeneralEnv(env_type, param_name, seed=seed)
        self.eval_environment = GeneralEnv(env_type, param_name, is_evaluation_environment=True, seed=seed)
        self.log_dir = './log_data/' + name + '/' + env_type
        self.name = name
        self.action_noise = action_noise

        self.use_action_noise = self.environment.param_data["use_action_noise"] == 1
        self.max_episode_steps = self.environment.param_data["max_episode_steps"]
        self.training_steps = self.environment.param_data["training_steps"]
        self.eval_freq = self.environment.param_data["eval_freq"]
        self.save_freq = self.environment.param_data["save_freq"]
        self.verbose = self.environment.param_data["verbose"] == 1
        self.render_eval = self.environment.param_data["render_eval"] == 1
        self.n_eval_episodes = self.environment.param_data["n_eval_episodes"]
        self.show_progressBar = self.environment.param_data["show_progress_bar"]

        if not self.use_action_noise:
            self.action_noise = None

    def train(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)

        agent = CustomSAC(
            policy_classes=self.policy_classes,
            replay_buffer_classes=self.replay_buffer_classes,
            env=envs,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            action_noise=self.action_noise,
            seed=self.environment.seed,
            decider=self.decider,
            **get_filtered_data(self.environment)
        )

        callback_list = self.get_callback_list(agent)
        agent.learn(self.training_steps, callback=callback_list)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def retrain(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".pkl"):
            raise Exception("model not found")

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)

        agent = CustomSAC.load(
            self.log_dir + model_path,
            env=envs,
            action_noise=self.action_noise,
            seed=self.environment.seed,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            **get_filtered_data(self.environment)
        )

        callback_list = self.get_callback_list(agent)
        agent.learn(self.training_steps, callback=callback_list, reset_num_timesteps=True)
        agent.save(os.path.join(self.log_dir, "saved_model", "trained_model"))

    def get_eval_envs(self, agent):
        eval_envs = self.eval_environment.get_envs(log_dir=self.log_dir)
        agent.connect_envs(eval_envs)
        for i in range(len(eval_envs.envs)):
            monitor = eval_envs.envs[i]
            if self.render_eval and i % self.eval_environment.render_every_envs == 0:
                monitor.env.render_mode = 'human'
        return eval_envs

    def evaluate(self, model_path):
        if not os.path.exists(self.log_dir + model_path + ".pkl"):
            raise Exception("model not found")

        self.environment.render_mode = None
        self.environment.reset()

        envs = self.environment.get_envs(log_dir=self.log_dir)

        agent = CustomSAC.load(
            self.log_dir + model_path,
            env=envs,
            action_noise=self.action_noise,
            seed=self.environment.seed,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs"),
            **get_filtered_data(self.environment)
        )

        eval_envs = self.get_eval_envs(agent)
        agent.set_env(eval_envs)

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

    def get_callback_list(self, agent):

        eval_envs = self.get_eval_envs(agent)

        eval_callback = CustomEvalCallback(
            eval_envs,
            best_model_save_path=os.path.join(self.log_dir, 'best_model'),
            log_path=self.log_dir,
            eval_freq=int(self.eval_freq / self.environment.n_envs),
            verbose=self.verbose,
            n_eval_episodes=self.n_eval_episodes * self.eval_environment.n_envs,
            render=self.render_eval,
            deterministic=True
        )

        checkpoint_callback = CheckpointCallback(save_freq=int(self.save_freq / self.environment.n_envs), save_path=os.path.join(self.log_dir, 'saved_model'), name_prefix="saved_model")
        progress_bar_callback = ProgressBarCallback(self.training_steps, self.log_dir, self.environment.param_data, self.environment.n_envs)

        if self.show_progressBar:
            return CallbackList([eval_callback, checkpoint_callback, progress_bar_callback])
        else:
            return CallbackList([eval_callback, checkpoint_callback])

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
    def __init__(self, environment: GeneralEnv, model_path):
        super().__init__()

        self.environment = environment
        environment.n_envs = 1

        envs = self.environment.get_envs(log_dir=None)
        envs.envs[0].env.reset_function = debug_reset
        envs.envs[0].env.reset()

        self.model = CustomSAC.load(
            model_path,
            env=envs,
            action_noise=None,
            seed=self.environment.seed,
            tensorboard_log=None,
            **get_filtered_data(self.environment)
        )

        self.simulation = environment.simulation
        self.dynamics_func = environment.dynamics_func
        self.dt = environment.dynamics_func.dt
        self.scaling = environment.dynamics_func.scaling
        self.integrator = environment.dynamics_func.integrator

        self.last_u = None
        self.last_action = 0.0

    # TODO:only use every 10th entry in full observation dict und damit zu jedem step neue action
    def get_control_output_(self, x, t=None):

        rounded_t = np.rint(t * 10000).astype(int)
        rounded_dt = np.rint(self.dt * 10000).astype(int)
        if rounded_t % rounded_dt == 0:
            env = self.model.env.envs[0].env
            obs = self.dynamics_func.normalize_state(x)
            if t != 0:
                env.append_observation_dict(obs, obs, self.last_action)
                env.observation_dict['U_con'].append(self.last_action)
            action, _ = self.model.predict(observation=obs.reshape(1, -1), deterministic=True)
            self.last_u = self.dynamics_func.unscale_action(action)
            self.last_action = action.item()

        return self.last_u.copy()
