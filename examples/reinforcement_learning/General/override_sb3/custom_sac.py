import ast
import re
import sys
import time
from copy import deepcopy
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit, GymEnv
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from examples.reinforcement_learning.General.override_sb3.common import OneEnvReplayBuffer
import torch as th
from torch.nn import functional as F

from examples.reinforcement_learning.General.override_sb3.sequence_policy import SequenceSACPolicy


def parse_args_kwargs(input_string):
    def parse_value(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    args = []
    kwargs = {}
    stack = []
    current = ""
    in_brackets = 0

    for char in input_string:
        if char == '[':
            in_brackets += 1
        elif char == ']':
            in_brackets -= 1

        if char == ',' and in_brackets == 0 and not stack:
            if '=' in current:
                key, value = current.split('=', 1)
                kwargs[key.strip()] = parse_value(value.strip())
            else:
                args.append(parse_value(current.strip()))
            current = ""
        else:
            current += char

    if current:
        if '=' in current:
            key, value = current.split('=', 1)
            kwargs[key.strip()] = parse_value(value.strip())
        else:
            args.append(parse_value(current.strip()))

    return args, kwargs

def create_lr_schedule(optimizer, schedule_str):
    if schedule_str.replace(".", "", 1).isdigit():
        lr = float(schedule_str)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['initial_lr'] = lr
        return None

    try:
        pattern = r"(\w+)\s*\((.*)\)"
        match = re.match(pattern, schedule_str)
        if not match:
            raise ValueError(f"Invalid schedule string format: '{schedule_str}'")

        name = match.group(1)
        params, kwargs = parse_args_kwargs(match.group(2))

        if 'lr' not in kwargs:
            raise ValueError("Learning rate ('lr') must be specified in the schedule string.")

        lr = kwargs.pop('lr')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['initial_lr'] = lr

        if name == 'ConstantLR':
            return None

        scheduler_dict = {
            "StepLR": lr_scheduler.StepLR,
            "MultiStepLR": lr_scheduler.MultiStepLR,
            "ExponentialLR": lr_scheduler.ExponentialLR,
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
            "CyclicLR": lr_scheduler.CyclicLR,
            "OneCycleLR": lr_scheduler.OneCycleLR,
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
        }

        if name not in scheduler_dict:
            raise ValueError(f"Scheduler '{name}' is not supported.")

        scheduler_class = scheduler_dict[name]
        scheduler = scheduler_class(optimizer, *params, **kwargs)

        return scheduler

    except Exception as e:
        raise ValueError(f"Error parsing schedule string '{schedule_str}': {e}")


class CustomSAC(SAC):
    def __init__(self, policy_classes, decider, *args, **kwargs):
        self.replay_buffer_classes = [OneEnvReplayBuffer, OneEnvReplayBuffer]
        self.schedulers = []
        self.active_policy = 0
        self.sample_policy = 0
        self.policies = []
        self.policy_classes = policy_classes
        self.policy_number = len(self.policy_classes)
        self.replay_buffers = []
        self.ent_coef_optimizers = []
        self.log_ent_coefs = []
        self.decider = decider

        schedule_params = {
            'actor_schedule': "",
            'critic_schedule': "",
            'entropy_schedule': ""
        }

        for key in list(kwargs.keys()):
            if key in schedule_params:
                schedule_params[key] = str(kwargs.pop(key))

        kwargs['policy'] = self.policy_classes[0]
        super().__init__(*args, **kwargs)

        self.connect_visualization()

        for i in range(len(self.policies)):
            self.select_policy(i)
            if schedule_params['actor_schedule']:
                self.schedulers.append(create_lr_schedule(self.actor.optimizer, schedule_params['actor_schedule']))
            if schedule_params['critic_schedule']:
                self.schedulers.append(create_lr_schedule(self.critic.optimizer, schedule_params['critic_schedule']))
            if schedule_params['entropy_schedule']:
                self.schedulers.append(create_lr_schedule(self.ent_coef_optimizer, schedule_params['entropy_schedule']))

    def connect_visualization(self, env=None):
        if env is None:
            env = self.env
        for monitor in env.envs:
            monitor.env.visualizer.model = self

    def select_policy(self, policy_id):
        if len(self.policies) > policy_id >= 0:
            self.active_policy = policy_id
            self.policy_class = self.policy_classes[self.active_policy]
            self.policy = self.policies[self.active_policy]
            self.actor = self.policy.actor
            self.critic = self.policy.critic
            self.critic_target = self.policy.critic_target
            self.ent_coef_optimizer = self.ent_coef_optimizers[self.active_policy]
            self.log_ent_coef = self.log_ent_coefs[self.active_policy]
            self.replay_buffer = self.replay_buffers[self.active_policy]

    def _setup_model(self) -> None:
        for i in range(self.policy_number):
            if len(self.replay_buffer_classes) > i:
                self.replay_buffer_class = self.replay_buffer_classes[i]
            else:
                self.replay_buffer_class = ReplayBuffer
            self.policy_class = self.policy_classes[i]
            self.replay_buffer = None
            translator = self.policy_class.get_translator()
            self.observation_space = translator.obs_space
            self.action_space = translator.act_space
            super()._setup_model()
            self.log_ent_coefs.append(self.log_ent_coef)
            self.ent_coef_optimizers.append(self.ent_coef_optimizer)
            self.replay_buffers.append(self.replay_buffer)
            self.policies.append(self.policy)

    def decide_policy(self, new_obs):
        assignments = []
        for func in self.decider:
            assignments.append([func(obs) for obs in new_obs])
        return np.array(assignments)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        return self.get_actions(observation, deterministic), None

    def add_to_buffer(self, next_obs, buffer_action, reward, dones, infos):
        selected_policies = self.decide_policy(self._last_original_obs)
        selected_policies_next = self.decide_policy(next_obs)
        envs = [m.env for m in self.env.envs]
        for env_index in range(len(envs)):
            for policy_index in range(self.policy_number):

                reward_index = policy_index
                if policy_index >= len(reward):
                    reward_index = len(reward) - 1

                if selected_policies[policy_index][env_index] == 1:
                    last_state = self.policies[policy_index].translator.build_state(self._last_original_obs[env_index], envs[env_index])

                    next_state_policy = 0
                    for i in range(len(selected_policies_next)):
                        if selected_policies_next[i][env_index] == 1:
                            next_state_policy = i
                            break

                    next_state = self.policies[next_state_policy].translator.build_state(next_obs[env_index], envs[env_index])
                    self.replay_buffers[policy_index].next_state_policy = next_state_policy
                    self.replay_buffers[policy_index].add(last_state, next_state, buffer_action[env_index], np.array([reward[reward_index][env_index]]), np.array([dones[env_index]]), [infos[env_index]])
                    break

    def get_last_state(self, env):
        obs = env.observation_dict['X_meas'][:-1]
        selected_policy = self.decide_policy(obs)
        for policy_index in range(len(selected_policy)):
            if selected_policy[policy_index][0] == 1:
                return self.policies[policy_index].translator.build_state(obs[0], env)
        return None

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        self.add_to_buffer(next_obs, buffer_action, reward, dones, infos)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def get_actions(self, obs, deterministic):
        selected_policies = self.decide_policy(obs)
        envs = [m.env for m in self.env.envs]
        states = [[] for _ in range(self.policy_number)]
        actions = [[] for _ in range(len(envs))]

        for env_index in range(len(envs)):
            for policy_index in range(self.policy_number):
                if selected_policies[policy_index, env_index] == 1:
                    states[policy_index].append(self.policies[policy_index].translator.build_state(obs[env_index], envs[env_index]))
                    break

        policy_actions = [None] * self.policy_number
        for policy_index in range(self.policy_number):
            if len(states[policy_index]) > 0:
                policy_actions[policy_index], _ = self.policies[policy_index].predict(np.array(states[policy_index]), deterministic=deterministic)

        action_index = [0] * self.policy_number
        for env_index in range(len(envs)):
            for policy_index in range(self.policy_number):
                if selected_policies[policy_index, env_index] == 1:
                    actions[env_index] = policy_actions[policy_index][action_index[policy_index]]
                    action_index[policy_index] += 1
                    break

        actions = np.array(actions)
        return actions

    def get_critic_target(self, next_observations, next_policies, device):
        next_observations = [
            [next_observations[i] for i in range(len(next_policies)) if next_policies[i] == j]
            for j in range(self.policy_number)
        ]

        reordered_next_q_values = [None] * len(next_policies)
        reordered_next_log_prob = [None] * len(next_policies)
        policy_outputs = []
        for i in range(self.policy_number):
            if len(next_observations[i]) > 0:
                obs = th.tensor(np.array(next_observations[i]), device=device)
                next_actions, next_log_prob = self.actor.action_log_prob(obs)
                next_q_values = th.cat(self.critic_target(obs, next_actions), dim=1)
                policy_outputs.append((next_q_values, next_log_prob))
            else:
                policy_outputs.append(None)

        index = [0] * self.policy_number
        for i, policy in enumerate(next_policies):
            policy = policy[0]
            if policy_outputs[policy] is not None:
                reordered_next_q_values[i] = policy_outputs[policy][0][index[policy]]
                reordered_next_log_prob[i] = policy_outputs[policy][1][index[policy]]
                index[policy] += 1

        next_q_values = th.stack(reordered_next_q_values)
        next_log_prob = th.stack(reordered_next_log_prob)

        return next_q_values, next_log_prob

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        result = super().collect_rollouts(*args, **kwargs)

        envs = [monitor.env for monitor in args[0].envs]
        # TODO: does not work anymore
        # self.policy.after_rollout(envs)

        total_training_steps = envs[0].training_steps
        schedule_interval = total_training_steps / 100
        if self.num_timesteps % schedule_interval < len(envs):
            self.step_schedules()

        return result

    def step_schedules(self):
        for s in self.schedulers:
            if s is not None:
                s.step()

    def prepare_print_gradients(self, gradient_step, replay_data):
        if self.num_timesteps % 100000 == 0 and gradient_step == 0:
            replay_data.observations.requires_grad = True

    def print_gradients(self, gradient_step, replay_data, logging_name):
        if self.num_timesteps % 100000 == 0 and gradient_step == 0:
            state_gradients = th.mean(th.abs(replay_data.observations.grad), axis=0)
            with SummaryWriter(self.logger.dir) as writer:
                plt.figure(figsize=(8, 6))

                colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(state_gradients)))

                plt.bar(np.arange(len(state_gradients)), state_gradients.detach().cpu().numpy(), color=colors)
                plt.xlabel('Feature Index')
                plt.ylabel('Gradient')
                plt.title('Mean Gradients of Input Features in Batch')
                plt.tight_layout()

                print("print gradients")
                plt.savefig(self.logger.dir + '/gradients.png')
                plt.close()

                writer.add_image('gradient_chart/' + logging_name, plt.imread(self.logger.dir + '/gradients.png'),
                                 self.num_timesteps, dataformats='HWC')

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for i in range(len(self.policies)):
            self.select_policy(i)
            if self.replay_buffer.full or self.replay_buffer.pos > 0:
                self.train_policy(i, gradient_steps, batch_size)

    def train_policy(self, policy_id, gradient_steps: int, batch_size: int = 64, critic_loss_goal=-1.0) -> None:
        logging_name = str(self.active_policy)
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        gradient_step = 0
        keep_training = True
        while keep_training:
            # Sample replay buffer
            replay_data, next_observations, next_policies = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            self.prepare_print_gradients(gradient_step, replay_data)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_q_values, next_log_prob = self.get_critic_target(next_observations, next_policies, replay_data.observations.device)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
            critic_loss_item = np.mean(critic_losses)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.after_critic_backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()

            self.print_gradients(gradient_step, replay_data, logging_name)

            self.policy.after_actor_backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            gradient_step += 1
            keep_training = gradient_step < gradient_steps
            if critic_loss_goal >= 0 and gradient_step < gradient_steps * 10:
                keep_training = keep_training or (critic_loss_item > critic_loss_goal)

        if policy_id == 0:
            self._n_updates += gradient_step

        self.logger.record("train/" + logging_name + "/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/" + logging_name + "/gradient_steps", gradient_step)
        self.logger.record("train/" + logging_name + "/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/" + logging_name + "/actor_loss", np.mean(actor_losses))
        self.logger.record("train/" + logging_name + "/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/" + logging_name + "/ent_coef_loss", np.mean(ent_coef_losses))

        actor_lr = self.actor.optimizer.param_groups[0]['lr']
        critic_lr = self.critic.optimizer.param_groups[0]['lr']
        entropy_lr = self.ent_coef_optimizer.param_groups[0]['lr']

        self.logger.record("train/" + logging_name + "/actor_lr", actor_lr)
        self.logger.record("train/" + logging_name + "/critic_lr", critic_lr)
        self.logger.record("train/" + logging_name + "/entropy_lr", entropy_lr)

        self.policy.after_train()

        def record_grads(logger, model, model_name, logging_name):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.record(f"{model_name}/{logging_name}/grads/{name}", param.grad.norm().item())
                logger.record(f"{model_name}/{logging_name}/weights/{name}", param.data.norm().item())

        record_grads(self.logger, self.actor, "actor", logging_name)
        record_grads(self.logger, self.critic, "critic", logging_name)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            for i in range(len(self.ep_info_buffer[0])):
                self.logger.record("rollout/ep_rew_mean_" + str(i), safe_mean([ep_info[i]["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info[0]["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def set_env(self, env: GymEnv, force_reset: bool = True, last_obs=None, last_original_obs=None):
        if last_obs is not None:
            self._last_obs = last_obs
            self._last_original_obs = last_original_obs

        last_obs = self._last_obs
        last_original_obs = self._last_original_obs
        self.n_envs = env.num_envs
        self.env = env

        if force_reset:
            self._last_obs = None

        return last_obs, last_original_obs

    def after_environment_reset(self, environment):
        progress = self.num_timesteps / self.env.envs[0].env.training_steps
        factor = (progress - 0.25) / 0.1
        factor = np.clip(factor, 0, 1) * 0.5
        # factor = 0.0
        #
        # changing_values = {
        #     'l': 0.0 * factor,
        #     'm': 0.25 * factor,
        #     'b': 0.1 * factor,
        #     'coulomb_fric': 0.2 * factor,
        #     'com': 0.25 * factor,
        #     'I': 0.25 * factor,
        #     'Ir': 0.0001 * factor,
        #     'start_delay': 0.0 * factor,
        #     'delay': 0.04 * factor,
        #     'velocity_noise': 0.025 * factor,
        #     'velocity_bias': 0.0 * factor,
        #     'position_noise': 0.0 * factor,
        #     'position_bias': 0.0 * factor,
        #     'action_noise': 0.22 * factor,
        #     'action_bias': 0.0 * factor,
        #     'n_pert_per_joint': 0,
        #     'min_t_dist': 1.0,
        #     'sigma_minmax': [0.05, 0.1],
        #     'amplitude_min_max': [0.1, 5.0],
        #     'responsiveness': np.random.uniform(1 - factor * 1.8, 1 + factor * 2)
        # }
        #

        changing_values = {
            'l': 0.05 * factor,
            'm': 0.05 * factor,
            'b': 0.05 * factor,
            'coulomb_fric': 0.05 * factor,
            'com': 0.05 * factor,
            'I': 0.05 * factor,
            'Ir': 0.0001 * factor,
            'start_delay': 0.0,
            'delay': 0.02 * factor,
            'velocity_noise': 0.005 * factor,
            'velocity_bias': 0.0,
            'position_noise': 0.005 * factor,
            'position_bias': 0.0,
            'action_noise': 0.005 * factor,
            'action_bias': 0.0,
            'n_pert_per_joint': 0,
            'min_t_dist': 1.0,
            'sigma_minmax': [0.05, 0.1],
            'amplitude_min_max': [0.1, 5.0],
            'responsiveness': np.random.uniform(1 - 0.1 * factor, 1 + 0.1 * factor)
        }

        if factor > 0:
            environment.change_dynamics(changing_values, progress)



