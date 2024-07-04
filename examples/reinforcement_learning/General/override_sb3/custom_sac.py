import ast
import re
import sys
import time
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from examples.reinforcement_learning.General.override_sb3.common import CustomPolicy, ScoreReplayBuffer
import torch as th
from torch.nn import functional as F


def parse_args_kwargs(input_string):
    arg_pattern = re.compile(r'(?P<arg>\[.*?\]|[^,]+)')
    kwarg_pattern = re.compile(r'(?P<key>\w+)\s*=\s*(?P<value>.+)')

    params = []
    kwargs = {}
    matches = arg_pattern.findall(input_string.replace(' ', ''))

    for match in matches:
        kwarg_match = kwarg_pattern.match(match)
        if kwarg_match:
            key = kwarg_match.group('key').strip()
            value = kwarg_match.group('value').strip()
            kwargs[key] = ast.literal_eval(value)
        else:
            params.append(ast.literal_eval(match.strip()))

    return params, kwargs


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
        if kwargs:
            scheduler = scheduler_class(optimizer, *params, **kwargs)
        else:
            scheduler = scheduler_class(optimizer, *params)

        return scheduler

    except Exception as e:
        raise ValueError(f"Error parsing schedule string '{schedule_str}': {e}")


class CustomSAC(SAC):
    def __init__(self, policy_number, *args, **kwargs):
        self.schedulers = []
        self.active_policy = 0
        self.sample_policy = 0
        self.policies = []
        self.policy_number = policy_number
        self.replay_buffers = []
        self.replay_buffer_classes = [ReplayBuffer, ScoreReplayBuffer]
        self.ent_coef_optimizers = []
        self.log_ent_coefs = []

        schedule_params = {
            'actor_schedule': "",
            'critic_schedule': "",
            'entropy_schedule': ""
        }

        for key in list(kwargs.keys()):
            if key in schedule_params:
                schedule_params[key] = str(kwargs.pop(key))

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
        self.active_policy = policy_id
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
            self.replay_buffer = None
            super()._setup_model()
            self.log_ent_coefs.append(self.log_ent_coef)
            self.ent_coef_optimizers.append(self.ent_coef_optimizer)
            self.replay_buffers.append(self.replay_buffer)
            self.policies.append(self.policy)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        self.select_policy(self.sample_policy)
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, reward_list, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            for i in range(self.policy_number):
                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(self.replay_buffers[i], buffer_actions, new_obs, reward_list[i], dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        result = RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

        envs = [monitor.env for monitor in env.envs]
        progress = self.num_timesteps / envs[0].training_steps
        self.policy_class.progress = progress
        self.policy.after_rollout(envs)

        total_training_steps = envs[0].training_steps
        schedule_interval = total_training_steps / 100
        if self.num_timesteps % schedule_interval < len(envs):
            self.step_schedules()

        return result

    def step_schedules(self):
        for s in self.schedulers:
            if s is not None:
                s.step()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for i in range(len(self.policies)):
            self.select_policy(i)
            # TODO: was wird alles zu oft gemacht?
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
        # self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        gradient_step = 0
        keep_training = True
        while keep_training:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            if self.num_timesteps % 100000 == 0 and gradient_step == 0:
                replay_data.observations.requires_grad = True
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
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
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

