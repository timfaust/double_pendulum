import ast
import re
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import RolloutReturn
from stable_baselines3.common.utils import polyak_update
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.override_sb3.common import CustomPolicy
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
    def __init__(self, *args, **kwargs):
        self.schedulers = []
        schedule_params = {
            'actor_schedule': "",
            'critic_schedule': "",
            'entropy_schedule': ""
        }

        for key in list(kwargs.keys()):
            if key in schedule_params:
                schedule_params[key] = str(kwargs.pop(key))

        super().__init__(*args, **kwargs)

        if schedule_params['actor_schedule']:
            self.schedulers.append(create_lr_schedule(self.actor.optimizer, schedule_params['actor_schedule']))
        if schedule_params['critic_schedule']:
            self.schedulers.append(create_lr_schedule(self.critic.optimizer, schedule_params['critic_schedule']))
        if schedule_params['entropy_schedule']:
            self.schedulers.append(create_lr_schedule(self.ent_coef_optimizer, schedule_params['entropy_schedule']))

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        result = super().collect_rollouts(*args, **kwargs)
        envs: List[GeneralEnv] = [monitor.env for monitor in args[0].envs]
        progress = self.num_timesteps / envs[0].training_steps
        self.policy_class.progress = progress
        self.policy.after_rollout(envs, *args, **kwargs)

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

        for gradient_step in range(gradient_steps):
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

                    writer.add_image('Gradient Bar Chart', plt.imread(self.logger.dir + '/gradients.png'),
                                     self.num_timesteps, dataformats='HWC')

            self.policy.after_actor_backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        actor_lr = self.actor.optimizer.param_groups[0]['lr']
        critic_lr = self.critic.optimizer.param_groups[0]['lr']
        entropy_lr = self.ent_coef_optimizer.param_groups[0]['lr']

        self.logger.record("train/actor_lr", actor_lr)
        self.logger.record("train/critic_lr", critic_lr)
        self.logger.record("train/entropy_lr", entropy_lr)

        self.policy.after_train()
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                self.logger.record(f"actor/grads/{name}", param.grad.norm().item())
            self.logger.record(f"actor/weights/{name}", param.data.norm().item())

        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                self.logger.record(f"critic/grads/{name}", param.grad.norm().item())
            self.logger.record(f"critic/weights/{name}", param.data.norm().item())

