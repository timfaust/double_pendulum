from typing import List, Tuple, Dict, Optional

import numpy as np
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import Actor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, DefaultActor, CustomPolicy, \
    DefaultCritic, get_additional_values
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, translator, channels=[4, 8, 16]):
        super().__init__()
        self.input_features = translator.feature_dim
        self.timesteps = translator.timesteps
        self.output_dim = translator.output_dim
        self.additional_features = translator.additional_features

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(3, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2], channels[1]),
            nn.ReLU(),
            nn.Linear(channels[1], channels[0]),
            nn.ReLU(),
            nn.Linear(channels[0], self.output_dim),
            nn.Tanh()
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        obs_1 = obs[:, :self.additional_features]
        obs_2 = obs[:, self.additional_features:]
        obs_reshaped = obs_2.view(-1, self.timesteps, self.input_features).unsqueeze(1)
        conv_output = self.conv_layers(obs_reshaped)
        fc_output = self.fc_layers(conv_output)
        output_concat = th.cat((obs_1, fc_output), dim=1)
        return output_concat


class ConvTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 50
        self.feature_dim = 3
        self.output_dim = 8
        self.additional_features = 4 + 17
        self.net_arch = [64, 32, 16]

        super().__init__(self.timesteps * self.feature_dim + self.additional_features)

    def build_state(self, env: GeneralEnv, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        if len(kwargs) > 0:
            conv_memory = [np.append(x[0:2], u) for x, u in zip(kwargs['X_meas'][-self.timesteps:], kwargs['U_con'][-self.timesteps:])]
            conv_memory = np.array(conv_memory)
        else:
            conv_memory = np.array([np.append(dirty_observation.copy()[0:2], clean_action)])

        output = conv_memory
        if output.shape[0] < self.timesteps:
            repeat_count = self.timesteps - output.shape[0]
            output = np.vstack((np.tile(np.zeros(output.shape[1]), (repeat_count, 1)), output))

        output = np.concatenate(output)
        output = np.append(dirty_observation.copy(), output)
        additional = get_additional_values(dirty_observation.copy())
        output = np.append(additional, output)
        return output


class ConvActor(DefaultActor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_net = None

    @classmethod
    def get_translator(cls) -> ConvTranslator:
        return ConvTranslator()

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.conv_net(obs)
        return super().get_action_dist_params(features)


class ConvCritic(DefaultCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_net = None

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        features = self.conv_net(obs)
        return super().forward(features, actions)


class ConvSACPolicy(CustomPolicy):
    actor_class = ConvActor
    critic_class = ConvCritic

    def __init__(self, *args, **kwargs):
        self.translator = self.actor_class.get_translator()
        conv_output_dim = self.translator.output_dim + self.translator.additional_features

        self.additional_actor_kwargs['net_arch'] = self.translator.net_arch
        self.additional_actor_kwargs['features_dim'] = conv_output_dim
        self.additional_critic_kwargs['net_arch'] = self.translator.net_arch
        self.additional_critic_kwargs['features_dim'] = conv_output_dim
        self.conv_net = None

        super().__init__(*args, **kwargs)

        self.critic.conv_net = self.get_conv_net()
        self.critic.add_module(f"conv_net", self.critic.conv_net)
        self.critic_target.conv_net = self.get_conv_net()
        self.critic_target.add_module(f"conv_net", self.critic_target.conv_net)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor = super().make_actor(features_extractor)
        actor.conv_net = self.get_conv_net()
        return actor

    def get_conv_net(self) -> ConvModule:
        if self.conv_net is None:
            self.conv_net = ConvModule(self.translator)
        return self.conv_net

    def after_train(self):
        pass

    def after_actor_backward(self):
        # th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        pass

    def after_critic_backward(self):
        # th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        pass