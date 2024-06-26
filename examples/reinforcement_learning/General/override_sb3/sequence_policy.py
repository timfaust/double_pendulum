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
import gymnasium as gym


class SequenceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, translator):
        super().__init__(observation_space, translator.output_dim + translator.additional_features)
        self.input_features = translator.feature_dim
        self.timesteps = translator.timesteps
        self.output_dim = translator.output_dim
        self.additional_features = translator.additional_features

    def _process_additional_features(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        if self.additional_features > 0:
            obs_1 = obs[:, :self.additional_features]
            obs_2 = obs[:, self.additional_features:]
            return obs_1, obs_2
        return None, obs

    def _combine_output(self, obs_1: th.Tensor, processed_output: th.Tensor) -> th.Tensor:
        if obs_1 is not None:
            return th.cat((obs_1, processed_output), dim=1)
        return processed_output

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs_1, obs_2 = self._process_additional_features(obs)
        processed_output = self._process_main_features(obs_2)
        return self._combine_output(obs_1, processed_output)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class LSTMExtractor(SequenceExtractor):
    def __init__(self, observation_space: gym.spaces.Box, translator, hidden_size=16, num_layers=2):
        super().__init__(observation_space, translator)
        self.lstm = nn.LSTM(self.input_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.activation = nn.Tanh()

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        obs_reshaped = obs.view(-1, self.timesteps, self.input_features)
        _, (hidden, _) = self.lstm(obs_reshaped)
        last_timestep = hidden[-1]
        fc_output = self.fc(last_timestep)
        return self.activation(fc_output)


class ConvExtractor(SequenceExtractor):
    def __init__(self, observation_space: gym.spaces.Box, translator, channels=[4, 8]):
        super().__init__(observation_space, translator)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(channels[0], channels[1], kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[1], channels[0]),
            nn.Tanh(),
            nn.Linear(channels[0], self.output_dim),
            nn.Tanh()
        )

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        obs_reshaped = obs.view(-1, self.timesteps, self.input_features).unsqueeze(1)
        conv_output = self.conv_layers(obs_reshaped)
        return self.fc_layers(conv_output)


class SequenceTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 50
        self.feature_dim = 5
        self.output_dim = 12
        self.additional_features = 4 #4 + 17
        self.net_arch = [64, 32]

        super().__init__(self.timesteps * self.feature_dim + self.additional_features)

    def build_state(self, env: GeneralEnv, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        if len(kwargs) > 0:
            conv_memory = [np.append(x[0:self.feature_dim - 1], u) for x, u in zip(kwargs['X_meas'][-self.timesteps:], kwargs['U_con'][-self.timesteps:])]
            conv_memory = np.array(conv_memory)
        else:
            conv_memory = np.array([np.append(dirty_observation.copy()[0:self.feature_dim - 1], clean_action)])

        output = conv_memory
        if output.shape[0] < self.timesteps:
            repeat_count = self.timesteps - output.shape[0]
            output = np.vstack((np.tile(np.zeros(output.shape[1]), (repeat_count, 1)), output))

        output = np.concatenate(output)

        output = np.append(dirty_observation.copy(), output)
        # additional = get_additional_values(dirty_observation.copy())
        # output = np.append(additional, output)

        return output


class SequenceSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> SequenceTranslator:
        return SequenceTranslator()

    def __init__(self, *args, **kwargs):
        self.translator = self.get_translator()
        self.additional_actor_kwargs['net_arch'] = self.translator.net_arch
        self.additional_critic_kwargs['net_arch'] = self.translator.net_arch

        kwargs.update(dict(
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(translator=self.translator),
        ))

        super().__init__(*args, **kwargs)
