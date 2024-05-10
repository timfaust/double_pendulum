from typing import List, Tuple, Dict

import numpy as np
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.policies.common import DefaultTranslator, DefaultActor, CustomPolicy, DefaultCritic


class LSTMModule(nn.Module):
    def __init__(self, translator):
        super().__init__()
        self.input_features = translator.observation_dim
        self.timesteps = translator.timesteps
        self.observation_dim = translator.observation_dim

        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=translator.lstm_hidden_dim,
            num_layers=translator.num_layers,
            batch_first=True
        )
        self.feature_mapper = nn.Linear(translator.lstm_hidden_dim, translator.lstm_output_dim)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        obs_reshaped = obs.view(-1, self.timesteps, self.observation_dim)
        lstm_output = self.lstm(obs_reshaped)
        outputs, (hidden, cell) = lstm_output
        lstm_last_timestep = hidden[-1]
        mapped_features = self.feature_mapper(lstm_last_timestep)
        original_features = obs_reshaped[:, -1, :]
        combined_features = th.cat((original_features, mapped_features), dim=1)
        return combined_features


class LSTMTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 20
        self.observation_dim = 5
        self.lstm_output_dim = 8
        self.lstm_hidden_dim = 64
        self.num_layers = 1
        self.net_arch = [256, 256]

        super().__init__(self.timesteps * self.observation_dim)

    def build_state(self, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        if len(kwargs) > 0:
            lstm_memory = [np.append(x, u) for x, u in zip(kwargs['X_meas'][-self.timesteps:], kwargs['U_con'][-self.timesteps:])]
            lstm_memory = np.array(lstm_memory)
        else:
            lstm_memory = np.array([np.append(dirty_observation.copy(), clean_action)])

        output = lstm_memory
        if output.shape[0] < self.timesteps:
            repeat_count = self.timesteps - output.shape[0]
            output = np.vstack((np.tile(output[0], (repeat_count, 1)), output))

        output = np.concatenate(output)
        return output


class LSTMActor(DefaultActor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_net = None

    @classmethod
    def get_translator(cls) -> LSTMTranslator:
        return LSTMTranslator()

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.lstm_net(obs)
        return super().get_action_dist_params(features)


class LSTMCritic(DefaultCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_net = None

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        features = self.lstm_net(obs)
        return super().forward(features, actions)


class LSTMSACPolicy(CustomPolicy):
    actor_class = LSTMActor
    critic_class = LSTMCritic

    def __init__(self, *args, **kwargs):
        translator = self.actor_class.get_translator()
        lstm_input_dim = translator.observation_dim
        lstm_output_dim = translator.lstm_output_dim

        self.additional_actor_kwargs['net_arch'] = translator.net_arch
        self.additional_actor_kwargs['features_dim'] = lstm_output_dim + lstm_input_dim
        self.additional_critic_kwargs['net_arch'] = translator.net_arch
        self.additional_critic_kwargs['features_dim'] = lstm_output_dim + lstm_input_dim

        super().__init__(*args, **kwargs)

        lstm_net = LSTMModule(translator)
        self.actor.lstm_net = lstm_net
        self.critic.lstm_net = lstm_net
        self.critic_target.lstm_net = lstm_net

    @classmethod
    def after_environment_reset(cls, environment: GeneralEnv):
        factor = (cls.progress - 0.1)/0.3
        factor = np.clip(factor, 0, 1)

        sigmas = {
            'l': 0.01 * factor,
            'm': 0.02 * factor,
            'b': 0.002 * factor,
            'cf': 0.2 * factor,
            'start_delay': 0.01 * factor,
            'delay': 0.005 * factor,
        }
        environment.change_dynamics(sigmas)


