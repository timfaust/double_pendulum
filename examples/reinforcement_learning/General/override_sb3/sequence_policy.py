import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.misc_helper import find_observation_index, find_index_and_dict
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy
import gymnasium as gym

from examples.reinforcement_learning.General.reward_functions import get_state_values


class SmoothingFilter(nn.Module):
    def __init__(self, num_features, kernel_size=11, padding='same'):
        super(SmoothingFilter, self).__init__()
        self.num_features = num_features
        self.conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_features,
            bias=False
        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)
        self.alpha = nn.Parameter(th.zeros(num_features))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        batch_size, sequence_length, num_features = x.size()
        x_t = x.transpose(1, 2)  # Transpose to (batch_size, num_features, sequence_length)
        smoothed = self.conv(x_t)
        smoothed = smoothed.transpose(1, 2)  # Transpose back to (batch_size, sequence_length, num_features)

        alpha_sigmoid = self.activation(self.alpha).view(1, 1, -1)
        alpha_sigmoid = alpha_sigmoid.expand(batch_size, sequence_length, num_features)  # Expand to match the input dimensions
        return alpha_sigmoid * x + (1 - alpha_sigmoid) * smoothed


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
    def __init__(self, observation_space: gym.spaces.Box, translator, hidden_size=32, num_layers=2, dropout=0.0):
        super().__init__(observation_space, translator)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, self.output_dim)
        self.activation = nn.Tanh()
        self.smoothing = SmoothingFilter(self.input_features)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.size(0)
        obs_reshaped = obs.view(batch_size, self.timesteps, self.input_features)
        obs_smoothed = self.smoothing(obs_reshaped)
        lstm_out, (h_n, c_n) = self.lstm(obs_smoothed)
        last_hidden = h_n[-1]
        last_cell = c_n[-1]
        concatenated = th.cat((last_hidden, last_cell), dim=1)
        fc_output = self.fc(concatenated)
        return self.activation(fc_output)


class SequenceTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 256
        self.feature_dim = 5
        self.output_dim = 16
        self.additional_features = 8
        self.net_arch = [128, 64, 32]

        super().__init__(self.timesteps * self.feature_dim + self.additional_features)

    def build_state(self, observation, env) -> np.ndarray:
        index, observation_dict = find_index_and_dict(observation, env)
        clean_action = observation_dict['U_con'][index]
        dirty_observation = observation
        sequence_start = max(0, index + 1 - self.timesteps)

        if observation_dict:
            X_meas = np.array(observation_dict['X_meas'])
            U_con = np.array(observation_dict['U_con'])
            conv_memory = np.hstack((
                X_meas[sequence_start:index + 1, :self.feature_dim - 1],
                U_con[sequence_start:index + 1, np.newaxis]
            ))
        else:
            conv_memory = np.append(dirty_observation[:self.feature_dim - 1], clean_action).reshape(1, -1)

        if index < 0:
            print("This should not happen :(")

        output = conv_memory
        if output.shape[0] < self.timesteps:
            padding = np.zeros((self.timesteps - output.shape[0], output.shape[1]))
            output = np.vstack((padding, output))

        output = output.flatten()
        output = np.append(dirty_observation, output)

        state_values = get_state_values(observation_dict, offset=index + 1 - len(observation_dict['T']))
        l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([
            state_values['x3'][1] / l_ges,
            state_values['v2'][0] / env.dynamics_func.max_velocity,
            state_values['c1'],
            state_values['c2']
        ])

        return np.append(additional, output)


class SequenceSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> SequenceTranslator:
        return SequenceTranslator()

    def __init__(self, *args, **kwargs):
        self.translator = self.get_translator()
        self.additional_actor_kwargs['net_arch'] = self.translator.net_arch
        self.additional_critic_kwargs['net_arch'] = self.translator.net_arch

        kwargs.update(
            dict(
                features_extractor_class=LSTMExtractor,
                features_extractor_kwargs=dict(translator=self.translator),
                share_features_extractor=False,
                optimizer_kwargs={'weight_decay': 0.000001}
            )
        )

        super().__init__(*args, **kwargs)

    def after_critic_backward(self):
        pass
        # th.nn.utils.clip_grad_norm_(self.critic.parameters(), 25)
