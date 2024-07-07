import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy
import gymnasium as gym

from examples.reinforcement_learning.General.reward_functions import get_state_values


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
    def __init__(self, observation_space: gym.spaces.Box, translator, hidden_size=64, num_layers=2):
        super().__init__(observation_space, translator)
        # self.filter = SmoothingFilter(self.input_features, 5)
        self.lstm = nn.LSTM(self.input_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(0.2)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        obs_reshaped = obs.view(-1, self.timesteps, self.input_features)
        # smoothed_seq = self.filter(obs_reshaped)
        _, (hidden, _) = self.lstm(obs_reshaped)
        last_timestep = hidden[-1]
        # dropped = self.dropout(last_timestep)
        fc_output = self.fc(last_timestep)
        return self.activation(fc_output)


class SequenceTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 200
        self.feature_dim = 5
        self.output_dim = 12
        self.additional_features = 8
        self.net_arch = [128, 64, 32]

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
        state_values = get_state_values(env.observation_dict)
        l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([state_values['x3'][1]/l_ges, state_values['v2'][0]/env.dynamics_func.max_velocity, state_values['c1'], state_values['c2']])
        output = np.append(additional, output)

        return output


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
                share_features_extractor=False
            )
        )

        super().__init__(*args, **kwargs)
