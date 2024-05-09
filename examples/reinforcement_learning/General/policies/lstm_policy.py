from typing import List, Tuple, Optional, Dict

import numpy as np
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import Actor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.policies.common import DefaultTranslator, DefaultActor, CustomPolicy, DefaultCritic


class ReshapeToLSTMInput(nn.Module):
    def __init__(self, timesteps, features_per_timestep):
        super().__init__()
        self.timesteps = timesteps
        self.features_per_timestep = features_per_timestep

    def forward(self, x):
        return x.view(-1, self.timesteps, self.features_per_timestep)


class ExtractLastTimestep(nn.Module):
    def forward(self, x):
        return x[0][:, -1, :]


class LSTMTranslator(DefaultTranslator):
    def __init__(self):
        self.lstm_memory = None
        self.reset()
        self.timesteps = 10
        self.features_per_timestep = 5
        self.lstm_hidden_dim = 64
        self.num_layers = 1
        self.net_arch = [64, 64]

        super().__init__(self.timesteps * self.features_per_timestep)

    def extract_observation(self, state: np.ndarray) -> np.ndarray:
        return self.lstm_memory[-self.features_per_timestep:-1]

    def build_state(self, observation: np.ndarray, action: float) -> np.ndarray:
        observation = np.append(observation, action)
        if self.lstm_memory.size == 0:
            self.lstm_memory = observation
        else:
            self.lstm_memory = np.concatenate((self.lstm_memory, observation), axis=0)

        num_required = self.timesteps * self.features_per_timestep
        output = self.lstm_memory
        if output.size < num_required:
            repeat_count = self.timesteps + 1
            repeated_memory = np.tile(output[0:self.features_per_timestep], repeat_count)
            output = np.concatenate((repeated_memory, output), axis=0)

        return output[-num_required:]

    def reset(self):
        self.lstm_memory = np.array([])


class LSTMActor(DefaultActor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_net = None

    @classmethod
    def get_translator(cls) -> LSTMTranslator:
        return LSTMTranslator()

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        obs = self.lstm_net(obs)
        return super().get_action_dist_params(obs)


class LSTMCritic(DefaultCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_net = None

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        obs = self.lstm_net(obs)
        return super().forward(obs, actions)


class LSTMSACPolicy(CustomPolicy):
    actor_class = LSTMActor
    critic_class = LSTMCritic

    def __init__(self, *args, **kwargs):
        translator = self.actor_class.get_translator()
        lstm_input_dim = translator.features_per_timestep
        lstm_output_dim = translator.net_arch[-1]
        self.additional_actor_kwargs['net_arch'] = translator.net_arch
        self.additional_actor_kwargs['features_dim'] = lstm_output_dim
        self.additional_critic_kwargs['net_arch'] = translator.net_arch
        self.additional_critic_kwargs['features_dim'] = lstm_output_dim

        super().__init__(*args, **kwargs)

        self.reshape_to_lstm_input = ReshapeToLSTMInput(translator.timesteps, lstm_input_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=translator.lstm_hidden_dim,
            num_layers=translator.num_layers,
            batch_first=True
        )
        self.extract_last_timestep = ExtractLastTimestep()
        self.action_head = nn.Linear(translator.lstm_hidden_dim, lstm_output_dim)

        self.lstm_net = nn.Sequential(
            self.reshape_to_lstm_input,
            self.lstm,
            self.extract_last_timestep,
            self.action_head,
            nn.ReLU(),
        )

        self.actor.lstm_net = self.lstm_net
        self.critic.lstm_net = self.lstm_net
        self.critic_target.lstm_net = self.lstm_net

    @classmethod
    def after_rollout(cls, envs: List[GeneralEnv], progress, *args, **kwargs):
        # for env in envs:
        #     cls.modify_env(env, progress)
        pass

    @classmethod
    def modify_env(cls, environment: GeneralEnv, progress):
        if progress > 0.3:
            environment.start_delay = 0.1
            environment.delay = 0.02
        elif progress > 0.1:
            factor = (progress - 0.1)/0.2
            environment.start_delay = 0.1 * factor
            environment.delay = 0.02 * factor


