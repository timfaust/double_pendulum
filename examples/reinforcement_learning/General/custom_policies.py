from typing import Optional
import gymnasium as gym
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor
from abc import ABC, abstractmethod


class Translator(ABC):
    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    @abstractmethod
    def extract_observation(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_state(self, observation: np.ndarray, action: float) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass


class CustomActor(Actor, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_translator(cls) -> Translator:
        pass


class CustomPolicy(SACPolicy, ABC):
    actor_class = CustomActor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return self.actor_class(**actor_kwargs).to(self.device)


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


class LSTMTranslator(Translator):
    def __init__(self):
        self.lstm_memory = np.array([])
        self.timesteps = 20
        self.features_per_timestep = 5
        self.lstm_hidden_dim = 64
        self.num_layers = 2

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


class LSTMActor(CustomActor):

    @classmethod
    def get_translator(cls) -> LSTMTranslator:
        return LSTMTranslator()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = np.random.random()
        translator: LSTMTranslator = LSTMActor.get_translator()
        lstm_input_dim = translator.features_per_timestep

        self.reshape_to_lstm_input = ReshapeToLSTMInput(translator.timesteps, lstm_input_dim)
        self.extract_last_timestep = ExtractLastTimestep()
        self.action_head = nn.Linear(translator.lstm_hidden_dim, self.observation_space.shape[0])

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=translator.lstm_hidden_dim,
            num_layers=translator.num_layers,
            batch_first=True
        )

        self.latent_pi = nn.Sequential(
            self.reshape_to_lstm_input,
            self.lstm,
            self.extract_last_timestep,
            self.action_head,
            nn.ReLU(),
            *self.latent_pi
        )


class LSTMSACPolicy(CustomPolicy):
    actor_class = LSTMActor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = np.random.random()
