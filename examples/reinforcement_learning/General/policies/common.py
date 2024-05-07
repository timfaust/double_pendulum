from typing import Optional
import gymnasium as gym
import numpy as np
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