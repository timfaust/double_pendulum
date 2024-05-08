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

    def print_architecture(self):
        print("\nActor Architecture\n" + "=" * 20)

        # Funktion zum Ausdrucken der Parameteranzahl und Beschreibung eines Modulteils
        def print_part_details(part, name, description):
            num_params = sum(p.numel() for p in part.parameters())
            print(f"{name}:\n{description}\n{part}\n - Number of parameters: {num_params}\n")

        # Ausgabe für jeden Teil des Modells
        print_part_details(self.features_extractor, "Input to Features",
                           "Extracts features from input observations to use in the policy network.")
        print_part_details(self.latent_pi, "Latent Policy Features",
                           "Transforms extracted features into a latent space from which actions can be decided.")
        print_part_details(self.mu, "Mean Actions",
                           "Computes the mean of the action distribution based on latent features.")

        if hasattr(self, 'log_std') and not callable(self.log_std):
            # Konstante log_std verwendet
            print(f"Log Std Deviations (constant):\n - Number of parameters: {self.log_std.numel()}\n")
        else:
            # Zustandsabhängige log_std
            print_part_details(self.log_std, "Log Std Deviations",
                               "Computes the log standard deviations for action exploration, based on latent features.")

        print("Actions for every environment are sampled from resulting distribution and action noise is added. Then environments are stepped with selected actions.\n"
              "Afterwards training is performed in gradient steps. For each gradient step actions are selected using current actor model for batch size from replay buffer (256 observations)\n"
              "Then q_values_pi are computed with those actions and corresponding past observations. Also current_q_values are computed with past actions and corresponding past observations.")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")
        print("=" * 20)
        print("\n\n")


class CustomPolicy(SACPolicy, ABC):
    actor_class = CustomActor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor = self.actor_class(**actor_kwargs).to(self.device)
        actor.print_architecture()
        return actor