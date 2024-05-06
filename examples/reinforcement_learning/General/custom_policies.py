from typing import Optional
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor


class LSTMSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LSTMActor(**actor_kwargs).to(self.device)


class LSTMActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lstm_input_dim = self.latent_pi[0].in_features
        lstm_hidden_dim = 64
        num_layers = 2

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.latent_pi = nn.Sequential(self.lstm, nn.Linear(lstm_hidden_dim, self.latent_pi[0].out_features), *self.latent_pi[1:])
