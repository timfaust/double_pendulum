from typing import Optional, List, Tuple, Dict
import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv


class DefaultTranslator:
    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        return dirty_observation

    def reset(self):
        pass


class DefaultCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kwargs['action_dim'] = get_action_dim(self.action_space)
        self.q_networks: List[nn.Module] = []
        for idx in range(kwargs['n_critics']):
            q_net = self.create_critic(**kwargs)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def create_critic(self, **kwargs):
        q_net_list = create_mlp(kwargs['features_dim'] + kwargs['action_dim'], 1, kwargs['net_arch'], kwargs['activation_fn'])
        q_net = nn.Sequential(*q_net_list)
        return q_net

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)


class DefaultActor(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_translator(cls) -> DefaultTranslator:
        return DefaultTranslator(4)

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class CustomPolicy(SACPolicy):
    actor_class = DefaultActor
    critic_class = DefaultCritic
    additional_actor_kwargs = {}
    additional_critic_kwargs = {}
    progress = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(self.additional_actor_kwargs)
        actor = self.actor_class(**actor_kwargs).to(self.device)
        return actor

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(self.additional_critic_kwargs)
        critic = self.critic_class(**critic_kwargs).to(self.device)
        return critic

    def after_rollout(self, envs: List[GeneralEnv], *args, **kwargs):
        pass

    def after_train(self):
        pass

    def after_actor_backward(self):
        pass

    def after_critic_backward(self):
        pass

    @classmethod
    def after_environment_reset(cls, environment: GeneralEnv):
        # factor = (cls.progress - 0.1) / 0.4
        # factor = np.clip(factor, 0, 1) * 0.5
        factor = 0.5

        sigmas = {
            'l': 0.02 * factor,
            'm': 0.04 * factor,
            'b': 0.005 * factor,
            'cf': 0.4 * factor,
            'start_delay': 0.15 * factor,
            'delay': 0.03 * factor,
            # 'velocity_noise': 0.01 * factor,
            # 'velocity_bias': 0.01 * factor,
            # 'position_noise': 0.01 * factor,
            # 'position_bias': 0.01 * factor,
            # 'action_noise': 0.005 * factor,
            # 'action_bias': 0.005 * factor
        }

        environment.change_dynamics(sigmas)

