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
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def get_all_knowing(env: GeneralEnv):
    plant = env.dynamics_func.simulator.plant
    all_knowing = np.array([plant.l[0], plant.l[1], plant.m[0], plant.m[1], plant.b[0], plant.b[1], plant.cf[0], plant.cf[1], env.delay, env.start_delay])
    return all_knowing


# TODO: same code as in reward function
def get_additional_values(dirty_observation: np.ndarray):
    observation = dirty_observation.copy()
    l = [0.2, 0.3]
    x = np.array(
        [
            observation[0] * 2 * np.pi + np.pi,
            observation[1] * 2 * np.pi,
            observation[2] * 20,
            observation[3] * 20,
        ]
    )

    y = wrap_angles_diff(x)
    s1 = np.sin(y[0])
    s2 = np.sin(y[0] + y[1])
    c1 = np.cos(y[0])
    c2 = np.cos(y[0] + y[1])

    x1 = np.array([s1, c1]) * l[0]
    x2 = x1 + np.array([s2, c2]) * l[1]

    # cartesian velocities of the joints
    v1 = np.array([c1, -s1]) * y[2] * l[0]
    v2 = v1 + np.array([c2, -s2]) * (y[2] + y[3]) * l[1]

    x3 = x2 + 0.05 * v2
    distance = np.linalg.norm(x3 - np.array([0, -0.5]))

    additional = np.array(
        [x1[0], x1[1], x2[0], x2[1], v1[0], v1[1], v2[0], v2[1], s1, s2, c1, c2, x[2] ** 2, x[3] ** 2, x3[0], x3[1],
         distance])

    return additional


class DefaultTranslator:
    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, env: GeneralEnv, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        return dirty_observation

    def reset(self):
        pass


class DefaultCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # exactly the same as in ContinuousCritic
    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)


class DefaultActor(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # exactly the same as in Actor
    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class CustomPolicy(SACPolicy):
    actor_class = DefaultActor
    critic_class = DefaultCritic
    additional_actor_kwargs = {}
    additional_critic_kwargs = {}
    progress = 0

    def __init__(self, *args, **kwargs):
        self.translator = self.get_translator()
        super().__init__(*args, **kwargs)

    @classmethod
    def get_translator(cls) -> DefaultTranslator:
        return DefaultTranslator(4)

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
            # 'velocity_noise': 0.005 * factor,
            # 'velocity_bias': 0.005 * factor,
            # 'position_noise': 0.005 * factor,
            # 'position_bias': 0.005 * factor,
            # 'action_noise': 0.005 * factor,
            # 'action_bias': 0.005 * factor
        }

        environment.change_dynamics(sigmas, cls.progress)

