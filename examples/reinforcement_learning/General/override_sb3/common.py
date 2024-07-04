from typing import Optional, List, Tuple, Dict, Any
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX
import torch as th
from gymnasium import spaces


class ScoreReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def update_rewards(self, reward):
        size = self.pos
        if self.full:
            size = self.buffer_size
        for env_id, r in enumerate(reward):
            if r > 0.0:
                episode_id = self.episode_ids[self.pos - 1][env_id]
                for i in range(size):
                    if self.episode_ids[i][env_id] == episode_id:
                        self.rewards[i][env_id] = r

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        self.episode_ids[self.pos - 1] = np.array([info['episode_id'] for info in infos])
        self.update_rewards(reward)

    # TODO: only sample when reward > 0
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


def get_all_knowing(env):
    plant = env.dynamics_func.simulator.plant
    all_knowing = np.array([plant.l[0], plant.l[1], plant.m[0], plant.m[1], plant.b[0], plant.b[1], plant.cf[0], plant.cf[1], env.delay, env.start_delay])
    return all_knowing


class DefaultTranslator:
    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, env, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
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

    @classmethod
    def after_rollout(cls, envs):
        pass

    @classmethod
    def after_train(cls):
        pass

    @classmethod
    def after_actor_backward(cls):
        pass

    @classmethod
    def after_critic_backward(cls):
        pass

    @classmethod
    def after_environment_reset(cls, environment):
        # factor = (cls.progress - 0.1) / 0.4
        # factor = np.clip(factor, 0, 1) * 0.5
        factor = 0.5

        changing_values = {
            'l': 0.02 * factor,
            'm': 0.04 * factor,
            'b': 0.005 * factor,
            'cf': 0.4 * factor,
            'start_delay': 0.15 * factor,
            'delay': 0.05 * factor,
            'velocity_noise': 0.005 * factor,
            'velocity_bias': 0.005 * factor,
            'position_noise': 0.005 * factor,
            'position_bias': 0.005 * factor,
            'action_noise': 0.005 * factor,
            'action_bias': 0.005 * factor,
            'n_pert_per_joint': 3,
            'min_t_dist': 1.0,
            'sigma_minmax': [0.01, 0.05],
            'amplitude_min_max': [0.1, 1.0],
            'responsiveness': np.random.uniform(0.3, 1)
        }

        environment.change_dynamics(changing_values, cls.progress)

