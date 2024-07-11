from typing import Optional, List, Tuple, Dict, Any
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX
import torch as th
from gymnasium import spaces


class OneEnvReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        kwargs['n_envs'] = 1
        super().__init__(*args, **kwargs)
        self.next_observations = [None] * self.buffer_size
        self.next_state_policy = 0
        self.next_policies = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array([next_obs])

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.next_policies[self.pos] = np.array(self.next_state_policy)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = [self._normalize_obs(self.next_observations[batch][env_indices[i], :], env) for i, batch in enumerate(batch_inds)]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            [],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), next_obs, self.next_policies[batch_inds, env_indices]


def get_all_knowing(env):
    plant = env.dynamics_func.simulator.plant
    all_knowing = np.array([plant.l[0], plant.l[1], plant.m[0], plant.m[1], plant.b[0], plant.b[1], plant.cf[0], plant.cf[1], env.delay, env.start_delay])
    return all_knowing


class DefaultTranslator:

    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, observation, env) -> np.ndarray:
        dirty_observation = env.observation_dict['X_meas'][-1]
        return dirty_observation

    def reset(self):
        pass


class CustomPolicy(SACPolicy):
    additional_actor_kwargs = {}
    additional_critic_kwargs = {}

    def __init__(self, *args, **kwargs):
        self.translator = self.get_translator()
        super().__init__(*args, **kwargs)

    @classmethod
    def get_translator(cls) -> DefaultTranslator:
        return DefaultTranslator(4)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(self.additional_actor_kwargs)
        actor = Actor(**actor_kwargs).to(self.device)
        return actor

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(self.additional_critic_kwargs)
        critic = ContinuousCritic(**critic_kwargs).to(self.device)
        return critic

    def after_rollout(self, envs):
        pass

    def after_train(self):
        pass

    def after_actor_backward(self):
        pass

    def after_critic_backward(self):
        pass
