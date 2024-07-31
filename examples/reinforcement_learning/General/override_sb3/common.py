from typing import Optional, List, Tuple, Dict, Any, Union
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

from examples.reinforcement_learning.General.misc_helper import softmax_and_select, stabilize, swing_up


# custom replay buffer that is split into multiple different which get filled and sampled depending on the state (chosen by the decider functions)
class SplitReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        self.decider = [swing_up, stabilize]
        single_buffer_size = buffer_size // len(self.decider)
        super().__init__(single_buffer_size * len(self.decider), observation_space, action_space, device, 1, optimize_memory_usage, handle_timeout_termination)
        self.buffers = [MultiplePoliciesReplayBuffer(buffer_size=single_buffer_size, observation_space=observation_space, action_space=action_space, device=device, n_envs=1, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination) for _ in self.decider]
        self.progress = 0
        self.original_obs = None
        self.next_state_policy = 0

    def decide_buffer(self):
        assignments = np.array([
            [func(obs, self.progress) for obs in self.original_obs.reshape(1, -1)]
            for func in self.decider
        ])
        return softmax_and_select(assignments)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        selected_buffer = self.decide_buffer()
        buffer_index = np.argmax(selected_buffer, axis=0)[0]
        self.buffers[buffer_index].next_state_policy = self.next_state_policy
        self.buffers[buffer_index].add(obs, next_obs, action, reward, done, infos)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def distribute_evenly(self, batch_size):
        non_empty = [i for i, b in enumerate(self.buffers) if b.size() > 0]
        if not non_empty:
            return [0] * len(self.buffers)

        per_buffer, remainder = divmod(batch_size, len(non_empty))
        return [per_buffer + (i < remainder) if i in non_empty else 0
                for i in range(len(self.buffers))]

    def combine_replay_buffer_samples(self, samples_list):
        if not samples_list:
            raise ValueError("The input list is empty")

        combined_samples = {}
        for field in ReplayBufferSamples._fields:
            tensors = [getattr(sample[0], field) for sample in samples_list]
            combined_samples[field] = th.cat(tensors, dim=0)
        combined_next_observations = [obs for sample in samples_list for obs in sample[1]]
        combined_next_policies = np.concatenate([sample[2] for sample in samples_list])

        return ReplayBufferSamples(**combined_samples), combined_next_observations, np.array(combined_next_policies)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        distribution = self.distribute_evenly(batch_size)
        samples = []
        for i, mini_batch in enumerate(distribution):
            samples.append(self.buffers[i].sample(mini_batch, env=env))
        return self.combine_replay_buffer_samples(samples)


# changes replay buffer history storage to allow multiple policies in parallel with possibly different reward functions
class MultiplePoliciesReplayBuffer(ReplayBuffer):
    """
        A specialized replay buffer for reinforcement learning, designed to handle multiple policies.

        This buffer stores experiences from different policies, allowing for efficient storage and retrieval of data for training multiple policies simultaneously. It extends the standard replay buffer by adding functionality to track which policy generated each experience.

        Attributes:
        -----------
        next_observations : list
            Stores the next observations in the buffer, initialized with `None` and sized to the buffer capacity.
        next_state_policy : int
            The policy identifier for the next state.
        next_policies : np.ndarray
            An array to store the policy index for each experience, facilitating multi-policy training.
    """

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
        """
            Adds a new experience to the buffer.

            Parameters:
            -----------
            obs : np.ndarray
                The current observation.

            next_obs : np.ndarray
                The next observation following the action.

            action : np.ndarray
                The action taken by the agent.

            reward : np.ndarray
                The reward received after taking the action.

            done : np.ndarray
                Boolean array indicating whether the episode ended.

            infos : List[Dict[str, Any]]
                Additional information about the episode, such as whether it ended due to a time limit.

            Returns:
            --------
            None
        """

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
        """
            Retrieves a batch of samples from the buffer, including policy indices.

            Parameters:
            -----------
            batch_inds : np.ndarray
                Indices of the experiences to sample.

            env : Optional[VecNormalize]
                Optional environment for normalizing observations and rewards.

            Returns:
            --------
            tuple
                Contains sampled data and next observations, along with policy indices.
        """
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


# return the actual current plant parameters as ground truth values
def get_all_knowing(env):
    plant = env.dynamics_func.simulator.plant
    all_knowing = np.array([plant.l[0], plant.l[1], plant.m[0], plant.m[1], plant.b[0], plant.b[1], plant.cf[0], plant.cf[1], env.delay, env.start_delay])
    return all_knowing


# default state builder, just returns the observation from the simulator
class DefaultTranslator:

    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, observation, env) -> np.ndarray:
        """
            Processes and returns the latest raw observation from the environment's observation dictionary.

            This method retrieves the most recent measurement from the environment's observation dictionary and returns it.
        """
        dirty_observation = env.observation_dict['X_meas'][-1]
        return dirty_observation

    def reset(self):
        pass


# custom policies which is almost the same but can return the default translator for generic state building
class CustomPolicy(SACPolicy):
    """
        A base class for custom Soft Actor-Critic (SAC) policies.

        This class extends the SACPolicy and serves as a template for more specialized policies. It includes additional mechanisms for
        handling actor and critic network configurations through keyword arguments, and it supports the use of a custom translator for preprocessing observations.
    """
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
