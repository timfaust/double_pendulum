from typing import Optional, List, Tuple, Dict, Any
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX
import torch as th


class ScoreReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        n_envs = kwargs['n_envs']
        kwargs['n_envs'] = 1
        super().__init__(*args, **kwargs)
        self.episode_ids = np.zeros((self.buffer_size, n_envs), dtype=np.int32)
        self.old_episode_ids = None
        self.replay_buffer = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                handle_timeout_termination=self.handle_timeout_termination
            )

    def update_score_buffer(self, infos):
        current_episode_ids = np.array([info['episode_id'] for info in infos])
        self.episode_ids[self.replay_buffer.pos - 1] = current_episode_ids

        if self.old_episode_ids is not None:
            difference = current_episode_ids - self.old_episode_ids
            for env_id, d in enumerate(difference):
                if d == 1:
                    episode_id = self.old_episode_ids[env_id]
                    size = self.replay_buffer.pos
                    if self.replay_buffer.full:
                        size = self.replay_buffer.buffer_size
                    for i in range(size):
                        if self.episode_ids[i][env_id] == episode_id:
                            super().add(
                                self.replay_buffer.observations[i][env_id],
                                self.replay_buffer.next_observations[i][env_id],
                                self.replay_buffer.actions[i][env_id],
                                self.replay_buffer.rewards[self.replay_buffer.pos - 2][env_id],
                                self.replay_buffer.dones[i][env_id],
                            [{}]
                                    )

        self.old_episode_ids = current_episode_ids

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.replay_buffer.add(obs, next_obs, action, reward, done, infos)
        self.update_score_buffer(infos)


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

