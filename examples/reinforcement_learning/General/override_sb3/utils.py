__all__ = ["DummyVecEnv", "ResultsWriter", "get_monitor_files",]

import csv
import json
import os
import time

from glob import glob
from typing import SupportsFloat, Tuple, Union, OrderedDict

import pandas
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.monitor import LoadMonitorResultsError, get_monitor_files, ResultsWriter, Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = CustomMonitor(env, filename=None, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


# overwritten
class CustomMonitor(Monitor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_list: List[List[float]] = []
        self.episode_returns: List[List[float]] = []

    def append_rewards(self, new_rewards: List[float]):
        if not self.reward_list:
            self.reward_list = [[] for _ in range(len(new_rewards))]
        for i, reward in enumerate(new_rewards):
            self.reward_list[i].append(reward)

    def append_return(self, reward_list: List[float]):
        if not self.reward_list:
            self.reward_list = [[] for _ in range(len(reward_list))]
        for i, reward in enumerate(reward_list):
            self.reward_list[i].append(reward)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        self.reward_list = []
        return super().reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward_list, terminated, truncated, info = self.env.step(action)
        self.append_rewards(reward_list)
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = [sum(self.reward_list[i]) for i in range(len(self.reward_list))]
            ep_len = len(self.reward_list[0])
            t = time.time()
            ep_info = [{"r": round(ep_rew[i], 6), "l": ep_len, "t": round(t - self.t_start, 6)} for i in range(len(self.reward_list))]
            for key in self.info_keywords:
                for info_entry in ep_info:
                    info_entry[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            for info_entry in ep_info:
                info_entry.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info[0])
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, np.array(reward_list), terminated, truncated, info


# overwritten
class CustomDummyVecEnv(DummyVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_rews = []

    # overwritten
    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        self.buf_rews = []
        for env_idx in range(self.num_envs):
            obs, reward_list, terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_rews.append(reward_list)
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.array(self.buf_rews, dtype=np.float32).T, np.copy(self.buf_dones), deepcopy(self.buf_infos))