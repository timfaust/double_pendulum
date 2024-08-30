from stable_baselines3 import SAC
from typing import List, Optional, Union, Dict, Any, Tuple, Iterable, Type
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutReturn, GymEnv
from stable_baselines3.common.utils import polyak_update, safe_mean
import pathlib
import io
import os
import torch as th
import pickle
import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def column_softmax(x):
    col_sums = x.sum(axis=0)
    needs_softmax = ~np.isclose(col_sums, 1.0)
    x[:, needs_softmax] = softmax(x[:, needs_softmax])
    return x

def softmax_and_select(arr):
    softmax_probs = column_softmax(arr)
    result = np.zeros_like(arr)
    selected_rows = [np.argmax(np.random.multinomial(1, softmax_probs[:, i])) for i in range(arr.shape[1])]
    result[selected_rows, np.arange(arr.shape[1])] = 1
    return result

class CustomSAC(SAC):

    def __init__(self, policy_classes, replay_buffer_classes, *args, **kwargs):
        self.replay_buffer_classes = replay_buffer_classes
        self.schedulers = []
        self.active_policy = 0
        self.sample_policy = 0

        self.policies = []
        self.policy_classes = policy_classes
        self.policy_number = len(self.policy_classes)

        self.replay_buffers = []
        self.ent_coef_optimizers = []
        self.log_ent_coefs = []

        # DIFFERENCE: Setup schedulers
        schedule_params = {
            'actor_schedule': "",
            'critic_schedule': "",
            'entropy_schedule': ""
        }

        for key in list(kwargs.keys()):
            if key in schedule_params:
                schedule_params[key] = kwargs.pop(key)

        kwargs['policy'] = self.policy_classes[0]
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        for i in range(self.policy_number):
            if len(self.replay_buffer_classes) > i:
                self.replay_buffer_class = self.replay_buffer_classes[i]
            else:
                self.replay_buffer_class = ReplayBuffer
            self.policy_class = self.policy_classes[i]
            self.replay_buffer = None
            translator = self.policy_class.get_translator()
            self.observation_space = translator.obs_space
            self.action_space = translator.act_space
            super()._setup_model()
            self.log_ent_coefs.append(self.log_ent_coef)
            self.ent_coef_optimizers.append(self.ent_coef_optimizer)
            self.replay_buffers.append(self.replay_buffer)
            self.policies.append(self.policy)

    @classmethod
    def load(  # noqa: C901
            cls: Type[SelfBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,

    ) -> SelfBaseAlgorithm:
        path = path + '.pkl'
        if not os.path.exists(path):
            path = os.path.basename(path)
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)

        model = CustomSAC(
            policy_classes=loaded_data["policy_classes"],
            replay_buffer_classes=loaded_data["replay_buffer_classes"],
            env=env,
            **kwargs
        )

        for i, optimizer_state in enumerate(loaded_data["ent_coef_optimizers"]):
            model.ent_coef_optimizers[i].load_state_dict(optimizer_state)

        for i, coef in enumerate(loaded_data["log_ent_coefs"]):
            old_coef = model.log_ent_coefs[i]
            old_coef.data = th.tensor(coef, dtype=old_coef.dtype, device=old_coef.device)

        for i, policy_state in enumerate(loaded_data["policies"]):
            model.policies[i].load_state_dict(policy_state)

        if "replay_buffers" in loaded_data:
            model.replay_buffers = loaded_data["replay_buffers"]

        return model

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        return self.get_actions(observation, deterministic), None

    def get_actions(self, obs, deterministic):
        selected_policies = self.decide_policy(obs)
        envs = [m.env for m in self.env.envs]
        n_envs = len(envs)

        policy_indices = np.argmax(selected_policies, axis=0)
        states = [[] for _ in range(self.policy_number)]

        for policy_index, env_index in enumerate(policy_indices):
            states[env_index].append(self.policies[env_index].translator.build_state(obs[policy_index], envs[policy_index]))

        actions = np.empty((n_envs,) + self.policies[0].action_space.shape, dtype=self.policies[0].action_space.dtype)

        for policy_index in range(self.policy_number):
            if states[policy_index]:
                policy_actions, _ = self.policies[policy_index].predict(np.array(states[policy_index]), deterministic=deterministic)
                actions = policy_actions

        return actions

    def decide_policy(self, new_obs):
        decider = [lambda x, y: 1]
        assignments = np.array([
            [func(obs, 0) for obs in new_obs]
            for func in decider
        ])
        return softmax_and_select(assignments)
