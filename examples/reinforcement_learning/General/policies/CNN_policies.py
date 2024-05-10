from typing import List, Tuple, Dict

import numpy as np
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn
import torch as th
from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.policies.common import DefaultTranslator, DefaultActor, CustomPolicy, \
    DefaultCritic

#TODO: Input dimension must be: batch_size, input features, sequence length for 1d conv

class CNNModule(nn.Module):
    def __init__(self, translator):
        super().__init__()
        self.input_features = translator.observation_dim
        self.timesteps = translator.timesteps
        self.observation_dim = translator.observation_dim

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=5,# #self.input_features, #TODO: parametrize
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2

            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_mapper = nn.Linear(32 * self.timesteps, translator.observation_dim)
        print(self.cnn_layers)
    def forward(self, obs: PyTorchObs) -> th.Tensor:
        input = obs
        obs_reshaped = obs.view(-1, self.input_features, self.timesteps)  # Verify dimensions order based on your data


        #cnn_output = self.cnn_layers(obs_reshaped)
        print("Conv1d Layers:")
        cnn_output_layer1 = self.cnn_layers(obs_reshaped)[0]
        print(cnn_output_layer1.shape)


        cnn_output_layer2 = self.cnn_layers(obs_reshaped)[2]


        mapped_features = self.feature_mapper(cnn_output)
        return mapped_features

class CNNTranslator(DefaultTranslator):
    def __init__(self):
        self.reset()
        self.timesteps = 30
        self.observation_dim = 5 #dont change
        self.cnn_output = 5 #TODO change
        self.cnn_input = 5 #TODO
        self.kernel_size = 3 #TODO
        self.padding = 0 #TODO
        self.stride = 1 #TODO

        input_dim = self.timesteps * self.observation_dim #must be 150
        super().__init__(input_dim)

    def build_state(self, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        if len(kwargs) > 0:
            lstm_memory = [np.append(x, u) for x, u in
                           zip(kwargs['X_meas'][-self.timesteps:], kwargs['U_con'][-self.timesteps:])]
            lstm_memory = np.array(lstm_memory)
        else:
            lstm_memory = np.array([np.append(dirty_observation.copy(), clean_action)])

        output = lstm_memory
        if output.shape[0] < self.timesteps:
            repeat_count = self.timesteps - output.shape[0]
            output = np.vstack((np.tile(output[0], (repeat_count, 1)), output))

        output = np.concatenate(output)
        return output
class CNNActor(DefaultActor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cnn_net = None

    @classmethod  #operate on the class itself rather than an instance of the class
    def get_translator(cls) -> CNNTranslator:
        return CNNTranslator()

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.cnn_net(obs)
        return super().get_action_dist_params(features)


class CNNCritic(DefaultCritic):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_net = None

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        features = self.cnn_net(obs)
        return super().forward(features, actions)

class CNN_SAC_Policy(CustomPolicy):
    actor_class = CNNActor
    critic_class = CNNCritic

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

        translator = self.actor_class.get_translator()
        #cnn_input_dim = self.translator.get_observation_dim
        #cnn_output_dim = self.translator.get_input_dim

        self.additional_actor_kwargs['stride'] = translator.stride
        self.additional_critic_kwargs['stride'] = translator.stride



        cnn_net = CNNModule(translator)
        self.actor.cnn_net = cnn_net
        self.critic.cnn_net = cnn_net
        self.critic_target.cnn_net = cnn_net


    @classmethod
    def after_rollout(cls, envs: List[GeneralEnv], progress, *args, **kwargs):
        pass

    @classmethod
    def modify_env(cls, environment: GeneralEnv, progress):
        if progress > 0.3:
            environment.start_delay = 0.01
            environment.delay = 0.02

        elif progress > 0.1:
            factor = (progress - 0.1)/0.2
            environment.start_delay = 0.1 * factor
            environment.delay = 0.02 * factor