import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.misc_helper import find_observation_index, find_index_and_dict
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy
import gymnasium as gym

from examples.reinforcement_learning.General.override_sb3.sequence_policy import SequenceExtractor
from examples.reinforcement_learning.General.reward_functions import get_state_values
import torch.nn.functional as F

class ConvExtractor(SequenceExtractor):
    def __init__(self, observation_space: gym.spaces.Box, translator, num_filters=12, num_heads=0, dropout=0.0):
        super().__init__(observation_space, translator)

        self.num_heads = num_heads

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(self.input_features, num_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)

        if num_heads > 0:
            # Multi-head self-attention
            self.self_attn = nn.MultiheadAttention(num_filters, num_heads, dropout=dropout)
        else:
            self.self_attn = None

        # Feature combination layers
        self.fc1 = nn.Linear(num_filters * self.timesteps, 256)
        self.fc2 = nn.Linear(256, self.output_dim)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_filters) if num_heads > 0 else None

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.size(0)
        # Reshape the input tensor to have the shape (batch_size, timesteps, input_features)
        x = obs.view(batch_size, self.timesteps, self.input_features)
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)

        # Apply 1D convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.self_attn is not None:
            # Apply self-attention
            x = x.transpose(1, 2)  # (batch_size, sequence_length, num_filters)
            x = self.layer_norm(x) if self.layer_norm is not None else x
            attn_output, _ = self.self_attn(x, x, x)
            x = x + attn_output
            x = x.transpose(1, 2)  # (batch_size, num_filters, sequence_length)

        # Combine features
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x


class ConvTranslator(DefaultTranslator):
    """
        SequenceTranslator is responsible for preparing and translating observations from an environment
        into a format suitable for input into a neural network model. It handles the conversion of sequences
        of observations and actions, along with additional features, into a structured state representation.

        Attributes:
            timesteps (int): the timesteps to start the sequence.
            feature_dim (int): Dimensionality of the feature vector at each timestep.
            output_dim (int): Dimensionality of the output feature vector.
            additional_features (int): Number of additional features to include in the state representation.
            net_arch (list): Architecture of the neural network, specifying the number of units in each layer.
    """
    def __init__(self):
        self.reset()
        self.timesteps = 8
        self.feature_dim = 3
        self.output_dim = 4
        self.additional_features = 4
        self.net_arch = [1024, 1024, 1024]

        super().__init__(self.timesteps * self.feature_dim + self.additional_features)

    def build_state(self, observation, env) -> np.ndarray:
        """
            Builds the state representation from the current observation and environment state.

            Args:
                observation (object): The current observation from the environment.
                env (object): The environment instance providing the observation.

            Returns:
                np.ndarray: A flattened array containing the processed state representation.
        """

        index, observation_dict = find_index_and_dict(observation, env)
        clean_action = observation_dict['U_con'][index]
        dirty_observation = observation
        sequence_start = max(0, index + 1 - self.timesteps)

        if observation_dict:
            X_meas = np.array(observation_dict['X_meas'])
            U_con = np.array(observation_dict['U_con'])
            conv_memory = np.hstack((
                X_meas[sequence_start:index + 1, :-2],
                U_con[sequence_start:index + 1, np.newaxis]
            ))
        else:
            conv_memory = np.append(dirty_observation[:self.feature_dim - 1], clean_action).reshape(1, -1)

        if index < 0:
            print("This should not happen :(")

        output = conv_memory
        if output.shape[0] < self.timesteps:
            padding = np.zeros((self.timesteps - output.shape[0], output.shape[1]))
            output = np.vstack((padding, output))

        output = output.flatten()
        output = np.append(dirty_observation, output)

        state_values = get_state_values(observation_dict, offset=index + 1 - len(observation_dict['T']))
        # l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([
            # state_values['x2'][1] / l_ges,
            # state_values['v2'][0] / env.dynamics_func.max_velocity,
            # state_values['c1'],
            # state_values['c2']
        ])

        return np.append(additional, output)


class ConvPolicy(CustomPolicy):
    """
        A custom Soft Actor-Critic (SAC) policy class that utilizes a sequence-based feature extractor (LSTM) for handling temporal dependencies in observations.

        This policy class extends the CustomPolicy class and integrates a SequenceTranslator for handling sequences of observations.
        It is specifically designed for environments where the temporal aspect of the data is crucial, such as in reinforcement learning
        tasks involving dynamic systems like robotics or control systems.

        Attributes:
        -----------
        translator : SequenceTranslator
            An instance of the SequenceTranslator class used for translating observations.

        additional_actor_kwargs : dict
            Additional keyword arguments for configuring the actor network, including its architecture.

        additional_critic_kwargs : dict
            Additional keyword arguments for configuring the critic network, including its architecture.

    """
    @classmethod
    def get_translator(cls) -> ConvTranslator:
        """
            A class method that returns an instance of the SequenceTranslator class. This translator is responsible for handling
            the preprocessing and translation of observations into a format suitable for the LSTM feature extractor.
        """
        return ConvTranslator()

    def __init__(self, *args, **kwargs):
        """
            Initializes the SequenceSACPolicy.

            Sets up the SequenceTranslator for the policy, configures the network architecture for the actor and critic networks,
            and initializes the feature extractor with an LSTM-based extractor.

            Parameters:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
        """
        self.translator = self.get_translator()
        self.additional_actor_kwargs['net_arch'] = self.translator.net_arch
        self.additional_critic_kwargs['net_arch'] = self.translator.net_arch

        kwargs.update(
            dict(
                features_extractor_class=ConvExtractor,
                features_extractor_kwargs=dict(translator=self.translator),
                share_features_extractor=False,
                # optimizer_kwargs={'weight_decay': 0.00001}
            )
        )

        super().__init__(*args, **kwargs)

    def after_critic_backward(self):
        pass
        # th.nn.utils.clip_grad_norm_(self.critic.parameters(), 25)
