import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th
from examples.reinforcement_learning.General.misc_helper import find_observation_index, find_index_and_dict
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, CustomPolicy
import gymnasium as gym

from examples.reinforcement_learning.General.reward_functions import get_state_values


class SmoothingFilter(nn.Module):
    """
        A smoothing filter for processing sequences of features using a 1D convolution.
        The filter applies a learnable smoothing operation to the input features.

        Attributes:
            num_features (int): Number of input features/channels.
            conv (nn.Conv1d): 1D convolutional layer that applies the smoothing operation.
            alpha (nn.Parameter): A learnable parameter that adjusts the effect of the smoothing.
            activation (nn.Module): An activation function applied to the alpha parameter.
    """
    def __init__(self, num_features, kernel_size=11, padding='same'):
        """
            Initializes the SmoothingFilter.

            Args:
                num_features (int): Number of input features/channels.
                kernel_size (int, optional): Size of the convolution kernel. Defaults to 11.
                padding (str or int, optional): Padding applied to the input sequence.
                    'same' padding maintains the input length. Defaults to 'same'.
        """
        super(SmoothingFilter, self).__init__()
        self.num_features = num_features

        # The `groups=num_features` means each input channel is convolved with its own filter
        self.conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_features,
            bias=False
        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

        # A learnable parameter to adjust the smoothing effect
        self.alpha = nn.Parameter(th.zeros(num_features))

        # Sigmoid activation to ensure alpha stays within [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x):
        batch_size, sequence_length, num_features = x.size()
        x_t = x.transpose(1, 2)  # Transpose to (batch_size, num_features, sequence_length)
        smoothed = self.conv(x_t)
        smoothed = smoothed.transpose(1, 2)  # Transpose back to (batch_size, sequence_length, num_features)

        alpha_sigmoid = self.activation(self.alpha).view(1, 1, -1)
        alpha_sigmoid = alpha_sigmoid.expand(batch_size, sequence_length, num_features)  # Expand to match the input dimensions
        return alpha_sigmoid * x + (1 - alpha_sigmoid) * smoothed


class SequenceExtractor(BaseFeaturesExtractor):
    """
        A feature extractor that processes sequences of observations. It separates additional features
        from the main feature set, processes them separately, and then combines them.

        Attributes:
            input_features (int): The dimensionality of the input features.
            timesteps (int): The timesteps in the sequence.
            output_dim (int): The output dimensionality after processing the features.
            additional_features (int): Number of additional features in the input.
    """
    def __init__(self, observation_space: gym.spaces.Box, translator):
        """
            Initializes the SequenceExtractor with the observation space and translator.

            Args:
                observation_space (gym.spaces.Box): The observation space of the environment.
                translator: An object that provides feature dimensionality and other configurations.
        """
        super().__init__(observation_space, translator.output_dim + translator.additional_features)
        self.input_features = translator.feature_dim
        self.timesteps = translator.timesteps
        self.output_dim = translator.output_dim
        self.additional_features = translator.additional_features

    def _process_additional_features(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
            Separates additional features from the main observation data.

            Args:
                obs (th.Tensor): The input observation tensor.

            Returns:
                tuple: A tuple containing:
                    - Additional features tensor if present.
                    - The remaining observation tensor.
        """
        if self.additional_features > 0:
            obs_1 = obs[:, :self.additional_features]
            obs_2 = obs[:, self.additional_features:]
            return obs_1, obs_2
        return None, obs

    def _combine_output(self, obs_1: th.Tensor, processed_output: th.Tensor) -> th.Tensor:
        """
            Combines the additional features with the processed main features.

            Args:
                obs_1 (th.Tensor): Additional features tensor.
                processed_output (th.Tensor): Processed main features tensor.

            Returns:
                th.Tensor: The combined tensor.
        """
        if obs_1 is not None:
            return th.cat((obs_1, processed_output), dim=1)
        return processed_output

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs_1, obs_2 = self._process_additional_features(obs)
        processed_output = self._process_main_features(obs_2)
        return self._combine_output(obs_1, processed_output)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class LSTMExtractor(SequenceExtractor):
    """
        A feature extractor that processes sequences using an LSTM network. It includes smoothing of input features,
        LSTM-based feature extraction, and a fully connected layer for output transformation.

        Attributes:
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers in the LSTM.
            lstm (nn.LSTM): The LSTM network for processing sequences.
            fc (nn.Linear): A fully connected layer for transforming the LSTM output.
            activation (nn.Tanh): An activation function applied to the output of the fully connected layer.
            smoothing (SmoothingFilter): A smoothing filter applied to the input features.
    """
    def __init__(self, observation_space: gym.spaces.Box, translator, hidden_size=32, num_layers=2, dropout=0.0):
        """
            Initializes the LSTMExtractor with specified parameters.

            Args:
                observation_space (gym.spaces.Box): The observation space of the environment.
                translator: An object providing feature dimensionality and other configurations.
                hidden_size (int, optional): Number of features in the LSTM hidden state. Defaults to 32.
                num_layers (int, optional): Number of layers in the LSTM. Defaults to 2.
                dropout (float, optional): Dropout probability for the LSTM. Defaults to 0.0.
        """
        super().__init__(observation_space, translator)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM network for processing the sequence of input features
        self.lstm = nn.LSTM(self.input_features, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer to produce the final output
        self.fc = nn.Linear(hidden_size * 2, self.output_dim)

        # Tanh activation function applied to the output of the fully connected layer
        self.activation = nn.Tanh()

        # Smoothing filter applied to the input features before feeding into the LSTM
        self.smoothing = SmoothingFilter(self.input_features)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        """
            Processes the main features using an LSTM and additional layers.

            Args:
                obs (th.Tensor): The input observation tensor.

            Returns:
                th.Tensor: The processed feature tensor.
        """
        batch_size = obs.size(0)

        # Reshape the input tensor to have the shape (batch_size, timesteps, input_features)
        obs_reshaped = obs.view(batch_size, self.timesteps, self.input_features)
        obs_smoothed = self.smoothing(obs_reshaped)

        lstm_out, (h_n, c_n) = self.lstm(obs_smoothed)

        last_hidden = h_n[-1]
        last_cell = c_n[-1]

        concatenated = th.cat((last_hidden, last_cell), dim=1)

        fc_output = self.fc(concatenated)
        return self.activation(fc_output)


class SequenceTranslator(DefaultTranslator):
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
        self.timesteps = 256
        self.feature_dim = 5
        self.output_dim = 16
        self.additional_features = 8
        self.net_arch = [128, 64, 32]

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
                X_meas[sequence_start:index + 1, :self.feature_dim - 1],
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
        l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([
            state_values['x3'][1] / l_ges,
            state_values['v2'][0] / env.dynamics_func.max_velocity,
            state_values['c1'],
            state_values['c2']
        ])

        return np.append(additional, output)


class SequenceSACPolicy(CustomPolicy):
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
    def get_translator(cls) -> SequenceTranslator:
        """
            A class method that returns an instance of the SequenceTranslator class. This translator is responsible for handling
            the preprocessing and translation of observations into a format suitable for the LSTM feature extractor.
        """
        return SequenceTranslator()

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
                features_extractor_class=LSTMExtractor,
                features_extractor_kwargs=dict(translator=self.translator),
                share_features_extractor=False,
                #optimizer_kwargs={'weight_decay': 0.000001}
            )
        )

        super().__init__(*args, **kwargs)

    def after_critic_backward(self):
        pass
        # th.nn.utils.clip_grad_norm_(self.critic.parameters(), 25)
