from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import BaseJaxPolicy
from sbx.common.type_aliases import RLTrainState, ActorTrainState

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]

class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    bn_mode: str = "bn"

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:

        BN = nn.BatchNorm

        x = jnp.concatenate([x, action], -1)

        if self.use_batch_norm:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation_fn()(x)

            if self.use_batch_norm:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    use_batch_norm: bool = False
    batch_norm_mode: str = "bn"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, train)
        return q_values


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    batch_norm_momentum: float
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = False
    bn_mode: str = "bn"

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    # type: ignore[name-defined]
    def __call__(self, x: jnp.ndarray, train) -> tfd.Distribution:

        BN = nn.BatchNorm

        if self.use_batch_norm:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
            if self.use_batch_norm:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        activation_fn: Type[nn.Module],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = False,
        batch_norm_momentum: float = 0.9,
        batch_norm_mode: str = "bn",
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[...,
                                  optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        td3_mode: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_mode = batch_norm_mode
        self.activation_fn = activation_fn
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

        if td3_mode:
            self._predict = self._predict_deterministic

    def build(self, key: jax.random.KeyArray, lr_schedule: Schedule, qf_learning_rate: float) -> jax.random.KeyArray:
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # params=self.actor.init(actor_key, obs)
        actor_init_variables = self.actor.init(
            {"params": actor_key, "batch_stats": bn_key},
            obs,
            train=False
        )
        self.actor_state = ActorTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_mode=self.batch_norm_mode,
            net_arch=self.net_arch_qf,
            activation_fn=self.activation_fn,
            n_critics=self.n_critics,
        )

        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.actor.apply = jax.jit(  # type: ignore[method-assign]
            self.actor.apply,
            static_argnames=("use_batch_norm", "batch_norm_momentum", "bn_mode")
        )
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    # type: ignore[override]
    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)
    
    def _predict_deterministic(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return BaseJaxPolicy.select_action(self.actor_state, observation)

    def predict_action_with_logprobs(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation, True)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
            
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key, True)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a, 
                rngs={"dropout": dropout_key},
                train=False
            ) 
        
        return jax.jit(Q)(
            self.qf_state.params, 
            self.qf_state.batch_stats, 
            observation, 
            action,
            self.noise_key,
        )

