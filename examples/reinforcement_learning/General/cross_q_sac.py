from sbx.sac.sac import SAC
import flax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from sbx.common.type_aliases import RLTrainState

class CROSSQ_SAC(SAC):

    @staticmethod
    @jax.jit
    def update_critic(
        gamma: float,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        key: jax.Array,
    ):
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)
        # sample action from the actor
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            # ----- CrossQ's One Weird Trick™ -----
            # concatenate current and next observations to double the batch size
            # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
            # apply critic to this bigger batch
            catted_q_values, state_updates = qf_state.apply_fn(
                params,
                jnp.concatenate([observations, next_observations], axis=0),
                jnp.concatenate([actions, next_state_actions], axis=0),
                rngs={"dropout": dropout_key}
            )
            current_q_values, next_q_values = jnp.split(catted_q_values, 2)

            next_q_values = jnp.min(next_q_values, axis=0)
            next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
            target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values  # shape is (batch_size, 1)

            loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()

            return loss


        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
        )
