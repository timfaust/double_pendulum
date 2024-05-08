from double_pendulum.controller.abstract_controller import AbstractController
import numpy as np
from sbx.sac.sac import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from gymnasium import spaces
from torch import Type


class GeneralController(AbstractController):
    def __init__(self, environment, model: SAC, robot, callbacks=None, fine_tune=False):
        super().__init__()

        self.model = model
        self.robot = robot
        self.callbacks = callbacks
        self.environment = environment
        self.fine_tune = fine_tune
        self.simulation = environment.dynamics_func.simulator
        self.dynamics_func = environment.dynamics_func
        self.dt = environment.dynamics_func.dt
        self.scaling = environment.dynamics_func.scaling
        self.integrator = environment.dynamics_func.integrator
        self.actions_in_state = environment.actions_in_state
        self.last_actions = [0, 0]
        self.steps = 0
        self.actions = [0]
        self.buffer_actions = [0]

        if self.fine_tune:
            total_timesteps, callback = self.model._setup_learn(
                1,
                None,
                True,
                "run",
                False,
            )

    def get_control_output_(self, x, t=None):

        if self.actions_in_state:
            x = np.append(x, self.last_actions)

        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            if self.fine_tune:
                if self.steps > 0:
                    self.environment.update_observation(obs)
                    self.finish_last_rollaout(actions=self.buffer_actions, buffer_actions=self.buffer_actions,
                                            env=self.model.get_env(),
                                            action_noise=self.model.action_noise,
                                            callback=None,
                                            learning_starts=self.model.learning_starts,
                                            replay_buffer=self.model.replay_buffer,
                                            log_interval=4)
                    self.model.train(self.model.gradient_steps, self.model.batch_size)

                action, buffer_action = self.collect_rollout(env=self.model.env, obs=obs,
                                                train_freq=self.model.train_freq,
                                                callback=None)

                self.actions = action
                self.buffer_actions = buffer_action
                self.steps += 1
            else:
                action = self.model.predict(observation=obs, deterministic=True)

            u = self.dynamics_func.unscale_action(action)
        else:
            action = self.model.predict(observation=x, deterministic=True)
            u = self.dynamics_func.unscale_action(action)

        if self.actions_in_state:
            self.last_actions[-1] = self.last_actions[-2]
            if not np.all(u == 0):
                max_abs_value_index = np.abs(u).argmax()
                self.last_actions[-2] = u[max_abs_value_index]
            else:
                self.last_actions[-2] = 0
        return u

    def collect_rollout(self, obs,
                        env,
                        callback,
                        train_freq : TrainFreq):

        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.model.use_sde:
            self.model.actor.reset_noise(env.num_envs)

        if not callback is None:
            callback.on_rollout_start()

        if self.model.use_sde and self.model.sde_sample_freq > 0:
            # Sample a new noise matrix
            self.model.actor.reset_noise(env.num_envs)

        action, buffer_action = self._sample_action(obs, 0, self.model.action_noise, self.environment.n_envs)

        return action, buffer_action

    def finish_last_rollaout(self, actions, buffer_actions, env, callback,
                            replay_buffer,
                            action_noise,
                            learning_starts: int = 0,
                            log_interval= None):

        new_obs, rewards, dones, infos = env.step(actions)

        self.model.num_timesteps += env.num_envs

        # Give access to local variables
        if not callback is None:
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return

        # Retrieve reward and episode length if using Monitor wrapper
        self.model._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self.model._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

        self.model._update_current_progress_remaining(self.model.num_timesteps, self.model._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self.model._on_step()

        for idx, done in enumerate(dones):
            if done:
                self.model._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and self.model._episode_num % log_interval == 0:
                    self.model._dump_logs()
        if not callback is None:
            callback.on_rollout_end()

    def _sample_action(
        self,
        obs,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ):
        unscaled_action, _ = self.model.predict(obs, deterministic=True)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.model.action_space, spaces.Box):
            scaled_action = self.model.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.model.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
