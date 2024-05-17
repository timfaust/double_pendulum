from double_pendulum.controller.abstract_controller import AbstractController
import numpy as np
from sbx.sac.sac import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
import pygame

class GeneralController(AbstractController):
    def __init__(self, environment, model: SAC, robot, eval_env=None, callbacks=None, fine_tune=False, train_freq=100):
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
        self.train_every_n_step = train_freq
        self.actions = [0]
        self.buffer_actions = [0]
        self.action = 0
        self.rewards = []
        self.eval_env = eval_env
        self.window_size = 800

        if self.fine_tune:
            total_timesteps, callback = self.model._setup_learn(
                1,
                callbacks,
                True,
                "run",
                False,
            )
            self.model.replay_buffer.reset()

            pygame.init()

            if not self.callbacks is None:
                self.callbacks.on_training_start(locals(), globals())

    def reset(self):
        self.simulation.set_state(0, [0, 0, 0, 0])
        self.simulation.reset()
        self.actions = [0]
        self.action = np.array([0])
        self.buffer_actions = [0]
        self.last_actions = [0, 0]
        self.steps = 0
        self.rewards.clear()
        if self.fine_tune:
            self.eval_env.reset()
            self.model.env.reset()

    def get_control_output_(self, x, t=None):
        if self.actions_in_state:
            x = np.append(x, self.last_actions)

        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            eval_obs = obs

            if self.fine_tune:
                if self.steps > 0:
                    envs = self.model.get_env().envs

                    for env in envs:
                        env.update_observation(obs)
                    if self.eval_env is not None:
                        eval_obs = self.eval_env.calculate_obs(self.action)

                    self.finish_last_rollaout(actions=self.actions, buffer_actions=self.buffer_actions,
                                            env=self.model.get_env(),
                                            action_noise=self.model.action_noise,
                                            callback=self.callbacks,
                                            learning_starts=self.model.learning_starts,
                                            replay_buffer=self.model.replay_buffer,
                                            log_interval=4)

                    if self.steps % (self.train_every_n_step) == 0:
                        self.model.train(self.model.gradient_steps, self.model.batch_size)

                self.collect_rollout(env=self.model.env,
                                                train_freq=self.model.train_freq,
                                                callback=self.callbacks)

                action, _ = self.model.predict(obs, deterministic=False)
                self.action = self.adjust_action(obs, eval_obs)

                self.eval_env.render()
                self.model.env.render()

                self.actions, self.buffer_actions = self._sample_action(action=action, n_envs=self.model.env.num_envs)

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

    def finish_training(self):
        self.model.train(self.model.gradient_steps, self.model.batch_size)
        mean_reward = np.mean(self.rewards)
        print("mean reward:", mean_reward)
        self.rewards.clear()
        print("n_steps:", self.steps)
        return mean_reward
    def adjust_action(self, obs, eval_obs):
        """
        diff = eval_obs[:4] - obs[:4]
        print("obs", obs[:2])
        print("eval_obs", eval_obs[:2])
        print("diff_x", diff[:2])

        K_p = np.array([.1, 0.1])
        K_v = np.array([0.0, 0.0])

        return np.clip(K_p @ diff[:2].T + K_v @ diff[-2:].T, -1, 1)
        """
        action, _ = self.model.predict(eval_obs, deterministic=True)
        return action

    def _sample_action(
        self,
        action,
        n_envs,
    ):
        buffer_actions = np.array([action for _ in range(n_envs)])
        actions = buffer_actions
        return actions, buffer_actions
    def collect_rollout(self,
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


    def finish_last_rollaout(self, actions, buffer_actions, env, callback,
                            replay_buffer,
                            action_noise,
                            learning_starts: int = 0,
                            log_interval= None):

        new_obs, rewards, dones, infos = env.step(actions)
        self.rewards.append(np.mean(rewards))
        self.model.num_timesteps += env.num_envs

        # Give access to local variables
        if not callback is None:
            callback.update_locals(locals())
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
