from custom_sac import CustomSAC
from double_pendulum.controller.abstract_controller import AbstractController
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from environments import GeneralEnv
import numpy as np

class GeneralController(AbstractController):
    def __init__(self, dynamics_func, model_path, torque_limit):
        super().__init__()

        envs = make_vec_env(
            env_id=GeneralEnv,
            n_envs=20,
            env_kwargs={
                "dynamic_function": dynamics_func,
                "reward_function": None,
                "termination_function": None,
                "reset_function": lambda: [0, 0, 0, 0],
                "torque_limit": torque_limit
            },
            vec_env_cls=DummyVecEnv
        )

        self.model = CustomSAC.load(
            model_path,
            env=envs,
            print_system_info=True
        )

        self.dynamics_func = dynamics_func
        self.simulator = self.dynamics_func.simulator
        self.dt = self.dynamics_func.dt
        self.scaling = dynamics_func.scaling
        self.integrator = dynamics_func.integrator
        self.controller_dt = np.rint(self.dt * 10000).astype(int)
        self.observation_dict = None
        self.last_action = None
        self.n = None
        self.last_u = None
        self.reset()

    def reset(self):
        super().reset()
        self.observation_dict = {'X': [], 'U': []}
        self.last_action = 0.0
        self.n = 1
        self.last_u = None
        self.model.env.envs[0].env.reset()

    def get_control_output_(self, x, t=None):
        if self.n == 1 and t > 0:
            self.n = np.rint(self.dt / np.round(t, decimals=5)).astype(int)

        env = self.model.env.envs[0].env
        obs = self.dynamics_func.normalize_state(x)
        rounded_t = np.rint(t * 10000).astype(int)
        if rounded_t % self.controller_dt == 0 and t > 0.0:
            env.observation_dict['T'].append(np.round(t, decimals=5))
        self.observation_dict['X'].append(obs)
        self.observation_dict['U'].append(self.last_action)

        env.observation_dict['U_con'] = self.observation_dict['U'][::-1][::self.n][::-1].copy()
        env.observation_dict['X_meas'] = self.observation_dict['X'][::-1][::self.n][::-1].copy()
        action, _ = self.model.predict(observation=obs.reshape(1, -1), deterministic=True)
        self.last_action = action.item()

        return self.dynamics_func.unscale_action(action)
