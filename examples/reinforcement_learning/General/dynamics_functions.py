import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters
from scipy.stats import norm

from src.python.double_pendulum.utils.wrap_angles import wrap_angles_diff


def load_param(torque_limit, simplify=True):
    design = "design_C.1"
    model = "model_1.0"
    torque_array = [torque_limit, torque_limit]

    model_par_path = (
            "../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    if simplify:
        mpar.set_motor_inertia(0.0)
        mpar.set_damping([0.0, 0.0])
        mpar.set_cfric([0.0, 0.0])

    return mpar


def default_dynamics(robot, dt, max_torque, class_obj):
    mpar = load_param(max_torque)
    plant = SymbolicDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque, class_obj)


def general_dynamics(robot, plant, dt, max_torque, class_obj):
    print("build new plant")
    simulator = Simulator(plant=plant)
    max_vel = 20.0
    if robot == "acrobot":
        max_vel = 50.0

    dynamics_function = class_obj(
        simulator=simulator,
        robot=robot,
        dt=dt,
        integrator="runge_kutta",
        max_velocity=max_vel,
        torque_limit=[max_torque, max_torque],
        scaling=True
    )
    return dynamics_function, simulator, plant


class custom_dynamics_func_4PI(double_pendulum_dynamics_func):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_angle = 2 * np.pi

    def unscale_action(self, action):
        if len(action) > 1:
            a = [
                float(self.torque_limit[0] * action[0]),
                float(self.torque_limit[1] * action[1]),
            ]
        elif self.robot == "pendubot":
            a = np.array([float(self.torque_limit[0] * action[0]), 0.0])
        elif self.robot == "acrobot":
            a = np.array([0.0, float(self.torque_limit[1] * action[0])])
        return a

    def unscale_state(self, observation):
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * self.max_angle + np.pi,
                    observation[1] * self.max_angle,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = super().unscale_state(observation)
        return x

    def normalize_state(self, state):
        if self.state_representation == 2:
            observation = np.array(
                [
                    ((state[0] + np.pi) % (2 * self.max_angle) - self.max_angle) / self.max_angle,
                    ((state[1] + self.max_angle) % (2 * self.max_angle) - self.max_angle) / self.max_angle,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = super().normalize_state(state)
        return observation


class custom_dynamics_func_PI(double_pendulum_dynamics_func):
    virtual_sensor_state = [0.0, 0.0]
    def __call__(self, state, action, scaling=True):
        if scaling:
            x = self.unscale_state(state)
            u = self.unscale_action(action)
            xn = self.integration(x, u)
            self.virtual_sensor_state = wrap_angles_diff(xn - x)[:2]
            obs = self.normalize_state(xn)
            return np.array(obs, dtype=np.float32)
        else:
            super().__call__(state, action, scaling)

    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        else:
            x = super().unscale_state(observation)
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
            observation = np.array(
                [
                    (state[0] % (2 * np.pi) - np.pi) / np.pi,
                    ((state[1] - np.pi) % (2 * np.pi) - np.pi) / np.pi,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        else:
            observation = super().normalize_state(state)
        return observation
