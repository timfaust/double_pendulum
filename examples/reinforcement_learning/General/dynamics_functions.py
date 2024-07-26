import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters


def load_param(torque_limit, simplify=True):
    design = "design_C.1"
    model = "model_1.1"
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
    max_vel = 30.0
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
        self.max_angle = 3 * np.pi

    def unscale_action(self, action):
        if isinstance(action, (float, list)):
            action = np.array(action).reshape(-1, 1)
        elif isinstance(action, np.ndarray) and action.shape == (2,):
            return action

        a = np.zeros((2,) if action.ndim == 1 else (action.shape[0], 2))

        if self.robot == "pendubot":
            a[..., 0] = self.torque_limit[0] * action[..., 0]
        elif self.robot == "acrobot":
            a[..., 1] = self.torque_limit[1] * action[..., -1]
        else:
            a = np.multiply(self.torque_limit, action[..., :2])

        return a.squeeze()

    def unscale_state(self, observation):
        observation = np.asarray(observation)
        scale = np.array([self.max_angle, self.max_angle, self.max_velocity, self.max_velocity])
        return observation * scale

    def normalize_state(self, state):
        state = np.asarray(state)
        angles = ((state[..., :2] + self.max_angle) % (2 * self.max_angle) - self.max_angle) / self.max_angle
        velocities = np.clip(state[..., 2:], -self.max_velocity, self.max_velocity) / self.max_velocity
        return np.concatenate([angles, velocities], axis=-1)
