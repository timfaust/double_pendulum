import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters

class PushDoublePendulum(SymbolicDoublePendulum):
    def __init__(self, model_pars):
        super().__init__(model_pars=model_pars)

    @staticmethod
    def normalize_angle(angle):
        normalized_angle = (angle / np.pi) - 1
        while normalized_angle < -1:
            normalized_angle += 2
        return normalized_angle

    def forward_dynamics(self, x, u):
        accn = super().forward_dynamics(x, u)

        angle_1_norm = self.normalize_angle(x[0])
        angle_2_norm = self.normalize_angle(x[1])

        if np.all(np.abs([angle_1_norm, angle_2_norm]) < 0.1):
            if np.random.uniform(0, 1) < 1.0 / 200.0:
                f = np.random.uniform(2, 10) * np.random.choice([-1, 1])
                force = np.array([(self.l[0] + self.l[1]) * f, self.l[1] * f])
                accn += np.linalg.inv(self.mass_matrix(x)).dot(force)

        return accn


def load_param(robot, torque_limit):
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"
        torque_array = [torque_limit, 0.0]

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"
        torque_array = [0.0, torque_limit]

    model_par_path = (
            "../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0., 0.])
    mpar.set_cfric([0., 0.])

    return mpar


def random_dynamics(robot, dt, max_torque, sigma=0.02, plant_class=SymbolicDoublePendulum, use_random=True):
    mpar = load_param(robot, max_torque)
    if use_random:
        mpar.g = np.random.normal(mpar.g, sigma)
        mpar.m = np.random.normal(mpar.m, sigma)
        mpar.l = np.random.normal(mpar.l, sigma).tolist()
        mpar.cf = abs(np.random.normal(mpar.cf, sigma)).tolist()
    plant = plant_class(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque)


def push_dynamics(robot, dt, max_torque):
    mpar = load_param(robot, max_torque)
    plant = PushDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque)


def random_push_dynamics(robot, dt, max_torque, sigma=0.02):
    return random_dynamics(robot, dt, max_torque, sigma, PushDoublePendulum)


def default_dynamics(robot, dt, max_torque):
    mpar = load_param(robot, max_torque)
    plant = SymbolicDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque)


def general_dynamics(robot, plant, dt, max_torque):
    print("build new plant")
    simulator = Simulator(plant=plant)
    dynamics_function = custom_double_pendulum_dynamics_func(
        simulator=simulator,
        robot=robot,
        dt=dt,
        integrator="runge_kutta",
        max_velocity=20.0,
        torque_limit=[max_torque, max_torque],
        scaling=True
    )
    return dynamics_function, simulator, plant


class custom_double_pendulum_dynamics_func(double_pendulum_dynamics_func):

    def unscale_state(self, observation):
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * 2 * np.pi + np.pi,
                    observation[1] * 2 * np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = np.array(
                [
                    np.arctan2(observation[0], observation[1]),
                    np.arctan2(observation[2], observation[3]),
                    observation[4] * self.max_velocity,
                    observation[5] * self.max_velocity,
                ]
            )
        if len(observation) > 4:
            return np.append(x, observation[-2:]*self.torque_limit)
        return x

    def normalize_state(self, state):
        if self.state_representation == 2:
            observation = np.array(
                [
                    ((state[0] + np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),
                    ((state[1] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = np.array(
                [
                    np.cos(state[0]),
                    np.sin(state[0]),
                    np.cos(state[1]),
                    np.sin(state[1]),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        if len(state) > 4:
            return np.append(observation, state[-2:]/self.torque_limit)
        return observation