import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters
from scipy.stats import norm

from src.python.double_pendulum.utils.wrap_angles import wrap_angles_diff


class PushDoublePendulum(SymbolicDoublePendulum):
    state_dict = None

    def __init__(self, model_pars):
        super().__init__(model_pars=model_pars)

    def normalize_state(self, state):
        observation = np.array(
            [
                ((state[0] + np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),
                ((state[1] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),

            ]
        )
        return observation

    def forward_dynamics(self, x, u):
        accn = super().forward_dynamics(x, u)
        push = self.check_push()

        if push:
            f = np.array(self.state_dict["current_force"])
            angle = wrap_angles_diff(x)
            l_00 = np.sin(angle[0]) * self.l[0]
            l_01 = np.cos(angle[0]) * self.l[0]
            l_10 = np.sin(angle[1]) * self.l[1]
            l_11 = np.cos(angle[1]) * self.l[1]
            force = np.array([(l_00 + l_10) * f[0] + (l_01 + l_11) * f[1], l_10 * f[0] + l_11 * f[1]])
            accn += np.linalg.inv(self.mass_matrix(x)).dot(force)

        return accn

    def check_push(self, start_time=4, sigma_start=0.5, end_time=0.5, sigma_end=0.1, force=[5,25]):
        def random_force():
            angle = np.random.uniform(0, 2 * np.pi)
            x = np.cos(angle)
            y = np.sin(angle)
            return np.array([x, y]) * np.sqrt(np.random.uniform(force[0], force[1]))

        push_list = self.state_dict["push"]
        if len(push_list) > len(self.state_dict["T"]):
            return push_list[-1]

        push_value = False
        if len(push_list) > 0:
            consecutive_falses = 0
            consecutive_trues = 0

            for value in reversed(push_list):
                if not value:
                    consecutive_falses += 1
                else:
                    break

            for value in reversed(push_list):
                if value:
                    consecutive_trues += 1
                else:
                    break

            false_time = self.state_dict["T"][-1] - self.state_dict["T"][-consecutive_falses]
            true_time = self.state_dict["T"][-1] - self.state_dict["T"][-consecutive_trues]
            if consecutive_falses == 0:
                false_time = 0
            if consecutive_trues == 0:
                true_time = 0

            start_push_probability = norm.cdf(false_time, loc=start_time, scale=sigma_start)
            end_push_probability = norm.cdf(true_time, loc=end_time, scale=sigma_end)

            if np.random.rand() < start_push_probability:
                self.state_dict["current_force"] = random_force().tolist()
                push_value = True

            elif np.random.rand() < end_push_probability:
                push_value = False

            else:
                push_value = push_list[-1]

        push_list.append(push_value)
        return push_value


def load_param(robot, torque_limit, simplify=True):
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
    if simplify:
        mpar.set_motor_inertia(0.0)
        mpar.set_damping([0., 0.])
        mpar.set_cfric([0., 0.])

    return mpar


def random_dynamics(robot, dt, max_torque, class_obj, sigma=0.02, plant_class=SymbolicDoublePendulum, use_random=True):
    mpar = load_param(robot, max_torque, simplify=False)
    if use_random:
        mpar.g = np.random.normal(mpar.g, sigma * mpar.g)
        mpar.m = np.random.normal(mpar.m, sigma * np.array(mpar.m)).tolist()
        mpar.l = np.random.normal(mpar.l, sigma * np.array(mpar.l)).tolist()
        mpar.cf = abs(np.random.normal(mpar.cf, sigma * np.array(mpar.cf))).tolist()
        mpar.b = abs(np.random.normal(mpar.b, sigma * np.array(mpar.b))).tolist()
    plant = plant_class(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque, class_obj)


def push_dynamics(robot, dt, max_torque, class_obj):
    mpar = load_param(robot, max_torque)
    plant = PushDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant, dt, max_torque, class_obj)


def random_push_dynamics(robot, dt, max_torque, class_obj, sigma=0.02):
    return random_dynamics(robot, dt, max_torque, class_obj, sigma, PushDoublePendulum)


def default_dynamics(robot, dt, max_torque, class_obj):
    mpar = load_param(robot, max_torque)
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
            x = super().unscale_state(observation)
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
            observation = super().normalize_state(state)
        if len(state) > 4:
            return np.append(observation, state[-2:]/self.torque_limit)
        return observation

class custom_dynamics_func_PI(double_pendulum_dynamics_func):
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
        if len(observation) > 4:
            return np.append(x, observation[-2:]*self.torque_limit)
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

        if len(state) > 4:
            return np.append(observation, state[-2:]/self.torque_limit)
        return observation
