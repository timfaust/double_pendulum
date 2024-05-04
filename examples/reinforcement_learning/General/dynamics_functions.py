import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters
from scipy.stats import norm
from matplotlib.animation import PillowWriter
from matplotlib import animation
from sympy import lambdify
import pandas as pd

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

    def check_push(self, start_time=8, sigma_start=1, end_time=0.5, sigma_end=0.1, force=[5, 25]):
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


def load_param(robot, torque_limit, simplify=True, real_robot=True):
    if robot == "pendubot":
        design = "design_C.1"
        model = "model_1.0"
        torque_array = [torque_limit, 0.0]

    elif robot == "acrobot":
        design = "design_C.1"
        model = "model_1.0"
        torque_array = [0.0, torque_limit]

    if real_robot:
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
        mpar.set_damping([0., 0.])
        mpar.set_cfric([0., 0.])

    return mpar


def random_dynamics(robot, dt, max_torque, class_obj, sigma=0.07, plant_class=SymbolicDoublePendulum, use_random=True):
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


def real_robot(robot, dt, max_torque, class_obj):
    mpar = load_param(robot, max_torque, simplify=False, real_robot=True)
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
            return np.append(x, observation[-2:] * self.torque_limit)
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
            return np.append(observation, state[-2:] / self.torque_limit)
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
        if len(observation) > 4:
            return np.append(x, observation[-2:] * self.torque_limit)
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
            return np.append(observation, state[-2:] / self.torque_limit)
        return observation


def update_plant(plant):
    M = plant.replace_parameters(plant.M)
    C = plant.replace_parameters(plant.C)
    G = plant.replace_parameters(plant.G)
    F = plant.replace_parameters(plant.F)
    plant.M_la = lambdify(plant.x, M)
    plant.C_la = lambdify(plant.x, C)
    plant.G_la = lambdify(plant.x, G)
    plant.F_la = lambdify(plant.x, F)


def simulate_test():
    print("create plant")
    mpar = load_param("pendubot", 5)
    plant = SymbolicDoublePendulum(model_pars=mpar)
    time = np.linspace(0, 2, num=1000)
    force = np.cos(time)

    initial_conditions = [0, 0, 0, 0]
    plant.m = [0.45, 0.52]
    plant.l = [0.29, 0.24]
    delay = 0.1
    update_plant(plant)

    def double_pendulum_model(t, x, delay, torque):
        q1, q2, v1, v2 = x
        delay_time = max(t - delay, 0)
        u = np.interp(delay_time, time, torque)
        accn = plant.forward_dynamics(x, np.array([u, 0]))
        return [v1, v2, accn[0], accn[1]]

    print("calculate true solution")
    true_solution = solve_ivp(double_pendulum_model, [time[0], time[-1]], initial_conditions, args=(delay, force), t_eval=time)
    measured_data = true_solution.y + np.random.normal(0, 0.05, true_solution.y.shape)

    # print("override true solution")
    #
    # data = pd.read_excel('trajectory.xlsx')
    # data = data[data['time'] <= 2]
    #
    # time = data['time'].to_numpy()
    # force = data['tau_con1'].to_numpy()
    # pos_meas1 = data['pos_meas1'].to_numpy()
    # pos_meas2 = data['pos_meas2'].to_numpy()
    # vel_meas1 = data['vel_meas1'].to_numpy()
    # vel_meas2 = data['vel_meas2'].to_numpy()
    # measured_data = np.vstack((pos_meas1, pos_meas2, vel_meas1, vel_meas2))

    def least_squares(params):
        m1, m2, l1, l2, delay = params
        plant.m = [m1, m2]
        plant.l = [l1, l2]
        update_plant(plant)
        print("calculate for: " + str(params))

        batch_length = 20
        batch_number = 20

        batch_distance = len(time) // batch_number
        total_error = 0

        initial_conditions = [measured_data[:, i * batch_distance] for i in range(batch_number)]

        for i in range(batch_number):
            start_index = i * batch_distance
            end_index = start_index + batch_length
            if end_index >= len(time):
                end_index = len(time) - 1
            initial_conditions_segment = initial_conditions[i]
            sol = solve_ivp(double_pendulum_model, [time[start_index], time[end_index]], initial_conditions_segment, args=(delay, force), t_eval=time[start_index:end_index])
            weighted_errors = np.sum((measured_data[:, start_index:end_index] - sol.y) ** 2)
            total_error += weighted_errors

        return total_error

    print("solve least squares")
    initial_guess = np.array([0.5, 0.5, 0.3, 0.2, 0.2])
    max_deviation = np.array([0.1, 0.1, 0.05, 0.05, 0.2])
    lower_bounds = initial_guess - max_deviation
    upper_bounds = initial_guess + max_deviation
    bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]

    result = minimize(least_squares, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-2)

    print("calculate model solution")
    m1, m2, l1, l2, delay = result.x
    plant.m = [m1, m2]
    plant.l = [l1, l2]
    update_plant(plant)
    model_solution = solve_ivp(double_pendulum_model, [time[0], time[-1]], initial_conditions, args=(delay, force), t_eval=time)
    print("results: " + str(result.x))

    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_data[0], 'g', label='q1')
    plt.plot(time, measured_data[1], 'r', label='q2')
    plt.plot(time, model_solution.y[0], 'b--', label='q1 model')
    plt.plot(time, model_solution.y[1], 'k--', label='q2 model')
    plt.title('Modell')
    plt.xlabel('Zeit [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # def get_x1y1x2y2(t, the1, the2, L1, L2):
    #     return (L1 * np.sin(the1),
    #             -L1 * np.cos(the1),
    #             L1 * np.sin(the1) + L2 * np.sin(the1 + the2),
    #             -L1 * np.cos(the1) - L2 * np.cos(the1 + the2))
    #
    # x1, y1, x2, y2 = get_x1y1x2y2(time, true_solution.y[0], true_solution.y[1], 1, 1)
    #
    # def animate(i):
    #     ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.set_facecolor('k')
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    # ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
    # ax.set_ylim(-4, 4)
    # ax.set_xlim(-4, 4)
    # ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
    # ani.save('pen.gif', writer='pillow', fps=25)