import numpy as np

from src.python.double_pendulum.model.model_parameters import model_parameters
from src.python.double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from src.python.double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from src.python.double_pendulum.simulation.simulation import Simulator
from src.python.double_pendulum.utils.wrap_angles import wrap_angles_diff


def load_param(robot="acrobot", torque_limit=5.0):
    """
    param: robot: String robot name
            torque_limit = float
    """
    # model parameter
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"
        torque_array = [torque_limit, 0.0]
        active_act = 0

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"
        torque_array = [0.0, torque_limit]
        active_act = 1

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

    return mpar, torque_array, active_act


def get_dynamics_function(mpar, robot, dt=0.01, integrator="runge_kutta"):
    plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant)
    dynamics_function = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=2,
    )
    return dynamics_function, simulator


def reward_func(observation, action):
    control_line = 0.4
    v_thresh = 8.0
    vflag = False
    flag = False
    bonus = False

    u = 5.0 * action
    goal = [np.pi, 0.0, 0.0, 0.0]

    # quadratic with roa attraction
    Q = np.zeros((4, 4))
    Q[0, 0] = 10
    Q[1, 1] = 10
    Q[2, 2] = 0.4
    Q[3, 3] = 0.3
    R = np.array([[0.0001]])

    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * 8.0,
            observation[3] * 8.0,
        ]
    )

    y = wrap_angles_diff(s)

    # openAI
    p1 = y[0]  # joint 1 pos
    p2 = y[1]  # joint 2 pos

    ee1_pos_x = 0.2 * np.sin(p1)
    ee1_pos_y = -0.2 * np.cos(p1)

    ee2_pos_x = ee1_pos_x + 0.3 * np.sin(p1 + p2)
    ee2_pos_y = ee1_pos_y - 0.3 * np.cos(p1 + p2)

    # criteria 4
    if ee2_pos_y >= control_line:
        flag = True
    else:
        flag = False

    #bonus, rad = check_if_state_in_roa(S, rho, y)

    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r
    if flag:
        reward += 100
        if bonus:
            # roa method
            reward += 1e3
            # print("!!!bonus = True")
    else:
        reward = reward

    return reward


def terminated_func(observation):
    return False


def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation
