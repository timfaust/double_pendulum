import numpy as np

from src.python.double_pendulum.analysis.leaderboard import get_max_tau, get_energy, \
    get_integrated_torque, get_torque_cost, get_tau_smoothness, get_velocity_cost


def get_swingup_time(
    T,
    X,
    plant,
    eps=[1e-2, 1e-2, 1e-2, 1e-2],
    has_to_stay=True,
    method="height",
    height=0.9,
):
    """get_swingup_time.
    get the swingup time from a data_dict.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    eps : list
        list with len(eps) = 4. The thresholds for the swingup to be
        successfull ([position, velocity])
        default = [1e-2, 1e-2, 1e-2, 1e-2]
    has_to_stay : bool
        whether the pendulum has to stay upright until the end of the trajectory
        default=True

    Returns
    -------
    float
        swingup time
    """
    if method == "epsilon":
        goal = np.array([np.pi, 0.0, 0.0, 0.0])

        dist_x0 = np.abs(np.mod(X.T[0] - goal[0] + np.pi, 2 * np.pi) - np.pi)
        ddist_x0 = np.where(dist_x0 < eps[0], 0.0, dist_x0)
        n_x0 = np.argwhere(ddist_x0 == 0.0)

        dist_x1 = np.abs(np.mod(X.T[1] - goal[1] + np.pi, 2 * np.pi) - np.pi)
        ddist_x1 = np.where(dist_x1 < eps[1], 0.0, dist_x1)
        n_x1 = np.argwhere(ddist_x1 == 0.0)

        dist_x2 = np.abs(X.T[2] - goal[2])
        ddist_x2 = np.where(dist_x2 < eps[2], 0.0, dist_x2)
        n_x2 = np.argwhere(ddist_x2 == 0.0)

        dist_x3 = np.abs(X.T[3] - goal[3])
        ddist_x3 = np.where(dist_x3 < eps[3], 0.0, dist_x3)
        n_x3 = np.argwhere(ddist_x3 == 0.0)

        n = np.intersect1d(n_x0, n_x1)
        n = np.intersect1d(n, n_x2)
        n = np.intersect1d(n, n_x3)

        time_index = len(T) - 1
        if has_to_stay:
            if len(n) > 0:
                for i in range(len(n) - 2, 0, -1):
                    if n[i] + 1 == n[i + 1]:
                        time_index = n[i]
                    else:
                        break
        else:
            if len(n) > 0:
                time_index = n[0]
        time = T[time_index]
    elif method == "height":
        fk = plant.forward_kinematics(X.T[:2])
        ee_pos_y = fk[1][1]

        goal_height = height * (plant.l[0] + plant.l[1])

        up = np.where(ee_pos_y > goal_height, True, False)

        time_index = len(T) - 1
        if has_to_stay:
            for i in range(len(up) - 2, 0, -1):
                if up[i]:
                    time_index = i
                else:
                    break

        else:
            time_index = np.argwhere(up)[0][0]
        time = T[time_index]

    else:
        time = np.inf

    return time


def calculate_score(
    observation_dict,
    weights={
        "swingup_time": 1.0,
        "max_tau": 0.0,
        "energy": 1.0,
        "integ_tau": 0.0,
        "tau_cost": 1.0,
        "tau_smoothness": 1.0,
        "velocity_cost": 1.0,
    },
    normalize={
        "swingup_time": 20.0,
        "max_tau": 1.0,  # not used
        "energy": 60.0,
        "integ_tau": 1.0,  # not used
        "tau_cost": 20.0,
        "tau_smoothness": 0.1,
        "velocity_cost": 400,
    },
    needs_success=True
):

    swingup_times = []
    max_taus = []
    energies = []
    integ_taus = []
    tau_costs = []
    tau_smoothnesses = []
    velocity_costs = []
    successes = []

    T = np.array(observation_dict["T"])
    X = np.array([observation_dict['dynamics_func'].unscale_state(x) for x in observation_dict['X_real']])
    U = np.array([observation_dict['dynamics_func'].unscale_action([u]) for u in observation_dict['U_real']])

    # plant = dynamics_func.simulator.plant
    swingup_times.append(
        get_swingup_time(
            T=T, X=X, plant=observation_dict['dynamics_func'].simulator.plant, has_to_stay=True, method="height", height=0.9
        )
    )
    max_taus.append(get_max_tau(U))
    energies.append(get_energy(X, U))
    integ_taus.append(get_integrated_torque(T, U))
    tau_costs.append(get_torque_cost(T, U))
    tau_smoothnesses.append(get_tau_smoothness(U))
    velocity_costs.append(get_velocity_cost(T, X))

    successes.append(int(swingup_times[-1] < T[-1]))

    nonzero_weigths = 0
    for w in weights.keys():
        if weights[w] != 0.0:
            nonzero_weigths += 1

    factor = 1
    if needs_success:
        factor = successes[-1]

    score = factor * (
        1.0
        - 1.0
        / nonzero_weigths
        * (
            np.tanh(
                np.pi
                * weights["swingup_time"]
                * swingup_times[-1]
                / normalize["swingup_time"]
            )
            + np.tanh(
                np.pi
                * weights["max_tau"]
                * max_taus[-1]
                / normalize["max_tau"]
            )
            + np.tanh(
                np.pi
                * weights["energy"]
                * energies[-1]
                / normalize["energy"]
            )
            + np.tanh(
                np.pi
                * weights["integ_tau"]
                * integ_taus[-1]
                / normalize["integ_tau"]
            )
            + np.tanh(
                np.pi
                * weights["tau_cost"]
                * tau_costs[-1]
                / normalize["tau_cost"]
            )
            + np.tanh(
                np.pi
                * weights["tau_smoothness"]
                * tau_smoothnesses[-1]
                / normalize["tau_smoothness"]
            )
            + np.tanh(
                np.pi
                * weights["velocity_cost"]
                * velocity_costs[-1]
                / normalize["velocity_cost"]
            )
        )
    )

    return score