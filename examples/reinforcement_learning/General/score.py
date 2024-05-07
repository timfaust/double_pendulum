import numpy as np
from src.python.double_pendulum.analysis.leaderboard import get_max_tau, get_energy, \
    get_integrated_torque, get_torque_cost, get_tau_smoothness, get_velocity_cost


def get_swingup_time(
    T,
    X,
    plant=None,
    height=0.9,
):
    fk = plant.forward_kinematics(X.T[:2])
    ee_pos_y = fk[1][1]

    goal_height = height * (plant.l[0] + plant.l[1])

    up = np.where(ee_pos_y > goal_height, True, False)

    time_index = len(T) - 1
    for i in range(len(up) - 2, 0, -1):
        if up[i]:
            time_index = i
        else:
            break

    else:
        time_index = np.argwhere(up)[0][0]
    time = T[time_index]

    return time


def get_score(observation_dict):
    step = len(observation_dict["T"])
    max_episode_steps = observation_dict["max_episode_steps"]
    if step == max_episode_steps:
        return calculate_score(observation_dict)
    return 0


def calculate_score(observation_dict, verbose=False, step=False, check_swingup=True):
    normalize = {
        "swingup_time": 10.0,
        "max_tau": 6.0,
        "energy": 100.0,
        "integ_tau": 60.0,
        "tau_cost": 360.0,
        "tau_smoothness": 12.0,
        "velocity_cost": 1000,
    }
    weights = {
        "swingup_time": 0.2,
        "max_tau": 0.1,
        "energy": 0.1,
        "integ_tau": 0.1,
        "tau_cost": 0.1,
        "tau_smoothness": 0.2,
        "velocity_cost": 0.2,
    }

    T = np.array(observation_dict["T"])
    X = np.array(observation_dict["X_meas"])
    U = np.array(observation_dict["U_con"])

    if len(np.array(observation_dict["T"])) < 2:
        return 1

    if step:
        T = np.copy(T)[-2:]
        X = np.copy(X)[-2:]
        U = np.copy(U)[-2:]

    if check_swingup:
        swingup_time = get_swingup_time(T=T, X=X, plant=observation_dict["plant"], height=0.9)
        success = int(swingup_time < T[-1])
    else:
        success = 1
        swingup_time = 0

    max_tau = get_max_tau(U)
    energy = get_energy(X, U)
    integ_tau = get_integrated_torque(T, U)
    tau_cost = get_torque_cost(T, U)
    tau_smoothness = get_tau_smoothness(U)
    velocity_cost = get_velocity_cost(T, X)

    score = success * (
            1.0
            - (
                    weights["swingup_time"]
                    * swingup_time
                    / normalize["swingup_time"]
                    + weights["max_tau"] * max_tau / normalize["max_tau"]
                    + weights["energy"] * energy / normalize["energy"]
                    + weights["integ_tau"] * integ_tau / normalize["integ_tau"]
                    + weights["tau_cost"] * tau_cost / normalize["tau_cost"]
                    + weights["tau_smoothness"]
                    * tau_smoothness
                    / normalize["tau_smoothness"]
                    + weights["velocity_cost"]
                    * velocity_cost
                    / normalize["velocity_cost"]
            )
    )

    if verbose:
        print("calculate_score")
        print("swingup_time: " + str(swingup_time))
        print("max_tau: " + str(max_tau))
        print("energy: " + str(energy))
        print("integ_tau: " + str(integ_tau))
        print("tau_cost: " + str(tau_cost))
        print("tau_smoothness: " + str(tau_smoothness))
        print("velocity_cost: " + str(velocity_cost))
        print("success: " + str(success))
        print("score: " + str(score))
    return score