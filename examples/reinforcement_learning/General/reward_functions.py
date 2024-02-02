import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff
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


def get_score(state_dict):
    step = len(state_dict["T"])
    max_episode_steps = state_dict["max_episode_steps"]
    if step == max_episode_steps:
        return calculate_score(state_dict)
    return 0


def calculate_score(state_dict, verbose=False):
    print("calculate_score")
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

    T = np.array(state_dict["T"])
    X = np.array(state_dict["X_meas"])
    U = np.array(state_dict["U_con"])

    swingup_time = get_swingup_time(T=T, X=X, plant=state_dict["plant"], height=0.9)
    max_tau = get_max_tau(U)
    energy = get_energy(X, U)
    integ_tau = get_integrated_torque(T, U)
    tau_cost = get_torque_cost(T, U)
    tau_smoothness = get_tau_smoothness(U)
    velocity_cost = get_velocity_cost(T, X)
    success = int(swingup_time < T[-1])

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


def get_state_values(observation, action, robot, dynamics):
    dt = dynamics[0]
    max_velocity = dynamics[1]
    torque_limit = dynamics[2][0]

    l = [0.2, 0.3]
    if robot == 'pendubot':
        l = [0.3, 0.2]

    s = np.array(
        [
            observation[0] * 2 * np.pi + np.pi,
            observation[1] * 2 * np.pi,
            observation[2] * max_velocity,
            observation[3] * max_velocity
        ]
    )

    y = wrap_angles_diff(s) #now both angles from -pi to pi

    #cartesians of elbow x1 and end effector x2
    x1 = np.array([np.sin(y[0]), np.cos(y[0])]) * l[0]
    x2 = x1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * l[1]

    #angular velocities of the joints
    v1 = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * l[0]
    v2 = v1 + np.array([np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * l[1]

    #goal for cartesian end effector position
    goal = np.array([0, -0.5])

    dt_goal = 0.05
    threshold_distance = 0.005

    u_p, u_pp = 0, 0
    if len(observation) > 4:
        u_p = (action - observation[-2]) / dt
        u_pp = (action - 2 * observation[-2] + observation[-1])/(dt * dt)

    return s, x1, x2, v1, v2, action * torque_limit, goal, dt_goal, threshold_distance, u_p * torque_limit, u_pp * torque_limit


def future_pos_reward(observation, action, env_type, dynamics, state_dict):
    y, x1, x2, v1, v2, action, goal, dt_goal, threshold_distance, u_p, u_pp = get_state_values(observation, action, env_type, dynamics)
    distance = np.linalg.norm(x2 + dt_goal * v2 - goal)
    reward = 1 / (distance + 0.01)
    if distance < threshold_distance:
        v_total = np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(action) + np.linalg.norm(u_p)
        reward += 1 / (v_total + 0.001)
    return reward


def pos_reward(observation, action, env_type, dynamics, state_dict):
    y, x1, x2, v1, v2, action, goal, dt, threshold, _, _ = get_state_values(observation, action, env_type, dynamics)
    distance = np.linalg.norm(x2 - goal)
    return 1 / (distance + 0.0001)


def saturated_distance_from_target(observation, action, env_type, dynamics, state_dict):
    l = [0.2, 0.3]
    if env_type == 'pendubot':
        l = [0.3, 0.2]

    R = np.array([[0.0001]])

    y, pos1, pos2, vel1, vel2, u, _, _, _, _, _ = get_state_values(observation, action, env_type, dynamics)

    goal = [np.pi, 0]
    diff = y[:2] - goal
    weight_vel = 1
    weight_torque = 1

    sigma_c = np.diag([1 / l[0], 1 / l[1]])
    #   encourage to minimize the distance
    sat_dist = np.dot(np.dot(diff.T, sigma_c), diff)

    #   encourage to have minimum torque change
    exp_indx = - sat_dist - weight_torque * np.abs(np.einsum("i, ij, j", u, R, u)) / (sat_dist + 0.01)

    #   encourage to have zero velocity as distance minimizes
    exp_indx -= weight_vel * np.abs(np.linalg.norm(y[2:])) / (sat_dist + 0.0001)

    exp_term = np.exp(exp_indx)

    squared_dist = 1.0 - exp_term

    return -squared_dist



def unholy_reward_4(observation, action, env_type, dynamics, state_dict):
    #quadtratic cost and quadtratic penalties
    l = [0.2, 0.3]
    if env_type == 'pendubot':
        l = [0.3, 0.2]

    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2],
            observation[3]
        ]
    )



    y, x1, x2, v1, v2, action, goal, dt, threshold, _, _ = get_state_values(observation, action, env_type, dynamics)


    #defining custom goal for state (pos1, pos2, angl_vel1, angl_vel2)
    goal = np.array([np.pi, 0., 0., 0.])

    #we want it to go up
    #we dont want rotations
    #we dont want oscillations

    #error scale matrix for state deviation
    Q = np.zeros((4, 4))
    Q[0, 0] = 10.0
    Q[1, 1] = 10.0
    Q[2, 2] = 0.2
    Q[3, 3] = 0.2

    #penalty for actuation
    R = np.array([[0.001]])

    #state error
    err = s - goal

    #"control" input penalty
    u = action * 5

    #quadratic cost for u, quadratic cost for state
    cost1 = np.einsum("i, ij, j", err, Q, err) + np.einsum("i, ij, j", u, R, u)


    #additional cartesian distance cost
    if env_type == 'pendubot':
        cart_goal_x1 = np.array([0,-0.3])
    elif env_type =='acrobot':
        cart_goal_x1 = np.array([0,-0.2])


    cart_goal_x2 = np.array([0, -0.5])
    cart_err_x2 = x2 - cart_goal_x2

    cart_err_x1 = x1 - cart_goal_x1

    Q2 = np.zeros((2,2))
    Q2[0, 0] = 10.0
    Q2[1, 1] = 10.0



    cost2= np.einsum("i, ij, j", cart_err_x2, Q2, cart_err_x2) + np.einsum("i, ij, j", cart_err_x1, Q2, cart_err_x1)

    reward = -1 * cost1 - 1 * cost2

    #try x.T Q x, then try vector norms

    #general:
    #try: LQR style --> Quadratic Costfunctions
    #else try: PID style --> "Damping"

    return reward
