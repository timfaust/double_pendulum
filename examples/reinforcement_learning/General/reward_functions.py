import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def get_state_values(observation, action, robot):
    l = [0.2, 0.3]
    if robot == 'pendubot':
        l = [0.3, 0.2]

    s = np.array(
        [
            observation[0] * np.pi + np.pi,
            observation[1] * np.pi + np.pi,
            observation[2] * 20,
            observation[3] * 20
        ]
    )

    y = wrap_angles_diff(s) #now both angles from -pi to pi

    #cartesians of elbow x1 and end effector x2
    x1 = np.array([np.sin(y[0]), np.cos(y[0])]) * l[0]
    x2 = x1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * l[1]

    #cartesian velocities of the joints: v = omega * r
    v1 = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * l[0]
    v2 = v1 + np.array([np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * l[1]

    #goal for cartesian end effector position
    goal = np.array([0, -0.5])

    return s, x1, x2, v1, v2, action * 5, goal, 0.05, 0.005


def future_pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, action, goal, dt, threshold = get_state_values(observation, action, env_type)
    distance = np.linalg.norm(x2 + dt * v2 - goal)
    reward = 1 / (distance + 0.01)
    if distance < threshold:
        v_total = np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(action)
        reward += 1 / (v_total + 0.001)
    return reward


def pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, action, goal, dt, threshold = get_state_values(observation, action, env_type)
    distance = np.linalg.norm(x2 - goal)
    return 1 / (distance + 0.0001)


def saturated_distance_from_target(observation, action, env_type):
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
    y = wrap_angles_diff(s)
    R = np.array([[0.0001]])
    u = action * 5
    goal = [np.pi, 0]

    diff = y[:2] - goal

    sigma_c = np.diag([1 / l[0], 1 / l[1]])
    #   encourage to minimize the distance
    exp_indx = -np.dot(np.dot(diff.T, sigma_c), diff)

    #   encourage to have zero torque change as distance minimizes
    threshold = np.maximum(np.abs(np.linalg.norm(diff)), 10)
    exp_indx -= np.abs(np.einsum("i, ij, j", u, R, u) / threshold)

    #   encourage to have zero velocity as distance minimizes
    exp_indx -= np.abs(np.linalg.norm(y[2:]) / threshold)

    exp_term = np.exp(exp_indx)

    squared_dist = 1.0 - exp_term
    #print(-squared_dist * 1000)
    return -squared_dist



def unholy_reward_4(observation, action, env_type):
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



    y, x1, x2, v1, v2, action, goal, dt, threshold = get_state_values(observation, action, env_type)


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



def unholy_reward_5(observation, action, env_type):
    #rein experimentell. nicht sehr wertvoll

    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2],
            observation[3]
        ]
    )

    y, x1, x2, v1, v2, action, goal, dt, threshold = get_state_values(observation, action, env_type)
    short_state = np.array([y[0], y[1]])
    # defining custom goal for state (pos1, pos2, angl_vel1, angl_vel2)
    goal = np.array([np.pi, 0.])
    if env_type == 'pendubot':
        d_goal_x1 = np.array([0, -0.3])
    elif env_type == 'acrobot':
        d_goal_x1 = np.array([0, -0.2])
    d_goal_x2 = np.array([0, -0.5])

    # error scale matrices
    Q = np.zeros((2, 2))
    Q[0, 0] = 10.0
    Q[1, 1] = 10.0


    # penalty for actuation
    R = np.array([[0.001]])

    # state error
    err = short_state - goal

    d_err_1 = x1 + dt * v1 - d_goal_x1
    d_err_2 = x2 + dt * v2 - d_goal_x2


    # "control" input penalty
    u = action * 5

    # quadratic cost for u, quadratic cost for state
    cost = np.einsum("i, ij, j", err, Q, err) + np.einsum("i, ij, j", u, R, u) + np.einsum("i, ij, j", d_err_1, Q, d_err_1) + np.einsum("i, ij, j", d_err_2, Q, d_err_2)
    reward = -1 * cost

    return reward

