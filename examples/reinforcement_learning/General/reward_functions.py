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
    x1 = np.array([np.sin(y[0]), np.cos(y[0])]) * l[0]
    x2 = x1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * l[1]
    v1 = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * l[0]
    v2 = v1 + np.array([np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * l[1]
    goal = np.array([0, -0.5])

    return s, x1, x2, v1, v2, action * 5, goal


def future_pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, action, goal = get_state_values(observation, action, env_type)
    threshold = 0.005
    distance = np.linalg.norm(x2 + 0.1 * v2 - goal)
    reward = 1 / (distance + 1)
    if distance < threshold:
        v_total = np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(action)
        reward += 1 / (v_total + 0.1)
    if y[1] > np.pi/2 or y[1] < -np.pi/2:
        reward -= 10
    return reward


def pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, action, goal = get_state_values(observation, action, env_type)
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
