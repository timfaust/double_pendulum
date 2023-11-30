import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def get_state_values(observation, robot):
    l = [0.2, 0.3]
    if robot == 'pendubot':
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
    x1 = np.array([np.sin(y[0]), np.cos(y[0])]) * l[0]
    x2 = x1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * l[1]
    v1 = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * l[0]
    v2 = v1 + np.array([np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * l[1]
    goal = np.array([0, -0.5])

    return y, x1, x2, v1, v2, goal


def future_pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, goal = get_state_values(observation, env_type)
    threshold = 0.01
    distance = np.linalg.norm(x2 + 0.01 * v2 - goal)
    reward = 1 / (distance + 0.0001)
    if distance < threshold:
        v_total = np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(action)
        reward += + 1 / (v_total + 0.0001)
    return reward


def pos_reward(observation, action, env_type):
    y, x1, x2, v1, v2, goal = get_state_values(observation, env_type)
    distance = np.linalg.norm(x2 - goal)
    return 1 / (distance + 0.0001)


def saturated_distance_from_target(observation, env_type):
    y, x1, x2, v1, v2, goal = get_state_values(observation, env_type)

    diff = x2 - goal

    sigma_c = np.diag([1 / 0.3, 1 / 0.2])
    exp_term = np.exp(-np.dot(np.dot(diff.T, sigma_c), diff))

    squared_dist = 1.0 - exp_term

    return squared_dist
