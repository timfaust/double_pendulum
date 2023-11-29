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
    x = np.array([np.sin(y[0]), np.cos(y[0])]) * l[0] + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * l[1]
    v = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * l[0] + np.array(
        [np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * l[1]
    goal = np.array([0, -0.5])

    return y, x, v, goal


def future_pos_reward(observation, action, env_type):
    y, x, v, goal = get_state_values(observation, env_type)
    v_total = np.linalg.norm(v)
    threshold = 0.005
    distance = np.maximum(np.linalg.norm(x + 0.01 * v - goal), threshold)
    if distance > threshold:
        return 1 / distance
    return 1 / distance + 1 / v_total


def pos_reward(observation, action, env_type):
    y, x, v, goal = get_state_values(observation, env_type)
    distance = np.linalg.norm(x - goal)
    return 1 / (distance + 0.0001)
