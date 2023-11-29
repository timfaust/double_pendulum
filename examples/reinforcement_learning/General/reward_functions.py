import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def future_pos_reward_acrobot(observation, action):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2],
            observation[3]
        ]
    )

    y = wrap_angles_diff(s)
    x = np.array([np.sin(y[0]), np.cos(y[0])]) * 0.2 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * 0.3
    v = np.array([np.cos(y[0]), -np.sin(y[0])]) * y[2] * 0.2 + np.array([np.cos(y[0] + y[1]), -np.sin(y[0] + y[1])]) * (y[2] + y[3]) * 0.3

    goal = np.array([0, -0.5])
    distance = np.sqrt(np.sum((x + 0.01 * v - goal)**2))

    return 1/(distance + 0.0001)
