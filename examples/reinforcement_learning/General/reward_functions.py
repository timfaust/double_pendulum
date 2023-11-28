import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def simple_reward_acrobot(observation, action):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2],
            observation[3]
        ]
    )

    y = wrap_angles_diff(s)
    start = np.array([0, 0])
    end_1 = start + np.array([np.sin(y[0]), np.cos(y[0])]) * 0.2
    end_2 = end_1 + np.array([np.sin(y[0] + y[1]), np.cos(y[0] + y[1])]) * 0.3

    goal = np.array([0, -0.5])
    distance = np.sqrt(np.sum((end_2 - goal)**2))

    return 1/(distance + 0.0001) - 10
