import numpy as np


def general_reset(x_values, dx_values):
    rand = np.random.rand(4)
    observation = [x - dx + 2 * dx * r for x, dx, r in zip(x_values, dx_values, rand)]
    return observation


def low_reset():
    return general_reset([1, 1, 0, 0], [0.05, 0.05, 0.05, 0.05])


def high_reset():
    return general_reset([0, 1, 0, 0], [0.05, 0.05, 0.05, 0.05])


def random_reset():
    return general_reset([0, 1, 0, 0], [1, 0.25, 0.5, 0.25])


def balanced_reset():
    r = np.random.random()
    if r < 0.25:
        return high_reset()
    elif r > 0.75:
        return low_reset()
    else:
        return random_reset()


def noisy_reset():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation


def no_termination(observation):
    return False
