import numpy as np


def general_reset(x_values, dx_values):
    rand = np.random.rand(4)
    observation = [x - dx + 2 * dx * r for x, dx, r in zip(x_values, dx_values, rand)]
    return observation


def low_reset(low_pos=[-0.5, 0, 0, 0]):
    return general_reset(low_pos, [0.01, 0.01, 0.01, 0.01])


def debug_reset(low_pos=[-0.5, 0, 0, 0]):
    return general_reset(low_pos, [0.0, 0.0, 0.0, 0.0])


def high_reset():
    return general_reset([0, 0, 0, 0], [0.025, 0.025, 0.05, 0.05])


def random_reset():
    return general_reset([0, 0, 0, 0], [0.5, 0.5, 0.75, 0.75])


def semi_random_reset():
    return general_reset([0, 0, 0, 0], [0.5, 0.1, 0.5, 0.2])


def balanced_reset(low_pos=[-0.5, 0, 0, 0]):
    r = np.random.random()
    if r < 0.25:
        return high_reset()
    elif r < 0.5:
        return low_reset(low_pos)
    elif r < 0.75:
        return semi_random_reset()
    else:
        return random_reset()


def updown_reset(low_pos=[-0.5, 0, 0, 0]):
    r = np.random.random()
    if r < 0.25:
        return high_reset()
    elif r < 0.5:
        return debug_reset(low_pos)
    else:
        return low_reset(low_pos)


def noisy_reset(low_pos=[-0.5, 0, 0, 0]):
    rand = np.random.rand(4) * 0.005
    rand[2:] = rand[2:] - 0.025
    observation = low_pos + rand
    return observation


def no_termination(observation):
    return False


def kill_switch(observation):
    if np.max(np.abs(observation[:2])) > 1.5 * np.pi:
        # print("Terminated at state: ", observation)
        return True
    if np.max(np.abs(observation[2:4])) > 19:
        # print("Terminated at velocity: ", observation)
        return True
    return False
