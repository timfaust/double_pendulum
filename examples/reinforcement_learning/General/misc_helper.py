from collections import deque
from double_pendulum.utils.wrap_angles import wrap_angles_diff
import numpy as np


def smooth_transition(value, threshold, sharpness=80):
    return 0.5 * (1 + np.tanh(sharpness * (value - threshold)))


def is_up(obs, progress):
    phi_1 = obs[0] * 3 * np.pi
    phi_2 = obs[1] * 3 * np.pi
    s1 = np.sin(phi_1)
    s2 = np.sin(phi_1 + phi_2)
    c1 = np.cos(phi_1)
    c2 = np.cos(phi_1 + phi_2)
    x1 = np.array([s1, c1]) * 0.2
    x2 = x1 + np.array([s2, c2]) * 0.3

    threshold = -0.425
    value = x2[1]
    out = 0.5
    if progress > 0.4:
        out = smooth_transition(value, threshold)

    return out


def swing_up(obs, progress):
    return is_up(obs, progress)


def stabilize(obs, progress):
    return 1 - is_up(obs, progress)


def default_decider(obs, progress):
    return 1


def general_reset(x_values, dx_values):
    rand = np.random.rand(4)
    observation = [x - dx + 2 * dx * r for x, dx, r in zip(x_values, dx_values, rand)]
    return observation


def low_reset(low_pos=[0, 0, 0, 0]):
    return general_reset(low_pos, [0.01, 0.01, 0.01, 0.01])


def debug_reset(low_pos=[0, 0, 0, 0]):
    return general_reset(low_pos, [0.0, 0.0, 0.0, 0.0])


def high_reset():
    return general_reset([0, 0, 0, 0], [0.025, 0.025, 0.05, 0.05])


def random_reset():
    return general_reset([0, 0, 0, 0], [0.5, 0.5, 0.75, 0.75])


def semi_random_reset():
    return general_reset([0, 0, 0, 0], [0.5, 0.1, 0.5, 0.2])


def balanced_reset(low_pos=[0, 0, 0, 0]):
    r = np.random.random()
    if r < 0.25:
        return high_reset()
    elif r < 0.5:
        return low_reset(low_pos)
    elif r < 0.75:
        return semi_random_reset()
    else:
        return random_reset()


def updown_reset(low_pos=[0, 0, 0, 0]):
    r = np.random.random()
    if r < 0.25:
        return high_reset()
    elif r < 0.5:
        return debug_reset(low_pos)
    else:
        return low_reset(low_pos)


def noisy_reset(low_pos=[0, 0, 0, 0]):
    rand = np.random.rand(4) * 0.005
    rand[2:] = rand[2:] - 0.025
    observation = low_pos + rand
    return observation


def no_termination(observation):
    return False


def punish_limit(observation, action, dynamics_function, k=50):
    thresholds = np.array([0.95] * 5)

    values = np.concatenate([np.abs(observation), np.array([np.abs(action)])])
    ratios = values / thresholds

    factors = np.where(ratios <= 1, 1 - np.exp(-k * np.abs(ratios - 1)), 0)

    return factors[:2].min(), factors[2:4].min(), 1# factors[4]


def kill_switch(observation, action, dynamics_func):
    return np.array(punish_limit(observation, action, dynamics_func)) == 0


def calculate_q_values(reward, gamma):
    actual_Q = deque()
    for r in reversed(reward):
        if actual_Q:
            q_value = r + actual_Q[0] * gamma
        else:
            q_value = r / (1 - gamma)
        actual_Q.appendleft(q_value)
    return np.array(actual_Q)


def get_i_decay(x, factor=2, offset=0.0):
    return np.where(x <= offset, 1, 1 / (factor * (x - offset) + 1))


def get_unscaled_action(observation_dict, t_minus=0, key='U_real'):
    unscaled_action = observation_dict['dynamics_func'].unscale_action(observation_dict[key][t_minus-1])
    max_value_index = np.argmax(np.abs(unscaled_action))
    max_action_value = unscaled_action[max_value_index]
    return max_action_value


def column_softmax(x):
    col_sums = x.sum(axis=0)
    needs_softmax = ~np.isclose(col_sums, 1.0)
    x[:, needs_softmax] = softmax(x[:, needs_softmax])
    return x


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def softmax_and_select(arr):
    softmax_probs = column_softmax(arr)
    result = np.zeros_like(arr)
    selected_rows = [np.argmax(np.random.multinomial(1, softmax_probs[:, i])) for i in range(arr.shape[1])]
    result[selected_rows, np.arange(arr.shape[1])] = 1
    return result


def find_index_and_dict(observation, env):
    observation_dict = env.observation_dict
    index = find_observation_index(observation, observation_dict)
    if index < 0:
        observation_dict = env.observation_dict_old
        index = find_observation_index(observation, observation_dict)

    return index, observation_dict


def find_observation_index(observation, observation_dict):
    X_meas = observation_dict['X_meas']
    for idx, array in reversed(list(enumerate(X_meas))):
        if np.array_equal(observation, array.astype(np.float32)) or np.array_equal(observation, array):
            return idx
    return -1


def get_stabilized(observation_dict, threshold=0.002):
    #TODO: add lowpass filter
    X_meas = np.array(observation_dict['X_real'])
    T = observation_dict['T']
    f = np.abs(np.sin((X_meas[-1][0] + X_meas[-1][1]) * 3 * np.pi / 2))

    # Check if the last measurement is within the specified range
    if f < 0.99:
        return 0.0

    n = len(X_meas)
    window_size = 1
    end_index = n - 1

    # Use cumulative sum for efficient standard deviation calculation
    cumsum = np.cumsum(X_meas[::-1], axis=0)
    cumsum_sq = np.cumsum(X_meas[::-1]**2, axis=0)

    while end_index >= 0:
        if window_size == 1:
            stds = np.zeros(2)
        else:
            means = cumsum[window_size-1] / window_size
            stds = np.sqrt((cumsum_sq[window_size-1] / window_size) - means**2)

        if np.all(stds < threshold):
            window_size += 1
            end_index -= 1
        else:
            break

    return T[-1] - T[end_index + 1] if end_index < n - 1 else 0.0


def get_state_values(observation_dict, key='X_meas', offset=0):
    l = observation_dict['mpar'].l
    action_key = 'U_con'
    if key == 'X_real':
        l = observation_dict['dynamics_func'].simulator.plant.l
        action_key = 'U_real'

    dt_goal = 0.05
    threshold_distance = (l[0] + l[1]) * 0.1

    unscaled_observation = observation_dict['dynamics_func'].unscale_state(observation_dict[key][offset-1])
    unscaled_action = get_unscaled_action(observation_dict, offset, action_key)

    y = wrap_angles_diff(unscaled_observation) #now both angles from -pi to pi

    s1 = np.sin(y[0])
    s2 = np.sin(y[0] + y[1])
    c1 = np.cos(y[0])
    c2 = np.cos(y[0] + y[1])

    #cartesians of elbow x1 and end effector x2
    x1 = np.array([s1, c1]) * l[0]
    x2 = x1 + np.array([s2, c2]) * l[1]

    #cartesian velocities of the joints
    v1 = np.array([c1, -s1]) * y[2] * l[0]
    v2 = v1 + np.array([c2, -s2]) * (y[2] + y[3]) * l[1]

    #goal for cartesian end effector position
    goal = np.array([0, -(l[0] + l[1])])

    x3 = x2 + dt_goal * v2
    distance = np.linalg.norm(x3 - goal)/(l[0] + l[1])

    u_p, u_pp = 0, 0
    if len(observation_dict['U_con']) + offset > 1:
        dt = observation_dict['T'][offset-1] - observation_dict['T'][offset-2]
        u_p = (unscaled_action - get_unscaled_action(observation_dict, offset-1, action_key)) / dt
        if len(observation_dict['U_con']) + offset > 2:
            u_pp = (unscaled_action - 2 * get_unscaled_action(observation_dict, offset-1, action_key) + get_unscaled_action(observation_dict, offset-2, action_key))/(dt * dt)

    state_values = {
        "unscaled_observation": unscaled_observation,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "s1": s1,
        "s2": s2,
        "c1": c1,
        "c2": c2,
        "v1": v1,
        "v2": v2,
        "omega_squared_1": unscaled_observation[2] ** 2,
        "omega_squared_2": unscaled_observation[3] ** 2,
        "goal": goal,
        "dt_goal": dt_goal,
        "threshold_distance": threshold_distance,
        "distance": distance,
        "unscaled_action": unscaled_action,
        "u_p": u_p,
        "u_pp": u_pp
    }

    return state_values
