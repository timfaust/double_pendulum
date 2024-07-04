from collections import deque
from double_pendulum.utils.wrap_angles import wrap_angles_diff
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


def punish_limit(observation, dynamics_function, k=100):
    angle_threshold = dynamics_function.max_angle * 0.95
    velocity_threshold = dynamics_function.max_velocity

    angle_max = np.max(np.abs(observation[:2]))/angle_threshold
    if angle_max > 1:
        return 0
    velocity_max = np.max(np.abs(observation[2:4]))/velocity_threshold
    if velocity_max > 1:
        return 0
    angle_factor = 1 - np.exp(-k * np.abs(angle_max - 1))
    velocity_factor = 1 - np.exp(-k * np.abs(velocity_max - 1))
    return min(angle_factor, velocity_factor)


def kill_switch(observation, dynamics_func):
    if punish_limit(observation, dynamics_func) > 0:
        return False
    return True


def calculate_q_values(reward, gamma):
    actual_Q = deque()
    for r in reversed(reward):
        if actual_Q:
            q_value = r + actual_Q[0] * gamma
        else:
            q_value = r / (1 - gamma)
        actual_Q.appendleft(q_value)
    return np.array(actual_Q)


def get_e_decay(x, x_max, factor=5):
    c = np.exp(-factor)
    return np.clip((np.exp(-x/x_max*factor) - c)/(1 - c), 0, 1)


def get_i_decay(x, factor=4):
    return 1/(factor * x + 1)


def get_unscaled_action(observation_dict, t_minus=0, key='U_real'):
    unscaled_action = observation_dict['dynamics_func'].unscale_action(np.array([observation_dict[key][t_minus-1]]))
    max_value_index = np.argmax(np.abs(unscaled_action))
    max_action_value = unscaled_action[max_value_index]
    return max_action_value


def get_state_values(observation_dict, key='X_meas'):
    l = observation_dict['mpar'].l
    action_key = 'U_con'
    if key == 'X_real':
        l = observation_dict['dynamics_func'].plant.l
        action_key = 'real'

    dt_goal = 0.05
    threshold_distance = 0.01

    unscaled_observation = observation_dict['dynamics_func'].unscale_state(observation_dict[key][-1])
    unscaled_action = get_unscaled_action(observation_dict, 0, action_key)

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
    distance = np.linalg.norm(x3 - goal)

    u_p, u_pp = 0, 0
    if len(observation_dict['U_con']) > 1:
        dt = observation_dict['T'][-1] - observation_dict['T'][-2]
        u_p = (unscaled_action - get_unscaled_action(observation_dict, -1, action_key)) / dt
        if len(observation_dict['U_con']) > 2:
            u_pp = (unscaled_action - 2 * get_unscaled_action(observation_dict, -1, action_key) + get_unscaled_action(observation_dict, -2, action_key))/(dt * dt)

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
