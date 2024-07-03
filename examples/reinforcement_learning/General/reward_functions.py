import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff

from examples.reinforcement_learning.General.misc_helper import punish_limit, get_state_values, get_i_decay


def future_pos_reward(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict, 'X_real')
    reward = get_i_decay(state_values['distance'], 4)
    # reward = get_e_decay(distance, 1)
    if (state_values['x3'] - state_values['goal'])[1] < state_values['threshold_distance']:
        abstract_distance = np.linalg.norm(state_values['v1']) + np.linalg.norm(state_values['v2']) + np.linalg.norm(action)/10# + np.linalg.norm(u_p)/10
        reward += get_i_decay(abstract_distance, 4)
        # reward += get_e_decay(abstract_distance, 10)
    reward = reward * punish_limit(state_values['unscaled_observation'], observation_dict['dynamics_func'])
    return [reward, reward]


def pos_reward(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict)
    return 1 / (state_values['distance'] + 0.0001)


def saturated_distance_from_target(observation, action, env_type, dynamic_func, observation_dict):
    u = dynamic_func.unscale_action(action)
    u_diff = 0
    if len(observation) > 4:
        u_diff = action - observation[-2]

    x = dynamic_func.unscale_state(observation)

    goal = [np.pi, 0]
    diff = x[:2] - goal
    diff = wrap_angles_diff(diff)

    sat_dist = np.dot(diff.T, diff)
    exp_indx = -sat_dist - np.linalg.norm(u) - np.linalg.norm(u_diff)

    exp_term = np.exp(exp_indx)
    return exp_term


def quadratic_rew(observation, action, env_type, dynamic_func, observation_dict):
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

    state_values = get_state_values(observation_dict)

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
    cart_err_x2 = state_values['x2'] - cart_goal_x2

    cart_err_x1 = state_values['x1'] - cart_goal_x1

    Q2 = np.zeros((2,2))
    Q2[0, 0] = 10.0
    Q2[1, 1] = 10.0



    cost2= np.einsum("i, ij, j", cart_err_x2, Q2, cart_err_x2) + np.einsum("i, ij, j", cart_err_x1, Q2, cart_err_x1)

    reward = -1 * cost1 - 1 * cost2



    return reward
