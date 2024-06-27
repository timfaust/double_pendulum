import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff

from examples.reinforcement_learning.General.misc_helper import punish_limit
from examples.reinforcement_learning.General.score import get_score

def get_e_decay(x, x_max, factor=5):
    c = np.exp(-factor)
    return np.clip((np.exp(-x/x_max*factor) - c)/(1 - c), 0, 1)

def get_i_decay(x, factor=4):
    return 1/(factor * x + 1)

def get_unscaled_action(observation_dict, t_minus=0):
    unscaled_action = observation_dict['dynamics_func'].unscale_action(np.array([observation_dict['U_con'][t_minus-1]]))
    max_value_index = np.argmax(np.abs(unscaled_action))
    max_action_value = unscaled_action[max_value_index]
    return max_action_value


def get_state_values(observation_dict, key='X_meas'):
    l = [0.2, 0.3]
    dt_goal = 0.05
    threshold_distance = 0.01

    unscaled_observation = observation_dict['dynamics_func'].unscale_state(observation_dict[key][-1])
    unscaled_action = get_unscaled_action(observation_dict)

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
    goal = np.array([0, -0.5])

    x3 = x2 + dt_goal * v2
    distance = np.linalg.norm(x3 - np.array([0, -0.5]))

    u_p, u_pp = 0, 0
    if len(observation_dict['U_con']) > 1:
        dt = observation_dict['T'][-1] - observation_dict['T'][-2]
        u_p = (unscaled_action - get_unscaled_action(observation_dict, -1)) / dt
        if len(observation_dict['U_con']) > 2:
            u_pp = (unscaled_action - 2 * get_unscaled_action(observation_dict, -1) + get_unscaled_action(observation_dict, -2))/(dt * dt)

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


def score_reward(observation, action, env_type, dynamic_func, observation_dict):
    return get_score(observation_dict) * int(observation_dict["max_episode_steps"])


def future_pos_reward(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict, 'X_real')
    reward = get_i_decay(state_values['distance'], 4)
    # reward = get_e_decay(distance, 1)
    if (state_values['x3'] - state_values['goal'])[1] < state_values['threshold_distance']:
        abstract_distance = np.linalg.norm(state_values['v1']) + np.linalg.norm(state_values['v2']) #+ np.linalg.norm(action)/100# + np.linalg.norm(u_p)/10
        reward += get_i_decay(abstract_distance, 4)
        # reward += get_e_decay(abstract_distance, 10)
    return reward * punish_limit(state_values['unscaled_observation'], observation_dict['dynamics_func'])


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
