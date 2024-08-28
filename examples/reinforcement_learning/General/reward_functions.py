import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_diff

from examples.reinforcement_learning.General.misc_helper import punish_limit, get_state_values, get_i_decay, \
    get_unscaled_action
from examples.reinforcement_learning.General.score import calculate_score


def pos_reward(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict, 'X_real')
    reward = get_i_decay(state_values['distance']) - get_i_decay(2)
    reward = reward * state_values['s2']
    return reward * np.min(punish_limit(observation_dict['X_meas'][-1], action, observation_dict['dynamics_func']))


def angle_distance(state_values):
    goal = [np.pi, 0]
    diff = state_values['y'][:2] - goal
    diff = wrap_angles_diff(diff)

    dist = np.dot(diff.T, diff)

    return dist * 0.05


def r1(observation_dict, state_values):
    d = effort_distance(observation_dict, state_values) + angle_distance(state_values) # + energy_distance(observation_dict, state_values)
    return -d


def energy_distance(observation_dict, state_values):
    Ekin = observation_dict['dynamics_func'].simulator.plant.kinetic_energy(state_values['y'])
    Epot = observation_dict['dynamics_func'].simulator.plant.potential_energy(state_values['y'])
    Etot = Ekin + Epot
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    Epot_goal = observation_dict['dynamics_func'].simulator.plant.potential_energy(goal)
    return (Epot_goal - Etot) ** 2 * 0.005


def f1(state_values):
    f = 1
    v = state_values['v1'][1]
    if state_values['distance'] > 0.5:
        f = 1-1/(1+np.exp(-10*(v + 0.2)))
    return f


def f2(state_values):
    y = state_values['y']
    y[1] += y[0]
    y = wrap_angles_diff(y)
    f = np.abs(np.sin(y[0]/2)) * np.abs(np.sin(y[1]/2))
    return f


def effort_distance(observation_dict, state_values):
    du = 0
    if len(observation_dict['U_con']) > 2:
        du = np.abs((observation_dict['U_con'][-1] - observation_dict['U_con'][-2]) / observation_dict['dynamics_func'].dt)

    velocity = state_values['omega_squared_1'] + state_values['omega_squared_2']
    torque = state_values['unscaled_action'] ** 2 + np.abs(state_values['unscaled_action']) * 2
    smoothness = du * observation_dict['dynamics_func'].torque_limit[0]
    i = 2
    if observation_dict['dynamics_func'].robot == 'acrobot':
        i = 3
    energy = np.abs(state_values['y'][i] * state_values['unscaled_action'])
    abstract_distance = 0.003 * velocity + 0.1 * torque + 0.01 * smoothness + 0.02 * energy

    return abstract_distance * 0.5


def future_pos_reward(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict, 'X_real')
    # score = calculate_score(observation_dict, needs_success=False)
    # abstract_distance = (state_values['omega_squared_1'] + state_values['omega_squared_2']) / 400.0 + (state_values['unscaled_action'] ** 2) / 20.0
    # print(abstract_distance)
    # reward = r3(observation_dict, state_values) + score * 4
    reward = r1(observation_dict, state_values)
    return reward + (np.min(punish_limit(observation_dict['X_meas'][-1], action, observation_dict['dynamics_func'])) - 1)


def exp_distance_from_target(observation, action, env_type, dynamic_func, observation_dict):
    u = dynamic_func.unscale_action(action)

    x = dynamic_func.unscale_state(observation)

    goal = [np.pi, 0]
    diff = x[:2] - goal
    diff = wrap_angles_diff(diff)

    sat_dist = np.dot(diff.T, diff)
    exp_indx = -sat_dist - np.linalg.norm(u)

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


def like_lqr(observation, action, env_type, dynamic_func, observation_dict):
    state_values = get_state_values(observation_dict, 'X_real')

    goal = np.array([np.pi, 0., 0., 0.])

    Q = np.zeros((4, 4))   #TODO learn Q and R with Bayesian Inference?
    Q[0, 0] = 50.0 #weight for state[0] etc
    Q[1, 1] = 50.0
    Q[2, 2] = 4
    Q[3, 3] = 2

    # penalty for actuation
    R = np.array([[1]])

    diff = state_values['y'] - goal

    # "control" input penalty
    u = action  # TODO is reward based on last action beneficial? Rather just depend on state

    # quadratic cost for u, quadratic cost for state
    cost1 = np.einsum("i, ij, j", diff, Q, diff) +  u**2 * R[0,0]#np.einsum("i, ij, j", u, R, u)
    #TODO: try including LQR gain Matrix K into u --> Static Information enough?
    #TODO: try including information about the energy
    return -0.001 * cost1