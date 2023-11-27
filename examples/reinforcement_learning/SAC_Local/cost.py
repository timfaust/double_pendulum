import numpy as np

from src.python.double_pendulum.utils.wrap_angles import wrap_angles_diff
from examples.reinforcement_learning.SAC_Local.unified_params import *

def unscale_action(action):
    """
    scale the action
    [-1, 1] -> [-limit, +limit]
    """
    if robot == "double_pendulum":
        a = [
            float(torque_limit[0] * action[0]),
            float(torque_limit[1] * action[1]),
        ]
    elif robot == "pendubot":
        a = np.array([float(torque_limit[0] * action[0]), 0.0])
    elif robot == "acrobot":
        a = np.array([0.0, float(torque_limit[1] * action[0])])
    return a

def unscale_state(observation):
    """
    scale the state
    [-1, 1] -> [-limit, +limit]
    """
    x = np.array(
        [
            observation[0] * np.pi + np.pi,
            observation[1] * np.pi + np.pi,
            observation[2] * max_velocity,
            observation[3] * max_velocity,
        ]
    )
    return x

def reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0]
    return observation

def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation

def check_if_state_in_roa(x, ignore_vel=False):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    if ignore_vel:
        xdiff[2:] = 0
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad

def simple_reward_func(observation, action):
    control_line = 0.4

    flag = False
    bonus = False

    Q = np.zeros((4, 4))
    Q[0, 0] = 10
    Q[1, 1] = 10
    Q[2, 2] = 0.4
    Q[3, 3] = 0.3
    R = np.array([[0.0001]])

    u = unscale_action(action)
    s = unscale_state(observation)

    y = wrap_angles_diff(s)
    # openAI
    p1 = y[0]  # joint 1 pos
    p2 = y[1]  # joint 2 pos

    ee1_pos_x = mpar.l[0] * np.sin(p1)
    ee1_pos_y = mpar.l[0] * np.cos(p1)

    ee2_pos_x = ee1_pos_x + mpar.l[1] * np.sin(p1 + p2)
    ee2_pos_y = ee1_pos_y - mpar.l[1] * np.cos(p1 + p2)

    # criteria 4
    if ee2_pos_y >= control_line:
        flag = True
    else:
        flag = False

    bonus, rad = check_if_state_in_roa(y)

    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r
    if flag:
        reward += 100
        if bonus:
            reward += 1e3
    else:
        reward = reward

    return reward

def terminated_func(observation):
    s = unscale_state(observation)
    y = wrap_angles_diff(s)
    bonus, rad = check_if_state_in_roa(y)

    if termination:
        if bonus:
            print("terminated")
            return bonus
    else:
        return False

def saturated_distance_from_target(observation):
    s = unscale_state(observation)

    input_state = wrap_angles_diff(s)[:2]
    target_state = goal[:2]

    diff = input_state - target_state

    sigma_c = np.diag([1/mpar.l[0], 1/mpar.l[1]])
    exp_term = np.exp(-np.dot(np.dot(diff.T, sigma_c), diff))

    squared_dist = 1.0 - exp_term

    return squared_dist

def saturated_reward_func(observation, action):
    u = unscale_action(action)
    R = np.array([[0.0001]])

    r = saturated_distance_from_target(observation) + np.einsum("i, ij, j", u, R, u)
    print(-r)
    return -r

