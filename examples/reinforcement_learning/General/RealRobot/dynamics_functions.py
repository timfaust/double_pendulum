import numpy as np
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters

def load_param(torque_limit):
    torque_array = [torque_limit, torque_limit]
    model_par_path = "model_data/model_parameters.yml"
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    return mpar

def general_dynamics(robot, dt, max_torque, class_obj):
    print("build new plant")
    max_vel = 20.0

    dynamics_function = class_obj(
        simulator=None,
        robot=robot,
        dt=dt,
        integrator="runge_kutta",
        max_velocity=max_vel,
        torque_limit=[max_torque, max_torque],
        scaling=True
    )
    return dynamics_function


class custom_dynamics_func_4PI(double_pendulum_dynamics_func):

    def unscale_state(self, observation):
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * 2 * np.pi + np.pi,
                    observation[1] * 2 * np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = super().unscale_state(observation)
        if len(observation) > 4:
            return np.append(x, observation[-2:]*self.torque_limit)
        return x

    def normalize_state(self, state):
        if self.state_representation == 2:
            observation = np.array(
                [
                    ((state[0] + np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),
                    ((state[1] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi) / (2 * np.pi),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = super().normalize_state(state)
        if len(state) > 4:
            return np.append(observation, state[-2:]/self.torque_limit)
        return observation

class custom_dynamics_func_PI(double_pendulum_dynamics_func):
    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        else:
            x = super().unscale_state(observation)
        if len(observation) > 4:
            return np.append(x, observation[-2:]*self.torque_limit)
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
            observation = np.array(
                [
                    (state[0] % (2 * np.pi) - np.pi) / np.pi,
                    ((state[1] - np.pi) % (2 * np.pi) - np.pi) / np.pi,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        else:
            observation = super().normalize_state(state)

        if len(state) > 4:
            return np.append(observation, state[-2:]/self.torque_limit)
        return observation
