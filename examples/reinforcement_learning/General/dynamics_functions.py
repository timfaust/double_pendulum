import time

import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters

class PushDoublePendulum(SymbolicDoublePendulum):
    def __int__(self, model_pars):
        super().__init__(model_pars=model_pars)

    def forward_dynamics(self, x, u):
        vel = np.copy(x[self.dof:])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        Minv = np.linalg.inv(M)

        force = np.dot(self.B, u) - np.dot(C, vel) + G - F

        test_1 = x[0]/np.pi - 1
        test_2 = x[1]/np.pi - 2
        while test_1 < -1:
            test_1 += 2
        while test_2 < -1:
            test_2 += 2

        if abs(test_1) < 0.1 and abs(test_2) < 0.1:
            if np.random.uniform(0, 1) < 1.0/200.0:
                f = np.random.uniform(2, 10) * np.random.choice([-1, 1])
                force += np.array([(self.l[0] + self.l[1])*f, self.l[1] * f])

        accn = Minv.dot(force)
        return accn


def load_param(robot, torque_limit=5.0):
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"
        torque_array = [torque_limit, 0.0]

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"
        torque_array = [0.0, torque_limit]

    model_par_path = (
            "../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0., 0.])
    mpar.set_cfric([0., 0.])

    return mpar


def random_dynamics(robot, sigma=0.02, plant_class=SymbolicDoublePendulum):
    start = time.time()
    mpar = load_param(robot)
    mpar.g = np.random.normal(mpar.g, sigma)
    mpar.m = np.random.normal(mpar.m, sigma)
    mpar.l = np.random.normal(mpar.l, sigma).tolist()
    mpar.cf = abs(np.random.normal(mpar.cf, sigma)).tolist()
    plant = plant_class(model_pars=mpar)
    print(time.time() - start)
    return general_dynamics(robot, plant)


def push_dynamics(robot):
    mpar = load_param(robot)
    plant = PushDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant)


def random_push_dynamics(robot, sigma=0.02):
    return random_dynamics(robot, sigma, PushDoublePendulum)


def default_dynamics(robot):
    mpar = load_param(robot)
    plant = SymbolicDoublePendulum(model_pars=mpar)
    return general_dynamics(robot, plant)


def general_dynamics(robot, plant):
    simulator = Simulator(plant=plant)
    dynamics_function = double_pendulum_dynamics_func(
        simulator=simulator,
        robot=robot,
        dt=0.01,
        integrator="runge_kutta",
        max_velocity=20.0,
        torque_limit=[5.0, 5.0],
        scaling=True
    )
    return dynamics_function, simulator, plant
