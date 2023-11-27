import os
import numpy as np
from src.python.double_pendulum.model.model_parameters import model_parameters
from src.python.double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from src.python.double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from src.python.double_pendulum.simulation.simulation import Simulator


#===================================================== Helper Methods ===========================================================================================#

def load_param(robot="acrobot", torque_limit=5.0):
    """
    param: robot: String robot name
            torque_limit = float
    """
    # model parameter
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"
        torque_array = [torque_limit, 0.0]
        active_act = 0

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"
        torque_array = [0.0, torque_limit]
        active_act = 1

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

    return mpar, torque_array, active_act


def load_lqr_param(robot="acrobot"):
    load_path = "../../../data/controller_parameters/design_C.1/model_1.1/"
    load_path = load_path + robot + "/lqr/"

    # import lqr parameters
    rho = np.loadtxt(os.path.join(load_path, "rho"))
    vol = np.loadtxt(os.path.join(load_path, "vol"))
    S = np.loadtxt(os.path.join(load_path, "Smatrix"))

    return rho, vol, S


def get_dynamics_function(mpar, robot, dt=0.01, integrator="runge_kutta", torque_limit=[5.0, 5.0], max_velocity=8.0,
                          scaling=True):
    plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant)
    dynamics_function = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=2,
        torque_limit=torque_limit,
        max_velocity=max_velocity
    )
    return dynamics_function, simulator

#===================================================== Params ===========================================================================================#

log_dir = "./log_data/SAC_training"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# hyperparam baseline

training_steps = 1e8
learning_rate = 0.01
verbose = True
n_envs = 1
reward_threshold = 3e7  # reward for best model
eval_freq = 500  # every 5000 steps
n_eval_episodes = 5  # do 5 evaluation sequences
max_episode_steps = 200  # with 200 steps

# hyperparam simulation
robot = "pendubot"
best_model_path = log_dir + "/best_model/" + robot + "/best_model"
saved_model_path = log_dir + "/saved_model/" + robot

mpar, torque_limit, active_act = load_param(robot=robot)
rho, vol, S = load_lqr_param(robot=robot)

termination = False
scaling = True
dt = 0.002
integrator = "runge_kutta"
t_final = 10.0
max_velocity = 8.0
goal = [np.pi, 0.0, 0.0, 0.0]
