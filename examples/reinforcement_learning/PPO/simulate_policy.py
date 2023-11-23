import numpy as np
from examples.reinforcement_learning.PPO import helpers
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.utils.plotting import plot_timeseries


log_dir = "./log_data/PPO_training"

# hyperparam simulation
robot = "acrobot"

dt = 0.002
integrator = "runge_kutta"
t_final = 10.0
goal = [np.pi, 0.0, 0.0, 0.0]

mpar, torque_limit, active_act = helpers.load_param(robot=robot)

dynamics_function, sim = helpers.get_dynamics_function(mpar, robot, dt=dt, integrator=integrator)
reward_func = helpers.reward_func
terminated_func = helpers.terminated_func
noisy_reset_func = helpers.noisy_reset_func

# initialize sac controller

model_path = log_dir + "/best_model/best_model.zip"
controller = SACController(
    model_path = model_path,
    dynamics_func=dynamics_function,
    dt=dt,
    scaling=False
)
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

# plot time series
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)
