import numpy as np
from examples.reinforcement_learning.SAC_Local.sac_controller import SACController
from examples.reinforcement_learning.SAC_Local.unified_params import *
from double_pendulum.utils.plotting import plot_timeseries

dynamics_function, sim = get_dynamics_function(mpar, robot, torque_limit=torque_limit, scaling=scaling)

# initialize sac controller
controller = SACController(
    model_path=best_model_path,
    dynamics_func=dynamics_function,
    dt=dt
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
