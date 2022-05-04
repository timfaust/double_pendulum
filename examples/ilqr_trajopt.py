import os
import numpy as np
from datetime import datetime
import yaml
import time

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController

robot = "acrobot"

# # model parameters
# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# damping = [0.081, 0.0]
# # damping = [0.0, 0.0]
# # cfric = [0.093, 0.186]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.02522]
# torque_limit = [0.0, 6.0]

# model parameters
# mass = [0.608, 0.63]
# length = [0.3, 0.4]
# com = [length[0], length[1]]
# damping = [0.081, 0.081]
# damping = [0.0005, 0.0001]
# damping = [0., 0.]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
# gravity = 9.81
# inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
motor_inertia = 0.
if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
# mpar.set_damping(damping)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 1000
max_iter = 2000
regu_init = 100.
max_regu = 1000000.
min_regu = 0.01
break_cost_redu = 1e-6

# simulation parameter
dt = 0.005
# t_final = 5.0
t_final = N*dt
integrator = "runge_kutta"

if robot == "acrobot":
    # looking good
    # [9.97938814e-02 2.06969312e-02 7.69967729e-02 1.55726136e-04
    #  5.42226523e-03 3.82623819e+02 7.05315590e+03 5.89790058e+01
    #  9.01459500e+01]
    # sCu = [9.97938814e-02, 9.97938814e-02]
    # sCp = [2.06969312e-01, 7.69967729e-01]
    # sCv = [1.55726136e-03, 5.42226523e-02]
    # sCen = 0.0
    # fCp = [3.82623819e+03, 7.05315590e+04]
    # fCv = [5.89790058e+01, 9.01459500e+01]
    # fCen = 0.0

    # # very good
    # sCu = [9.97938814e-02, 9.97938814e-02]
    # sCp = [2.06969312e-02, 7.69967729e-02]
    # sCv = [1.55726136e-04, 5.42226523e-03]
    # sCen = 0.0
    # fCp = [3.82623819e+02, 7.05315590e+03]
    # fCv = [5.89790058e+01, 9.01459500e+01]
    # fCen = 0.0
    sCu = [9.97938814e+01, 9.97938814e+01]
    sCp = [2.06969312e+01, 7.69967729e+01]
    sCv = [1.55726136e-01, 5.42226523e-00]
    sCen = 0.0
    fCp = [3.82623819e+02, 7.05315590e+03]
    fCv = [5.89790058e+01, 9.01459500e+01]
    fCen = 0.0

    # sCu = [100., 100.]
    # sCp = [0.1, 0.1]
    # sCv = [0.5, 0.5]
    # sCen = 0.
    # fCp = [50000., 50000.]
    # fCv = [2000., 2000.]
    # fCen = 0.

    # sCu = [60., 60.]
    # sCp = [3.6, 3.6]
    # sCv = [50., 50.]
    # sCen = 0.
    # fCp = [20000., 35000.]
    # fCv = [3651., 2068.]
    # fCen = 0.


    # [8.26303186e+01 2.64981012e+01 3.90215591e+01 3.87432205e+00
    #  2.47715889e+00 5.72238144e+04 9.99737172e+04 7.16184205e+03
    #  2.94688061e+03]
    # sCu = [89., 89.]
    # sCp = [40., 0.2]
    # sCv = [11., 1.0]
    # sCen = 0.0
    # fCp = [66000., 210000.]
    # fCv = [55000., 92000.]
    # fCen = 0.0

    # sCu = [89.53168298604868, 89.53168298604868]
    # sCp = [39.95840603845028, 0.220281011195961]
    # sCv = [10.853380829038803, 0.9882211066793491]
    # sCen = 0.
    # fCp = [65596.70698843336, 208226.67812877183]
    # fCv = [54863.83385207141, 91745.39489510724]
    # fCen = 0.
    # looking good
    # [9.96090757e-02 2.55362809e-02 9.65397113e-02 2.17121720e-05
    #  6.80616778e-03 2.56167942e+02 7.31751057e+03 9.88563736e+01
    #  9.67149494e+01]
    # sCu = [9.96090757e-02, 9.96090757e-02]
    # sCp = [2.55362809e-02, 9.65397113e-02]
    # sCv = [2.17121720e-05, 6.80616778e-03]
    # sCen = 0.0
    # fCp = [2.56167942e+02, 7.31751057e+03]
    # fCv = [9.88563736e+01, 9.67149494e+01]
    # fCen = 0.0

    # sCu = [0.2, 0.2]
    # sCp = [0.1, 0.2]
    # sCv = [0.2, 0.2]
    # sCen = 0.
    # fCp = [1500., 500.]
    # fCv = [10., 10.]
    # fCen = 0.

    # [9.64008003e-04 3.69465206e-04 9.00160028e-04 8.52634075e-04
    #  3.62146682e-05 3.49079107e-04 1.08953921e-05 9.88671633e+03
    #  7.27311031e+03 6.75242351e+01 9.95354381e+01 6.36798375e+01]

    # [6.83275883e-01 8.98205799e-03 5.94690881e-04 2.60169706e-03
    # 4.84307636e-03 7.78152311e-03 2.69548072e+02 9.99254272e+03
    # 8.55215256e+02 2.50563565e+02 2.57191000e+01]
if robot == "pendubot":
    sCu = [0.2, 0.2]
    sCp = [0.1, 0.2]
    sCv = [0., 0.]
    sCen = 0.
    fCp = [2500., 500.]
    fCv = [100., 100.]
    fCen = 0.

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

t0 = time.time()
il = ilqr_calculator()
# il.set_model_parameters(mass=mass,
#                         length=length,
#                         com=com,
#                         damping=damping,
#                         gravity=gravity,
#                         coulomb_fric=cfric,
#                         inertia=inertia,
#                         torque_limit=torque_limit)
il.set_model_parameters(model_pars=mpar)
il.set_parameters(N=N,
                  dt=dt,
                  max_iter=max_iter,
                  regu_init=regu_init,
                  max_regu=max_regu,
                  min_regu=min_regu,
                  break_cost_redu=break_cost_redu,
                  integrator=integrator)
il.set_cost_parameters(sCu=sCu,
                       sCp=sCp,
                       sCv=sCv,
                       sCen=sCen,
                       fCp=fCp,
                       fCv=fCv,
                       fCen=fCen)
il.set_start(start)
il.set_goal(goal)

# computing the trajectory
T, X, U = il.compute_trajectory()
print("Computing time: ", time.time() - t0, "s")

# saving

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "trajopt", timestamp)
os.makedirs(save_dir)

traj_file = os.path.join(save_dir, "trajectory.csv")

il.save_trajectory_csv()
os.system("mv trajectory.csv " + traj_file)
# save_trajectory(csv_path=filename,
#                 T=T, X=X, U=U)

par_dict = {
            # "mass1": mass[0],
            # "mass2": mass[1],
            # "length1": length[0],
            # "length2": length[1],
            # "com1": com[0],
            # "com2": com[1],
            # "inertia1": inertia[0],
            # "inertia2": inertia[1],
            # "damping1": damping[0],
            # "damping2": damping[1],
            # "coulomb_friction1": cfric[0],
            # "coulomb_friction2": cfric[1],
            # "gravity": gravity,
            # "torque_limit1": torque_limit[0],
            # "torque_limit2": torque_limit[1],
            "dt": dt,
            "t_final": t_final,
            "integrator": integrator,
            "start_pos1": start[0],
            "start_pos2": start[1],
            "start_vel1": start[2],
            "start_vel2": start[3],
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            "N": N,
            "max_iter": max_iter,
            "regu_init": regu_init,
            "max_regu": max_regu,
            "min_regu": min_regu,
            "break_cost_redu": break_cost_redu,
            "sCu1": sCu[0],
            "sCu2": sCu[1],
            "sCp1": sCp[0],
            "sCp2": sCp[1],
            "sCv1": sCv[0],
            "sCv2": sCv[1],
            "sCen": sCen,
            "fCp1": fCp[0],
            "fCp2": fCp[1],
            "fCv1": fCv[0],
            "fCv2": fCv[1],
            "fCen": fCen
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

# plotting
U = np.append(U, [[0.0, 0.0]], axis=0)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                save_to=os.path.join(save_dir, "timeseries"))

# simulation
# plant = SymbolicDoublePendulum(mass=mass,
#                                length=length,
#                                com=com,
#                                damping=damping,
#                                gravity=gravity,
#                                coulomb_fric=cfric,
#                                inertia=inertia,
#                                torque_limit=torque_limit)
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=traj_file,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, phase_plot=False,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   plot_inittraj=True)
