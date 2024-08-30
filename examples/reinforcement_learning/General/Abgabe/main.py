import argparse
import os
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from dynamics_functions import *
from controller import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default="test_2")
    parser.add_argument('--model_path', default="../log_data/")
    parser.add_argument('--mode', default="simulate", choices=["run", "simulate"])
    parser.add_argument('--env_type', default="acrobot", choices=["pendubot", "acrobot"])
    parser.add_argument('--design', default="design_C.1")

    args = parser.parse_args()
    env_type = args.env_type
    design = args.design

    save_path = args.model_path + args.model_name + "/" + args.env_type
    model_path = save_path + "/best_model/best_score"

    dt = 0.02
    max_torque = 6

    dynamics_function = general_dynamics(env_type, dt, max_torque)
    print(model_path)
    controller = GeneralController(dynamics_function, model_path=model_path, torque_limit=max_torque)

    controller.init()

    if args.mode == "run":
        # run experiment
        run_experiment(
            controller=controller,
            dt=dt,
            t_final=10.0,
            can_port="can0",
            motor_ids=[1, 2],
            tau_limit=max_torque,
            save_dir=os.path.join("data", design, env_type, args.model_name),
        )
    elif args.mode == "simulate":

        controller.simulator.set_state(0, [0, 0, 0, 0])

        T, X, U = controller.simulator.simulate_and_animate(
            t0=0.0,
            x0=[0.0, 0.0, 0.0, 0.0],
            tf=10.0,
            dt=controller.dt * 0.1,
            controller=controller,
            integrator=controller.integrator,
            save_video=True,
            video_name=os.path.join(save_path, "sim_video.gif"),
            scale=0.25
        )

        save_trajectory(os.path.join(save_path, "sim_swingup.csv"), T=T, X_meas=X, U_con=U)

        plot_timeseries(
            T,
            X,
            U,
            X_meas=controller.simulator.meas_x_values,
            pos_y_lines=[-np.pi, 0.0, np.pi],
            vel_y_lines=[0.0],
            tau_y_lines=[-5.0, 0.0, 5.0],
            save_to=os.path.join(save_path, "timeseries"),
            show=False,
            scale=0.5,
        )
