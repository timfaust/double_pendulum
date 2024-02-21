import argparse
import os
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
import json
from dynamics_functions import *
from controller import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default="real_Robot_pendu_4state_8")
    parser.add_argument('--model_path', default="/best_model/best_model")
    parser.add_argument('--mode', default="run", choices=["run", "connect"])
    parser.add_argument('--env_type', default="pendubot", choices=["pendubot", "acrobot"])
    parser.add_argument('--param', default="real_robot_1")
    parser.add_argument('--design', default="design_C.1")

    args = parser.parse_args()
    env_type = args.env_type
    design = args.design
    model_full_path = "best_models/" + args.model_name + "/" + env_type + "/" + args.model_path

    data = json.load(open("parameters.json"))[args.param]

    dynamic_class = globals()[data["dynamic_class"]]
    dt = data["dt"]
    max_torque = data["max_torque"]
    actions_in_state = data["actions_in_state"]

    dynamics_function = general_dynamics(env_type, dt, max_torque, dynamic_class)
    controller = GeneralController(dynamics_function, model_full_path, actions_in_state, args.mode)

    controller.init()

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

