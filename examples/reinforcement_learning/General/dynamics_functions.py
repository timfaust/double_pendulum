from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters


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


def default_dynamics(robot):
    mpar = load_param(robot)
    print("create plant")
    plant = SymbolicDoublePendulum(model_pars=mpar)
    print("plant created")
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
