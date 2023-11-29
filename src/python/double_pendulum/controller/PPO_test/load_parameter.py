from double_pendulum.model.model_parameters import model_parameters
"""
param: Robot: String robot name
        torque_limit = np.array(1,2)
"""

def load_param(robot="acrobot", torque_limit=[0, 5.0]):
    # model parameter
    if robot == "pendubot":
        design = "design_A.0"
        model = "model_2.0"

    elif robot == "acrobot":
        design = "design_C.0"
        model = "model_3.0"

    model_par_path = (
        "../../../data/system_identification/identified_parameters/"
        + design
        + "/"
        + model
        + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_limit)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0., 0.])
    mpar.set_cfric([0., 0.])

    return mpar
