import numpy as np

from double_pendulum.model.model_parameters import model_parameters

design = "design_C.1"
model = "model_1.0"
robot = "pendubot"

model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

dt = 0.002
t0 = 0.0
t_final = 10.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]
