from stable_baselines3 import SAC

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)

robot = "acrobot"

"""from sim_parameters import (
    mpar,
    dt,
    t_final,
    t0,
    x0,
    goal,
    integrator,
    design,
    model,
    robot,
)"""

name = "general_sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "SAC LQR",
    "short_description": "Swing-up with an RL Policy learned with SAC.",
    "readme_path": f"readmes/{name}.md",
    "username": "chiniklas",
    }

def general_dynamics(robot):
    dynamics_function = double_pendulum_dynamics_func(
        simulator=None,
        robot=robot,
        dt=0.01,
        integrator="runge_kutta",
        max_velocity=20.0,
        torque_limit=[5.0, 5.0],
        scaling=True
    )
    return dynamics_function

class GeneralController(AbstractController):
    def __init__(self, dynamics_func, model_path):
        super().__init__()

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.model.predict([0.0, 0.0, 0.0, 0.0])
        self.dt = dynamics_func.dt
        self.scaling = dynamics_func.scaling
        self.simulation = None
        self.integrator = dynamics_func.integrator

    def get_control_output_(self, x, t=None):
        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            action = self.model.predict(obs)
            u = self.dynamics_func.unscale_action(action)
        else:
            action = self.model.predict(x)
            u = self.dynamics_func.unscale_action(action)

        return u

model_path = "../../../examples/reinforcement_learning/General/log_data/SAC_MLP_4/" + robot +"/best_model/best_model.zip"
dynamics_function = general_dynamics(robot)
controller = GeneralController(dynamics_func=dynamics_function, model_path=model_path)

controller.init()
