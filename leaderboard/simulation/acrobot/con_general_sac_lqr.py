from stable_baselines3 import SAC
import numpy as np
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.controller.lqr.lqr_controller import LQRController, LQRController_nonsymbolic
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)

"""robot = "acrobot"
mpar = None
goal = [np.pi, 0.0, 0.0, 0.0]"""

from sim_parameters import (
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
)

name = "general_sac_lqr"
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


def condition1(t, x):
    return False


def condition2(t, x):
    y = wrap_angles_top(x)
    goal = [np.pi, 0]
    diff = y[:2] - goal
    return np.linalg.norm(diff) < 0.1

model_path = "../../../examples/reinforcement_learning/General/log_data/SAC_MLP_4/" + robot +"/best_model/best_model.zip"
dynamics_function = general_dynamics(robot)
controller1 = GeneralController(dynamics_func=dynamics_function, model_path=model_path)

controller2 = stabilization_controller = LQRController_nonsymbolic(model_pars=mpar)
controller2.set_cost_parameters(u2u2_cost=100, p1p1_cost=100, p2p2_cost=100)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=10**3)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
    verbose=True
)

controller.set_goal(goal)
controller.init()
