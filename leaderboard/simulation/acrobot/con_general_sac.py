from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.trainer import GeneralController

from sim_parameters import (
    robot,
)

name = "general_sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "General SAC",
    "short_description": "Swing-up with an RL Policy learned with SAC.",
    "readme_path": f"readmes/{name}.md",
    "username": "erfan",
    }

model_path = "../../../examples/reinforcement_learning/General/log_data/debug/" + robot +"/best_model/best_model.zip"
controller = GeneralController(GeneralEnv(robot, "test"), model_path=model_path)

controller.init()
