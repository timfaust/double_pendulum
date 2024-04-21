from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.trainer import GeneralController

from sim_parameters import (
    robot,
)

name = "6states_fut"
controller_name = "general_sac"

leaderboard_config = {
    "csv_path": controller_name + "/sim_swingup.csv",
    "name": controller_name,
    "simple_name": "General SAC",
    "short_description": "Swing-up with an RL Policy learned with SAC.",
    "readme_path": f"readmes/{controller_name}.md",
    "username": "erfan",
    }

model_path = "../../../examples/reinforcement_learning/General/log_data/" + name + "/" + robot + "/best_model/best_model.zip"
controller = GeneralController(GeneralEnv(robot, "test", "../../../examples/reinforcement_learning/General/parameters.json"), model_path=model_path)

controller.init()
