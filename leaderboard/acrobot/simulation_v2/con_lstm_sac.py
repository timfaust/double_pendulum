import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from examples.reinforcement_learning.General.misc_helper import default_decider
from examples.reinforcement_learning.General.override_sb3.common import MultiplePoliciesReplayBuffer
from examples.reinforcement_learning.General.override_sb3.sequence_policy import SequenceSACPolicy
from examples.reinforcement_learning.General.trainer import Trainer

name = "lstm_sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "LSTM SAC",
    "short_description": "SAC using custom model architecture",
    "readme_path": f"readmes/{name}.md",
    "username": "tfaust",
}

sac = Trainer("default_6", "acrobot", "default", [SequenceSACPolicy], [MultiplePoliciesReplayBuffer], [default_decider], 42, None)
controller = sac.get_controller("/saved_model/saved_model_1300000_steps")
controller.init()

