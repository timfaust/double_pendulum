from reward_functions import *
from dynamics_functions import *
from misc_helper import *
import json

def load_env_attributes(data):

    dynamics_function = globals()[data["dynamics_function"]]
    reset_function = globals()[data["reset_function"]]
    reward_function = globals()[data["reward_function"]]
    number_of_envs = data["n_envs"]
    use_same_env = data["same_env"] == 1

    return dynamics_function, reset_function, reward_function, number_of_envs, use_same_env


def load_json_params(param_name, path="parameters.json"):

    return json.load(open(path))[param_name]



