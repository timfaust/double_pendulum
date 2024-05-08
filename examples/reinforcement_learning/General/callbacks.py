import json
import re
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from examples.reinforcement_learning.General.reward_functions import calculate_score
import os

class ScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ScoreCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.training_env.get_attr('state_dict')[0]['T']) == self.training_env.get_attr('max_episode_steps')[0] - 1:
            sum = 0
            state_dicts = self.training_env.get_attr('state_dict')
            for state_dict in state_dicts:
                sum += calculate_score(state_dict)
            self.logger.record("rollout/score_mean", sum/len(state_dicts))
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_steps, log_dir, data, n_envs):
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_steps = total_steps
        self.log_dir = log_dir
        self.data = data
        self.n_envs = n_envs

    def find_next_log_dir(self):
        tb_log_dir = os.path.join(self.log_dir, "tb_logs")
        os.makedirs(tb_log_dir, exist_ok=True)

        sac_dirs = [d for d in os.listdir(tb_log_dir) if re.match(r'SAC_\d+', d)]
        highest_number = 0
        for d in sac_dirs:
            num = int(d.split('_')[-1])
            highest_number = max(highest_number, num)

        next_sac_dir = f"SAC_{highest_number}"
        return os.path.join(tb_log_dir, next_sac_dir)

    def _on_training_start(self):
        sac_log_dir = self.find_next_log_dir()
        self.pbar = tqdm(total=self.total_steps, desc='Training Progress')
        with SummaryWriter(sac_log_dir) as writer:
            config_str = json.dumps(self.data, indent=4)
            writer.add_text("Configuration", f"```json\n{config_str}\n```", 0)

    def _on_step(self):
        self.pbar.update(self.n_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()
