import numpy as np

from examples.reinforcement_learning.General.environments import GeneralEnv
from examples.reinforcement_learning.General.override_sb3.common import DefaultTranslator, DefaultActor, CustomPolicy, \
    DefaultCritic
from double_pendulum.utils.wrap_angles import wrap_angles_diff


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.clean_action_memory = None
        self.past_action_number = 200
        self.reset()
        super().__init__(21 + self.past_action_number)

    def build_state(self, env: GeneralEnv, dirty_observation: np.ndarray, clean_action: float, **kwargs) -> np.ndarray:
        observation = dirty_observation.copy()
        l = [0.2, 0.3]
        x = np.array(
            [
                observation[0] * 2 * np.pi + np.pi,
                observation[1] * 2 * np.pi,
                observation[2] * 20,
                observation[3] * 20,
            ]
        )

        y = wrap_angles_diff(x)
        s1 = np.sin(y[0])
        s2 = np.sin(y[0] + y[1])
        c1 = np.cos(y[0])
        c2 = np.cos(y[0] + y[1])

        x1 = np.array([s1, c1]) * l[0]
        x2 = x1 + np.array([s2, c2]) * l[1]

        # cartesian velocities of the joints
        v1 = np.array([c1, -s1]) * y[2] * l[0]
        v2 = v1 + np.array([c2, -s2]) * (y[2] + y[3]) * l[1]

        x3 = x2 + 0.05 * v2
        distance = np.linalg.norm(x3 - np.array([0, -0.5]))

        additional = np.array([x1[0], x1[1], x2[0], x2[1], v1[0], v1[1], v2[0], v2[1], s1, s2, c1, c2, x[2]**2, x[3]**2, x3[0], x3[1], distance])

        self.clean_action_memory = np.append(self.clean_action_memory, clean_action)
        state = np.append(dirty_observation.copy(), self.clean_action_memory[-self.past_action_number:])
        state = np.append(state, additional)
        return state

    def reset(self):
        self.clean_action_memory = np.zeros(self.past_action_number)


class PastActionsActor(DefaultActor):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PastActionsSACPolicy(CustomPolicy):
    actor_class = PastActionsActor

    def __init__(self, *args, **kwargs):
        self.additional_actor_kwargs['net_arch'] = [256, 128, 64, 32]
        self.additional_critic_kwargs['net_arch'] = self.additional_actor_kwargs['net_arch']
        super().__init__(*args, **kwargs)
