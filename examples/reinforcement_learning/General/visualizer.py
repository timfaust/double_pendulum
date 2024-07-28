import pygame
import numpy as np
import torch as th
from examples.reinforcement_learning.General.score import calculate_score
from examples.reinforcement_learning.General.misc_helper import calculate_q_values, get_stabilized, get_i_decay
from examples.reinforcement_learning.General.reward_functions import get_state_values, r1, r2, f1, f2


class Visualizer:
    def __init__(self, env):
        self.acc_reward = 0
        self.reward = 0
        self.policy = 0
        self.pendulum_length_visualization = 700
        self.graph_window_width = 1500
        self.graph_window_height = 1500
        self.metrics_width = 3000
        self.full_window_width = self.graph_window_width + self.metrics_width
        self.window = None
        self.clock = None
        self.metadata_visualization = {"render_modes": ["human"], "render_fps": 144}
        self.env = env
        self.model = None
        self.past_scores = []
        self.predicted_Q = []
        self.used_policies = []

        # Pre-calculate some values
        self.graph_x = self.graph_window_width
        self.graph_width = (self.full_window_width - self.graph_window_width) // 2
        self.graph_height = self.graph_window_height // 2

        # Create surfaces once
        self.canvas = pygame.Surface((self.full_window_width, self.graph_window_height))
        self.graph_surface = pygame.Surface((self.full_window_width, self.graph_window_height), pygame.SRCALPHA)
        self.pendulum_surface = pygame.Surface((self.graph_window_width, self.graph_window_height), pygame.SRCALPHA)

        # Preload fonts
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 28)

    def reset(self):
        if len(self.past_scores) > 0:
            self.render()
        self.past_scores = []
        self.predicted_Q = []
        self.used_policies = []

    def init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.full_window_width, self.graph_window_height))
        self.clock = pygame.time.Clock()

    def render(self):
        if self.window is None or self.clock is None:
            self.init_pygame()

        self.canvas.fill((255, 255, 255))
        metrics = self.draw_environment()
        self.draw_graph()
        metrics['score'] = self.past_scores[-1] if self.past_scores else 0
        self.blit_texts(metrics)
        self.update_display()

    def draw_graph(self):
        reward_name = 'reward_0'
        gamma = 0.99

        if self.model is not None:
            with th.no_grad():
                device = self.model.critic.device
                actions = th.tensor(np.array([self.env.observation_dict['U_con'][-1]]), dtype=th.float32, device=device).unsqueeze(1)
                state, self.policy = self.model.get_last_state(self.env)
                critic = self.model.policies[self.policy].critic
                state = th.tensor(np.array([state]), dtype=th.float32, device=device)
                q_values, _ = th.min(th.cat(critic(state, actions), dim=1), dim=1, keepdim=True)
                self.predicted_Q.append(q_values.squeeze(1).cpu().numpy()[0])
                self.used_policies.append(self.policy)
                extracted_features = critic.extract_features(state, critic.features_extractor)[-1, :].cpu().numpy().tolist()

                gamma = self.model.gamma
                reward_index = min(self.policy, len(self.env.observation_dict) - 10)
                reward_name = f'reward_{reward_index}'

        self.graph_surface.fill((0, 0, 0, 0))

        dirty_actions = self.env.observation_dict['U_real'][1:]
        clean_actions = self.env.observation_dict['U_con'][1:]
        dirty_x = [x[1] for x in self.env.observation_dict['X_meas'][1:]]
        clean_x = [x[1] for x in self.env.observation_dict['X_real'][1:]]
        dirty_v = [x[3] for x in self.env.observation_dict['X_meas'][1:]]
        clean_v = [x[3] for x in self.env.observation_dict['X_real'][1:]]
        reward_history = np.array(self.env.observation_dict[reward_name][1:])

        self.past_scores.append(calculate_score(self.env.observation_dict, needs_success=True))

        if len(reward_history) == 0:
            return

        self.reward = reward_history[-1]
        self.acc_reward = np.sum(reward_history)
        actual_Q = calculate_q_values(reward_history, gamma)

        reward_shifted = reward_history - 1
        actual_Q_scaled, predicted_Q_scaled = self.scale_arrays_together(actual_Q, self.predicted_Q)
        past_scores_scaled = [score * 2 - 1 for score in self.past_scores]

        graphs = [
            (clean_actions, (0, 0, 255, 100), 0),
            (dirty_actions, (255, 0, 0, 255), 0),
            (dirty_x, (100, 0, 0, 150), 1),
            (clean_x, (255, 0, 0, 255), 1),
            (dirty_v, (0, 100, 0, 150), 1),
            (clean_v, (0, 255, 0, 255), 1),
            (predicted_Q_scaled, (60, 60, 230, 150), 2),
            (actual_Q_scaled, (0, 0, 255, 255), 2),
            (reward_shifted.tolist(), (0, 200, 0, 255), 2),
            (past_scores_scaled, (200, 0, 0, 255), 2)
        ]

        for graph, color, graph_num in graphs:
            self.draw_line_graph(graph, color, graph_num)

        if extracted_features:
            self.draw_feature_bars(extracted_features)

        # Draw the zero lines
        pygame.draw.line(self.graph_surface, (0, 0, 0), (self.graph_x, self.graph_height * 0.5), (2 * self.graph_width + self.graph_x, self.graph_height * 0.5), 1)
        pygame.draw.line(self.graph_surface, (0, 0, 0), (self.graph_x + self.graph_width, self.graph_height * 1.5), (2 * self.graph_width + self.graph_x, self.graph_height * 1.5), 1)

        self.canvas.blit(self.graph_surface, (0, 0))

        for i in range(4):
            x = (i % 2) * self.graph_width
            y = (i // 2) * self.graph_height
            pygame.draw.rect(self.canvas, (0, 0, 0), (x + self.graph_x, y, self.graph_width, self.graph_height), 1)

    def draw_line_graph(self, graph, color, graph_num):
        if len(graph) > 1:
            points = [(
                (graph_num % 2) * self.graph_width + i * (self.graph_width / (len(graph) - 1)) + self.graph_x,
                (graph_num // 2) * self.graph_height + self.graph_height - ((value + 1) / 2 * self.graph_height)
            ) for i, value in enumerate(graph)]

            pygame.draw.lines(self.graph_surface, color, False, points, 2)

    def draw_feature_bars(self, extracted_features):
        bar_width = self.graph_width / len(extracted_features)
        width_factor = 0.8
        for i, feature in enumerate(extracted_features):
            x = i * bar_width + self.graph_x + self.graph_width + (bar_width * (1 - width_factor)) // 2
            y = self.graph_height * 1.5
            height = -feature * (self.graph_height / 2)
            if feature >= 0:
                height = -height
                y -= height
            color = (0, 255, 0) if feature >= 0 else (255, 165, 0)
            pygame.draw.rect(self.graph_surface, color, (x, y, bar_width * width_factor, height))

    def draw_environment(self):
        x1, x2, x3, goal, threshold, metrics = self.calculate_positions('X_real')
        x1_meas, x2_meas, _, _, _, _ = self.calculate_positions('X_meas')

        if metrics['distance_next'] < threshold:
            self.canvas.fill((184, 255, 191))
        pygame.draw.rect(self.canvas, (240, 240, 240), (self.graph_window_width, 0, self.full_window_width, self.graph_window_height))

        self.draw_grid()
        self.draw_goals(goal, threshold, x3)
        self.draw_pendulum(x1_meas, x2_meas, alpha=50)
        self.draw_pendulum(x1, x2)

        return metrics

    def draw_grid(self, line_color=(200, 200, 200), spacing=75):
        for x in range(0, self.full_window_width, spacing):
            pygame.draw.line(self.canvas, line_color, (x, 0), (x, self.graph_window_height), 1)
        for y in range(0, self.graph_window_height, spacing):
            pygame.draw.line(self.canvas, line_color, (0, y), (self.full_window_width, y), 1)

    def calculate_positions(self, key):
        state_values = get_state_values(self.env.observation_dict, key)
        distance_next = (state_values['x3'] - state_values['goal'])[1]
        y = state_values['y']
        dynamics_func = self.env.observation_dict['dynamics_func']

        metrics = {
            'acc_reward': round(self.acc_reward, 5),
            'reward': round(self.reward, 5),
            'step_counter': len(self.env.observation_dict['T']) - 1,
            'x_1': round(y[0] / dynamics_func.max_angle, 4),
            'x_2': round(y[1] / dynamics_func.max_angle, 4),
            'distance_next': round(distance_next, 4),
            'v_1': round(y[2] / dynamics_func.max_velocity, 4),
            'v_2': round(y[3] / dynamics_func.max_velocity, 4),
            'action': round(state_values['unscaled_action'] / dynamics_func.torque_limit[0], 4),
            'time': self.env.observation_dict['T'][-1],
            'policy': self.policy,
            'killed': self.env.killed_because,
            'stabilized': get_stabilized(self.env.observation_dict),
            'r1': r1(state_values),
            'r2': r2(state_values),
            'v2[1]': state_values['v2'][1],
            'f1': f1(state_values),
            'f2': f2(state_values)
        }

        return state_values['x1'], state_values['x2'], state_values['x3'], state_values['goal'], state_values[
            'threshold_distance'], metrics

    def draw_pendulum(self, x1, x2, alpha=255):
        # Clear the pendulum surface
        self.pendulum_surface.fill((0, 0, 0, 0))

        black = (0, 0, 0, alpha)
        joint_color = (60, 60, 230, alpha) if self.policy != 1 else (230, 193, 60, alpha)

        pygame.draw.line(self.pendulum_surface, black, self.getXY(np.array([0, 0])), self.getXY(x1), 10)
        pygame.draw.line(self.pendulum_surface, black, self.getXY(x1), self.getXY(x2), 10)
        pygame.draw.circle(self.pendulum_surface, joint_color, self.getXY(np.array([0, 0])), 20)
        pygame.draw.circle(self.pendulum_surface, joint_color, self.getXY(x1), 20)
        pygame.draw.circle(self.pendulum_surface, joint_color, self.getXY(x2), 10)

        # Blit the pendulum surface onto the main canvas
        self.canvas.blit(self.pendulum_surface, (0, 0))

    def draw_goals(self, goal, threshold, x3):
        pygame.draw.line(self.canvas, (255, 50, 50),
                         self.getXY(np.array([-0.5 * 30 / 28, goal[1] + threshold])),
                         self.getXY(np.array([0.5 * 30 / 28, goal[1] + threshold])), 3)

    def blit_texts(self, metrics):
        for i, (label, value) in enumerate(metrics.items()):
            text = self.font.render(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}", True,
                                    (0, 0, 0))
            self.canvas.blit(text, (10, i * 40 + 150))

    def update_display(self):
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata_visualization["render_fps"])

    def getXY(self, point):
        return (
            self.graph_window_width // 2 + int(point[0] * self.pendulum_length_visualization * 2),
            self.graph_window_width // 2 + int(point[1] * self.pendulum_length_visualization * 2)
        )

    @staticmethod
    def scale_arrays_together(arr1, arr2, new_min=-1, new_max=1):
        combined = np.concatenate((arr1, arr2))
        old_min, old_max = np.min(combined), np.max(combined)

        if old_min == old_max:
            return np.full_like(arr1, new_min).tolist(), np.full_like(arr2, new_min).tolist()

        scale_func = lambda arr: (arr - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        return scale_func(arr1).tolist(), scale_func(arr2).tolist()