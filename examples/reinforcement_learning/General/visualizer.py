from examples.reinforcement_learning.General.misc_helper import calculate_q_values
from examples.reinforcement_learning.General.reward_functions import get_state_values
import pygame
import numpy as np
import torch as th


class Visualizer:
    def __init__(self, env_type, observation_dict):
        self.pendulum_length_visualization = 700
        self.reward_visualization = 0
        self.action_visualization = None
        self.acc_reward_visualization = 0
        self.window_width = 1500
        self.window_height = 1500
        self.metrics_width = 3000
        self.full_window_width = self.window_width + self.metrics_width
        self.window = None
        self.clock = None
        self.metadata_visualization = {"render_modes": ["human"], "render_fps": 30}
        self.env_type = env_type
        self.observation_dict = observation_dict
        self.model = None

    def init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.full_window_width, self.window_height))
        self.clock = pygame.time.Clock()

    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.window is None or self.clock is None:
            self.init_pygame()

        canvas = self.setup_canvas()
        metrics = self.draw_environment(canvas)
        self.draw_graph(canvas)
        self.blit_texts(canvas, metrics)
        self.update_display(canvas)

    def setup_canvas(self):
        canvas = pygame.Surface((self.full_window_width, self.window_height))
        canvas.fill((255, 255, 255))
        return canvas

    def draw_graph(self, canvas):
        reward_name = 'reward_0'
        gamma = 0.99
        if self.model is not None:
            gamma = self.model.gamma
            reward_name = 'reward_' + str(self.model.active_policy)

        # Basis-Einstellungen für den Graphen
        graph_x, graph_y, graph_width, graph_height = self.window_width, 0, (self.full_window_width - self.window_width) // 2, self.window_height
        max_value = 1.02
        min_value = -1.02

        # Create a transparent surface for drawing
        graph_surface = pygame.Surface((self.full_window_width, self.window_height), pygame.SRCALPHA)
        graph_surface.fill((0, 0, 0, 0))  # Fill with transparent color

        dirty_actions = self.observation_dict['U_real'][1:]
        clean_actions = self.observation_dict['U_con'][1:]
        dirty_x = [x[1] for x in self.observation_dict['X_meas'][1:]]
        clean_x = [x[1] for x in self.observation_dict['X_real'][1:]]
        dirty_v = [x[3] for x in self.observation_dict['X_meas'][1:]]
        clean_v = [x[3] for x in self.observation_dict['X_real'][1:]]
        reward = np.array(self.observation_dict[reward_name][1:])
        actual_Q = calculate_q_values(reward, gamma)

        if self.model is not None:
            with th.no_grad():
                device = self.model.critic.device
                actions = th.tensor(self.observation_dict['U_con'][1:], dtype=th.float32, device=device).unsqueeze(1)
                states_np = np.array(self.observation_dict['state'][:-1])
                states = th.tensor(states_np, dtype=th.float32, device=device)
                q_values, _ = th.min(th.cat(self.model.critic(states, actions), dim=1), dim=1, keepdim=True)
                predicted_Q = q_values.squeeze(1).cpu().numpy()
        else:
            predicted_Q = np.array([])

        reward_shifted = reward - 1
        actual_Q_scaled = actual_Q / 500 - 1
        predicted_Q_scaled = predicted_Q / 500 - 1

        reward_shifted = reward_shifted.tolist()
        actual_Q_scaled = actual_Q_scaled.tolist()
        predicted_Q_scaled = predicted_Q_scaled.tolist()

        graphs_left = [
            (clean_actions, (0, 0, 255, 150)),
            (dirty_actions, (255, 0, 0, 255)),
            (reward_shifted, (0, 255, 0, 255))
        ]

        graphs_right = [
            (dirty_x, (100, 0, 0, 150)),
            (clean_x, (255, 0, 0, 255)),
            (dirty_v, (0, 100, 0, 150)),
            (clean_v, (0, 255, 0, 255)),
            (predicted_Q_scaled, (0, 0, 100, 150)),
            (actual_Q_scaled, (0, 0, 255, 255))
        ]

        def draw_line(surface, color, start_pos, end_pos):
            pygame.draw.line(surface, color, start_pos, end_pos, 2)

        for (graph, color) in graphs_left + graphs_right:
            if len(graph) > 1:
                points = []
                for i, value in enumerate(graph):
                    x = graph_x + i * (graph_width / (len(graph) - 1))
                    if graph in [graph for graph, _ in graphs_right]:
                        x += graph_width
                    y = graph_y + graph_height - ((value - min_value) / (max_value - min_value) * graph_height)
                    points.append((x, y))

                for i in range(len(points) - 1):
                    draw_line(graph_surface, color, points[i], points[i + 1])

        # Draw the transparent surface onto the main canvas
        canvas.blit(graph_surface, (0, 0))

        # Draw border
        pygame.draw.rect(canvas, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)
        pygame.draw.rect(canvas, (0, 0, 0), (graph_x + graph_width, graph_y, graph_width, graph_height), 1)

    def draw_environment(self, canvas):
        x1, x2, x3, goal, threshold, metrics = self.calculate_positions('X_real')
        x1_meas, x2_meas, _, _, _, _ = self.calculate_positions('X_meas')

        if metrics['distance_next'] < threshold:
            canvas.fill((184, 255, 191))
        pygame.draw.rect(canvas, (240, 240, 240), (self.window_width, 0, self.full_window_width, self.window_height))

        self.draw_grid(canvas)
        self.draw_goals(canvas, goal, threshold, x3)

        self.draw_pendulum(canvas, x1_meas, x2_meas, alpha=50)
        self.draw_pendulum(canvas, x1, x2)

        return metrics

    def draw_grid(self, canvas, line_color=(200, 200, 200), spacing=50):
        for x in range(0, self.full_window_width, spacing):
            pygame.draw.line(canvas, line_color, (x, 0), (x, self.window_height), 1)
        for y in range(0, self.window_height, spacing):
            pygame.draw.line(canvas, line_color, (0, y), (self.full_window_width, y), 1)

    def calculate_positions(self, key):
        state_values = get_state_values(self.observation_dict, key)

        distance_next = (state_values['x3'] - state_values['goal'])[1]
        y = state_values['unscaled_observation']

        metrics = {
            'acc_reward': np.round(self.acc_reward_visualization, 5),
            'reward': np.round(self.reward_visualization, 5),
            'step_counter': len(self.observation_dict['T']) - 1,
            'x_1': round(y[0]/(2 * np.pi), 4),
            'x_2': round(y[1]/(2 * np.pi), 4),
            # 'distance': round(distance, 4),
            'distance_next': round(distance_next, 4),
            'v_1': round(y[2]/20, 4),
            'v_2': round(y[3]/20, 4),
            'action': round(state_values['unscaled_action']/5, 4),
            'time': self.observation_dict["T"][-1]
        }

        return state_values['x1'], state_values['x2'], state_values['x3'], state_values['goal'], state_values['threshold_distance'], metrics

    def draw_pendulum(self, canvas, x1, x2, alpha=255):
        transparent_surface = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        black = (0, 0, 0, alpha)
        blue = (60, 60, 230, alpha)
        pygame.draw.line(transparent_surface, black, self.getXY(np.array([0, 0])), self.getXY(x1), 10)
        pygame.draw.line(transparent_surface, black, self.getXY(x1), self.getXY(x2), 10)
        pygame.draw.circle(transparent_surface, blue, self.getXY(np.array([0, 0])), 20)
        pygame.draw.circle(transparent_surface, blue, self.getXY(x1), 20)
        pygame.draw.circle(transparent_surface, blue, self.getXY(x2), 10)
        canvas.blit(transparent_surface, (0, 0))

    def draw_goals(self, canvas, goal, threshold, x3):
        pygame.draw.line(canvas, (255, 50, 50), self.getXY(np.array([-0.5*30/28, goal[1] + threshold])), self.getXY(np.array([0.5*30/28, goal[1] + threshold])), 3)
        # pygame.draw.circle(canvas, (95, 2, 99), self.getXY(x3), 10)

    def blit_texts(self, canvas, metrics):
        myFont = pygame.font.SysFont("Arial", 28)
        base_x = self.window_width + 10  # Beginne rechts von der Hauptvisualisierung
        positions = [(base_x, i * 40 + 10) for i, _ in enumerate(metrics.items())]
        for (label, value), position in zip(metrics.items(), positions):
            if isinstance(value, float):
                text_format = f"{label}: {value:.4f}"
            else:
                text_format = f"{label}: {value}"
            text = myFont.render(text_format, True, (0, 0, 0))
            canvas.blit(text, position)

    def update_display(self, canvas):
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata_visualization["render_fps"])

    def getXY(self, point):
        return (
            self.window_width // 2 + int(point[0] * self.pendulum_length_visualization * 2),
            self.window_width // 2 + int(point[1] * self.pendulum_length_visualization * 2)
        )

    def reset(self):
        self.reward_visualization = 0
        self.acc_reward_visualization = 0
        self.action_visualization = np.array([0, 0])
