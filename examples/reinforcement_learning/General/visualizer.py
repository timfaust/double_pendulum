from examples.reinforcement_learning.General.reward_functions import get_state_values
import pygame
import numpy as np


class Visualizer:
    def __init__(self, env_type, observation_dict):
        self.pendulum_length_visualization = 350
        self.reward_visualization = 0
        self.action_visualization = None
        self.acc_reward_visualization = 0
        self.window_width = 1000
        self.window_height = 1000
        self.metrics_width = 2000
        self.full_window_width = self.window_width + self.metrics_width
        self.window = None
        self.clock = None
        self.metadata_visualization = {"render_modes": ["human"], "render_fps": 120}
        self.env_type = env_type
        self.observation_dict = observation_dict

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
        # self.blit_texts(canvas, metrics)
        self.update_display(canvas)

    def setup_canvas(self):
        canvas = pygame.Surface((self.full_window_width, self.window_height))
        canvas.fill((255, 255, 255))
        return canvas

    def draw_graph(self, canvas):
        # Basis-Einstellungen für den Graphen
        graph_x, graph_y, graph_width, graph_height = self.window_width, 0, self.full_window_width, self.window_height
        max_value = 6
        min_value = -6

        dirty_actions = [arr[np.argmax(np.abs(arr))] for arr in self.observation_dict['U_con']]
        clean_actions = [arr[np.argmax(np.abs(arr))] for arr in self.observation_dict['U_clean']]

        graphs = [
            (dirty_actions, (255, 0, 0)),
            (clean_actions, (0, 0, 255))
        ]

        for (graph, color) in graphs:
            if len(graph) > 1:
                points = []
                for i, value in enumerate(graph):
                    x = graph_x + i * (graph_width / (len(graph) - 1))
                    y = graph_y + graph_height - ((value - min_value) / (max_value - min_value) * graph_height)
                    points.append((x, y))
                pygame.draw.lines(canvas, color, False, points, 2)

        pygame.draw.rect(canvas, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)

    def draw_environment(self, canvas):
        x1, x2, x3, goal, threshold, metrics = self.calculate_positions()

        if metrics['distance_next'] < threshold:
            canvas.fill((184, 255, 191))
        pygame.draw.rect(canvas, (240, 240, 240), (self.window_width, 0, self.full_window_width, self.window_height))

        self.draw_grid(canvas)
        self.draw_pendulum(canvas, x1, x2)
        self.draw_goals(canvas, goal, threshold, x3)

        return metrics

    def draw_grid(self, canvas, line_color=(200, 200, 200), spacing=50):
        for x in range(0, self.full_window_width, spacing):
            pygame.draw.line(canvas, line_color, (x, 0), (x, self.window_height), 1)
        for y in range(0, self.window_height, spacing):
            pygame.draw.line(canvas, line_color, (0, y), (self.full_window_width, y), 1)

    def calculate_positions(self):
        y, x1, x2, v1, v2, action, goal, dt, threshold, u_p, u_pp = get_state_values(self.env_type, self.observation_dict)
        x3 = x2 + dt * v2

        distance = np.linalg.norm(x2 - goal)
        distance_next = np.linalg.norm(x3 - goal)
        v1_total = np.linalg.norm(v1)
        v2_total = np.linalg.norm(v2)

        x_1 = y[0]
        x_2 = y[1]

        metrics = {
            'acc_reward': np.round(self.acc_reward_visualization, 5),
            'reward': np.round(self.reward_visualization, 5),
            'step_counter': len(self.observation_dict['T']) - 1,
            'x_1': round(x_1, 4),
            'x_2': round(x_2, 4),
            'distance': round(distance, 4),
            'distance_next': round(distance_next, 4),
            'v1_total': round(v1_total, 4),
            'v2_total': round(v2_total, 4),
            'action': round(action, 4)
        }

        return x1, x2, x3, goal, threshold, metrics

    def draw_pendulum(self, canvas, x1, x2):
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(np.array([0, 0])), self.getXY(x1), 5)
        pygame.draw.line(canvas, (0, 0, 0), self.getXY(x1), self.getXY(x2), 5)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(np.array([0, 0])), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x1), 10)
        pygame.draw.circle(canvas, (60, 60, 230), self.getXY(x2), 5)

    def draw_goals(self, canvas, goal, threshold, x3):
        pygame.draw.circle(canvas, (255, 200, 200), self.getXY(goal), threshold * 4 * self.pendulum_length_visualization)
        pygame.draw.circle(canvas, (255, 50, 50), self.getXY(goal), threshold * 2 * self.pendulum_length_visualization)
        pygame.draw.circle(canvas, (95, 2, 99), self.getXY(x3), threshold * 2 * self.pendulum_length_visualization)

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
