import matplotlib.pyplot as plt
from particle_filter import State
import numpy as np


def draw_robot(state: State, ax=None, radius=0.2):
    ax = ax or plt.gca()
    robot_circle = plt.Circle((state.x, state.y), radius=radius, edgecolor='blue', facecolor='cyan', linewidth=1, zorder=5)
    ax.add_patch(robot_circle)
    
    arrow_length = radius * 1.5
    dx = arrow_length * np.cos(state.theta)
    dy = arrow_length * np.sin(state.theta)
    
    ax.arrow(state.x, state.y, dx, dy, 
              head_width=radius*0.4, head_length=radius*0.4, 
              fc='red', ec='red', linewidth=2, zorder=6)


def draw_particles(particles: np.ndarray, particle_color='green', ax=None):
    ax = ax or plt.gca()
    X = particles[:, 0]
    Y = particles[:, 1]
    Angles = particles[:, 2]
    
    # Вычисляем направление стрелок для каждой частицы
    U = np.cos(Angles)
    V = np.sin(Angles)
    
    # quiver рисует тысячи стрелок мгновенно. 
    # color — цвет частиц, scale=30 — размер стрелочек
    ax.quiver(X, Y, U, V, color=particle_color, scale=25, width=0.003, zorder=4, alpha=0.6)


def draw_field(ax=None,
               top_left=[-1.3, 6.0], top_right=[1.3, 6.0], 
               bottom_left=[-1.3, -6.0], bottom_right=[1.3, -6.0], 
               length=12.0, width=9.0):
    ax = ax or plt.gca()
    
    half_L = length / 2.0
    half_W = width / 2.0
    
    # Границы поля
    ax.plot([-half_W, half_W, half_W, -half_W, -half_W], [-half_L, -half_L, half_L, half_L, -half_L], color='black', linewidth=2)
    # Центральная линия
    ax.plot([-half_W, half_W], [0.0, 0.0], color='black', linewidth=2)
    
    # Центральный круг
    center_circle = plt.Circle((0, 0), radius=1.5, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(center_circle)
    
    goal_width = top_right[0] - top_left[0]
    
    # Верхние и нижние ворота
    top_goal = plt.Rectangle((top_left[0], half_L - 1.2), goal_width, 1.2, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(top_goal)
    
    bottom_goal = plt.Rectangle((bottom_left[0], -half_L), goal_width, 1.2, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(bottom_goal)
    
    return ax


def draw_field_with_robot(robot_state: State, particles: np.ndarray, 
                          top_left=[-1.3, 6.0], top_right=[1.3, 6.0], 
                          bottom_left=[-1.3, -6.0], bottom_right=[1.3, -6.0], 
                          length=12.0, width=9.0, particle_color='green',
                          ax=None, clear=True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    elif clear:
        # Очищаем оси, чтобы перерисовать всё заново на том же поле
        ax.cla()
    
    draw_field(ax, top_left, top_right, bottom_left, bottom_right, length, width)
    draw_particles(particles, particle_color, ax)
    
    # Отрисовка робота
    draw_robot(robot_state, ax)
    
    ax.set_aspect('equal')
    ax.axis('off')
    return ax