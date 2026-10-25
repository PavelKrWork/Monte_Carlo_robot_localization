from dataclasses import dataclass

import numpy as np


# 2D particles cloud
@dataclass
class State:
    x: float
    y: float
    theta: float

# Land mark coords initialization
@dataclass
class Coord:
    x: float
    y: float

@dataclass
class LandMark:
    left_coord: Coord
    right_coord: Coord


# Приведение угла к диапазону [-pi, pi)
def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Положение робота по двум дальностям и двум углам до ориентиров.
# У двух окружностей есть два пересечения; выбираем то, которое
# согласует углы alpha_l и alpha_r (а при равенстве — ближайшее к previous).
def camera_pose_from_measurement(land_mark: LandMark, zl: float, zr: float,
                                 alpha_l: float, alpha_r: float,
                                 previous: State = None) -> State:
    xl, yl = land_mark.left_coord.x, land_mark.left_coord.y
    xr, yr = land_mark.right_coord.x, land_mark.right_coord.y

    dx = xr - xl
    dy = yr - yl
    d2 = dx ** 2 + dy ** 2

    # Вырожденный случай: ориентиры совпали, положение не восстановить.
    if d2 < 1e-12:
        if previous is not None:
            return State(previous.x, previous.y, previous.theta)
        return State(0.0, 0.0, 0.0)

    d = np.sqrt(d2)
    a = (zl ** 2 - zr ** 2 + d2) / (2 * d)
    h = np.sqrt(max(zl ** 2 - a ** 2, 0.0))

    base_x = xl + a * dx / d
    base_y = yl + a * dy / d

    perp_x = -dy / d
    perp_y = dx / d

    candidates = [
        (base_x + h * perp_x, base_y + h * perp_y),
        (base_x - h * perp_x, base_y - h * perp_y),
    ]

    best_score = np.inf
    best_pose = None

    for x, y in candidates:
        phi_l = np.arctan2(yl - y, xl - x)
        phi_r = np.arctan2(yr - y, xr - x)

        theta_l = wrap_angle(phi_l - alpha_l)
        theta_r = wrap_angle(phi_r - alpha_r)

        # Насколько два угла дают согласованный theta
        score = abs(wrap_angle(theta_l - theta_r))

        if previous is not None:
            score += 1e-6 * np.hypot(x - previous.x, y - previous.y)

        theta = np.arctan2(
            np.sin(theta_l) + np.sin(theta_r),
            np.cos(theta_l) + np.cos(theta_r),
        )

        if score < best_score:
            best_score = score
            best_pose = State(float(x), float(y), float(theta))

    return best_pose
