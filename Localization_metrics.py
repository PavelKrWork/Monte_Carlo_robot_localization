import numpy as np
from structures import *

class LocalizationMetrics:
    def __init__(self):
        self.position_errors = []
        self.angle_errors = []

    def update(self, true_state: State, est_state: State):
        position_error = np.sqrt((true_state.x - est_state.x) ** 2 + (true_state.y - est_state.y) ** 2)
        angle_error = np.abs((true_state.theta - est_state.theta + np.pi) % (2 * np.pi) - np.pi)

        self.position_errors.append(position_error)
        self.angle_errors.append(angle_error)
    
    def summary(self) -> dict:
        position_errors = np.array(self.position_errors)
        angle_errors = np.array(self.angle_errors)

        return {
            "position_rmse": np.sqrt(np.mean(position_errors ** 2)),
            "position_mae": np.mean(position_errors),
            "position_max": np.max(position_errors),
            "position_final": position_errors[-1],
            "angle_rmse": np.sqrt(np.mean(angle_errors ** 2))
        }
