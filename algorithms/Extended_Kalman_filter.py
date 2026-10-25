import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from utils.structures import *

class ExtendedKalmanFilter:
    def __init__(self, initial_state: State, land_mark: LandMark,
                 system_noise: State, sensor_noise: np.ndarray,
                 init_cov=None):
        """
        initial_state - начальное состояние
        land_mark - координаты ориентиров
        system_noise - стандартные отклонения шума процесса (движения)
        sensor_noise - стандартные отклонения шума сенсора [r_left, r_right, alpha]
        init_cov - начальная ковариация оценки: scalar, вектор дисперсий (3,)
                   или матрица (3, 3). Если None, используется тот же prior,
                   что и в PF: diag(system_noise^2).
        """

        self.ideal_state = State(
            initial_state.x, initial_state.y, initial_state.theta)
        self.state = State(initial_state.x, initial_state.y,
                           initial_state.theta)
        self.dead_reckoning_state = State(
            initial_state.x, initial_state.y, initial_state.theta)
        self.camera_state = State(
            initial_state.x, initial_state.y, initial_state.theta)

        self.land_mark = land_mark
        self.system_noise = system_noise
        self.sensor_noise = sensor_noise

        # Оценка фильтра: среднее и ковариация
        self.mu = np.array([initial_state.x,
                            initial_state.y,
                            wrap_angle(initial_state.theta)], dtype=float)

        if init_cov is None:
            init_cov = np.array([
                system_noise.x ** 2,
                system_noise.y ** 2,
                system_noise.theta ** 2,
            ])

        init_cov = np.asarray(init_cov, dtype=float)
        if init_cov.ndim == 0:
            init_cov = np.eye(3) * float(init_cov)
        elif init_cov.ndim == 1:
            if init_cov.shape != (3,):
                raise ValueError(
                    "init_cov должен быть scalar, вектором (3,) или матрицей (3, 3)")
            init_cov = np.diag(init_cov)
        elif init_cov.shape != (3, 3):
            raise ValueError(
                "init_cov должен быть scalar, вектором (3,) или матрицей (3, 3)")

        self.Sigma = 0.5 * (init_cov + init_cov.T)

        # Матрица шума процесса Q = diag(sigma_x^2, sigma_y^2, sigma_theta^2)
        self.Q = np.diag([system_noise.x ** 2,
                          system_noise.y ** 2,
                          system_noise.theta ** 2])

        # Матрица шума измерений R = diag(sigma_r1^2, sigma_r2^2,
        #                                   sigma_alpha1^2, sigma_alpha2^2)
        self.R = np.diag([sensor_noise[0] ** 2,
                          sensor_noise[1] ** 2,
                          sensor_noise[2] ** 2,
                          sensor_noise[2] ** 2])

    # Движение робота без шума (идеальная траектория)
    def ideal_robot_move(self, step: State) -> State:
        dx_global = step.x * np.cos(self.ideal_state.theta) - step.y * np.sin(self.ideal_state.theta)
        dy_global = step.x * np.sin(self.ideal_state.theta) + step.y * np.cos(self.ideal_state.theta)

        self.ideal_state = State(
            self.ideal_state.x + dx_global,
            self.ideal_state.y + dy_global,
            wrap_angle(self.ideal_state.theta + step.theta),
        )

        return self.ideal_state

    def robot_move(self, step: State) -> State:
        dx_global = step.x * np.cos(self.state.theta) - step.y * np.sin(self.state.theta)
        dy_global = step.x * np.sin(self.state.theta) + step.y * np.cos(self.state.theta)

        self.state = State(
            self.state.x + dx_global + np.random.normal(0, self.system_noise.x),
            self.state.y + dy_global + np.random.normal(0, self.system_noise.y),
            wrap_angle(self.state.theta + step.theta + np.random.normal(0, self.system_noise.theta)),
        )

        self.ideal_robot_move(step)

        return self.state

    # Движение по чистой одометрии: та же зашумлённая модель, но без коррекции измерениями
    def dead_reckoning_move(self, step: State):
        dx_global = step.x * np.cos(self.dead_reckoning_state.theta) - step.y * np.sin(self.dead_reckoning_state.theta)
        dy_global = step.x * np.sin(self.dead_reckoning_state.theta) + step.y * np.cos(self.dead_reckoning_state.theta)

        self.dead_reckoning_state = State(
            self.dead_reckoning_state.x + dx_global + np.random.normal(0, self.system_noise.x),
            self.dead_reckoning_state.y + dy_global + np.random.normal(0, self.system_noise.y),
            wrap_angle(self.dead_reckoning_state.theta + step.theta + np.random.normal(0, self.system_noise.theta)),
        )

    # Измерение сенсора робота: расстояния и углы до ориентиров
    def sensor_measurement(self) -> tuple:
        dl = np.sqrt(
            (self.land_mark.left_coord.x - self.state.x) ** 2 +
            (self.land_mark.left_coord.y - self.state.y) ** 2
        )

        dr = np.sqrt(
            (self.land_mark.right_coord.x - self.state.x) ** 2 +
            (self.land_mark.right_coord.y - self.state.y) ** 2
        )

        phi_l = np.arctan2(
            self.land_mark.left_coord.y - self.state.y,
            self.land_mark.left_coord.x - self.state.x
        )

        phi_r = np.arctan2(
            self.land_mark.right_coord.y - self.state.y,
            self.land_mark.right_coord.x - self.state.x
        )

        alpha_l = wrap_angle(phi_l - self.state.theta)
        alpha_r = wrap_angle(phi_r - self.state.theta)

        alpha_l += np.random.normal(0, self.sensor_noise[2])
        alpha_r += np.random.normal(0, self.sensor_noise[2])

        alpha_l = wrap_angle(alpha_l)
        alpha_r = wrap_angle(alpha_r)

        zl = dl + np.random.normal(0, self.sensor_noise[0])
        zr = dr + np.random.normal(0, self.sensor_noise[1])

        return zl, zr, alpha_l, alpha_r

    def motion_model(self, mu: np.ndarray, step: State) -> np.ndarray:
        theta = mu[2]
        new_x = mu[0] + step.x * np.cos(theta) - step.y * np.sin(theta)
        new_y = mu[1] + step.x * np.sin(theta) + step.y * np.cos(theta)
        new_theta = wrap_angle(mu[2] + step.theta)
        return np.array([new_x, new_y, new_theta])

    # Якобиан матрицы движения A = df/dmu.
    def motion_jacobian(self, mu: np.ndarray, step: State) -> np.ndarray:
        theta = mu[2]
        A = np.array([
            [1.0, 0.0, -step.x * np.sin(theta) - step.y * np.cos(theta)],
            [0.0, 1.0, step.x * np.cos(theta) - step.y * np.sin(theta)],
            [0.0, 0.0, 1.0]
        ])
        return A

    def measurement_model(self, mu: np.ndarray) -> np.ndarray:
        # До левого ориентира
        dx_l = self.land_mark.left_coord.x - mu[0]
        dy_l = self.land_mark.left_coord.y - mu[1]
        r_l = np.sqrt(dx_l ** 2 + dy_l ** 2)
        alpha_l = wrap_angle(np.arctan2(dy_l, dx_l) - mu[2])

        # До правого ориентира
        dx_r = self.land_mark.right_coord.x - mu[0]
        dy_r = self.land_mark.right_coord.y - mu[1]
        r_r = np.sqrt(dx_r ** 2 + dy_r ** 2)
        alpha_r = wrap_angle(np.arctan2(dy_r, dx_r) - mu[2])

        return np.array([r_l, r_r, alpha_l, alpha_r])

    def measurement_jacobian(self, mu: np.ndarray) -> np.ndarray:
        landmarks = [self.land_mark.left_coord, self.land_mark.right_coord]

        H = np.zeros((4, 3))
        for i, lm in enumerate(landmarks):
            dx = lm.x - mu[0]
            dy = lm.y - mu[1]
            r_sq = dx ** 2 + dy ** 2

            # В точке ориентира угол не определён, а H сингулярен.
            # Такой ориентир пропускаем, чтобы не получить inf/nan.
            if r_sq < 1e-12:
                continue

            r = np.sqrt(r_sq)

            # Строка для расстояния r (i = 0 -> левый, i = 1 -> правый)
            H[i, 0] = -dx / r
            H[i, 1] = -dy / r
            H[i, 2] = 0.0

            # Строка для угла alpha (i = 0 -> левый, i = 1 -> правый)
            H[i + 2, 0] = dy / r_sq
            H[i + 2, 1] = -dx / r_sq
            H[i + 2, 2] = -1.0

        return H

    def predict(self, step: State):
        A = self.motion_jacobian(self.mu, step)
        self.mu = self.motion_model(self.mu, step)
        self.Sigma = A @ self.Sigma @ A.T + self.Q
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

    def update(self, zl: float, zr: float, alpha_l: float, alpha_r: float,
               gate_threshold=None):
        z = np.array([zl, zr, alpha_l, alpha_r])
        H = self.measurement_jacobian(self.mu)
        h = self.measurement_model(self.mu)

        y = z - h
        y[2] = wrap_angle(y[2])
        y[3] = wrap_angle(y[3])

        S = H @ self.Sigma @ H.T + self.R

        # Опциональная отбраковка выбросов по расстоянию Махаланобиса.
        if gate_threshold is not None:
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > gate_threshold:
                return False

        # K = P H^T S^{-1}, без явного обращения S
        K = np.linalg.solve(S, (self.Sigma @ H.T).T).T

        self.mu = self.mu + K @ y
        self.mu[2] = wrap_angle(self.mu[2])

        # Joseph form: сохраняет симметрию и положительную определённость
        I_KH = np.eye(3) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        return True

    def camera_estimate_state(self, zl: float, zr: float, alpha_l: float, alpha_r: float) -> State:
        return camera_pose_from_measurement(
            self.land_mark, zl, zr, alpha_l, alpha_r,
            previous=self.camera_state,
        )

    # Оцениваем состояние робота (для EKF — среднее из mu)
    def estimate_state(self) -> State:
        return State(float(self.mu[0]), float(self.mu[1]), float(self.mu[2]))

    # Полный шаг симуляции: движение -> измерение -> предсказание -> коррекция
    def simulate(self, step: State) -> State:
        self.robot_move(step)
        sm = self.sensor_measurement()

        camera_est = self.camera_estimate_state(*sm)
        self.camera_state = camera_est

        self.dead_reckoning_move(step)  # зашумленная одометрия

        self.predict(step)
        self.update(*sm)

        return self.estimate_state()
