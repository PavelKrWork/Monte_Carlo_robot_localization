import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from utils.structures import *

# Алгоритм отсеивания частиц: колесо отсева
def resampling_wheel(initial_particles: list, particles_weights: list, particles_cnt: int) -> list:
    new_particles = []

    beta = 0
    index = int(random.randint(0, particles_cnt - 1))
    max_weight = max(particles_weights)

    for i in range(particles_cnt):
        beta += random.uniform(0, 2 * max_weight)

        while beta > particles_weights[index]:
            beta -= particles_weights[index]
            index = (index + 1) % particles_cnt
        new_particles.append(
            [initial_particles[index], particles_weights[index]])

    return new_particles


# Основная логика фильтра частиц
class ParticleFilter:
    def __init__(self, num_particles: int, initial_state: State, land_mark: LandMark, system_noise: State, sensor_noise: np.ndarray, is_adapt_resampl_enabled_: bool, init_cov=None):
        """
        num_particles - количество частиц
        initial_state - начальное состояние
        land_mark - координаты ориентиров
        system_noise - шум системы (используется только при движении)
        sensor_noise - шум измерений
        particles - состояния частиц
        weights - веса частиц
        is_adapt_resampl_enabled - включен ли адаптивный ресемплинг
        init_cov - начальная ковариация облака частиц: scalar, вектор
                   дисперсий (3,) или матрица (3, 3). Это prior, не шум процесса.
                   Если None, для обратной совместимости берётся system_noise.
        """

        self.num_particles = num_particles
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

        self.init_cov = 0.5 * (init_cov + init_cov.T)

        theta_center = wrap_angle(initial_state.theta)
        particles = np.random.multivariate_normal(
            mean=[initial_state.x, initial_state.y, theta_center],
            cov=self.init_cov,
            size=num_particles,
        )
        particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        self.particles = particles
        self.weights = np.ones(num_particles) / num_particles
        self.is_adapt_resampl_enabled = is_adapt_resampl_enabled_

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

    # Движение робота с гауссовым шумом (истинная траектория в симуляции)
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

    # Совершаем движение частицами
    def particles_move(self, step: State) -> np.array:
        for i in range(self.num_particles):
            dx_global = step.x * \
                np.cos(self.particles[i][2]) - step.y * \
                np.sin(self.particles[i][2])
            dy_global = step.x * \
                np.sin(self.particles[i][2]) + step.y * \
                np.cos(self.particles[i][2])

            self.particles[i][0] += dx_global + \
                np.random.normal(loc=0, scale=self.system_noise.x)
            self.particles[i][1] += dy_global + \
                np.random.normal(loc=0, scale=self.system_noise.y)
            self.particles[i][2] = (self.particles[i][2] + step.theta + np.pi +
                                    np.random.normal(0, self.system_noise.theta)) % (2 * np.pi) - np.pi

        return self.particles

    def randomize_particles(self):
        x_min, x_max = -4.5, 4.5
        y_min, y_max = -6.0, 6.0

        self.particles[:, 0] = np.random.uniform(
            x_min, x_max, self.num_particles
        )
        self.particles[:, 1] = np.random.uniform(
            y_min, y_max, self.num_particles
        )
        self.particles[:, 2] = np.random.uniform(
            -np.pi, np.pi, self.num_particles
        )

        self.weights.fill(1.0 / self.num_particles)

    # Измерение каждой частицей
    def particle_measurement(self) -> list:
        particle_measurements = [0] * self.num_particles
        for i in range(self.num_particles):
            dl = np.sqrt((self.land_mark.left_coord.x - self.particles[i][0]) ** 2 + (
                self.land_mark.left_coord.y - self.particles[i][1]) ** 2)
            dr = np.sqrt((self.land_mark.right_coord.x - self.particles[i][0]) ** 2 + (
                self.land_mark.right_coord.y - self.particles[i][1]) ** 2)

            dfi_l = np.arctan2(
                self.land_mark.left_coord.y - self.particles[i][1],
                self.land_mark.left_coord.x - self.particles[i][0]
            )

            dfi_r = np.arctan2(
                self.land_mark.right_coord.y - self.particles[i][1],
                self.land_mark.right_coord.x - self.particles[i][0]
            )

            alpha_l = dfi_l - self.particles[i][2]
            alpha_r = dfi_r - self.particles[i][2]

            alpha_l = (alpha_l + np.pi) % (2 * np.pi) - np.pi
            alpha_r = (alpha_r + np.pi) % (2 * np.pi) - np.pi

            particle_measurements[i] = (dl, dr, alpha_l, alpha_r)

        return particle_measurements

    # Вычисление веса для каждой частицы из ф-лы плотности нормального распределения
    def weight_calc(self, measurement: tuple) -> np.ndarray:
        particle_measurements = self.particle_measurement()
        zl, zr, alpha_l, alpha_r = measurement
        sigma_l = self.sensor_noise[0]
        sigma_r = self.sensor_noise[1]
        sigma_alpha = self.sensor_noise[2]

        positive_weights = self.weights > 0
        log_weights = np.full(self.num_particles, -np.inf)
        log_weights[positive_weights] = np.log(
            self.weights[positive_weights])

        for i in range(self.num_particles):
            if not positive_weights[i]:
                continue

            dl, dr, dalpha_l, dalpha_r = particle_measurements[i]
            error_alpha_l = (alpha_l - dalpha_l + np.pi) % (2 * np.pi) - np.pi
            error_alpha_r = (alpha_r - dalpha_r + np.pi) % (2 * np.pi) - np.pi

            log_weights[i] += (
                -0.5 * ((zl - dl) / sigma_l) ** 2
                - 0.5 * ((zr - dr) / sigma_r) ** 2
                - 0.5 * (error_alpha_l / sigma_alpha) ** 2
                - 0.5 * (error_alpha_r / sigma_alpha) ** 2
            )

        max_log_weight = np.max(log_weights)
        if not np.isfinite(max_log_weight):
            weights = np.full(self.num_particles, 1 / self.num_particles)
        else:
            weights = np.exp(log_weights - max_log_weight)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights.fill(1 / self.num_particles)

        self.weights = weights

        return weights

    # Делаем ресемплинг всех частиц с помощью колеса отсева
    def resample(self) -> np.array:
        initial_particles = np.arange(self.num_particles)
        new_particles = resampling_wheel(
            initial_particles, self.weights, self.num_particles)
        particle_indexes = [el_i[0] for el_i in new_particles]
        self.particles = self.particles[particle_indexes]

        # rescaling weights
        self.weights = np.ones(self.num_particles) / self.num_particles

        return self.particles

    def effective_particles_calc(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def camera_estimate_state(self, zl: float, zr: float, alpha_l: float, alpha_r: float) -> State:
        return camera_pose_from_measurement(
            self.land_mark, zl, zr, alpha_l, alpha_r,
            previous=self.camera_state,
        )

    # Оцениваем состояние робота (для PF — взвешенное среднее частиц)
    def estimate_state(self) -> State:
        estimate_x = 0
        estimate_y = 0
        sin_theta = 0
        cos_theta = 0
        for i in range(self.num_particles):
            estimate_x += self.particles[i][0] * self.weights[i]
            estimate_y += self.particles[i][1] * self.weights[i]
            sin_theta += np.sin(self.particles[i][2]) * self.weights[i]
            cos_theta += np.cos(self.particles[i][2]) * self.weights[i]

        estimate_theta = np.arctan2(sin_theta, cos_theta)

        return State(estimate_x, estimate_y, estimate_theta)

    def simulate(self, step: State) -> State:
        self.robot_move(step)
        sm = self.sensor_measurement()
        
        camera_est = self.camera_estimate_state(*sm)
        self.camera_state = camera_est

        self.dead_reckoning_move(step)  # зашумленная одометрия

        self.particles_move(step)
        self.weight_calc(sm)

        est_state = self.estimate_state()

        if self.is_adapt_resampl_enabled:
            if self.effective_particles_calc() < self.num_particles // 2:
                self.resample()
        else:
            self.resample()
        
        return est_state
