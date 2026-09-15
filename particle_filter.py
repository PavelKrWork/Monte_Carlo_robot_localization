import numpy as np
import random
from structures import *

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
    def __init__(self, num_particles: int, initial_state: State, land_mark: LandMark, system_noise: State, sensor_noise: np.ndarray):
        """
        num_particles - количество частиц
        initial_state - начальное состояние
        land_mark - координаты ориентиров
        system_noise - шум системы
        sensor_noise - шум измерений
        particles - состояния частиц
        weights - веса частиц
        """

        self.num_particles = num_particles
        self.ideal_state = State(
            initial_state.x, initial_state.y, initial_state.theta)
        self.state = State(initial_state.x, initial_state.y,
                           initial_state.theta)
        self.no_pf_state = State(
            initial_state.x, initial_state.y, initial_state.theta)

        self.land_mark = land_mark
        self.system_noise = system_noise
        self.sensor_noise = sensor_noise

        # apologise that: x, y, thate are independent, loc = mean, scale = standard deviation
        x_particles = np.random.normal(
            loc=initial_state.x, scale=system_noise.x, size=num_particles)
        y_particles = np.random.normal(
            loc=initial_state.y, scale=system_noise.y, size=num_particles)

        theta_center = (initial_state.theta + np.pi) % (2 * np.pi) - np.pi
        theta_particles = np.random.normal(
            loc=theta_center, scale=system_noise.theta, size=num_particles)
        theta_particles = (theta_particles + np.pi) % (2 * np.pi) - np.pi

        particles = np.column_stack(
            (x_particles, y_particles, theta_particles))
        self.particles = particles
        self.weights = np.ones(num_particles) / num_particles

    """
    Предсказываем положение робота:
    сначало робот совершает движение, а потом делает поворот
    """

    # Совершаем движение робота без шума
    def ideal_robot_move(self, step: State) -> State:
        dx_global = step.x * \
            np.cos(self.ideal_state.theta) - step.y * \
            np.sin(self.ideal_state.theta)
        dy_global = step.x * \
            np.sin(self.ideal_state.theta) + step.y * \
            np.cos(self.ideal_state.theta)

        self.ideal_state = State(
            self.ideal_state.x + dx_global,
            self.ideal_state.y + dy_global,
            (self.ideal_state.theta + step.theta + np.pi) % (2 * np.pi) - np.pi,
        )

        return self.ideal_state

    # Совершаем движение робота с добавлением Гауссова шума
    def robot_move(self, step: State) -> State:
        dx_global = (step.x * np.cos(self.state.theta) -
                     step.y * np.sin(self.state.theta))
        dy_global = (step.x * np.sin(self.state.theta) +
                     step.y * np.cos(self.state.theta))

        self.state = State(
            self.state.x + dx_global +
            np.random.normal(0, self.system_noise.x),
            self.state.y + dy_global +
            np.random.normal(0, self.system_noise.y),
            (self.state.theta
             + step.theta
             + np.random.normal(0, self.system_noise.theta)
             + np.pi) % (2 * np.pi) - np.pi
        )

        self.ideal_robot_move(step)

        return self.state

    # Совершаем движение робота с шумом, без дальнейшего использования PF
    def no_pf_move(self, step: State):
        dx_global = (
            step.x * np.cos(self.no_pf_state.theta)
            - step.y * np.sin(self.no_pf_state.theta)
        )

        dy_global = (
            step.x * np.sin(self.no_pf_state.theta)
            + step.y * np.cos(self.no_pf_state.theta)
        )

        self.no_pf_state.x += (
            dx_global + np.random.normal(0, self.system_noise.x)
        )

        self.no_pf_state.y += (
            dy_global + np.random.normal(0, self.system_noise.y)
        )

        self.no_pf_state.theta += (
            step.theta + np.random.normal(0, self.system_noise.theta)
        )

        self.no_pf_state.theta = (
            self.no_pf_state.theta + np.pi
        ) % (2 * np.pi) - np.pi

    # Измерение сенсора робота
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

        alpha_l = phi_l - self.state.theta
        alpha_r = phi_r - self.state.theta

        alpha_l = (alpha_l + np.pi) % (2 * np.pi) - np.pi
        alpha_r = (alpha_r + np.pi) % (2 * np.pi) - np.pi

        alpha_l += np.random.normal(0, self.sensor_noise[2])
        alpha_r += np.random.normal(0, self.sensor_noise[2])

        alpha_l = (alpha_l + np.pi) % (2 * np.pi) - np.pi
        alpha_r = (alpha_r + np.pi) % (2 * np.pi) - np.pi

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
        weights = np.zeros(self.num_particles)
        particle_measurements = self.particle_measurement()
        zl, zr, alpha_l, alpha_r = measurement
        sigma_l = self.sensor_noise[0]
        sigma_r = self.sensor_noise[1]
        sigma_alpha = self.sensor_noise[2]
        for i in range(self.num_particles):
            dl, dr, dalpha_l, dalpha_r = particle_measurements[i]
            wl = 1 / np.sqrt(2 * np.pi * sigma_l ** 2) * \
                np.exp(-(zl - dl) ** 2 / (2 * sigma_l ** 2))
            wr = 1 / np.sqrt(2 * np.pi * sigma_r ** 2) * \
                np.exp(-(zr - dr) ** 2 / (2 * sigma_r ** 2))

            error_alpha_l = (alpha_l - dalpha_l + np.pi) % (2 * np.pi) - np.pi
            error_alpha_r = (alpha_r - dalpha_r + np.pi) % (2 * np.pi) - np.pi
            walphal = 1 / np.sqrt(2 * np.pi * sigma_alpha ** 2) * \
                np.exp(-error_alpha_l ** 2 / (2 * sigma_alpha ** 2))
            walphar = 1 / np.sqrt(2 * np.pi * sigma_alpha ** 2) * \
                np.exp(-error_alpha_r ** 2 / (2 * sigma_alpha ** 2))

            weights[i] = wl * wr * walphal * walphar

        # безопасная нормализация
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

    # Оцениваем положение робота
    def estimate_robot_state(self) -> State:
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

        self.no_pf_move(step)  # зашумленная одометрия

        self.particles_move(step)
        self.weight_calc(sm)

        est_state = self.estimate_robot_state()

        self.resample()

        return est_state
