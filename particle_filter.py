import numpy as np
from dataclasses import dataclass

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

# TODO: resampling wheel algo implementation
def resampling_wheel(weights: dict, particleList: list) -> list:
    pass

# main algo logic of Particle filter
class ParticleFilter:
    def __init__(self, num_particles: int, state: State, land_mark: LandMark, system_noise: State, sensor_noise: np.ndarray):
        
        """
        add here params description 
        """

        self.num_particles = num_particles
        self.state = state
        self.land_mark = land_mark
        self.system_noise = system_noise
        self.sensor_noise = sensor_noise

        # apologise that: x, y, thate are independent, loc = mean, scale = var
        x_particles = np.random.normal(loc=state.x, scale=system_noise.x, size=num_particles)
        y_particles = np.random.normal(loc=state.y, scale=system_noise.y, size=num_particles)
        
        theta_center = (state.theta + np.pi) % (2 * np.pi) - np.pi
        theta_particles = np.random.normal(loc=theta_center, scale=system_noise.theta, size=num_particles)
        theta_particles = (theta_particles + np.pi) % (2 * np.pi) - np.pi

        particles = np.column_stack((x_particles, y_particles, theta_particles))
        self.particles = particles
        self.weights = np.ones(num_particles) / num_particles

    # предсказываем положение робота
    def robot_move(self, step: State) -> State:
        dx_global = step.x * np.cos(self.state.theta) - step.y * np.sin(self.state.theta)
        dy_global = step.x * np.sin(self.state.theta) + step.y * np.cos(self.state.theta)

        self.state = State(
            self.state.x + dx_global,
            self.state.y + dy_global,
            (self.state.theta + step.theta + np.pi) % (2 * np.pi) - np.pi,
        )

        return self.state

    # измерение сенсора
    def sensor_measurement(self) -> tuple:
        dl = np.sqrt(
            (self.land_mark.left_coord.x - self.state.x) ** 2 +
            (self.land_mark.left_coord.y - self.state.y) ** 2
        )

        dr = np.sqrt(
            (self.land_mark.right_coord.x - self.state.x) ** 2 +
            (self.land_mark.right_coord.y - self.state.y) ** 2
        )

        zl = dl + np.random.normal(0, self.sensor_noise[0])
        zr = dr + np.random.normal(0, self.sensor_noise[1])

        return zl, zr

    # предсказываем положение робота
    def particles_move(self, step: State) -> np.array:
        for i in range(self.num_particles):
            dx_global = step.x * np.cos(self.particles[i][2]) - step.y * np.sin(self.particles[i][2])
            dy_global = step.x * np.sin(self.particles[i][2]) + step.y * np.cos(self.particles[i][2])

            self.particles[i][0] += dx_global + np.random.normal(loc=0, scale=self.system_noise.x)
            self.particles[i][1] += dy_global + np.random.normal(loc=0, scale=self.system_noise.y)
            self.particles[i][2] = (self.particles[i][2] + step.theta + np.pi + np.random.normal(0, self.system_noise.theta)) % (2 * np.pi) - np.pi
        
        return self.particles

    def particle_measurement(self) -> list:
        particle_measurements = [0] * self.num_particles
        for i in range(self.num_particles):
            dl = np.sqrt((self.land_mark.left_coord.x - self.particles[i][0]) ** 2 + (self.land_mark.left_coord.y - self.particles[i][1]) ** 2)
            dr = np.sqrt((self.land_mark.right_coord.x - self.particles[i][0]) ** 2 + (self.land_mark.right_coord.y - self.particles[i][1]) ** 2)
            particle_measurements[i] = (dl, dr)
        
        return particle_measurements

    def weight_calc(self) -> list:
        weights = [0] * self.num_particles
        particle_measurements = self.particle_measurement()
        zl, zr = self.sensor_measurement()
        sigma_l = self.sensor_noise[0]
        sigma_r = self.sensor_noise[1]
        for i in range(self.num_particles):
            dl, dr = particle_measurements[i]
            wl = 1 / np.sqrt(2 * np.pi * sigma_l ** 2) * np.exp(-(zl - dl) ** 2 / (2 * sigma_l ** 2))
            wr = 1 / np.sqrt(2 * np.pi * sigma_r ** 2) * np.exp(-(zr - dr) ** 2 / (2 * sigma_r ** 2))
            weights[i] = wl * wr

        weights = np.array(weights)
        weights /= np.sum(weights)

        self.weights = weights
        
        return weights
    
    def resample(self) -> np.array:
        pass

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


    def simulate(self, iters_count: int):
        pass

