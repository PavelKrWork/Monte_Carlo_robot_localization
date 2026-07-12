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
def resampling_wheel(weights: dict, N: int, particleList: list) -> list:
    pass

# main algo logic of Particle filter
class ParticleFilter:
    # TODO: generate particles list using normal distr with mean = state0 and var = noise
    def __init__(self, num_particles: int, state0: State, partice_states: list, land_mark: LandMark, system_noise: State, sensor_noise: np.ndarray):
        
        """
        add here params description 
        """

        self.num_particles = num_particles
        self.state0 = state0
        self.partice_states = partice_states
        self.land_mark = land_mark
        self.system_noise = system_noise
        self.sensor_noise = sensor_noise

        # apologise that: x, y, thate are independent, loc = mean, scale = var
        x_particles = np.random.normal(loc=state0.x, scale=system_noise.x, size=num_particles)
        y_particles = np.random.normal(loc=state0.y, scale=system_noise.y, size=num_particles)
        
        theta_center = (state0.theta + np.pi) % (2 * np.pi) - np.pi
        theta_particles = np.random.normal(loc=theta_center, scale=system_noise.theta, size=num_particles)
        theta_particles = (theta_particles + np.pi) % (2 * np.pi) - np.pi

        particles = np.column_stack((x_particles, y_particles, theta_particles))
        self.particles = particles
        self.weights = np.ones(num_particles) / num_particles

    def robot_move(self, step: State) -> State:
        dx_global = step.x * np.cos(self.state0.theta) - step.y * np.sin(self.state0.theta)
        dy_global = step.x * np.sin(self.state0.theta) + step.y * np.cos(self.state0.theta)
        
        x_new = self.state0.x + dx_global
        y_new = self.state0.y + dy_global
        theta_new = (self.state0.theta + step.theta + np.pi) % (2 * np.pi) - np.pi

        new_state = State(x_new, y_new, theta_new)

        return new_state

    # update land_mark coords with sensor_noise
    def get_measurement(self) -> LandMark:
        left_land_coord_noise = np.diag(np.random.normal(loc=np.array([self.land_mark.left_coord.x, self.land_mark.left_coord.y]), scale=self.sensor_noise))
        right_land_coord_noise = np.diag(np.random.normal(loc=np.array([self.land_mark.right_coord.x, self.land_mark.right_coord.y]), scale=self.sensor_noise))
        lc = Coord(left_land_coord_noise[0], left_land_coord_noise[1])
        rc = Coord(right_land_coord_noise[0], right_land_coord_noise[1])
        noise_land_mark = LandMark(lc, rc)

        return noise_land_mark

    def particles_move(self):
        pass

    def weight_calc(self):
        pass

    def particle_screening(self):
        pass

    def estimate_robot_state(self):
        pass

