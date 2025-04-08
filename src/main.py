from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from numpy import array, ndarray
from numpy.typing import NDArray

# --------------------------
# Physical Constants
# --------------------------
# All units in Astronomical Units (AU), Solar Masses, and Years
GRAVITATIONAL_CONSTANT: float = 4 * np.pi**2 # AU^3/(Solar Mass * Year^2)

# --------------------------
# Core Physics Terminology
# --------------------------
"""
1. Newtonian Gravity: Force between masses calculated using F = G*(m1*m2)/r²
2. ODE (Ordinary Differential Equation): Equations describing system evolution
3. Numerical Integration: Solving equations computationally through approximation
4. State Vector: [x_pos, x_vel, y_pos, y_vel] describing an object's motion
"""

class CelestialBody:
    """Base class for astronomical objects with mass and orbital dynamics"""
    def __init__(self, name: str, mass: float, initial_state: ndarray, color: str) -> None:
        """
        Args:
            name: Identifier for the body
            mass: Mass in solar masses
            initial_state: [x_pos (AU), x_vel (AU/yr), y_pos (AU), y_vel (AU/yr)]
        """
        self.name: str = name
        self.mass: float = mass
        self.color: str = color

        self.x_pos: float = initial_state[0]
        self.y_pos: float = initial_state[2]
        self.x_vel: float = initial_state[1]
        self.y_vel: float = initial_state[3]

        self.trajectory: Dict[str, NDArray] = {
            'x': np.array([]),
            'y': np.array([]),
            'vx': np.array([]),
            'vy': np.array([]),
        }


    def update_trajectory(self, solution: ndarray) -> None:
        """Store computed orbital positions and velocities"""
        self.trajectory['x'] = solution[:, 0]
        self.trajectory['y'] = solution[:, 2]
        self.trajectory['vx'] = solution[:, 1]
        self.trajectory['vy'] = solution[:, 3]


class Star(CelestialBody):
    """Central massive body governing planetary orbits"""
    def __init__(self, name: str, mass: float, inital_state: ndarray, color: str):
        super().__init__(name, mass, inital_state, color)


class Planet(CelestialBody):
    """Orbiting body influenced by stellar gravity"""

    def __init__(
        self, name: str, mass: float, initial_state: ndarray, parent_star: Star, color: str
    ):
        """
        Args:
            parent_star: Central star providing gravitational influence
        """
        super().__init__(name, mass, initial_state, color)
        self.parent_star = parent_star


    def gravitationnal_acceleration_solver(self, state: ndarray) -> ndarray:
        """Calculates acceleration from gravitational interaction with parent star."""
        # vector from planet to star
        dx = state[0] - self.parent_star.x_pos
        dy = state[2] - self.parent_star.y_pos

        distance: float = float( np.linalg.norm([dx, dy]) )

        # Newtonian gravity: a = -G*M*r_vect/r^3
        acceleration_scale: float = (
            -GRAVITATIONAL_CONSTANT * self.parent_star.mass / distance**3
        )
        return array([acceleration_scale * dx, acceleration_scale * dy])


    def orbital_equation(self, state: ndarray, t: float) -> ndarray:
        """ODE system for orbital mechanics"""
        ax, ay = self.gravitationnal_acceleration_solver(state)
        return array([state[1], ax, state[3], ay])


class PlanetarySystem:
    """Container for celestial bodies and orbital computations"""

    def __init__(self, star: Star, planets: List[Planet]):
        self.star = star
        self.planets: List[Planet] = planets
        self.simulation_time: ndarray = array([])

    def compute_orbits(
        self, duration: float, time_step: float
    ) -> List[CelestialBody]:
        """
        Numerically integrate orbits over specified time period

        Args:
            duration: Total simulation time (years)
            time_step: Temporal resolution (years)
        """
        self.simulation_time = np.arange(0, duration + time_step, time_step)
        shape: tuple = (*self.simulation_time.shape, 4)

        for planet in self.planets:
            solution: ndarray = spi.odeint(
                func=planet.orbital_equation,
                y0=array(
                    [planet.x_pos, planet.x_vel, planet.y_pos, planet.y_vel]
                ),
                t=self.simulation_time,
            )

            planet.update_trajectory(solution=solution)

        star_solution: ndarray = np.zeros(shape)
        self.star.update_trajectory(solution=star_solution)

        return [
            self.star, # TODO: need to evolve state of star
            *self.planets,
        ]

class OrbitalVisualiser:
    """Handles visualisation of computed orbits"""

    @staticmethod
    def plot_orbits(system_data: List[CelestialBody]) -> None:
        """Generate 2D plot of celestial trajectories"""
        fig, ax = plt.subplots(figsize=(16, 9))

        for celestial_body in system_data:
            ax.plot(
                celestial_body.trajectory["x"],
                celestial_body.trajectory["y"],
                alpha=0.7,
                label=f"{celestial_body.name}",
                color=celestial_body.color,
            )
            ax.scatter(
                celestial_body.trajectory["x"][0],
                celestial_body.trajectory["y"][0],
                s=50,
                marker="o",
                color=celestial_body.color,
                edgecolors=celestial_body.color,
            )

        ax.set_title(f"Solar System Orbital Dynamics", fontsize=14)
        ax.set_xlabel("X Position (AU)", fontsize=12)
        ax.set_ylabel("Y Position (AU)", fontsize=12)

        # Set logarithmic scale for better outer planet visibility
        # ax.set_xscale('symlog', linthresh=10)
        # ax.set_yscale('symlog', linthresh=10)

        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    sun = Star(
        name="Sun",
        mass=1,
        inital_state=array([0.0, 0.0, 0.0, 0.0]),
        color="gold",
    )

    # Planetary orbital parameters (semi-major axis in AU, circular velocity in AU/yr)
    planetary_data = [
        # Inner rocky planets
        {"name": "Mercury", "mass": 1.65e-7, "semi_major_axis": 0.387,
         "velocity": 10.09, "color": "gray"},
        {"name": "Venus",   "mass": 2.45e-6, "semi_major_axis": 0.723,
         "velocity": 7.38,  "color": "goldenrod"},
        {"name": "Earth",   "mass": 3.0e-6,  "semi_major_axis": 1.0,
         "velocity": 6.28,  "color": "dodgerblue"},
        {"name": "Mars",    "mass": 3.3e-7,  "semi_major_axis": 1.52,
         "velocity": 5.06,  "color": "firebrick"},

        # Gas giants
        # {"name": "Jupiter", "mass": 9.54e-4, "semi_major_axis": 5.20,
        #  "velocity": 2.75,  "color": "darkorange"},
        # {"name": "Saturn",  "mass": 2.85e-4, "semi_major_axis": 9.58,
        #  "velocity": 2.03,  "color": "khaki"},

        # Ice giants
        # {"name": "Uranus",  "mass": 4.4e-5,  "semi_major_axis": 19.22,
        #  "velocity": 1.43,  "color": "lightsteelblue"},
        # {"name": "Neptune", "mass": 5.15e-5, "semi_major_axis": 30.05,
        #  "velocity": 1.14,  "color": "mediumslateblue"},
    ]

    # Create planet objects
    planets = [
        Planet(
            name=data["name"],
            mass=data["mass"],
            initial_state=np.array(
                [
                    data["semi_major_axis"],
                    0.0,  # Initial position (x, y)
                    0.0,
                    data["velocity"],  # Initial velocity (vx, vy)
                ]
            ),
            parent_star=sun,
            color=data["color"],
        )
        for data in planetary_data
    ]

    solar_system = PlanetarySystem(
        star=sun,
        planets=planets
    )

    # Run simulation for 2 years, with weekly temporal resolution (0.0192 yr)
    # system_data: Dict[str, List[CelestialBody]] = solar_system.compute_orbits(duration=2, time_step=0.0192)
    system_data: List[CelestialBody] = solar_system.compute_orbits(
        duration=20,  # Years
        time_step=0.01  # ~3.65 day temporal resolution
    )

    # visualise results
    OrbitalVisualiser.plot_orbits(system_data=system_data)

