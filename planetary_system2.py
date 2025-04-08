import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import Callable, Dict, List

    import numpy as np
    from numpy import ndarray, array
    import scipy.integrate as spi
    import matplotlib
    import matplotlib.pyplot as plt

    # %matplotlib inline
    # matplotlib.use('nbagg')
    return Callable, Dict, List, array, matplotlib, ndarray, np, plt, spi


@app.cell
def _(np):
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
    return (GRAVITATIONAL_CONSTANT,)


@app.cell
def _(Dict, ndarray):
    class CelestialBody:
        """Base class for astronomical objects with mass and orbital dynamics"""
        def __init__(self, name: str, mass: float, initial_state: ndarray) -> None:
            """
            Args:
                name: Identifier for the body
                mass: Mass in solar masses
                initial_state: [x_pos (AU), x_vel (AU/yr), y_pos (AU), y_vel (AU/yr)]
            """
            self.name = name
            self.mass = mass

            self.x_pos = initial_state[0]
            self.y_pos = initial_state[2]
            self.x_vel = initial_state[1]
            self.y_vel = initial_state[3]

            self.trajectory: Dict = { 'x': [], 'y': [], 'vx': [], 'vy': [] }


        def update_trajectory(self, solution: ndarray) -> None:
            """Store computed orbital positions and velocities"""
            self.trajectory['x'] = solution[:, 0]
            self.trajectory['y'] = solution[:, 2]
            self.trajectory['vx'] = solution[:, 1]
            self.trajectory['vy'] = solution[:, 3]
    return (CelestialBody,)


@app.cell
def _(CelestialBody, ndarray):
    class Star(CelestialBody):
        """Central massive body governing planetary orbits"""
        def __init__(self, name: str, mass: float, inital_state: ndarray):
            super().__init__(name, mass, inital_state)
    return (Star,)


@app.cell
def _(CelestialBody, GRAVITATIONAL_CONSTANT, Star, array, ndarray, np):
    class Planet(CelestialBody):
        """Orbiting body influenced by stellar gravity"""

        def __init__(
            self, name: str, mass: float, initial_state: ndarray, parent_star: Star, color: str
        ):
            """
            Args:
                parent_star: Central star providing gravitational influence
            """
            super().__init__(name, mass, initial_state)
            self.parent_star = parent_star
            self.color = color

        def gravitationnal_acceleration_solver(self, state: ndarray) -> ndarray:
            """Calculates acceleration from gravitational interaction with parent star."""
            # vector from planet to star
            dx = state[0] - self.parent_star.x_pos
            dy = state[2] - self.parent_star.y_pos

            distance = np.linalg.norm([dx, dy])

            # Newtonian gravity: a = -G*M*r_vect/r^3
            acceleration_scale: float = (
                -GRAVITATIONAL_CONSTANT * self.parent_star.mass / distance**3
            )
            return acceleration_scale * dx, acceleration_scale * dy

        def orbital_equation(self, state: ndarray, t: float) -> ndarray:
            """ODE system for orbital mechanics"""
            ax, ay = self.gravitationnal_acceleration_solver(state)
            return array([state[1], ax, state[3], ay])
    return (Planet,)


@app.cell
def _(CelestialBody, Dict, List, Planet, Star, array, ndarray, np, spi):
    class PlanetarySystem:
        """Container for celestial bodies and orbital computations"""

        def __init__(self, star: Star, planets: List[Planet]):
            self.star = star
            self.planets: List[Planet] = planets
            self.simulation_time = None

        def compute_orbits(
            self, duration: float, time_step: float
        ) -> Dict[str, List[CelestialBody]]:
            """
            Numerically integrate orbits over specified time period

            Args:
                duration: Total simulation time (years)
                time_step: Temporal resolution (years)
            """
            self.simulation_time = np.arange(0, duration + time_step, time_step)

            for planet in self.planets:
                solution: ndarray = spi.odeint(
                    func=planet.orbital_equation,
                    y0=array(
                        [planet.x_pos, planet.x_vel, planet.y_pos, planet.y_vel]
                    ),
                    t=self.simulation_time,
                )

                # print(f"{solution}")
                planet.update_trajectory(solution=solution)

            return {"star": self.star, "planets": self.planets}
    return (PlanetarySystem,)


@app.cell
def _(CelestialBody, Dict, List, plt):
    class OrbitalVisualiser:
        """Handles visualisation of computed orbits"""

        @staticmethod
        def plot_orbits(system_data: Dict[str, List[CelestialBody]]) -> None:
            """Generate 2D plot of celestial trajectories"""
            fig, ax = plt.subplots(figsize=(16, 9))

            # plot star
            star = system_data["star"]
            ax.scatter(
                star.x_pos,
                star.y_pos,
                c="gold",
                s=500,
                label=star.name,
                marker="o",
                edgecolor='orange'
            )

            # Plot planetary orbits
            for planet in system_data["planets"]:
                ax.plot(
                    planet.trajectory["x"],
                    planet.trajectory["y"],
                    alpha=0.7,
                    label=f"{planet.name}'s orbit",
                )
                ax.scatter(
                    planet.trajectory["x"][0],
                    planet.trajectory["y"][0],
                    s=50,
                    marker="o",
                )

            ax.set_title(f"Solar System Orbital Dynamics", fontsize=14)
            ax.set_xlabel("X Position (AU)", fontsize=12)
            ax.set_ylabel("Y Position (AU)", fontsize=12)

            # Set logarithmic scale for better outer planet visibility
            # ax.set_xscale('symlog', linthresh=10)
            # ax.set_yscale('symlog', linthresh=10)

            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            # ax.set_aspect("equal")
            plt.axis('equal')
            plt.show()
    return (OrbitalVisualiser,)


@app.cell
def _(Star, array):
    sun = Star(name="Sun", mass=1, inital_state=array([0.0, 0.0, 0.0, 0.0]))
    return (sun,)


@app.cell
def _():
    # Planetary orbital parameters (semi-major axis in AU, circular velocity in AU/yr)
    planetary_data = [
        # Inner rocky planets
        # {"name": "Mercury", "mass": 1.65e-7, "semi_major_axis": 0.387,
        #  "velocity": 10.09, "color": "gray"},
        {"name": "Venus",   "mass": 2.45e-6, "semi_major_axis": 0.723,
         "velocity": 7.38,  "color": "goldenrod"},
        {"name": "Earth",   "mass": 3.0e-6,  "semi_major_axis": 1.0,
         "velocity": 6.28,  "color": "dodgerblue"},
        {"name": "Mars",    "mass": 3.3e-7,  "semi_major_axis": 1.52,
         "velocity": 5.06,  "color": "firebrick"},

        # Gas giants
        {"name": "Jupiter", "mass": 9.54e-4, "semi_major_axis": 5.20,
         "velocity": 2.75,  "color": "darkorange"},
        # {"name": "Saturn",  "mass": 2.85e-4, "semi_major_axis": 9.58,
        #  "velocity": 2.03,  "color": "khaki"},

        # Ice giants
        # {"name": "Uranus",  "mass": 4.4e-5,  "semi_major_axis": 19.22,
        #  "velocity": 1.43,  "color": "lightsteelblue"},
        # {"name": "Neptune", "mass": 5.15e-5, "semi_major_axis": 30.05,
        #  "velocity": 1.14,  "color": "mediumslateblue"},
    ]
    return (planetary_data,)


@app.cell
def _(Planet, np, planetary_data, sun):
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
    return (planets,)


@app.cell
def _(PlanetarySystem, planets, sun):
    solar_system = PlanetarySystem(
        star=sun,
        planets=planets
    )
    return (solar_system,)


@app.cell
def _(solar_system):
    # Run simulation for 2 years, with weekly temporal resolution (0.0192 yr)
    # system_data: Dict[str, List[CelestialBody]] = solar_system.compute_orbits(duration=2, time_step=0.0192)
    system_data = solar_system.compute_orbits(
        duration=20,  # Years
        time_step=0.01  # ~3.65 day temporal resolution
    )
    return (system_data,)


@app.cell
def _(OrbitalVisualiser, system_data):
    # visualise results
    OrbitalVisualiser.plot_orbits(system_data=system_data)
    return


if __name__ == "__main__":
    app.run()
