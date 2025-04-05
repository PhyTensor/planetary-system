import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import Callable
    import numpy as np
    from numpy import ndarray
    import scipy.integrate as spi

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # For handling animations within the python notebook
    from matplotlib import rc, rcParams
    rc('animation', html='jshtml')
    rcParams['animation.embed_limit'] = 2**256
    return Callable, FuncAnimation, ndarray, np, plt, rc, rcParams, spi


@app.cell
def _(np):
    # Defining constants
    G: float = 4 * np.pi**2 # 39.
    solar_mass: float = 1
    return G, solar_mass


@app.cell
def Star():
    class Star:
        def __init__(self, mass, init_state):
            self.mass = mass
            self.init_state = init_state
            self.x: float = 0 # np.zeros(1)
            self.y: float = 0 # np.zeros(1)
    return (Star,)


@app.cell
def _(G, Star, ndarray, np):
    class Planet:
        def __init__(self, name, star, init_state):
            self.name = name
            self.init_state = init_state
            self.parent_star: Star = star

            # coordinates components of position and velocity
            self.x: ndarray = np.zeros(1)
            self.y: ndarray = np.zeros(1)
            self.vx: ndarray = np.zeros(1)
            self.vy: ndarray = np.zeros(1)


        def state_evolution(self, u: list, t: float) -> ndarray:
            d: float = np.linalg.norm( [ u[0] - self.parent_star.x, u[2] - self.parent_star.y ] )

            dudt: ndarray = np.zeros(4)
        
            xvec = ( u[0] - self.parent_star.x )
            yvec = ( u[2] - self.parent_star.y )
            # print(type(a), "\t", type(u[0]), "\t", type(self.parent_star.x))
            dudt[0] = u[1]
            dudt[1] = -G * self.parent_star.mass * xvec / d**3
            dudt[2] = u[3]
            dudt[3] = -G * self.parent_star.mass * yvec / d**3

            return dudt
    return (Planet,)


@app.cell
def _(Planet, Star, ndarray, np, plt, spi):
    class StellarSystem:
        def __init__(self, star: Star, planets: list):
            self.star = star
            self.planets = planets


        def execute(self, interval: float) -> None:
            t_span: tuple = (0, 2)
            n = int(20 / interval)

            t_eval: ndarray = np.linspace(*t_span, num=n)

            for i in range(len(self.planets)):
                planet: Planet = self.planets[i]
                U: ndarray = spi.odeint(planet.state_evolution, planet.init_state, t_eval)
                planet.x = U[:, 0] # np.append( planet.x,  U[:, 0] )
                planet.y = U[:, 2] # np.append( planet.y,  U[:, 2] )
                planet.vx = U[:, 1] # np.append( planet.vx,  U[:, 1] )
                planet.vy = U[:, 3] # np.append( planet.vy,  U[:, 3] )
                # print(planet.x)


        def make_plot(self) -> None:
            fig = plt.figure(figsize=(10, 6))

            for i in range(len(self.planets)):
                planet: Planet = self.planets[i]
                plt.plot(planet.x, planet.y, label=f'{planet.name}')
                # print(f'plotted')

            plt.plot(self.star.x, self.star.y, "ro", markersize=12)

            plt.title("Stellar System Planetary Orbit (x, y)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis('equal')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
    return (StellarSystem,)


@app.cell
def _(Planet, Star, StellarSystem, solar_mass):
    sun = Star(
        mass = solar_mass,
        init_state=[0, 0, 0, 0]
    )

    earth = Planet(
        # mass=(3e-6)*solar_mass,
        init_state=[1.017, 0, 0, 6.288],
        name = 'Earth',
        star=sun
    )

    mars = Planet(
        init_state=[1.52, 0, 0, 5.4],
        name='Mars',
        star=sun
    )

    system = StellarSystem(
        star=sun,
        planets=[earth, mars]
    )
    return earth, mars, sun, system


@app.cell
def _(system):
    system.execute(interval=1)
    return


@app.cell
def _(system):
    system.make_plot()
    return


if __name__ == "__main__":
    app.run()
