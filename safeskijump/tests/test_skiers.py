from math import isclose

import numpy as np
import matplotlib.pyplot as plt

from ..skiers import Skier
from ..utils import vel2speed
from ..classes import Surface


def test_skier(plot=False):

    mass = 75.0
    area = 0.5
    drag_coeff = 1.0
    friction_coeff = 0.3
    air_density = 0.85

    skier = Skier(mass, area, drag_coeff, friction_coeff)

    assert isclose(skier.mass, mass)
    assert isclose(skier.area, area)
    assert isclose(skier.drag_coeff, drag_coeff)
    assert isclose(skier.friction_coeff, friction_coeff)

    vel = -10.0

    assert isclose(skier.drag_force(vel), 1 / 2 * vel**2 * air_density *
                   drag_coeff * area)

    assert isclose(skier.friction_force(vel, slope=10.0),
                   friction_coeff * mass * 9.81 * np.cos(np.tan(10.0)))

    takeoff_pos = (4.0, 3.0)  # x, y
    takeoff_vel = (1.0, 10.0)  # vx, vy

    surf = Surface(np.linspace(0.0, 10.0, num=10), np.zeros(10))

    flight_traj = skier.fly_to(surf, takeoff_pos, takeoff_vel)

    if plot:
        ax = surf.plot()
        flight_traj.plot(ax=ax)

    landing_pos = tuple(flight_traj.pos[-1])
    landing_vel = tuple(flight_traj.vel[-1])

    takeoff_speed, takeoff_angle = vel2speed(*takeoff_vel)

    takeoff_speed2, landing_vel2 = skier.speed_to_land_at(landing_pos,
                                                          takeoff_pos,
                                                          takeoff_angle,
                                                          surf=surf)

    assert isclose(takeoff_speed, takeoff_speed2, rel_tol=1e-5)
    assert isclose(landing_vel[0], landing_vel2[0], rel_tol=1e-5)
    assert isclose(landing_vel[1], landing_vel2[1], rel_tol=1e-5)

    x = np.linspace(0, 20)
    y = -1.0 * x + 10.0

    x = np.linspace(0, 6 * np.pi)
    y = np.sin(x)

    surf = Surface(x, y)

    times, traj = skier.slide_on(surf, 50.0)

    if plot:
        plt.plot(times, traj.T)

    if plot:
        plt.show()
