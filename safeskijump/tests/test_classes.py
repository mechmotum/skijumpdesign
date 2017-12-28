from math import isclose

import numpy as np
import matplotlib.pyplot as plt

from ..classes import Surface, Skier


def test_surface():

    x = np.linspace(0.0, 10.0)
    y = np.ones_like(x)

    surface = Surface(x, y)

    assert isclose(surface.interp_y(3.21), 1.0)
    assert isclose(surface.distance_from(0.0, 2.0), 1.0)

    x = np.linspace(0.0, 10.0)
    y = 5.0 * x - 1.0

    surface = Surface(x, y)

    assert isclose(surface.interp_y(0.0), -1.0)
    assert isclose(surface.distance_from(0.0, -1.0), 0.0)
    assert isclose(surface.distance_from(1.0 / 5.0, 0.0), 0.0, abs_tol=1E-10)
    assert isclose(surface.distance_from(-5.0, 0.0), np.sqrt(26),
                   abs_tol=1E-10)
    assert isclose(surface.distance_from(-10.0, 1.0), np.sqrt(10**2 + 2**2),
                   abs_tol=1E-10)


def test_linear_surface():
    pass


def test_circular_surface():
    pass


def test_clothoid_surface():
    pass


def test_exponential_surface():
    pass


def test_skier():

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

    slope = np.tan(10)

    #assert isclose(skier.friction_force(vel, slope=slope),
                   #friction_coeff * mass * 9.81 * np.cos(10.0))

    loc = (4.0, 3.0)  # x, y
    speed = (1.0, 10.0)  # vx, vy

    surf = Surface(np.linspace(0.0, 30.0, num=50), np.zeros(50))

    flight_traj = skier.fly_to(surf, loc, speed)

    x = np.linspace(0, 20)
    y = -1.0 * x + 10.0

    x = np.linspace(0, 6 * np.pi)
    y = np.sin(x)

    surf = Surface(x, y)

    times, traj = skier.slide_on(surf, 50.0)

    plt.plot(times, traj.T)
    plt.show()


def test_all():

    """
    skier
    trajectory
    slope/surface
    weather/environment
    """
    skier = Skier(mass, cross_sectional_area, friction_coeff)

    approach = ApproachSurface(angle, length)

    pos, vel, acc = skier.slide_over(approach, init_speed)

    takeoff = TakeoffSurface(angle, entry_speed, tolerable_acc)

    pos, vel, acc = skier.slide_over(takeoff)

    flight_pos, flight_vel, flight_acc = skier.fly(init_pos, init_speed)

    landing = LandingSurface(slope_angle, flight_pos, flight_vel)

    pos, vel, acc = skier.slide_over(landing, init_speed)
