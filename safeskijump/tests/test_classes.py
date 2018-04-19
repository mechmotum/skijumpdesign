from math import isclose

import numpy as np
import matplotlib.pyplot as plt

from ..classes import vel2speed
from ..classes import (Surface, FlatSurface, ClothoidCircleSurface,
                       TakeoffSurface, Skier)


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


def test_flat_surface():

    fsurf = FlatSurface(-np.deg2rad(10), 40, init_pos=(5.0, 5.0))

    assert isclose(fsurf.x[0], 5.0)
    assert isclose(fsurf.y[0], 5.0)
    assert isclose(np.mean(np.arctan(fsurf.slope)), -np.deg2rad(10))

    length = np.sqrt(10**2 + 10**2)

    fsurf = FlatSurface(np.deg2rad(45.0), length, num_points=100000)

    assert isclose(10.0 * 10.0 / 2.0, fsurf.area_under(), abs_tol=1e-2)
    assert isclose(5.0 * 5.0 / 2.0, fsurf.area_under(x_end=5.0), abs_tol=1e-2)
    assert isclose(5.0 * 5.0 * 1.5, fsurf.area_under(x_start=5.0), abs_tol=1e-2)
    assert isclose(2.5 * 5.0 + 2.5**2 / 2, fsurf.area_under(x_start=5.0,
                                                            x_end=7.5),
                   abs_tol=1e-2)

    assert isclose(length, fsurf.length())


def test_clothoid_circle_surface(plot=False):

    fsurf = FlatSurface(-np.deg2rad(10), 40)
    csurf = ClothoidCircleSurface(fsurf.angle, np.deg2rad(20), 15, 1.5)

    if plot:
        ax = fsurf.plot()
        ax = csurf.plot(ax=ax)
        plt.show()


def test_takeoff_surface(plot=False):

    skier = Skier()

    fsurf = FlatSurface(-np.deg2rad(10.0), 2.0)
    tsurf = TakeoffSurface(skier, fsurf.angle, np.deg2rad(10), 5.0,
                           init_pos=fsurf.end)

    if plot:
        ax = fsurf.plot()
        ax = tsurf.plot(ax=ax)
        plt.show()


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
