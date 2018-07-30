from math import isclose

import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
import pytest

from ..skiers import Skier
from ..surfaces import (Surface, FlatSurface, ClothoidCircleSurface,
                        TakeoffSurface, LandingTransitionSurface)
from ..utils import InvalidJumpError


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

    surface.shift_coordinates(3.0, 5.0)
    assert isclose(surface.start[0], 3.0)
    assert isclose(surface.start[1], 4.0)


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


def test_landing_trans_surface(plot=False):
    slope_angle = -10.0
    start_pos = 0.0
    approach_len = 50.0
    takeoff_angle = 20.0
    fall_height = 1.5

    skier = Skier()

    slope_angle = np.deg2rad(slope_angle)
    takeoff_angle = np.deg2rad(takeoff_angle)

    init_pos = (start_pos * np.cos(slope_angle),
                start_pos * np.sin(slope_angle))

    approach = FlatSurface(slope_angle, approach_len, init_pos=init_pos)

    takeoff_entry_speed = skier.end_speed_on(approach)
    takeoff = TakeoffSurface(skier, slope_angle, takeoff_angle,
                             takeoff_entry_speed, init_pos=approach.end)

    slope = FlatSurface(slope_angle, 100 * approach_len)

    takeoff_vel = skier.end_vel_on(takeoff, init_speed=takeoff_entry_speed)

    flight = skier.fly_to(slope, init_pos=takeoff.end, init_vel=takeoff_vel)

    landing_trans = LandingTransitionSurface(slope, flight, fall_height,
                                             skier.tolerable_landing_acc)

    xpara, ypara = landing_trans.find_parallel_traj_point()

    x_trans, char_dist = landing_trans.find_transition_point()

    if plot:
        ax = slope.plot()
        ax = takeoff.plot(ax=ax)
        ax = flight.plot(ax=ax)
        ax = landing_trans.plot(ax=ax)
        ax.plot(xpara, ypara, marker='o')
        ax.axvline(x_trans)
        plt.show()


def test_area_under():

    x = sm.symbols('x')
    y = 2.3 * x**3 + x/2 * sm.cos(x**2)
    y_func = sm.lambdify(x, y)

    x0, xf = 0.0, 15.0

    x_vals = np.linspace(x0, xf, num=1000)
    y_vals = y_func(x_vals)

    expected_area = float(sm.integrate(y, (x, x0, xf)).evalf())

    surf = Surface(x_vals, y_vals)

    assert isclose(surf.area_under(), expected_area, rel_tol=1e-4)

    x0, xf = 0.34, 10.24

    expected_area = float(sm.integrate(y, (x, x0, xf)).evalf())

    assert isclose(surf.area_under(x_start=x0, x_end=xf), expected_area,
                   rel_tol=1e-4)
