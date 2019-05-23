from math import isclose

import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
import pytest

from ..skiers import Skier
from ..functions import make_jump
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


def test_calculate_efh(profile=False):

    slope_angle = -15.0
    approach_len = 40
    takeoff_angle = 25.0
    fall_height = 0.5
    skier = Skier()

    slope, approach, takeoff, landing, landing_trans, flight, outputs = \
        make_jump(slope_angle, 0.0, approach_len, takeoff_angle, fall_height)

    if profile:
        from pyinstrument import Profiler
        p = Profiler()
        p.start()

    dist, efh = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff.end,
                                      skier)

    if profile:
        p.stop()
        print(p.output_text(unicode=True, color=True))

    np.testing.assert_allclose(np.diff(dist), 0.2 * np.ones(len(dist) - 1))
    np.testing.assert_allclose(efh[0], 0.0)
    np.testing.assert_allclose(efh[1:], fall_height, rtol=0.0, atol=8e-3)

    dist, _ = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff.end,
                                    skier, increment=0.1)
    np.testing.assert_allclose(np.diff(dist), 0.1 * np.ones(len(dist) - 1))

    # Check if a surface that is before the takeoff point gives an error
    with pytest.raises(InvalidJumpError):
        dist, _ = takeoff.calculate_efh(np.deg2rad(takeoff_angle), takeoff.end,
                                        skier)

    # Create a surface with takeoff and landing to check if function only
    # calculates takeoff point and beyond
    x = np.concatenate([takeoff.x, landing.x])
    y = np.concatenate([takeoff.y, landing.y])
    new_surf = Surface(x, y)
    dist, efh = new_surf.calculate_efh(np.deg2rad(takeoff_angle), takeoff.end, skier)
    np.testing.assert_allclose(efh[0], 0.0)
    np.testing.assert_allclose(efh[1:], fall_height, rtol=0.0, atol=8e-3)
    np.testing.assert_allclose(np.diff(dist), 0.2 * np.ones(len(dist) - 1))

    # Create a surface where distance values are not monotonic
    nonmonotonic_surf = Surface([2, 1, 3], [2, 4, 5])
    with pytest.raises(InvalidJumpError):
        nonmonotonic_surf.calculate_efh(np.deg2rad(takeoff_angle), (0, 0), skier)

    # Test function when takeoff point is in the first quadrant relative to
    # initial takeoff point (takeoff.end)
    takeoff_quad1 = (takeoff.end[0] + 2, takeoff.end[1] + 2)
    _, efh1 = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff_quad1,
                                    skier, increment=0.2)
    expected_quad1 = \
        np.array([2.28111451, 2.28166619, 2.27965337, 2.27491615, 2.26950806,
                 2.26108497, 2.25135037, 2.24015864, 2.22771691, 2.2142149,
                 2.19970402, 2.1843063, 2.16816956, 2.15140652, 2.13414037,
                 2.11641317, 2.09830531, 2.07988788, 2.06124532, 2.04244922,
                 2.02351379, 2.00449806, 1.98544223, 1.96639986, 1.94740442,
                 1.92846889, 1.90963346, 1.89090178, 1.87231766, 1.85389861,
                 1.83565513, 1.81759277, 1.79973638, 1.78209441, 1.76467057,
                 1.74746951, 1.73050309, 1.71377598, 1.69729497, 1.6810601,
                 1.66507211, 1.64933293, 1.63383985, 1.61859881, 1.60360605,
                 1.58888161, 1.57444257, 1.56023023, 1.5461859, 1.53229224,
                 1.51878114, 1.50552378, 1.49248902, 1.47967254, 1.46705516,
                 1.45470359, 1.442567, 1.43065526, 1.41895593, 1.40746713,
                 1.39619489, 1.3851331, 1.37427177, 1.36358857, 1.3531055,
                 1.34279368, 1.33250341, 1.32236275, 1.31250295, 1.30299843,
                 1.29359953, 1.28436784, 1.27530281, 1.26640151, 1.25789669])
    np.testing.assert_allclose(expected_quad1, efh1, rtol=1e-5)

    # Test function quadrant 2, negative takeoff angle
    takeoff_quad2 = (takeoff.end[0] - 2, takeoff.end[1] + 2)
    with pytest.raises(InvalidJumpError):
        dist, _ = landing.calculate_efh(np.deg2rad(-takeoff_angle), takeoff_quad2,
                                        skier, increment=0.2)

    # Test quadrant 2, positive takeoff angle
    _, efh2 = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff_quad2,
                                    skier, increment=0.2)
    expected_quad2 = \
        np.array([2.55544333, 2.80511855, 2.81025242, 2.74435538, 2.74534714,
                  2.72620195, 2.69649282, 2.66121638, 2.62764705, 2.60167897,
                  2.5684536, 2.53356526, 2.49748301, 2.46074811, 2.42379182,
                  2.38600488, 2.34767862, 2.30903868, 2.27030479, 2.2316696,
                  2.19303455, 2.15451746, 2.11623567, 2.07829158, 2.04079621,
                  2.00373476, 1.96716306, 1.93112212, 1.89568305, 1.86088816,
                  1.82673471, 1.79324093, 1.76042482, 1.72831694, 1.69693405,
                  1.66626193, 1.63629984, 1.60705506, 1.57853497, 1.55072874,
                  1.52363363, 1.49723158, 1.47152776, 1.44650197, 1.42214666,
                  1.39845328, 1.37540193, 1.35297996, 1.33117669, 1.30998353,
                  1.28937945, 1.26927482, 1.24993235, 1.23098841, 1.21262954,
                  1.19477266, 1.17745842, 1.16061509, 1.14416824, 1.12809673,
                  1.112611, 1.09755312, 1.08296561, 1.0687353, 1.05488337,
                  1.04145067, 1.02874884, 1.0160538, 1.00336374, 0.99171395,
                  0.98004604, 0.96871201, 0.95768052, 0.94693535, 0.93648955,
                  0.92630525, 0.91624628, 0.90642318, 0.8969507, 0.88788671,
                  0.87900833, 0.87036115, 0.86194071, 0.85374213, 0.84588294])
    np.testing.assert_allclose(expected_quad2, efh2, rtol=1e-5)

    # Test function quadrant 3
    takeoff_quad3 = (takeoff.end[0] - 2, takeoff.end[1] - 2)
    with pytest.raises(InvalidJumpError):
        dist, _ = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff_quad3,
                                        skier, increment=0.2)

    # Test function quadrant 4
    takeoff_quad4 = (takeoff.end[0] + 2, takeoff.end[1] - 2)
    with pytest.raises(InvalidJumpError):
        dist, _ = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff_quad4,
                                        skier, increment=0.2)
