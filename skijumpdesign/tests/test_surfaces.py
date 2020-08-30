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

    dist, efh, speeds = landing.calculate_efh(np.deg2rad(takeoff_angle),
                                              takeoff.end, skier)
    if profile:
        p.stop()
        print(p.output_text(unicode=True, color=True))

    expected_speeds = \
        np.array([ 0.        ,  0.64634268,  1.2356876 ,  1.76885108,  2.24964887,
                   2.69168606,  3.09866358,  3.47053895,  3.81252508,  4.12895563,
                   4.42290851,  4.69632597,  4.95135992,  5.1898158 ,  5.41341136,
                   5.62352591,  5.82141688,  6.00814636,  6.18473831,  6.35203728,
                   6.51082732,  6.66178867,  6.80554642,  6.94264205,  7.07360671,
                   7.19885863,  7.31883812,  7.43389184,  7.54435492,  7.65055735,
                   7.75276447,  7.85122643,  7.94619036,  8.03785699,  8.12644327,
                   8.21211438,  8.29503769,  8.37538282,  8.45328143,  8.5288697 ,
                   8.60226895,  8.67359403,  8.74294598,  8.81043762,  8.87615365,
                   8.94017827,  9.00259044,  9.06346839,  9.12288563,  9.18090629,
                   9.2375861 ,  9.29299052,  9.34717415,  9.40018621,  9.45207528,
                   9.50288513,  9.55266139,  9.60144541,  9.64915056,  9.69601049,
                   9.74202917,  9.78719679,  9.83154369,  9.87510007,  9.91785869,
                   9.95991937, 10.00126913, 10.04193202, 10.08192875, 10.12129032,
                  10.16002962, 10.19816754, 10.23572199, 10.27229304, 10.30913058,
                  10.34504771, 10.3800144 , 10.41491013, 10.44931605, 10.4832593 ,
                  10.51674639, 10.54978999, 10.5824021 , 10.61459589, 10.64642594])

    np.testing.assert_allclose(np.diff(dist), 0.2 * np.ones(len(dist) - 1))
    np.testing.assert_allclose(efh[0], 0.0)
    np.testing.assert_allclose(efh[1:], fall_height, rtol=0.0, atol=8e-3)
    np.testing.assert_allclose(speeds, expected_speeds, rtol=3.0e-5,
                               atol=3.0e-4)

    dist, _, _ = landing.calculate_efh(np.deg2rad(takeoff_angle), takeoff.end,
                                       skier, increment=0.1)
    np.testing.assert_allclose(np.diff(dist), 0.1 * np.ones(len(dist) - 1))

    # Check if a surface that is before the takeoff point gives an error
    with pytest.raises(InvalidJumpError):
        dist, _, _ = takeoff.calculate_efh(np.deg2rad(takeoff_angle),
                                           takeoff.end, skier)

    # Create a surface with takeoff and landing to check if function only
    # calculates takeoff point and beyond
    x = np.concatenate([takeoff.x, landing.x])
    y = np.concatenate([takeoff.y, landing.y])
    new_surf = Surface(x, y)
    dist, efh, _ = new_surf.calculate_efh(np.deg2rad(takeoff_angle),
                                          takeoff.end, skier)
    np.testing.assert_allclose(efh[0], 0.0)
    np.testing.assert_allclose(efh[1:], fall_height, rtol=0.0, atol=8e-3)
    np.testing.assert_allclose(np.diff(dist), 0.2 * np.ones(len(dist) - 1))

    # Create a surface where distance values are not monotonic
    with pytest.raises(InvalidJumpError):
        Surface([2, 1, 3], [2, 4, 5])

    # Test takeoff angle greater than pi/2
    with pytest.raises(InvalidJumpError):
        new_surf.calculate_efh(np.pi, takeoff.end, skier)

    # Test function when takeoff point is in the first quadrant relative to
    # initial takeoff point (takeoff.end)
    takeoff_quad1 = (landing.start[0] + 2, landing.start[1] + 2)
    _, efh1, _ = landing.calculate_efh(np.deg2rad(takeoff_angle),
                                       takeoff_quad1,
                                    skier, increment=0.2)
    expected_quad1 = \
        np.array([1.7835165, 1.78401272, 1.78217823, 1.77987242, 1.77628465,
                  1.76998945, 1.76337001, 1.75578132, 1.7473959, 1.73834729,
                  1.72865671, 1.71840936, 1.70768518, 1.69658602, 1.68518452,
                  1.67349779, 1.66157619, 1.64946937, 1.6372259, 1.62488912,
                  1.61246859, 1.59999239, 1.58749806, 1.57501062, 1.5625446,
                  1.55011972, 1.53774518, 1.52543527, 1.51321161, 1.50108674,
                  1.48906131, 1.47714836, 1.4653533, 1.45368229, 1.44214059,
                  1.43073275, 1.41946312, 1.40833569, 1.3973575, 1.38652761,
                  1.37583968, 1.3653022, 1.35491758, 1.34467922, 1.33459477,
                  1.3246687, 1.314938, 1.3053352, 1.2958138, 1.28635201,
                  1.27717301, 1.26814732, 1.25926159, 1.25050445, 1.24186084,
                  1.2333902, 1.22506106, 1.21686473, 1.20880351, 1.20087464,
                  1.19308445, 1.18542684, 1.17789294, 1.17046884, 1.16317604,
                  1.15598451, 1.14875674, 1.14161781, 1.13468529, 1.12802913,
                  1.12142533, 1.11492911, 1.1085392, 1.1022529, 1.09629121])
    np.testing.assert_allclose(expected_quad1, efh1, rtol=1e-3)

    # Test function quadrant 2, negative takeoff angle, skier reaches 100mph
    takeoff_quad2 = (landing.start[0] - 2, landing.start[1] + 2)
    _, efh_speed, _ = landing.calculate_efh(np.deg2rad(-takeoff_angle),
                                            takeoff_quad2, skier,
                                            increment=0.2)
    expected_speedskier = \
        np.array([2.19864804, 2.81796169, 3.02391351, 3.06098472, 3.37912743,
                  3.79123023, 4.43642894, 5.67024268, 9.02957195, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(expected_speedskier, efh_speed, rtol=1e-3)

    # Test quadrant 2, positive takeoff angle
    _, efh2, _ = landing.calculate_efh(np.deg2rad(takeoff_angle),
                                       takeoff_quad2, skier, increment=0.2)
    expected_quad2 = \
        np.array([2.06869294, 2.32862611, 2.3347512, 2.26367959, 2.26529656,
                  2.24632669, 2.21713456, 2.18302593, 2.1512251, 2.12735662,
                  2.09678855, 2.06501121, 2.03247095, 1.99966889, 1.96699731,
                  1.93383061, 1.90044012, 1.86702338, 1.83375702, 1.80080359,
                  1.76806412, 1.7356011, 1.70352141, 1.67190225, 1.64082434,
                  1.6102607, 1.58024025, 1.5507907, 1.52196707, 1.49379066,
                  1.46623922, 1.43932379, 1.41305071, 1.38744381, 1.36250567,
                  1.33821319, 1.31455562, 1.29153864, 1.269153, 1.24739955,
                  1.22625899, 1.20571586, 1.18576279, 1.16638949, 1.14758039,
                  1.12932359, 1.11160082, 1.0944017, 1.07765136, 1.06159372,
                  1.0458878, 1.03062399, 1.01586807, 1.00152386, 0.98760244,
                  0.97411286, 0.96106147, 0.94838167, 0.93598971, 0.92389684,
                  0.91232011, 0.90106331, 0.8901389, 0.87949807, 0.86953028,
                  0.85918949, 0.84947943, 0.8403958, 0.83124043, 0.82234776,
                  0.81371588, 0.8053377, 0.79719569, 0.78926783, 0.78157332,
                  0.77407325, 0.76664867, 0.75940032, 0.75243023, 0.7457996,
                  0.73929607, 0.73297045, 0.72681849, 0.72083236, 0.71508179])
    np.testing.assert_allclose(expected_quad2, efh2, rtol=1e-3)

    # Test quadrant 2, negative takeoff angle less than 45
    with pytest.raises(InvalidJumpError):
        dist, _, _ = landing.calculate_efh(np.deg2rad(-46), takeoff_quad2,
                                           skier, increment=0.2)

    # Test function quadrant 3
    takeoff_quad3 = (landing.start[0] - 2, landing.start[1] - 2)
    with pytest.raises(InvalidJumpError):
        dist, _, _ = landing.calculate_efh(np.deg2rad(takeoff_angle),
                                           takeoff_quad3, skier, increment=0.2)

    # Test function quadrant 4
    takeoff_quad4 = (landing.start[0] + 2, landing.start[1] - 2)
    with pytest.raises(InvalidJumpError):
        dist, _, _ = landing.calculate_efh(np.deg2rad(takeoff_angle),
                                           takeoff_quad4, skier, increment=0.2)
