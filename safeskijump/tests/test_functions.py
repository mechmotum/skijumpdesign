from math import isclose

import matplotlib.pyplot as plt

from ..functions import *


def test_compute_approach_exit_speed():

    parent_slope_angle = 10.0  # degrees
    start_pos = 10.0  # meters
    approach_len = 50.0  # meters

    approach_exit_speed = compute_approach_exit_speed(parent_slope_angle,
                                                      start_pos, approach_len)

    # NOTE : The Matlab app does much more manipulation of the numbers
    # (interpolation, etc) and gets a slightly diferent result. Not sure if
    # this is due to an error of mine or the Matlab app being less precise.
    assert isclose(approach_exit_speed, 11.4815)

def test_generate_launch_curve(plot=False):

    slope_angle = 10.0  # degrees
    start_pos = 10.0  # meters
    approach_len = 50.0  # meters
    takeoff_angle = 20.0  # degrees
    tolerable_acc = 1.5  # G

    approach_exit_speed = compute_approach_exit_speed(slope_angle, start_pos,
                                                      approach_len)

    curve_X, curve_Y, dydx, angle = generate_launch_curve(slope_angle,
                                                          approach_exit_speed,
                                                          takeoff_angle,
                                                          tolerable_acc)

    if plot:
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(start_pos, start_pos + approach_len, num=len(curve_X))
        y = x * np.tan(-np.deg2rad(slope_angle))
        ax.plot(x[0], y[0], marker='x', markersize=14, linewidth=4)
        ax.plot(x, y)
        ax.plot(curve_X + x[-1], curve_Y + y[-1])
        plt.show()
