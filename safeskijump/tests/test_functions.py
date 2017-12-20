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
        plot_jump(start_pos, approach_len, slope_angle, takeoff_angle, curve_X,
                  curve_Y)
        plt.show()


def test_compute_design_speed(plot=False):
    slope_angle = 10.0  # degrees
    start_pos = 10.0  # meters
    approach_len = 50.0  # meters
    takeoff_angle = 20.0  # degrees
    tolerable_acc = 1.5  # G

    launch_entry_speed = compute_approach_exit_speed(slope_angle, start_pos,
                                                      approach_len)

    curve_x, curve_y, _, _ = generate_takeoff_curve(slope_angle,
                                                    launch_entry_speed,
                                                    takeoff_angle,
                                                    tolerable_acc)

    ramp_entry_speed = compute_design_speed(launch_entry_speed, slope_angle,
                                            curve_x, curve_y)

    curve_x, curve_y = add_takeoff_ramp(takeoff_angle, ramp_entry_speed,
                                        curve_x, curve_y)

    design_speed = compute_design_speed(launch_entry_speed, slope_angle,
                                        curve_x, curve_y)

    if plot:
        plot_jump(start_pos, approach_len, slope_angle, takeoff_angle, curve_x,
                  curve_y)
        plt.show()
