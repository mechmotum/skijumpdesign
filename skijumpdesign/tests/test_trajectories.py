from math import isclose

import numpy as np
import matplotlib.pyplot as plt

from ..skiers import Skier
from ..surfaces import Surface
from ..trajectories import Trajectory


def test_trajectory(plot=False):
    skier = Skier()

    takeoff_pos = (4.0, 3.0)  # x, y
    takeoff_vel = (1.0, 10.0)  # vx, vy

    surf = Surface(np.linspace(0.0, 10.0, num=10), np.zeros(10))

    traj = skier.fly_to(surf, takeoff_pos, takeoff_vel)

    if plot:
        traj.plot_time_series()
        plt.plot()


def test_gradients():

    t = np.linspace(4.0, 16.0, num=1000)
    x = 1.27 * t**3
    y = np.sin(1.27 * t)
    vx = 3 * 1.27 * t**2
    vy = 1.27 * np.cos(1.27 * t)
    ax = 2 * 3 * 1.27 * t
    ay = -1.27**2 * np.sin(1.27 * t)

    traj = Trajectory(t, np.vstack((x, y)).T,
                      vel=np.vstack((vx, vy)).T,
                      acc=np.vstack((ax, ay)).T)

    assert isclose(traj.duration, 12.0)

    np.testing.assert_allclose(vy / vx, traj.slope)
    np.testing.assert_allclose(np.arctan(vy / vx), traj.angle)

    # if vel isn't supplied then numerical differentiation is used which gives
    # less accuracy
    traj = Trajectory(t, np.vstack((x, y)).T)
    np.testing.assert_allclose(vy / vx, traj.slope, atol=1e-5)
    np.testing.assert_allclose(np.arctan(vy / vx), traj.angle, atol=1e-5)


def test_interp_wrt_x():
    t = np.linspace(5.0, 15.0)
    x = np.linspace(0.0, 10.0)
    y = 5.0 * x + 2.0

    traj = Trajectory(t, np.vstack((x, y)).T)

    assert isclose(traj.duration, 10.0)

    res = traj.interp_wrt_x(2.33)

    assert isclose(res[1], 2.33)  # x
    assert isclose(res[2], 5.0 * 2.33 + 2.0)  # y
    assert isclose(res[7], 5.0)  # slope
