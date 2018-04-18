import numpy as np
import matplotlib.pyplot as plt

from ..classes import Surface, Skier


def test_trajectory():
    skier = Skier()

    takeoff_pos = (4.0, 3.0)  # x, y
    takeoff_vel = (1.0, 10.0)  # vx, vy

    surf = Surface(np.linspace(0.0, 10.0, num=10), np.zeros(10))

    traj = skier.fly_to(surf, takeoff_pos, takeoff_vel)

    traj.plot_time_series()
    plt.show()
