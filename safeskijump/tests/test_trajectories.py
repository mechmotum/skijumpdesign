import numpy as np

from ..classes import Surface, Skier
from ..trajectories import Trajectory


def test_trajectory():
    skier = Skier()

    takeoff_pos = (4.0, 3.0)  # x, y
    takeoff_vel = (1.0, 10.0)  # vx, vy

    surf = Surface(np.linspace(0.0, 10.0, num=10), np.zeros(10))

    times, flight_traj = skier.fly_to(surf, takeoff_pos, takeoff_vel)

    traj = Trajectory(times, flight_traj[:2].T, vel=flight_traj[2:].T)

    traj.plot_path()
