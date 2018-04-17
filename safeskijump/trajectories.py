import os

import numpy as np
from scipy.interpolate import interp1d
if 'ONHEROKU' in os.environ:
    plt = None
else:
    import matplotlib.pyplot as plt


class Trajectory(object):
    """Class that describes a 2D trajectory."""

    def __init__(self, t, pos, vel=None, acc=None, jer=None):
        """

        t : array_like, shape(n,)
        pos : array_like, shape(n, 2)
            The x and y coordinates of the position.
        vel : array_like, shape(n, 2)
        acc : array_like, shape(n, 2)
        jer : array_like, shape(n, 2)

        traj is [pos, vel, acc, jer] shape(n, 8)

        """

        self.t = t
        self.pos = pos

        if vel is None:
            vel = np.gradient(pos, t, axis=0, edge_order=2)

        self.vel = vel

        self.speed = np.sqrt(np.sum(vel**2, axis=1))
        self.slope = self.vel[:, 1] / self.vel[:, 0]
        self.angle = np.arctan(self.slope)

        if acc is None:
            acc = np.gradient(self.vel, t, axis=0, edge_order=2)

        self.acc = acc

        if jer is None:
            jer = np.gradient(self.acc, t, axis=0, edge_order=2)

        self.jer = jer

        self._traj = np.hstack((self.t,     # 0
                                self.pos,   # 1, 2
                                self.vel,   # 3, 4
                                self.acc,   # 5, 6
                                self.jer,   # 7, 8
                                self.slope, # 9
                                self.angle, # 10
                                self.speed, # 11
                                ))

        interp_kwargs = {'fill_value': 'extrapolate', 'axis': 0}
        self.interp_wrt_t = interp1d(self.t, self._traj, **interp_kwargs)
        self.interp_pos_wrt_t = interp1d(t, self.pos, **interp_kwargs)
        self.interp_slope_wrt_t = interp1d(t, self.slope, **interp_kwargs)
        self.interp_angle_wrt_t = interp1d(t, self.angle, **interp_kwargs)
        self.interp_vel_wrt_t = interp1d(t, self.vel, **interp_kwargs)
        self.interp_acc_wrt_t = interp1d(t, self.acc, **interp_kwargs)
        self.interp_jer_wrt_t = interp1d(t, self.jer, **interp_kwargs)

        self.interp_wrt_x = interp1d(self.pos[:, 0], self.traj,
                                     **interp_kwargs)

    def duration(self):
        return self.t[-1] - self.t[0]

    def plot_path(self, ax=None, **plot_kwargs):
        """Returns a matplotlib axes containing a plot of the surface.

        Parameters
        ==========
        ax : Axes
            An existing matplotlib axes to plot to.
        plot_kwargs : dict
            Arguments to be passed to Axes.plot().

        """

        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_ylabel('Vertical Position [m]')
            ax.set_xlabel('Horizontal Position [m]')

        ax.plot(self.pos[:, 0], self.pos[:, 1], **plot_kwargs)

        ax.set_aspect('equal')

        return ax
