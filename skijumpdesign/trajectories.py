import os

import numpy as np
from scipy.interpolate import interp1d
if 'ONHEROKU' in os.environ:
    plt = None
else:
    import matplotlib.pyplot as plt

from .utils import EPS


class Trajectory(object):
    """Class that describes a 2D trajectory."""

    def __init__(self, t, pos, vel=None, acc=None, speed=None):
        """Instantiates a trajectory.

        Parameters
        ==========
        t : array_like, shape(n,)
            The time values of the trajectory.
        pos : array_like, shape(n, 2)
            The x and y coordinates of the position.
        vel : array_like, shape(n, 2), optional
            The x and y components of velocity. If not provided numerical
            differentiation of position will be used.
        acc : array_like, shape(n, 2), optional
            The x and y components of acceleration. If not provided numerical
            differentiation of velocity will be used.
        speed : array_like, shape(n, 2), optional
            The magnitude of the velocity. If not provided it will be
            calculated from the velocity components.

        """

        self.t = t
        self.pos = pos

        if vel is None:
            self.slope = np.gradient(self.pos[:, 1], self.pos[:, 0],
                                     edge_order=2)
            vel = np.gradient(pos, t, axis=0, edge_order=2)
        else:
            # assumes that the velocity was calculated more accurately
            self.slope = vel[:, 1] / (vel[:, 0] + EPS)

        self.vel = vel

        if speed is None:
            self.speed = np.sqrt(np.sum(self.vel**2, axis=1))
        else:
            self.speed = speed

        self.angle = np.arctan(self.slope)

        # TODO : Would be nice to be able to pass in acc=False to skip this
        # gradient.
        if acc is None:
            acc = np.gradient(self.vel, t, axis=0, edge_order=2)

        self.acc = acc

        self._initialize_trajectory()

    def _initialize_trajectory(self):

        self._construct_traj()
        self._initialize_interpolators()

    def _construct_traj(self):

        self._traj = np.hstack((np.atleast_2d(self.t).T,  # 0
                                self.pos,  # 1, 2
                                self.vel,  # 3, 4
                                self.acc,  # 5, 6
                                np.atleast_2d(self.slope).T,  # 7
                                np.atleast_2d(self.angle).T,  # 8
                                np.atleast_2d(self.speed).T,  # 9
                                ))

    def _initialize_interpolators(self):

        kwargs = {'fill_value': 'extrapolate',
                  'copy': False,
                  'assume_sorted': True,
                  'axis': 0}
        self.interp_pos_wrt_x = interp1d(self.pos[:, 0], self.pos, **kwargs)
        self.interp_wrt_x = interp1d(self.pos[:, 0], self._traj, **kwargs)
        self.interp_pos_wrt_slope = interp1d(self.slope, self.pos, **kwargs)

    @property
    def duration(self):
        """Returns the duration of the trajectory in seconds."""
        return self.t[-1] - self.t[0]

    def shift_coordinates(self, delx, dely):
        """Shifts the x and y coordinates by delx and dely respectively. This
        modifies the surface in place."""
        self.pos[:, 0] += delx
        self.pos[:, 1] += dely
        self._initialize_trajectory()

    def plot_time_series(self):
        """Plots all of the time series stored in the trajectory."""
        fig, axes = plt.subplots(2, 2)

        idxs = [1, 2, 7, 8]
        labels = ['Horizontal Position [m]',
                  'Vertical Position [m]',
                  'Slope [m/m]',
                  'Angle [rad]']
        for traj, ax, lab in zip(self._traj[:, idxs].T, axes.flatten(), labels):
            ax.plot(self.t, traj)
            ax.set_ylabel(lab)
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 1].set_xlabel('Time [s]')
        plt.tight_layout()

        def make_plot(data, word, unit):
            fig, axes = plt.subplots(2, 1, sharex=True)
            axes[0].set_title('{} Plots'.format(word))
            axes[0].plot(self.t, data)
            axes[0].set_ylabel('{} [{}]'.format(word, unit))
            axes[0].legend([r'${}_x$'.format(word[0].lower()),
                            r'${}_y$'.format(word[0].lower())])
            axes[1].plot(self.t, np.sqrt(np.sum(data**2, axis=1)))
            axes[1].set_ylabel('Magnitude of {} [{}]'.format(word, unit))
            axes[1].set_xlabel('Time [s]')

        make_plot(self.vel, 'Velocity', 'm/s')
        make_plot(self.acc, 'Acceleration', 'm/s/s')

        return axes

    def plot(self, ax=None, **plot_kwargs):
        """Returns a matplotlib axes containing a plot of the trajectory
        position.

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
