import os
import time
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapz, quad

from .utils import InvalidJumpError
from .utils import GRAV_ACC, EPS
from .utils import compute_dist_from_flat, vel2speed


if 'ONHEROKU' in os.environ:
    plt = None
else:
    import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class Surface(object):
    """Base class for a 2D curve that represents the cross section of a surface
    expressed in a standard Cartesian coordinate system."""

    def __init__(self, x, y):
        """Instantiates an arbitrary 2D surface.

        Parameters
        ==========
        x : array_like, shape(n,)
            The horizontal, x, coordinates of the slope. x[0] should be the
            left most horizontal position and corresponds to the start of the
            surface. This should be monotonically increasing.
        y : array_like, shape(n,)
            The vertical, y, coordinates of the slope. y[0] corresponds to the
            start of the surface.

        """

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self._initialize_surface()

    def _initialize_surface(self):

        self._initialize_gradients()
        self._initialize_interpolators()

    def _initialize_gradients(self):

        self.slope = np.gradient(self.y, self.x, edge_order=2)
        slope_deriv = np.gradient(self.slope, self.x, edge_order=2)
        self.curvature = slope_deriv / (1 + self.slope**2)**1.5

    def _initialize_interpolators(self):

        kwargs = {'fill_value': 'extrapolate'}
        self.interp_y = interp1d(self.x, self.y, **kwargs)
        self.interp_slope = interp1d(self.x, self.slope, **kwargs)
        self.interp_curvature = interp1d(self.x, self.curvature, **kwargs)

    @property
    def start(self):
        """Returns the x and y coordinates at the start point of the
        surface."""
        return self.x[0], self.y[0]

    @property
    def end(self):
        """Returns the x and y coordinates at the end point of the surface."""
        return self.x[-1], self.y[-1]

    def shift_coordinates(self, delx, dely):
        """Shifts the x and y coordinates by delx and dely respectively. This
        modifies the surface in place."""
        self.x += delx
        self.y += dely
        # NOTE : Only the interpolators have to be reinitialized, the gradients
        # don't have to be computed again. For now, this method is here for
        # consistency among *Surface classes.
        self._initialize_surface()

    def distance_from(self, xp, yp):
        """Returns the shortest distance from point (xp, yp) to the surface.

        Parameters
        ==========
        xp : float
            The horizontal, x, coordinate of the point.
        yp : float
            The vertical, y, coordinate of the point.

        Returns
        =======
        distance : float
            The shortest distance from the point to the surface. If the point
            is above the surface a positive distance is returned, else a
            negative distance.

        Note
        ====
        This general implementation can be slow, so implement overloaded
        ``distance_from()`` methods in subclasses when you can.

        """

        def distance_squared(x):
            return (xp - x)**2 + (yp - self.interp_y(x))**2

        distances = np.sqrt((self.x - xp)**2 + (self.y - yp)**2)

        x = fsolve(distance_squared, self.x[np.argmin(distances)])

        return np.sign(yp - self.interp_y(x)) * np.sqrt(distance_squared(x))

    def length(self):
        """Returns the length of the surface in meters via a numerical line
        integral."""
        def func(x):
            return np.sqrt(1.0 + self.interp_slope(x)**2)
        return quad(func, self.x[0], self.x[-1])[0]

    def area_under(self, x_start=None, x_end=None, interval=0.05):
        """Returns the area under the curve integrating wrt to the x axis at
        0.05 m intervals using the trapezoidal rule."""
        if x_start is not None:
            if x_start < self.start[0] or x_start > self.end[0]:
                raise ValueError('x_start has to be between start and end.')
        else:
            x_start = self.start[0]
        if x_end is not None:
            if x_end < self.start[0] or x_end > self.end[0]:
                raise ValueError('x_end has to be between start and end.')
        else:
            x_end = self.end[0]
        x = np.linspace(x_start, x_end, num=(x_end - x_start) / interval)
        y = self.interp_y(x)
        return trapz(y, x)

    def height_above(self, surface):
        """Returns an array of values giving the height each point in this
        surface is above the provided surface."""
        return self.y - surface.interp_y(self.x)

    def plot(self, ax=None, **plot_kwargs):
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

        ax.plot(self.x, self.y, **plot_kwargs)

        ax.set_aspect('equal')

        return ax


class HorizontalSurface(Surface):
    def __init__(self, height, length, start=0.0, num_points=100):
        """Instantiates a class that represents a horizontal surface at a
        height above the x axis.abs

        Parameters
        ==========
        height : float
            The height of the surface above the horizontal x axis in meters.
        length : float
            The length of the surface in meters.
        start : float, optional
            The x location of the start of the left most point of the surface.
        num_points : integer, optional
            The number of (x,y) coordinates.

        """
        x = np.linspace(start, start + length, num=num_points)
        y = height * np.ones_like(x)
        super(HorizontalSurface, self).__init__(x, y)

    def distance_from(self, xp, yp):
        """Returns the shortest distance from point (xp, yp) to the surface.

        Parameters
        ==========
        xp : float
            The horizontal, x, coordinate of the point.
        yp : float
            The vertical, y, coordinate of the point.

        Returns
        =======
        distance : float
            The shortest distance from the point to the surface. If the point
            is above the surface a positive distance is returned, else a
            negative distance.

        """
        return yp - self.y[0]


class FlatSurface(Surface):
    """Class that represents a flat surface angled relative to the
    horizontal."""

    def __init__(self, angle, length, init_pos=(0.0, 0.0), num_points=100):
        """Instantiates a flat surface that is oriented at a counterclockwise
        angle from the horizontal.

        Parameters
        ==========
        angle : float
            The angle of the surface in radians. Counterclockwise (about z) is
            positive, clockwise is negative.
        length : float
            The distance in meters along the surface from the initial position.
        init_pos : 2-tuple of floats, optional
            The x and y coordinates in meters that locate the start of the
            surface.
        num_points : integer, optional
            The number of points used to define the surface coordinates.

        """

        if angle >= np.pi / 2.0 or angle <= -np.pi / 2.0:
            raise InvalidJumpError('Angle must be between -90 and 90 degrees')

        self._angle = angle

        x = np.linspace(init_pos[0], init_pos[0] + length * np.cos(angle),
                        num=num_points)
        y = np.linspace(init_pos[1], init_pos[1] + length * np.sin(angle),
                        num=num_points)

        super(FlatSurface, self).__init__(x, y)

    @property
    def angle(self):
        """Returns the angle wrt to horizontal in radians of the surface."""
        return self._angle

    def distance_from(self, xp, yp):
        """Returns the shortest distance from point (xp, yp) to the surface.

        Parameters
        ==========
        xp : float
            The horizontal, x, coordinate of the point.
        yp : float
            The vertical, y, coordinate of the point.

        Returns
        =======
        distance : float
            The shortest distance from the point to the surface. If the point
            is above the surface a positive distance is returned, else a
            negative distance.

        """

        if compute_dist_from_flat is None:
            m = np.tan(self.angle)
            d = (yp - m * xp) * np.cos(self.angle)
            return d
        else:
            return compute_dist_from_flat(self.angle, xp, yp)


class ClothoidCircleSurface(Surface):
    """Class that represents a surface made up of a circle bounded by two
    clothoids."""

    def __init__(self, entry_angle, exit_angle, entry_speed, tolerable_acc,
                 init_pos=(0.0, 0.0), gamma=0.99, num_points=200):
        """Instantiates a clothoid-circle-clothoid curve.

        Parameters
        ==========
        entry_angle : float
            The entry angle tangent to the start of the left clothoid in
            radians.
        exit_angle : float
            The exit angle tangent to the end of the right clothoid in radians.
        entry_speed : float
            The magnitude of the skier's velocity in meters per second as they
            enter the left clothiod.
        tolerable_acc : float
            The tolerable normal acceleration of the skier in G's.
        init_pos : 2-tuple of floats
            The x and y coordinates of the start of the left clothoid.
        gamma : float
            Fraction of circular section.
        num_points : integer, optional
            The number of points in each of the three sections of the curve.

        """
        self.entry_angle = entry_angle
        self.exit_angle = exit_angle
        self.entry_speed = entry_speed
        self.tolerable_acc = tolerable_acc
        self.init_pos = init_pos
        self.gamma = gamma
        self.num_points = num_points

        X, Y = self._create_surface()

        super(ClothoidCircleSurface, self).__init__(X, Y)

    def _create_surface(self):
        # TODO : Break this function into smaller functions.

        lam = -self.entry_angle
        beta = self.exit_angle

        rotation_clothoid = (lam - beta) / 2
        # used to rotate symmetric clothoid so that left side is at lam and
        # right sid is at beta

        # radius_min is the radius of the circular part of the transition.
        # Every other radius length (in the clothoid) will be longer than that,
        # as this will ensure the g - force felt by the skier is always less
        # than a desired value. This code ASSUMES that the velocity at the
        # minimum radius is equal to the velocity at the end of the approach.
        radius_min = self.entry_speed**2 / (self.tolerable_acc * GRAV_ACC)

        #  x,y data for circle
        thetaCir = 0.5 * self.gamma * (lam + beta)
        xCirBound = radius_min * np.sin(thetaCir)
        xCirSt = -radius_min * np.sin(thetaCir)
        xCir = np.linspace(xCirSt, xCirBound, num=self.num_points)

        # x,y data for one clothoid
        A_squared = radius_min**2 * (1 - self.gamma) * (lam + beta)
        A = np.sqrt(A_squared)
        clothoid_length = A * np.sqrt((1 - self.gamma) * (lam + beta))

        # generates arc length points for one clothoid
        s = np.linspace(clothoid_length, 0, num=self.num_points)

        X1 = s - (s**5) / (40*A**4) + (s**9) / (3456*A**8)
        Y1 = (s**3) / (6*A**2) - (s**7) / (336*A**6) + (s**11) / (42240*A**10)

        X2 = X1 - X1[0]
        Y2 = Y1 - Y1[0]

        theta = (lam + beta) / 2
        X3 = np.cos(theta)*X2 + np.sin(theta)*Y2
        Y3 = -np.sin(theta)*X2 + np.cos(theta)*Y2

        X4 = X3
        Y4 = Y3

        X5 = -X4 + 2*X4[0]
        Y5 = Y4

        X4 = X4 - radius_min*np.sin(thetaCir)
        Y4 = Y4 + radius_min*(1 - np.cos(thetaCir))
        X4 = X4[::-1]
        Y4 = Y4[::-1]

        X5 = X5 + radius_min*np.sin(thetaCir)
        Y5 = Y5 + radius_min*(1 - np.cos(thetaCir))

        # stitching together clothoid and circular data
        xLCir = xCir[xCir <= 0]
        yLCir = radius_min - np.sqrt(radius_min**2 - xLCir**2)

        xRCir = xCir[xCir >= 0]
        yRCir = radius_min - np.sqrt(radius_min**2 - xRCir**2)

        X4 = np.hstack((X4, xLCir[1:-1]))
        Y4 = np.hstack((Y4, yLCir[1:-1]))

        X5 = np.hstack((xRCir[0:-2], X5))
        Y5 = np.hstack((yRCir[0:-2], Y5))

        X6 = np.cos(rotation_clothoid)*X4 + np.sin(rotation_clothoid)*Y4
        Y6 = -np.sin(rotation_clothoid)*X4 + np.cos(rotation_clothoid)*Y4
        X7 = np.cos(rotation_clothoid)*X5 + np.sin(rotation_clothoid)*Y5
        Y7 = -np.sin(rotation_clothoid)*X5 + np.cos(rotation_clothoid)*Y5

        X = np.hstack((X6, X7))
        Y = np.hstack((Y6, Y7))

        # Shift the entry point of the curve to be at X=0, Y=0.
        X -= np.min(X)
        Y -= Y[np.argmin(X)]

        # Shift the entry point of the curve to be at the end of the flat
        # surface.

        X += self.init_pos[0]
        Y += self.init_pos[1]

        return X, Y


class TakeoffSurface(Surface):
    """Class that represents a surface made up of a circle bounded by two
    clothoids with a flat exit surface."""

    def __init__(self, skier, entry_angle, exit_angle, entry_speed,
                 time_on_ramp=0.25, gamma=0.99, init_pos=(0.0, 0.0),
                 num_points=200):
        """Instantiates the takeoff curve with the flat takeoff ramp added to
        the terminus of the clothoid-circle-clothoid curve.

        Parameters
        ==========
        skier : Skier
            A skier instance.
        entry_angle : float
            The entry angle tangent to the start of the left clothoid in
            radians.
        exit_angle : float
            The exit angle tangent to the end of the right clothoid in radians.
        entry_speed : float
            The magnitude of the skier's velocity in meters per second as they
            enter the left clothiod.
        time_on_ramp : float, optional
            The time in seconds that the skier should be on the takeoff ramp
            before launch.
        gamma : float, optional
            Fraction of circular section.
        init_pos : 2-tuple of floats, optional
            The x and y coordinates of the start of the left clothoid.
        num_points : integer, optional
            The number of points in each of the three sections of the curve.

        """
        self.skier = skier
        self.entry_angle = entry_angle
        self.exit_angle = exit_angle
        self.entry_speed = entry_speed
        self.time_on_ramp = time_on_ramp
        self.gamma = gamma
        self.init_pos = init_pos
        self.num_points = num_points

        clt_cir_clt = ClothoidCircleSurface(entry_angle, exit_angle,
                                            entry_speed,
                                            skier.tolerable_sliding_acc,
                                            init_pos=init_pos, gamma=gamma,
                                            num_points=num_points)

        ramp_entry_speed = skier.end_speed_on(clt_cir_clt,
                                              init_speed=self.entry_speed)

        ramp_len = time_on_ramp * ramp_entry_speed  # meters

        start_x = clt_cir_clt.x[-1]
        start_y = clt_cir_clt.y[-1]

        points_per_meter = len(clt_cir_clt.x) / (start_x - clt_cir_clt.x[0])

        stop_x = start_x + ramp_len * np.cos(clt_cir_clt.exit_angle)
        ramp_x = np.linspace(start_x, stop_x,
                             num=int(points_per_meter * stop_x - start_x))

        stop_y = start_y + ramp_len * np.sin(clt_cir_clt.exit_angle)
        ramp_y = np.linspace(start_y, stop_y, num=len(ramp_x))

        ext_takeoff_curve_x = np.hstack((clt_cir_clt.x[:-1], ramp_x))
        ext_takeoff_curve_y = np.hstack((clt_cir_clt.y[:-1], ramp_y))

        super(TakeoffSurface, self).__init__(ext_takeoff_curve_x,
                                             ext_takeoff_curve_y)


class LandingTransitionSurface(Surface):
    """Class representing a acceleration limited exponential curve that
    transitions the skier from the landing surface to the parent slope."""

    acc_error_tolerance = 0.001
    max_iterations = 1000
    delta = 0.01  # used for central difference approximation

    def __init__(self, parent_surface, flight_traj, fall_height, tolerable_acc,
                 num_points=100):
        """Instantiates an exponentially decaying surface that connects the
        landing surface to the parent slope.

        Parameters
        ==========
        parent_surface : FlatSurface
            The parent slope in which the landing transition should be tangent
            to on exit.
        flight_traj : Trajectory
            The flight trajectory from the takeoff point to the parent slope.
        fall_height : float
            The desired equivalent fall height for the jump design in meters.
        tolerable_acc : float
            The maximum normal acceleration the skier should experience in the
            landing.
        num_points : integer
            The number of points in the surface.

        """
        if fall_height <= 0.0:
            raise InvalidJumpError('Fall height must be greater than zero.')

        self.fall_height = fall_height
        self.parent_surface = parent_surface
        self.flight_traj = flight_traj
        self.tolerable_acc = tolerable_acc

        trans_x, char_dist = self.find_transition_point()

        x, y = self._create_trans_curve(trans_x, char_dist, num_points)

        super(LandingTransitionSurface, self).__init__(x, y)

    @property
    def allowable_impact_speed(self):
        """Returns the perpendicular speed one would reach if dropped from the
        provided fall height."""
        return np.sqrt(2 * GRAV_ACC * self.fall_height)

    def calc_trans_acc(self, x):
        """Returns the acceleration in G's the skier feels at the exit
        transition occurring if the transition starts at the provided
        horizontal location, x."""

        # TODO : This code seems to be repeated some in the LandingSurface
        # creation code.

        # NOTE : "slope" means dy/dx here

        flight_y, flight_speed, flight_angle = \
            self.flight_traj.interp_wrt_x(x)[[2, 9, 8]]

        # NOTE : Not sure if setting this to pi/2 if the flight speed is
        # greater than the allowable impact speed is a correct thing to do but
        # it prevents some arcsin RunTimeWarnings for invalid values.
        ratio = self.allowable_impact_speed / flight_speed
        if ratio > 1.0:
            flight_rel_landing_angle = np.pi / 2
        else:
            flight_rel_landing_angle = np.arcsin(ratio)

        landing_angle = flight_angle + flight_rel_landing_angle
        landing_slope = np.tan(landing_angle)  # y'E(x0)

        parent_slope = self.parent_surface.interp_slope(x)
        parent_rel_landing_slope = landing_slope - parent_slope

        parent_y = self.parent_surface.interp_y(x)
        height_above_parent = flight_y - parent_y  # C in Mont's paper

        # required exponential characteristic distance, using three
        # characteristic distances for transition
        char_dist = np.abs(height_above_parent / parent_rel_landing_slope)

        ydoubleprime = height_above_parent / char_dist**2

        curvature = np.abs(ydoubleprime / (1 + landing_slope**2)**1.5)

        trans_acc = (curvature * flight_speed**2 + GRAV_ACC *
                     np.cos(landing_angle))

        return np.abs(trans_acc / GRAV_ACC), char_dist

    def _find_dgdx(self, x):

        x_plus = x + self.delta
        x_minus = x - self.delta

        acc_plus, _ = self.calc_trans_acc(x_plus)
        acc_minus, _ = self.calc_trans_acc(x_minus)

        return (acc_plus - acc_minus) / 2 / self.delta

    def find_transition_point(self):
        """Returns the horizontal position indicating the intersection of the
        flight path with the beginning of the landing transition. This is the
        last possible transition point, that by definition minimizes the
        transition snow budget, that satisfies the allowable transition
        acceleration.

        Notes
        =====
        This uses Newton's method to find an adequate point but may fail to do
        so with some combinations of flight trajectories, parent slope
        geometry, and allowable acceleration. A warning will be emitted if the
        maximum number of iterations is reached in this search and the curve is
        likely invalid.

        """

        i = 0
        g_error = np.inf
        x, _ = self.find_parallel_traj_point()
        xpara = float(x)  # copy

        while g_error > .001:  # tolerance

            transition_Gs, char_dist = self.calc_trans_acc(x)

            g_error = abs(transition_Gs - self.tolerable_acc)

            dx = -g_error / self._find_dgdx(x)

            x += dx

            if x >= self.flight_traj.pos[-1, 0]:
                msg = ('No landing transition point was found, backing up to '
                       'last possible point.')
                logging.info(msg)
                x = self.flight_traj.pos[-1, 0] - 2 * self.delta

            if i > self.max_iterations:
                msg = 'Landing transition while loop ran more than {} times.'
                logging.warning(msg.format(self.max_iterations))
                break
            else:
                i += 1

        logging.debug('{} iterations in the landing transition loop.'.format(i))

        x -= dx  # loop stops after dx is added, so take previous

        msg = ("The maximum landing transition acceleration is {} G's and the "
               "tolerable landing transition acceleration is {} G's.")
        logging.info(msg.format(transition_Gs, self.tolerable_acc))

        if x < xpara:
            msg = 'Not able to find valid landing transition point.'
            raise InvalidJumpError(msg)

        return x, char_dist

    def find_parallel_traj_point(self):
        """Returns the position of a point on the flight trajectory where its
        tangent is parallel to the parent slope. This is used as a starting
        guess for the start of the landing transition point."""

        slope_angle = self.parent_surface.angle

        flight_traj_slope = self.flight_traj.slope

        # TODO : Seems like these two interpolations can be combined into a
        # single interpolation call by adding the y coordinate to the following
        # line.
        xpara_interpolator = interp1d(flight_traj_slope,
                                      self.flight_traj.pos[:, 0])

        xpara = xpara_interpolator(np.tan(slope_angle))

        ypara = self.flight_traj.interp_wrt_x(xpara)[2]

        return xpara, ypara

    def _create_trans_curve(self, trans_x, char_dist, num_points):

        xTranOutEnd = trans_x + 3 * char_dist

        xParent = np.linspace(trans_x, xTranOutEnd, num_points)

        yParent0 = self.parent_surface.interp_y(trans_x)

        yParent = (yParent0 + (xParent - trans_x) *
                   np.tan(self.parent_surface.angle))

        xTranOut = np.linspace(trans_x, xTranOutEnd, num_points)

        dy = (self.flight_traj.interp_wrt_x(trans_x)[2] -
              self.parent_surface.interp_y(trans_x))

        yTranOut = yParent + dy * np.exp(-1*(xTranOut - trans_x) / char_dist)

        return xTranOut, yTranOut


class LandingSurface(Surface):
    """Class that defines an equivalent fall height landing surface."""

    def __init__(self, skier, takeoff_point, takeoff_angle, max_landing_point,
                 fall_height, surf):
        """Instantiates a surface that ensures impact velocity is equivalent to
        that from a vertical fall.

        Parameters
        ==========
        skier : Skier
            A skier instance.
        takeoff_point : 2-tuple of floats
            The point at which the skier leaves the takeoff ramp.
        takeoff_angle : float
            The takeoff angle in radians.
        max_landing_point : 2-tuple of floats
            The maximum x position that the landing surface will attain in
            meters. In the standard design, this is the start of the landing
            transition point.
        fall_height : float
            The desired equivalent fall height in meters. This should always be
            greater than zero.
        surf : Surface
            A surface below the full flight trajectory, the parent slope is a
            good choice. It is useful if the distance_from() method runs very
            fast, as it is called a lot internally.

        """
        if fall_height <= 0.0:
            raise InvalidJumpError('Fall height must be greater than zero.')

        self.skier = skier
        self.takeoff_point = takeoff_point
        self.takeoff_angle = takeoff_angle
        self.max_landing_point = max_landing_point
        self.fall_height = fall_height
        self.surf = surf

        x, y = self._create_safe_surface()

        super(LandingSurface, self).__init__(x, y)

    @property
    def allowable_impact_speed(self):
        """Returns the perpendicular speed one would reach if dropped from the
        provided fall height."""
        # NOTE : This is used in the LandingTransitionSurface class too and is
        # duplicate code. May need to be a simple function.
        return np.sqrt(2 * GRAV_ACC * self.fall_height)

    def _create_safe_surface(self):
        """Returns the x and y coordinates of the equivalent fall height
        landing surface."""

        def rhs(x, y):
            """Returns the slope of the safe surface that ensures the impact
            speed is equivalent to the impact speed from the equivalent fall
            height.

            dy
            -- = ...
            dx

            x : integrating through x instead of time
            y : single state variable

            equivalent to safe_surface.m

            integrates from the impact location backwards

            If the direction of the velocity vector is known, and the mangitude
            at impact is known, and the angle between v and the slope is known,
            then we can find out how the slope should be oriented.

            """

            # NOTE : y is an array of length 1
            y = y[0]

            logging.debug('x = {}, y = {}'.format(x, y))

            takeoff_speed, impact_vel = self.skier.speed_to_land_at(
                (x, y), self.takeoff_point, self.takeoff_angle, self.surf)

            if takeoff_speed > 0.0:
                impact_speed, impact_angle = vel2speed(*impact_vel)
            else:  # else takeoff_speed == 0, what about < 0?
                impact_speed = self.allowable_impact_speed
                impact_angle = -np.pi / 2.0

            speed_ratio = self.allowable_impact_speed / impact_speed

            logging.debug('speed ratio = {}'.format(speed_ratio))

            # beta is the allowed angle between slope and path at speed vImpact

            if speed_ratio > 1.0:
                beta = np.pi / 2.0 + EPS
            else:
                beta = np.arcsin(speed_ratio)

            logging.debug('impact angle = {} deg'.format(
                np.rad2deg(impact_angle)))
            logging.debug('beta = {} deg'.format(np.rad2deg(beta)))

            safe_surface_angle = beta + impact_angle

            logging.debug('safe_surface_angle = {} deg'.format(
                np.rad2deg(safe_surface_angle)))

            dydx = np.tan(safe_surface_angle)

            logging.debug('dydx = {}'.format(dydx))

            return dydx

        # NOTE : This is working for this range (back to 16.5), I think it is
        # getting hung in the find skier.speed_to_land_at().

        x_eval = np.linspace(self.max_landing_point[0], self.takeoff_point[0],
                             num=1000)

        logging.debug(x_eval)

        y0 = np.array([self.max_landing_point[1]])

        logging.debug('Making sure rhs() works.')
        logging.debug(rhs(self.max_landing_point[0], y0))

        logging.info('Integrating landing surface.')
        start_time = time.time()
        sol = solve_ivp(rhs, (x_eval[0], x_eval[-1]), y0, t_eval=x_eval,
                        max_step=1.0)
        msg = 'Landing surface finished in {} seconds.'
        logging.info(msg.format(time.time() - start_time))

        x = sol.t[::-1]
        y = sol.y.squeeze()[::-1]

        return x, y
