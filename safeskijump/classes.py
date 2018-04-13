import os
import time
import logging
from math import isclose

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, trapz, quad
import sympy as sm
from sympy.utilities.autowrap import autowrap


if 'ONHEROKU' in os.environ:
    plt = None
else:
    import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# NOTE : These parameters are more associated with an environment, but this
# doesn't warrant making a class for them. Maybe a namedtuple would be useful
# though.
GRAV_ACC = 9.81  # m/s/s
AIR_DENSITY = 0.85  # kg/m/m/m

EPS = np.finfo(float).eps


def generate_fast_drag_func():
    v, A, ro, C = sm.symbols('v, A, ro, C')
    drag_expr = -sm.sign(v) / 2 * ro * C * A * v**2
    return autowrap(drag_expr, backend='cython', args=(ro, v, C, A))


def gen_fast_distance_from():
    theta, x, y = sm.symbols('theta, x, y')
    expr = (y - sm.tan(theta) * x) * sm.cos(theta)
    return autowrap(expr, backend='cython', args=(theta, x, y))

#if 'ONHEROKU' in os.environ:
    #compute_drag = None
    #compute_dist_from_flat = None
#else:
compute_drag = generate_fast_drag_func()
compute_dist_from_flat = gen_fast_distance_from()


def speed2vel(speed, angle):
    """Returns the x and y components of velocity given the magnitude and angle
    of the velocity vector.

    Parameters
    ==========
    speed : float
        Magnitude of the velocity vector in meters per second.
    angle : float
        Angle of velocity vector in radians. Clockwise is negative and counter
        clockwise is positive.

    Returns
    =======
    vel_x : float
        X component of velocity in meters per second.
    vel_y : float
        Y component of velocity in meters per second.

    """
    vel_x = speed * np.cos(angle)
    vel_y = -speed * np.sin(angle)
    return vel_x, vel_y


def vel2speed(hor_vel, ver_vel):
    """Returns the magnitude and angle of the velocity vector given the
    horizontal and vertical components.

    Parameters
    ==========
    hor_vel : float
        X component of velocity in meters per second.
    ver_vel : float
        Y component of velocity in meters per second.

    Returns
    =======
    speed : float
        Magnitude of the velocity vector in meters per second.
    angle : float
        Angle of velocity vector in radians. Clockwise is negative and counter
        clockwise is positive.

    """
    speed = np.sqrt(hor_vel**2 + ver_vel**2)
    slope = ver_vel / hor_vel
    angle = np.arctan(slope)
    return speed, angle


class InvalidJumpError(Exception):
    """Custom class to signal that a poor combination of parameters have been
    supplied to the surface building functions."""
    pass


class Surface(object):
    """Base class for a 2D surface tied to a standard xy coordinate system."""

    def __init__(self, x, y):
        """Instantiates an arbitrary 2D surface.

        Parameters
        ==========
        x : ndarray, shape(n,)
            The horizontal, x, coordinates of the slope. x[0] should be the
            left most horizontal position and corresponds to the start of the
            surface.
        y : ndarray, shape(n,)
            The vertical, y, coordinates of the slope. y[0] corresponds to the
            start of the surface.

        """

        self.x = x
        self.y = y

        self.slope = np.gradient(y, x, edge_order=2)

        slope_deriv = np.gradient(self.slope, x, edge_order=2)
        self.curvature = slope_deriv / (1 + self.slope**2)**1.5

        interp_kwargs = {'fill_value': 'extrapolate'}
        self.interp_y = interp1d(x, y, **interp_kwargs)
        self.interp_slope = interp1d(x, self.slope, **interp_kwargs)
        self.interp_curvature = interp1d(x, self.curvature, **interp_kwargs)

    @property
    def start(self):
        """Returns the x and y coordinates at the start of the surface."""
        return self.x[0], self.y[0]

    @property
    def end(self):
        """Returns the x and y coordinates at the end of the surface."""
        return self.x[-1], self.y[-1]

    @property
    def xy(self):
        """Returns a tuple of the x and y coordinates."""
        return self.x, self.y

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

        # NOTE : This general implementation can be slow, so implement
        # overloaded distance_from methods in subclasses when you can.

        def distance_squared(x):
            return (xp - x)**2 + (yp - self.interp_y(x))**2

        distances = np.sqrt((self.x - xp)**2 + (self.y - yp)**2)

        x = fsolve(distance_squared, self.x[np.argmin(distances)])

        return np.sign(yp - self.interp_y(x)) * np.sqrt(distance_squared(x))

    def _limits(self, x_start=None, x_end=None):

        if x_start is not None:
            if x_start < self.start[0] or x_start > self.end[0]:
                raise ValueError('x_start has to be between start and end.')
            start_idx = np.argmin(np.abs(x_start - self.x))
        else:
            start_idx = 0

        if x_end is not None:
            if x_end < self.start[0] or x_end > self.end[0]:
                raise ValueError('x_end has to be between start and end.')
            end_idx = np.argmin(np.abs(x_end - self.x))
        else:
            end_idx = -1

        # TODO : make sure end_idx < start_idx
        #if end_idx <= start_idx:
            #raise ValueError('x_end has to be greater than x_start.')

        return start_idx, end_idx

    def length(self):
        """Returns the length of the surface in meters via a numerical line
        integral."""
        def func(x):
            return np.sqrt(1.0 + self.interp_slope(x)**2)
        return quad(func, self.x[0], self.x[-1])[0]

    def area_under(self, x_start=None, x_end=None):
        """Returns the area under the curve integrating wrt to the x axis."""
        start_idx, end_idx = self._limits(x_start, x_end)
        return trapz(self.y[start_idx:end_idx], self.x[start_idx:end_idx])

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

        ax.plot(self.x, self.y, **plot_kwargs)

        ax.set_aspect('equal')

        return ax


class FlatSurface(Surface):
    """Class that represents a flat angled surface."""

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
        init_pos : 2-tuple of floats
            The x and y coordinates in meters that locate the start of the
            surface.

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
        """Instantiates a clothoid-circle-clothoid takeoff curve (without the
        flat takeoff ramp).

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
        # TODO : Break this function into smaller functions.

        self.gamma = gamma
        self.entry_angle = entry_angle
        self.exit_angle = exit_angle

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
        radius_min = entry_speed**2 / (tolerable_acc * GRAV_ACC)

        #  x,y data for circle
        thetaCir = 0.5 * self.gamma * (lam + beta)
        xCirBound = radius_min * np.sin(thetaCir)
        xCirSt = -radius_min * np.sin(thetaCir)
        xCir = np.linspace(xCirSt, xCirBound, num=num_points)

        # x,y data for one clothoid
        A_squared = radius_min**2 * (1 - self.gamma) * (lam + beta)
        A = np.sqrt(A_squared)
        clothoid_length = A * np.sqrt((1 - self.gamma) * (lam + beta))

        # generates arc length points for one clothoid
        s = np.linspace(clothoid_length, 0, num=num_points)

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

        X += init_pos[0]
        Y += init_pos[1]

        super(ClothoidCircleSurface, self).__init__(X, Y)


class TakeoffSurface(Surface):
    """Class representing a takeoff surface made up of a
    clothoid-circle-clothoid-flat."""

    def __init__(self, clth_surface, ramp_entry_speed, time_on_ramp):
        """Returns the x and y coordinates of the takeoff curve with the flat
        takeoff ramp added to the terminus of the clothoid curve.

        Parameters
        ==========
        clth_surface : ClothoidCircleSurface
            The approach-takeoff transition curve.
        ramp_entry_speed : float
            The magnitude of the skier's speed at the exit of the second
            clothoid (entry to the flat ramp) in meters per second.
        time_on_ramp : float
            The time in seconds that the skier should be on the takeoff ramp
            before launch.

        """

        ramp_len = time_on_ramp * ramp_entry_speed  # meters

        start_x = clth_surface.x[-1]
        start_y = clth_surface.y[-1]

        points_per_meter = len(clth_surface.x) / (start_x - clth_surface.x[0])

        stop_x = start_x + ramp_len * np.cos(clth_surface.exit_angle)
        ramp_x = np.linspace(start_x, stop_x,
                             num=int(points_per_meter * stop_x - start_x))

        stop_y = start_y + ramp_len * np.sin(clth_surface.exit_angle)
        ramp_y = np.linspace(start_y, stop_y, num=len(ramp_x))

        ext_takeoff_curve_x = np.hstack((clth_surface.x[:-1], ramp_x))
        ext_takeoff_curve_y = np.hstack((clth_surface.y[:-1], ramp_y))

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
        """Instantiaties an exponentially decaying surface that connects the
        landing surface to the parent slope.

        Parameters
        ==========
        parent_surface : FlatSurface
            The parent slope in which the landing transition should be tangent
            to on exit.
        flight_traj : ndarray, shape(4, n)
            The flight trajectory from the takeoff point to the parent sloped.
            Rows correspond to [x, y, vx, vy] and columns to time.
        fall_height : float
            The desired equivalnent fall height for the jump design in meters.
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

        self._create_flight_interpolator()

        trans_x, char_dist = self.find_transition_point()

        x, y = self._create_trans_curve(trans_x, char_dist, num_points)

        super(LandingTransitionSurface, self).__init__(x, y)

    @property
    def allowable_impact_speed(self):
        """Returns the perpendicular speed one would reach if dropped from the
        provided fall height."""
        return np.sqrt(2 * GRAV_ACC * self.fall_height)

    def _create_flight_interpolator(self):
        """Creates a method that interpolates the veritcal position, slope,
        magnitude of the velocity, and angle of the velocity of the flight
        trajectory given a horizontal distance."""

        # TODO : This might be nicer to have in a Trajectory class that is
        # similar to a surface but includes velocity and acceleration.

        x = self.flight_traj[0]
        y = self.flight_traj[1]
        vx = self.flight_traj[2]
        vy = self.flight_traj[3]

        speed = np.sqrt(vx**2 + vy**2)

        slope = vy / vx
        angle = np.arctan(slope)

        data = np.vstack((y, slope, speed, angle))

        self._flight_interpolator = interp1d(x, data, fill_value='extrapolate')

    def interp_flight(self, x):
        """Returns the flight trajectory height, magnitude of the velocity, and
        the angle of the velocity given a horizontal position x.

        Returns
        =======

        y : float
            Trajectory height at position x.
        slope : float
            Trajectory slope at position x.
        speed : float
            Trajectory speed at position x.
        angle : float
            Trajectory speed direction at position x.

        """

        vals = self._flight_interpolator(x)

        return vals[0], vals[1], vals[2], vals[3]

    def calc_trans_acc(self, x):
        """Returns the acceleration in G's the skier feels at the exit
        transition occuring if the transition starts at the provided horizontal
        location, x."""

        # TODO : This code seems to be repeated some in the LandingSurface
        # creation code.

        # NOTE : "slope" means dy/dx here

        flight_y, _, flight_speed, flight_angle = self.interp_flight(x)

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

        while g_error > .001:  # tolerance

            transition_Gs, char_dist = self.calc_trans_acc(x)

            g_error = abs(transition_Gs - self.tolerable_acc)

            dx = -g_error / self._find_dgdx(x)

            x += dx

            if x >= self.flight_traj[0, -1]:
                x = self.flight_traj[0, -1] - 2 * self.delta

            if i > self.max_iterations:
                msg = 'Landing transition while loop ran more than {} times.'
                logging.warning(msg.format(self.max_iterations))
                break
            else:
                i += 1

        logging.debug('{} iterations in the landing transition loop.'.format(i))

        x -= dx  # loop stops after dx is added, so take previous

        # TODO : This should be uncommented, but need to make sure it doesn't
        # break anything.
        #transition_Gs, char_dist = self.calc_trans_acc(x)

        msg = ("The maximum landing transition acceleration is {} G's and the "
               "tolerable landing transition acceleration is {} G's.")
        logging.info(msg.format(transition_Gs, self.tolerable_acc))

        return x, char_dist

    def find_parallel_traj_point(self):
        """Returns the position of a point on the flight trajectory where its
        tagent is parallel to the parent slope. This is used as a starting
        guess for the start of the landing transition point."""

        slope_angle = self.parent_surface.angle

        flight_traj_slope = self.flight_traj[3] / self.flight_traj[2]

        # TODO : Seems like these two interpolations can be combined into a
        # single interpolation call by adding the y coordinate to the following
        # line.
        xpara_interpolator = interp1d(flight_traj_slope, self.flight_traj[0])

        xpara = xpara_interpolator(np.tan(slope_angle))

        ypara, _, _, _ = self.interp_flight(xpara)

        return xpara, ypara

    def _create_trans_curve(self, trans_x, char_dist, num_points):

        # TODO : Mont's code has 3 * char_dist
        xTranOutEnd = trans_x + 4 * char_dist

        xParent = np.linspace(trans_x, xTranOutEnd, num_points)

        yParent0 = self.parent_surface.interp_y(trans_x)

        yParent = (yParent0 + (xParent - trans_x) *
                   np.tan(self.parent_surface.angle))

        xTranOut = np.linspace(trans_x, xTranOutEnd, num_points)

        dy = (self.interp_flight(trans_x)[0] -
              self.parent_surface.interp_y(trans_x))

        yTranOut = yParent + dy * np.exp(-1*(xTranOut - trans_x) / char_dist)

        return xTranOut, yTranOut


class LandingSurface(Surface):
    """Class that defines an equivalent fall height landing surface."""

    def __init__(self, skier, takeoff_point, takeoff_angle, max_landing_point,
                 fall_height, surf=None):
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

        x, y = self.create_safe_surface()

        super(LandingSurface, self).__init__(x, y)

    @property
    def allowable_impact_speed(self):
        """Returns the perpendicular speed one would reach if dropped from the
        provided fall height."""
        # NOTE : This is used in the LandingTransitionSurface class too and is
        # duplicate code. May need to be a simple function.
        return np.sqrt(2 * GRAV_ACC * self.fall_height)

    def create_safe_surface(self):
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
                (x, y), self.takeoff_point, self.takeoff_angle, surf=self.surf)

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


class Skier(object):
    """Class that represents a skier which can slide on surfaces and fly in the
    air."""

    samples_per_sec = 360
    max_flight_time = 30.0  # seconds

    def __init__(self, mass=75.0, area=0.34, drag_coeff=0.821,
                 friction_coeff=0.03, tolerable_sliding_acc=1.5,
                 tolerable_landing_acc=3.0):
        """Instantiates a skier with default properties.

        Parameters
        ==========
        mass : float
            The mass of the skier.
        area : float
            The frontal area of the skier.
        drag_coeff : float
            The air drag coefficient of the skier.
        friction_coeff : float
            The sliding friction coefficient between the skis and the slope.
        tolerable_sliding_acc : float
            The maximum normal acceleration in G's that a skier can withstand
            while sliding.
        tolerable_landing_acc : float
            The maximum normal acceleration in G's that a skier can withstand
            when landing.

        """

        self.mass = mass
        self.area = area
        self.drag_coeff = drag_coeff
        self.friction_coeff = friction_coeff
        self.tolerable_sliding_acc = tolerable_sliding_acc
        self.tolerable_landing_acc = tolerable_landing_acc

    def drag_force(self, speed):
        """Returns the drag force in Newtons opposing the speed of the
        skier."""

        if compute_drag is None:
            return (-np.sign(speed) / 2 * AIR_DENSITY * self.drag_coeff *
                    self.area * speed**2)
        else:
            return compute_drag(AIR_DENSITY, speed, self.drag_coeff,
                                self.area)

    def friction_force(self, speed, slope=0.0, curvature=0.0):
        """Returns the friction force in Newtons opposing the speed of the
        skier.

        Parameters
        ==========
        speed : float
            The tangential speed of the skier in meters per second.
        slope : float
            The slope of the surface at the point of contact.
        curvature : float
            The curvature of the surface at the point of contact.

        """

        theta = np.tan(slope)

        normal_force = self.mass * (GRAV_ACC * np.cos(theta) + curvature *
                                    speed**2)

        return -np.sign(speed) * self.friction_coeff * normal_force

    def fly_to(self, surface, init_pos, init_vel, fine=True,
               logging_type='info'):
        """Returns the flight trajectory of the skier given the initial
        conditions and a surface which the skier contacts at the end of the
        flight trajectory.

        Parameters
        ==========
        surface : Surface
            A landing surface. This surface must intersect the flight path.
        init_pos : 2-tuple of floats
            The X and Y coordinates of the starting point of the flight.
        init_vel : 2-tuple of floats
            The X and Y components of the skier's velocity at the start of the
            flight.
        fine : boolean
            If True two integrations occur. The first finds the landing time
            with coarse time steps and the second integrates over a finer
            equally spaced time steps. False will skip the second integration.
        logging_type : string
            The logging level desired for the non-debug logging calls in this
            function. Useful for suppressing too much information since this
            runs a lot.

        Returns
        =======
        times : ndarray, shape(n,)
            The values of time corresponding to each state instance.
        states : ndarray, shape(n, 4)
            The states: (x, y, vx, vy) for each instance of time. The last
            value of the state corresponds to the skier touching the surface.

        Raises
        ======
        InvalidJumpError if the skier does not contact a surface within
        Skier.max_flight_time.

        """

        def rhs(t, state):

            xdot = state[2]
            ydot = state[3]

            vxdot = self.drag_force(xdot) / self.mass
            vydot = -GRAV_ACC + self.drag_force(ydot) / self.mass

            return xdot, ydot, vxdot, vydot

        def touch_surface(t, state):

            x = state[0]
            y = state[1]

            return surface.distance_from(x, y)

        touch_surface.terminal = True
        # NOTE: always from above surface, positive to negative crossing
        touch_surface.direction = -1

        logging_call = getattr(logging, logging_type)

        # NOTE : For a more accurate event time, the error tolerances on the
        # states need to be lower.
        logging_call('Integrating skier flight.')
        start_time = time.time()

        # integrate to find the final time point
        sol = solve_ivp(rhs,
                        (0.0, self.max_flight_time),
                        init_pos + init_vel,
                        events=(touch_surface, ),
                        rtol=1e-6, atol=1e-9)

        if isclose(sol.t[-1], self.max_flight_time):
            msg = ('Flying skier did not contact ground within {:1.3.f} '
                   'seconds, integration aborted.')
            raise InvalidJumpError(msg.format(self.max_flight_time))

        msg = 'Flight integration terminated at {:1.3f} s'
        logging_call(msg.format(sol.t[-1]))
        msg = 'Flight contact event occurred at {:1.3f} s'
        logging_call(msg.format(float(sol.t_events[0])))

        logging.debug(sol.t[-1] - sol.t_events[0])
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(sol.t[-1], sol.y[:, -1]))

        te = sol.t_events[0]

        if fine:
            # integrate at desired resolution
            times = np.linspace(0.0, sol.t[-1],
                                num=int(self.samples_per_sec * sol.t[-1]))
            sol = solve_ivp(rhs, (0.0, sol.t[-1]), init_pos + init_vel,
                            t_eval=times, rtol=1e-6, atol=1e-9)

        msg = 'Flight integration finished in {:1.3f} seconds.'
        logging_call(msg.format(time.time() - start_time))

        logging.debug(sol.t[-1])
        logging.debug(sol.t[-1] - te)
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(sol.t[-1], sol.y[:, -1]))

        return sol.t, sol.y

    def slide_on(self, surface, init_speed=0.0, fine=True):
        """Returns the trajectory of the skier sliding over a surface.

        Parameters
        ==========
        surface : Surface
            A surface that the skier will slide on.
        init_speed : float
            The magnitude of the velocity of the skier at the start of the
            surface which is directed tangent to the surface.
        fine : boolean
            If True two integrations occur. The first finds the exit time with
            coarse time steps and the second integrates over a finer equally
            spaced time steps. False will skip the second integration.

        Raises
        ======
        InvalidJumpError if skier can't reach the end of the surface within
        1000 seconds.

        """

        def rhs(t, state):

            x = state[0]  # horizontal position
            v = state[1]  # velocity tangent to slope

            slope = surface.interp_slope(x)
            kurva = surface.interp_curvature(x)

            theta = np.arctan(slope)

            xdot = v * np.cos(theta)
            vdot = -GRAV_ACC * np.sin(theta) + (
                (self.drag_force(v) + self.friction_force(v, slope, kurva)) /
                self.mass)

            return xdot, vdot

        def reach_end(t, state):
            """Returns zero when the skier gets to the end of the approach
            length."""
            return state[0] - surface.x[-1]

        reach_end.terminal = True

        logging.info('Integrating skier sliding.')
        start_time = time.time()

        sol = solve_ivp(rhs,
                        (0.0, 1000.0),  # time span
                        (surface.x[0], init_speed),  # initial conditions
                        events=(reach_end, ))

        if fine:
            times = np.linspace(0.0, sol.t[-1],
                                num=int(self.samples_per_sec * sol.t[-1]))
            sol = solve_ivp(rhs, (0.0, sol.t[-1]), (surface.x[0], init_speed),
                            t_eval=times)

        msg = 'Sliding integration finished in {} seconds.'
        logging.info(msg.format(time.time() - start_time))

        logging.info('Skier slid for {} seconds.'.format(sol.t[-1]))

        if np.any(sol.y[1] < 0.0):  # if tangential velocity is ever negative
            msg = ('Skier does not have a high enough velocity to make it to '
                   'the end of the surface.')
            raise InvalidJumpError(msg)

        return sol.t, sol.y

    def end_speed_on(self, surface, **kwargs):
        """Returns the ending speed after sliding on the provided surface.
        Keyword args are passed to Skier.slide_on()."""

        _, traj = self.slide_on(surface, **kwargs)

        return traj[1, -1]

    def end_vel_on(self, surface, **kwargs):
        """Returns the ending velocity (vx, vy) after sliding on the provided
        surface.  Keyword args are passed to Skier.slide_on()."""

        _, traj = self.slide_on(surface, **kwargs)
        end_angle = np.tan(surface.slope[-1])
        speed_x = traj[1, -1] * np.cos(end_angle)
        speed_y = traj[1, -1] * np.sin(end_angle)
        return speed_x, speed_y

    def speed_to_land_at(self, landing_point, takeoff_point, takeoff_angle,
                         surf=None):
        """Returns the magnitude of the velocity required to land at a specific
        point given launch position and angle.

        Parameters
        ==========
        landing_point : 2-tuple of floats
            The (x, y) coordinates of the desired landing point in meters.
        takeoff_point : 2-tuple of floats
            The (x, y) coordinates of the takeoff point in meters.
        takeoff_angle : float
            The takeoff angle in radians.
        surf : Surface
            This should most likely be the parent slope but needs to be
            something that ensures the skier flys past the landing point.

        Returns
        =======
        takeoff_speed : float
            The magnitude of the takeoff velocity.

        Notes
        =====
        This method corresponds to Mont's Matlab function findVoWithDrag.m.

        """

        # NOTE : This may only work if the landing surface is below the takeoff
        # point.

        # TODO : Is it possible to solve a boundary value problem here instead
        # using this iterative approach with an initial value problem?

        x, y = landing_point

        if isclose(landing_point[0] - takeoff_point[0], 0.0):
            return 0.0, (0.0, 0.0)

        theta = takeoff_angle
        cto = np.cos(takeoff_angle)
        sto = np.sin(takeoff_angle)
        tto = np.tan(takeoff_angle)

        # guess init. velocity for impact at x,y based on explicit solution
        # for the no drag case
        delx = landing_point[0] - takeoff_point[0]
        dely = landing_point[1] - takeoff_point[1]
        logging.debug('delx = {}, dely = {}'.format(delx, dely))
        logging.debug(delx**2 * GRAV_ACC / (2*cto**2 * (delx*tto - dely)))
        logging.debug(2*cto**2 * (delx*tto - dely))
        vo = np.sqrt(delx**2 * GRAV_ACC / (2*cto**2 * (delx*tto - dely)))
        logging.debug('vo = {}'.format(vo))
        # dvody is calculated from the explicit solution without drag @ (x,y)
        dvody = ((delx**2 * GRAV_ACC / 2 / cto**2)**0.5 *
                 ((delx*tto-dely)**(-3/2)) / 2)
        # TODO : This gets a negative under the sqrt for some cases, e.g.
        # make_jump(-10.0, 0.0, 30.0, 20.0, 0.1)
        dvody = (np.sqrt(GRAV_ACC*(delx)**2/((delx)*np.sin(2*theta) -
                                             2*(dely)*cto**2))*cto**2 /
                 ((delx)*np.sin(2*theta) - 2*(dely)*cto**2))

        # TODO : Make this should take in the parent slope and use it.
        # creates a flat landing surface that starts at the landing point x
        # position and 1 meter below the y position, this ensures we get a
        # flight trajectory that passes through a horizontal line through the
        # landing position
        if surf is None:
            #surf = FlatSurface(np.deg2rad(45.0), 10.0, init_pos=(x + 6, y),
                            #num_points=100)
            surf = FlatSurface(np.deg2rad(-20.0), 40.0)

        deltay = np.inf

        #ax = surf.plot()

        while abs(deltay) > 0.001:
            vox = vo*cto
            voy = vo*sto

            times, flight_traj = self.fly_to(surf,
                                             init_pos=takeoff_point,
                                             init_vel=(vox, voy),
                                             logging_type='debug')
            x_traj = flight_traj[0]

            #ax.plot(*flight_traj[:2])

            interpolator = interp1d(x_traj, flight_traj[1:], fill_value='extrapolate')

            traj_at_impact = interpolator(x)

            ypred = traj_at_impact[0]
            logging.debug('ypred = {}'.format(ypred))

            deltay = ypred - y
            logging.debug('deltay = {}'.format(deltay))
            dvo = -deltay * dvody
            logging.debug('dvo = {}'.format(dvo))
            vo = vo + dvo
            logging.debug('vo = {}'.format(vo))

        #ax.plot(*takeoff_point, 'o')
        #ax.plot(*landing_point, 'o')
        #ax.set_xlim((10.0, 35.0))
        #plt.show()

        # the takeoff velocity is adjsted by dvo before the while loop ends
        vo = vo - dvo

        takeoff_speed = vo

        impact_vel = (traj_at_impact[1], traj_at_impact[2])

        return takeoff_speed, impact_vel
