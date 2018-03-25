from math import isclose

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# NOTE : These parameters are more associated with an environment, but this
# doesn't warrant making a class for them. Maybe a namedtuple would be useful
# though.
GRAV_ACC = 9.81  # m/s/s
AIR_DENSITY = 0.85  # kg/m/m/m

EPS = np.finfo(float).eps


def speed2vel(speed, angle):
    """
    speed : float
        Magnitude of the speed in meters per second.
    angle : float
        Angle in radians. Clockwise is negative and counter clockwise is
        positive.

    Returns
    =======
    vel_x : float
    vel_y : float

    """
    vel_x = speed * np.cos(angle)
    vel_y = -speed * np.sin(angle)
    return vel_x, vel_y

def vel2speed(hor_vel, ver_vel):

    speed = np.sqrt(hor_vel**2 + ver_vel**2)

    slope = ver_vel / hor_vel

    angle = np.arctan(slope)

    return speed, angle


class Surface(object):

    def __init__(self, x, y):
        """Instantiates an arbitrary 2D surface.

        Parameters
        ==========
        x : ndarray, shape(n,)
            The horizontal, X, coordinates of the slope. x[0] should be the
            left most horizontal position and corresponds to the start of the
            surface.
        y : ndarray, shape(n,)
            The vertical, Y, coordinates of the slope. y[0] corresponds to the
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
        """Returns the X and Y coordinates at the start of the surface."""
        return self.x[0], self.y[0]

    @property
    def end(self):
        """Returns the X and Y coordinates at the end of the surface."""
        return self.x[-1], self.y[-1]

    def distance_from(self, xp, yp):
        """Returns the shortest distance from point (xp, yp) to the surface.

        Parameters
        ==========
        xp : float
            The horizontal, X, coordinate of the point.
        yp : float
            The vertical, Y, coordinate of the point.

        Returns
        =======
        distance : float
            The shortest distance from the point to the surface. If the point
            is above the surface a positive distance is returned, else a
            negative distance.

        """

        def distance_squared(x):
            return (xp - x)**2 + (yp - self.interp_y(x))**2

        distances = np.sqrt((self.x - xp)**2 + (self.y - yp)**2)

        x = fsolve(distance_squared, self.x[np.argmin(distances)])

        return np.sign(yp - self.interp_y(x)) * np.sqrt(distance_squared(x))

    def plot(self, ax=None, **plot_kwargs):
        """Creates a matplotlib plot of the surface."""

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.x, self.y, **plot_kwargs)

        ax.set_aspect('equal')

        return ax


class FlatSurface(Surface):

    def __init__(self, angle, length, init_pos=(0.0, 0.0), num_points=100):
        """Instantiates a flat surface that is oriented at an angle from the X
        axis.

        Parameters
        ==========
        angle : float
            The angle of the surface in degrees. This is the angle about the
            positive Z axis.
        length : float
            The distance in meters along the surface from the initial position.
        init_pos : two tuple of floats
            The X and Y coordinates in meters that locate the start of the
            surface.

        """

        if angle >= 90.0 or angle <= -90.0:
            raise ValueError('Angle must be between -90 and 90 degrees')

        self.angle_in_deg = angle

        angle = np.deg2rad(angle)

        self.angle_in_rad = angle

        x = np.linspace(init_pos[0], init_pos[0] + length * np.cos(angle),
                        num=num_points)
        y = np.linspace(init_pos[1], init_pos[1] + length * np.sin(angle),
                        num=num_points)

        super(FlatSurface, self).__init__(x, y)


class ClothoidCircleSurface(Surface):

    def __init__(self, entry_angle, exit_angle, entry_speed, tolerable_acc,
                 init_pos=(0.0, 0.0), gamma=0.99, num_points=200):
        """Instantiates a clothoid-circle-clothoid takeoff curve (without the
        flat takeoff ramp).

        Parameters
        ==========
        entry_angle : float
            The entry angle tangent to the left clothoid in degrees.
        exit_angle : float
            The exit angle tangent to the right clothoid in degrees.
        entry_speed : float
            The magnitude of the skier's velocity in meters per second as they
            enter the takeoff curve (i.e. approach exit speed).
        tolerable_acc : float
            The tolerable acceleration of the skier in G's.
        gamma : float
            Fraction of circular section.
        num_points : integer, optional
            The n number of points in each section of the curve.

        """
        self.gamma = gamma

        self.entry_angle_in_deg = entry_angle
        self.entry_angle_in_rad = np.deg2rad(entry_angle)

        self.exit_angle_in_deg = exit_angle
        self.exit_angle_in_rad = np.deg2rad(exit_angle)

        lam = -self.entry_angle_in_rad
        beta = self.exit_angle_in_rad

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

    def __init__(self, clth_surface, ramp_entry_speed, time_on_ramp):
        """Returns the X and Y coordinates of the takeoff curve with the flat
        takeoff ramp added to the terminus of the clothoid curve.

        Parameters
        ==========
        clth_surface : ClothoidCircleSurface
            The approach-takeoff transition curve.
        ramp_entry_speed : float
            The magnitude of the skier's speed at the exit of the second
            clothoid.
        time_on_ramp : float
            The time in seconds that the skier is on the takeoff ramp.

        """

        ramp_len = time_on_ramp * ramp_entry_speed  # meters

        start_x = clth_surface.x[-1]
        start_y = clth_surface.y[-1]

        points_per_meter = len(clth_surface.x) / (start_x - clth_surface.x[0])

        stop_x = start_x + ramp_len * np.cos(clth_surface.exit_angle_in_rad)
        ramp_x = np.linspace(start_x, stop_x,
                             num=int(points_per_meter * stop_x - start_x))

        stop_y = start_y + ramp_len * np.sin(clth_surface.exit_angle_in_rad)
        ramp_y = np.linspace(start_y, stop_y, num=len(ramp_x))

        ext_takeoff_curve_x = np.hstack((clth_surface.x, ramp_x))
        ext_takeoff_curve_y = np.hstack((clth_surface.y, ramp_y))

        super(TakeoffSurface, self).__init__(ext_takeoff_curve_x,
                                             ext_takeoff_curve_y)


class LandingTransitionSurface(Surface):

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
        flight_traj : ndarray, shape(4, n)
        fall_height : float
        tolerable_acc : float
        num_points : integer

        """

        self.fall_height = fall_height
        self.parent_surface = parent_surface
        self.flight_traj = flight_traj
        self.tolerable_acc = tolerable_acc

        self.create_flight_interpolator()

        trans_x, char_dist = self.find_transition_point()

        x, y = self.create_trans_curve(trans_x, char_dist, num_points)

        super(LandingTransitionSurface, self).__init__(x, y)

    @property
    def allowable_impact_speed(self):
        """Returns the perpendicular speed one would reach if dropped from the
        provided fall height."""
        return np.sqrt(2 * GRAV_ACC * self.fall_height)

    def create_flight_interpolator(self):
        """Creates a method that interpolates the veritcal position, slope,
        magnitude of the velocity, and angle of the velocity of the flight
        trajectory given a horizontal distance."""

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
        the angle of the velocity given a horizontal position.

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
        transition occuring at the provided horizontal location."""

        # NOTE : "slope" means dy/dx here

        flight_y, _, flight_speed, flight_angle = self.interp_flight(x)

        flight_rel_landing_angle = np.arcsin(self.allowable_impact_speed /
                                             flight_speed)

        landing_angle = flight_angle + flight_rel_landing_angle
        landing_slope = np.tan(landing_angle)

        parent_slope = self.parent_surface.interp_slope(x)
        parent_rel_landing_slope = landing_slope - parent_slope

        parent_y = self.parent_surface.interp_y(x)
        height_above_parent = flight_y - parent_y

        # required exponential characteristic distance, using three
        # characteristic distances for transition
        dx = np.abs(height_above_parent / parent_rel_landing_slope)

        ydoubleprime = height_above_parent / dx**2

        curvature = np.abs(ydoubleprime / (1 + landing_slope**2)**1.5)

        trans_acc = (curvature * flight_speed**2 + GRAV_ACC *
                     np.cos(landing_angle))

        return np.abs(trans_acc / GRAV_ACC), dx

    def find_dgdx(self, x):

        x_plus = x + self.delta
        x_minus = x - self.delta

        acc_plus, _ = self.calc_trans_acc(x_plus)
        acc_minus, _ = self.calc_trans_acc(x_minus)

        return (acc_plus - acc_minus) / 2 / self.delta

    def find_transition_point(self):
        """Returns the horizontal position indicating the start of the landing
        transition."""

        # goal is to find the last possible transition point (that by
        # definition minimizes the transition snow budget) that satisfies the
        # allowable transition G's

        i = 0
        g_error = np.inf
        x, _ = self.find_parallel_traj_point()

        while g_error > .001:  # tolerance

            transition_Gs, char_dist = self.calc_trans_acc(x)

            g_error = abs(transition_Gs - self.tolerable_acc)

            dx = -g_error / self.find_dgdx(x)

            x += dx

            if x > self.flight_traj[0, -1]:
                x = self.flight_traj[0, -1] - 2 * self.delta

            if i > self.max_iterations:
                msg = 'ERROR: while loop ran more than {} times'
                print(msg.format(self.max_iterations))
                break
            else:
                i += 1

        x -= dx  # loop stops after dx is added, so take previous

        return x, char_dist

    def find_parallel_traj_point(self):

        slope_angle = self.parent_surface.angle_in_rad

        flight_traj_slope = self.flight_traj[3] / self.flight_traj[2]

        xpara_interpolator = interp1d(flight_traj_slope, self.flight_traj[0])

        xpara = xpara_interpolator(np.tan(slope_angle))

        ypara, _, _, _ = self.interp_flight(xpara)

        return xpara, ypara

    def create_trans_curve(self, trans_x, char_dist, num_points):

        xTranOutEnd = trans_x + 4 * char_dist

        xParent = np.linspace(trans_x, xTranOutEnd, num_points)

        yParent0 = self.parent_surface.interp_y(trans_x)

        yParent = (yParent0 + (xParent - trans_x) *
                   np.tan(self.parent_surface.angle_in_rad))

        xTranOut = np.linspace(trans_x, xTranOutEnd, num_points)

        dy = (self.interp_flight(trans_x)[0] -
              self.parent_surface.interp_y(trans_x))

        yTranOut = yParent + dy * np.exp(-1*(xTranOut - trans_x) / char_dist)

        return xTranOut, yTranOut


class LandingSurface(Surface):

    def __init__(self, skier, takeoff_point, takeoff_angle, max_landing_point,
                 fall_height):
        """
        skier : Skier
        takeoff_point : 2-tuple of floats

        takeoff_angle : float
            Degrees
        max_landing_point : 2-tuple of floats
            meters
        fall_height : float
            Meters

        """

        self.skier = skier
        self.takeoff_point = takeoff_point
        self.takeoff_angle = takeoff_angle
        self.max_landing_point = max_landing_point
        self.fall_height = fall_height

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

            If the direction of the velocity vector is known, and
            the mangitude at impact is known, and
            the angle between v and the slope is known, then we can find out how the
            slope should be oriented.

            """
            print(type(y))

            y = y[0]

            takeoff_speed, impact_vel = self.skier.speed_to_land_at(
                (x, y), self.takeoff_point, self.takeoff_angle)

            if takeoff_speed > 0.0:
                impact_speed, impact_angle = vel2speed(*impact_vel)
            else:  # else takeoff_speed == 0, what about < 0?
                impact_speed = self.allowable_impact_speed
                impact_angle = -np.pi / 2.0  # this is vertical?

            speed_ratio = self.allowable_impact_speed / impact_speed

            # beta is the allowed angle between slope and path at speed vImpact

            if isclose(speed_ratio, 1.0):
                beta = -np.pi / 2.0 + EPS
            else:
                beta = -np.arcsin(speed_ratio)

            safe_surface_angle = impact_angle - beta

            dydx = np.tan(safe_surface_angle)

            print('x = {}, y = {}'.format(x, y))

            print('dydx = ', dydx)

            return dydx

        # NOTE : This is working for this range (back to 16.5), I think it is
        # getting hung in the find skier.speed_to_land_at().

        x_eval = np.linspace(self.max_landing_point[0], self.takeoff_point[0],
                             num=100)

        print(x_eval)

        y0 = np.array([self.max_landing_point[1]])

        print('Making sure rhs() works.')
        print(rhs(self.max_landing_point[0], y0))

        print('Integrating...')
        sol = solve_ivp(rhs, (x_eval[0], x_eval[-1]), y0, t_eval=x_eval)
        print('Integrating done.')

        x = sol.t[::-1]
        y = sol.y.squeeze()[::-1]

        return x, y


class Skier(object):

    samples_per_sec = 120
    tolerable_acc = 1.5  # G

    def __init__(self, mass=75.0, area=0.34, drag_coeff=0.821,
                 friction_coeff=0.03):
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

        """

        self.mass = mass
        self.area = area
        self.drag_coeff = drag_coeff
        self.friction_coeff = friction_coeff

    def drag_force(self, velocity):
        """Returns the drag force in Newtons opposing the velocity of the
        skier."""

        return (-np.sign(velocity) / 2 * AIR_DENSITY * self.drag_coeff *
                self.area * velocity**2)

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

    def fly_to(self, surface, init_pos, init_vel):
        """Returns the flight trajectory of the skier given the initial
        conditions and a surface which the skier contacts at the end of the
        flight trajectory.

        Parameters
        ==========
        surface : Surface
            A landing surface. This surface must intersect the flight path.
        init_pos : two tuple of floats
            The X and Y coordinates of the starting point of the flight.
        init_vel : 2-tuple of floats
            The X and Y components of the skier's velocity at the start of the
            flight.

        Returns
        =======
        times : ndarray, shape(n,)
            The values of time corresponding to each state instance.
        states : ndarray, shape(n, 4)
            The states: (X, Y, X', Y') for each instance of time. The last
            value of the state corresponds to the skier touching the surface.

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

        # integrate to find the final time point
        sol = solve_ivp(rhs,
                        (0.0, np.inf),
                        init_pos + init_vel,
                        events=(touch_surface, ))

        # integrate at higher resolution
        sol = solve_ivp(rhs,
                        (0.0, sol.t[-1]),
                        init_pos + init_vel,
                        t_eval=np.linspace(0.0, sol.t[-1],
                                           num=int(self.samples_per_sec *
                                                   sol.t[-1])))

        return sol.t, sol.y

    def slide_on(self, surface, init_speed=0.0):
        """Returns the trajectory of the skier sliding over a surface.

        Parameters
        ==========
        surface : Surface
            A surface that the skier will slide on.
        init_speed : float
            The magnitude of the velocity of the skier at the start of the
            surface which is directed tangent to the surface.

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

        sol = solve_ivp(rhs,
                        (0.0, np.inf),  # time span
                        (surface.x[0], init_speed),  # initial conditions
                        events=(reach_end, ))

        sol = solve_ivp(rhs,
                        (0.0, sol.t[-1]),
                        (surface.x[0], init_speed),
                        t_eval=np.linspace(0.0, sol.t[-1],
                                           num=int(self.samples_per_sec *
                                                   sol.t[-1])))

        return sol.t, sol.y

    def end_speed_on(self, surface, **kwargs):

        _, traj = self.slide_on(surface, **kwargs)

        return traj[1, -1]

    def end_vel_on(self, surface, **kwargs):

        _, traj = self.slide_on(surface, **kwargs)

        end_angle = np.tan(surface.slope[-1])

        speed_x = traj[1, -1] * np.cos(end_angle)
        speed_y = traj[1, -1] * np.sin(end_angle)

        return speed_x, speed_y

    def speed_to_land_at(self, landing_point, takeoff_point, takeoff_angle):
        """Returns the magnitude of the velocity required to land at a specific
        point.

        Parameters
        ==========
        landing_point : 2-tuple of floats
            The (x, y) coordinates of the desired landing point in meters.
        takeoff_point : 2-tuple of floats
            The (x, y) coordinates of the takeoff point in meters.
        takeoff_angle : float
            The takeoff angle in degrees.

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

        print('Landing point')
        print(landing_point)

        cto = np.cos(np.deg2rad(takeoff_angle))
        sto = np.sin(np.deg2rad(takeoff_angle))
        tto = np.tan(np.deg2rad(takeoff_angle))

        # guess init. velocity for impact at x,y based on explicit solution
        # for the no drag case
        x_norm = landing_point[0] - takeoff_point[0]
        y_norm = landing_point[1] - takeoff_point[1]
        print('normed', x_norm, y_norm)
        vo = np.sqrt(x_norm**2 * GRAV_ACC / (2*cto**2 * (x_norm*tto - y_norm)))
        # dvody is calculated from the explicit solution without drag @ (x,y)
        dvody = (x_norm**2 * GRAV_ACC / 2 / cto**2)**0.5 * ((x_norm*tto-y_norm)**(-3/2)) / 2
        print('dvody')
        print(dvody)

        # creates a flat landing surface that starts at the landing point x
        # position and 1 meter below the y position, this ensures we get a
        # flight trajectory that passes through a horizontal line through the
        # landing position
        surf = FlatSurface(45.0, 10.0, init_pos=(x + 6, y),
                           num_points=100)
        surf = FlatSurface(-20.0, 40.0)

        print('Takeoff Point')
        print(takeoff_point)

        print('Surface: x, y')
        print(surf.x)
        print(surf.y)

        deltay = np.inf

        ax = surf.plot()

        while abs(deltay) > 0.001:
            vox = vo*cto
            voy = vo*sto

            times, flight_traj = self.fly_to(surf,
                                             init_pos=takeoff_point,
                                             init_vel=(vox, voy))

            ax.plot(*flight_traj[:2])

            interpolator = interp1d(*flight_traj[:2], fill_value='extrapolate')

            ypred = interpolator(x)
            print('ypred', ypred)

            deltay = ypred - y
            print('deltay', deltay)
            dvo = -deltay * dvody
            print('dvo', dvo)
            vo = vo + dvo
            print('vo', vo)

        ax.plot(*takeoff_point, 'o')
        ax.plot(*landing_point, 'o')
        ax.set_xlim((10.0, 35.0))
        plt.show()

        # the takeoff velocity is adjsted by dvo before the while loop ends
        vo = vo - dvo

        takeoff_speed = vo

        impact_vel = tuple(flight_traj[2:, -1])

        return takeoff_speed, impact_vel
