import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

GRAV_ACC = 9.81  # m/s/s
AIR_DENSITY = 0.85  # kg/m/m/m


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

    def __init__(self, angle, length, init_pos=(0.0, 0.0)):
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

        if angle > 90.0 or angle < -90.0:
            raise ValueError('Angle must be between -90 and 90 degrees')

        self.angle_in_deg = angle

        angle = np.deg2rad(angle)

        self.angle_in_rad = angle

        x = np.linspace(init_pos[0], init_pos[0] + length * np.cos(angle),
                        num=100)
        y = np.linspace(init_pos[1], init_pos[1] + length * np.sin(angle),
                        num=100)

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


class Skier(object):

    samples_per_sec = 120
    tolerable_acc = 1.5

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

    def fly_to(self, surface, init_pos, init_speed):
        """Returns the flight trajectory of the skier given the initial
        conditions and a surface which the skier contacts at the end of the
        flight trajectory.

        Parameters
        ==========
        surface : Surface
            A landing surface. This surface must intersect the flight path.
        init_pos : two tuple of floats
            The X and Y coordinates of the starting point of the flight.
        init_speed : two tuple of floats
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
                        init_pos + init_speed,
                        events=(touch_surface, ))

        # integrate at higher resolution
        sol = solve_ivp(rhs,
                        (0.0, sol.t[-1]),
                        init_pos + init_speed,
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
