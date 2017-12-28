import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


class Surface(object):

    def __init__(self, x, y):
        """Instantiates an arbitrary 2D surface.

        Parameters
        ==========
        x : ndarray, shape(n,)
            The horizontal, X, coordinates of the slope. x[0] should be the
            left most horizontal position and correpsonds to the start of the
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

        self.interp_y = interp1d(x, y, fill_value='extrapolate')
        self.interp_slope = interp1d(x, self.slope, fill_value='extrapolate')
        self.interp_curvature = interp1d(x, self.curvature,
                                         fill_value='extrapolate')

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
            is above the surface a positie distance is returned, else a
            negative distance.

        """

        def distance_squared(x):
            return (xp - x)**2 + (yp - self.interp_y(x))**2

        distances = np.sqrt((self.x - xp)**2 + (self.y - yp)**2)

        x = fsolve(distance_squared, self.x[np.argmin(distances)])

        return np.sign(yp - self.interp_y(x)) * np.sqrt(distance_squared(x))


class FlatSurface(Surface):

    def __init__(self, angle, length, start_pos=0.0):
        """Returns the speed of the skier in meters per second at the end of the
        approach (entry to approach-takeoff transition).

        Parameters
        ==========
        angle : float
            The angle of the suface in degrees. This is the angle about the
            negative Z axis.
        approach_len : float
            The distance in meters along the slope from the skier starting
            position to the beginning of the approach transition.
        start_pos : float
            The position in meters along the slope from the top (beginning) of
            the slope.

        """

        if angle > 90.0 or angle < -90.0:
            raise ValueError('Angle must be between -90 and 90 degrees')

        angle = np.deg2rad(angle)

        start_x = start_pos * np.cos(angle)
        start_y = start_pos * np.sin(angle)

        x = np.linspace(start_x, start_x + length * np.cos(angle), num=100)
        y = np.linspace(start_y, start_y + length * np.sin(angle), num=100)

        super(FlatSurface, self).__init__(x, y)


class Skier(object):

    grav_acc = 9.81
    air_density = 0.85
    samples_per_sec = 120
    tolerable_acc = 1.5

    def __init__(self, mass=75.0, area=0.34, drag_coeff=0.821,
                 friction_coeff=0.03):
        """

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

        return (-np.sign(velocity) / 2 * self.air_density * self.drag_coeff *
                self.area * velocity**2)

    def friction_force(self, velocity,  slope=0.0, curvature=0.0):

        theta = np.tan(slope)

        normal_force = self.mass * (self.grav_acc * np.cos(theta) + curvature *
                                    velocity**2)

        return -np.sign(velocity) * self.friction_coeff * normal_force

    def fly_to(self, surface, init_pos, init_speed):
        """Returns the flight trajectory of the skier given the inititial
        conditionas and a surface which the skier contacts at the end of the
        flight trajectory.

        """

        def rhs(t, state):

            xdot = state[2]
            ydot = state[3]

            vxdot = self.drag_force(xdot) / self.mass
            vydot = -self.grav_acc + self.drag_force(ydot) / self.mass

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
                                           num=int(120 * sol.t[-1])))

        return sol.y[0], sol.y[1], sol.y[2], sol.y[3]

    def slide_on(self, surface, init_speed=0.0):
        """

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
            vdot = -self.grav_acc * np.sin(theta) + (
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
                                           num=int(120 * sol.t[-1])))

        return sol.t, sol.y
