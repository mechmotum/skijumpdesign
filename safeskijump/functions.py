import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .classes import Surface, FlatSurface, Skier

PARAMETERS = {
              'friction_coeff': 0.03,
              'grav_acc': 9.81,  # m/s**2
              'skier_mass': 75,  # kg
              'drag_coeff_times_area': 0.279,
              'air_density': 0.85,
              'gamma': 0.99, # fraction of circular section in inrun transition
              'takeoff_tolerable_acc': 1.5,  # G
              'landing_tolerable_acc': 3, # use this to find transition point
              'time_on_takeoff_ramp': 0.2,  # seconds
              }


def compute_approach_exit_speed(slope_angle, start_pos, approach_len):
    """Returns the speed of the skier in meters per second at the end of the
    approach (entry to approach-takeoff transition).

    Parameters
    ==========
    slope_angle : float
        The angle of the parent slope in degrees. This is the angle about the
        negative Z axis.
    start_pos : float
        The position in meters along the slope from the top (beginning) of the
        slope.
    approach_len : float
        The distance in meters along the slope from the skier starting position
        to the beginning of the approach transition.

    Returns
    =======
    exit_speed : float
        The speed the skier is traveling in meters per second at the entrance
        to the takeoff curve.

    """
    surf = FlatSurface(-slope_angle, approach_len, start_pos=start_pos)

    skier = Skier()

    times, states = skier.slide_on(surf)

    return states[1, -1]


def generate_takeoff_curve(slope_angle, entry_speed, takeoff_angle,
                           tolerable_acc, numpoints=500):
    """Returns the X and Y coordinates of the clothoid-circle-clothoid takeoff
    curve (without the flat takeoff ramp).

    Parameters
    ==========
    slope_angle : float
        The angle of the parent slope in degrees.
    entry_speed : float
        The magnitude of the skier's velocity in meters per second as they
        enter the takeoff curve (i.e. approach exit speed).
    takeoff_angle : float
        The desired takeoff angle at the takeoff point in degrees.
    tolerable_acc : float
        The tolerable acceleration of the skier in G's.
    numpoints : integer, optional
        The n number of points in the produced curve.

    Returns
    =======
    X : ndarray, shape(n,)
        The n X coordinates of the curve.
    Y : ndarray, shape(n,)
        The n Y coordinates of the curve.
    dYdX : ndarray, shape(n-1,)
        The slope of the curve.
    angle : ndarray, shape(n-1,)
        The angle of the slope of the curve.

    Notes
    =====

    lam = parent slope angle, radian
    beta = takeoff angle, radians
    gamma = percent of circular segment desired in transition

    s1 = left side clothoid length at any point (downhill part)
    s2 = right side clothoid at any point length (uphill part)
    L1 = left side (longest) clothoid length (downhill part)
    L2 = right side (longest) clothoid length (uphill part)

    """

    g = PARAMETERS['grav_acc']
    gamma = PARAMETERS['gamma']

    lam = np.deg2rad(slope_angle)
    beta = np.deg2rad(takeoff_angle)

    rotation_clothoid = (lam - beta) / 2
    # used to rotate symmetric clothoid so that left side is at lam and right
    # sid is at beta

    # radius_min is the radius of the circular part of the transition. Every
    # other radius length (in the clothoid) will be longer than that, as this
    # will ensure the g - force felt by the skier is always less than a desired
    # value. This code ASSUMES that the velocity at the minimum radius is equal
    # to the velocity at the end of the approach.
    radius_min = entry_speed**2 / (tolerable_acc * g)

    #  x,y data for circle
    thetaCir = 0.5 * gamma * (lam + beta)
    xCirBound = radius_min * np.sin(thetaCir)
    xCirSt = -radius_min * np.sin(thetaCir)
    xCir = np.linspace(xCirSt, xCirBound, num=numpoints)
    yCir = radius_min - np.sqrt(radius_min**2 - xCir**2)

    # x,y data for one clothoid
    A_squared = radius_min**2 * (1 - gamma) * (lam + beta)
    A = np.sqrt(A_squared)
    clothoid_length = A * np.sqrt((1 - gamma) * (lam + beta))

    # generates arc length points for one clothoid
    s = np.linspace(clothoid_length, 0, numpoints)

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

    # Shift the entry point of the curve to be at X=0, Y=0
    X = X - np.min(X)
    Y = Y - Y[np.argmin(X)]

    dYdX = np.diff(Y) / np.diff(X)
    angle = np.arctan(dYdX)

    return X, Y, dYdX, angle


def compute_design_speed(slope_angle, entry_speed, takeoff_curve_x,
                         takeoff_curve_y):
    """Returns the magnitude of the takeoff velocity at the takeoff point.

    Parameters
    ==========
    slope_angle : float
        The angle of the parent slope in degrees.
    entry_speed : float
        The magnitude of the skier's speed at the entry of the takeoff curve
        (same as approach exit speed).
    takeoff_curve_x : ndarray, shape(n,)
        The X coordinates in meters of points on the takeoff curve.
    takeoff_curve_y : ndarray, shape(n,)
        The Y coordinates in meters of points on the takeoff curve.

    Returns
    =======
    design_speed : float
        The magnitude of the skier's velocity at the takeoff point, this is
        called the "design speed".

    """
    surf = Surface(takeoff_curve_x, takeoff_curve_y)

    skier = Skier()

    times, states = skier.slide_on(surf, entry_speed)

    return states[1, -1]


def add_takeoff_ramp(takeoff_angle, ramp_entry_speed, takeoff_curve_x,
                     takeoff_curve_y):
    """Returns the X and Y coordinates of the takeoff curve with the flat
    takeoff ramp added to the terminus.

    Parameters
    ==========
    takeoff_angle : float
        The desired takeoff angle at the takeoff point in degrees, measured as
        a positive Z rotation from the horizontal X axis.
    ramp_entry_speed : float
        The magnitude of the skier's speed at the exit of the second clothoid.
    takeoff_curve_x : ndarray, shape(n,)
        The X coordinates in meters of points on the takeoff curve without the
        takeoff ramp.
    takeoff_curve_y : ndarray, shape(n,)
        The Y coordinates in meters of points on the takeoff curve without the
        takeoff ramp.

    Returns
    =======
    ext_takeoff_curve_x : ndarray, shape(n,)
        The X coordinates in meters of points on the takeoff curve with the
        takeoff ramp added as an extension.
    ext_takeoff_curve_y : ndarray, shape(n,)
        The Y coordinates in meters of points on the takeoff curve with the
        takeoff ramp added as an extension.

    """

    ramp_time = PARAMETERS['time_on_takeoff_ramp']
    ramp_len = ramp_time * ramp_entry_speed  # meters
    start_x = takeoff_curve_x[-1]
    start_y = takeoff_curve_y[-1]
    points_per_meter = len(takeoff_curve_x) / (start_x - takeoff_curve_x[0])
    stop_x = start_x + ramp_len * np.cos(np.deg2rad(takeoff_angle))
    ramp_x = np.linspace(start_x, stop_x, num=int(points_per_meter * stop_x -
                                                  start_x))
    stop_y = start_y + ramp_len * np.sin(np.deg2rad(takeoff_angle))
    ramp_y = np.linspace(start_y, stop_y, num=len(ramp_x))

    ext_takeoff_curve_x = np.hstack((takeoff_curve_x, ramp_x))
    ext_takeoff_curve_y = np.hstack((takeoff_curve_y, ramp_y))

    return ext_takeoff_curve_x, ext_takeoff_curve_y


def compute_flight_trajectory(slope_angle, takeoff_point, takeoff_angle,
                              takeoff_speed, num_points=1000000):
    """Returns the X and Y coordinates of the skier's flight trajectory
    beginning at the launch point and ending when the trajectory intersects the
    parent slope.

    Parameters
    ==========
    slope_angle : float
        The angle of the parent slope in degrees.
    takeoff_point : tuple of floats
        The X and Y coordinates of the takeoff point, i.e. where the skier
        leaves the jump and exits into the air.
    takeoff_angle : float
        The angle of the takeoff surface (positive Z rotation) in degrees.
    takeoff_speed : float
        The magnitude of the skier's speed at the takeoff point in meters per
        second.

    Returns
    =======
    trajectory_x : ndarray, shape(n, )
        The X coordinates of the flight trajectory.
    trajectory_y : ndarray, shape(n, )
        The Y coordinates of the flight trajectory.

    """
    slope_angle = np.deg2rad(slope_angle)
    takeoff_angle = np.deg2rad(takeoff_angle)

    m = PARAMETERS['skier_mass']
    g = PARAMETERS['grav_acc']
    CdA = PARAMETERS['drag_coeff_times_area']
    rho = PARAMETERS['air_density']

    eta = (CdA * rho) / (2 * m)

    def rhs(t, state):

        vx = state[2]
        vy = state[3]

        xdot = vx
        ydot = vy

        vxdot = -eta*np.sign(vx)*vx**2
        vydot = -g - eta*np.sign(vy)*vy**2

        return xdot, ydot, vxdot, vydot

    def touch_slope(t, state):
        """Returns zero when the skier gets to the end of the approach
        length."""
        x = state[0]
        y = state[1]

        m = np.tan(-slope_angle)
        d = (y - m * x) * np.cos(slope_angle)

        return d

    touch_slope.terminal = True

    init_conds = (takeoff_point[0],
                  takeoff_point[1],
                  takeoff_speed * np.cos(takeoff_angle),
                  takeoff_speed * np.sin(takeoff_angle))

    # TODO : The linspace call here takes some time. We could call solve_ivp
    # twice, the first to find the ending time and the second with the smaller
    # number of points. This would likely only be a little savings though.

    sol = solve_ivp(rhs,
                    (0.0, 1E4),
                    init_conds,
                    t_eval=np.linspace(0.0, 1E4, num=num_points),
                    events=(touch_slope, ))

    return sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def find_parallel_traj_point(slope_angle, flight_traj_x, flight_traj_y,
                             flight_speed_x, flight_speed_y):
    """Returns the X and Y coordinates of the point on the flight trajectory
    where the flight trajectory slope is parallel to the parent slope."""

    slope_angle = np.deg2rad(slope_angle)

    flight_traj_slope = flight_speed_y / flight_speed_x

    xpara_interpolator = interp1d(flight_traj_slope, flight_traj_x)
    y_interpolator = interp1d(flight_traj_x, flight_traj_y)

    xpara = xpara_interpolator(np.tan(-slope_angle))
    ypara = y_interpolator(xpara)

    return xpara, ypara


def find_landing_transition_point():
    # beginning of transition curve

    return x, y


def compute_landing_surface():

    return x, y


def calculate_landing_transition_curve():

    return x, y


def make_jump(slope_angle, start_pos, approach_len, takeoff_angle):
    """Returns the takeoff curve and the flight trajectory curve."""

    takeoff_entry_speed = compute_approach_exit_speed(
        slope_angle, start_pos, approach_len)

    tolerable_acc = PARAMETERS['takeoff_tolerable_acc']

    curve_x, curve_y, _, _ = generate_takeoff_curve(
        slope_angle, takeoff_entry_speed, takeoff_angle, tolerable_acc)

    ramp_entry_speed = compute_design_speed(
        takeoff_entry_speed, slope_angle, curve_x, curve_y)

    curve_x, curve_y = add_takeoff_ramp(
        takeoff_angle, ramp_entry_speed, curve_x, curve_y)

    design_speed = compute_design_speed(
        takeoff_entry_speed, slope_angle, curve_x, curve_y)

    takeoff_dist = start_pos + approach_len
    init_x = takeoff_dist * np.cos(np.deg2rad(slope_angle)) + curve_x[-1]
    init_y = -takeoff_dist * np.sin(np.deg2rad(slope_angle)) + curve_y[-1]

    traj_x, traj_y, vel_x, vel_y = compute_flight_trajectory(
        slope_angle, (init_x, init_y), takeoff_angle, design_speed)

    xpara, ypara = find_parallel_traj_point(slope_angle, traj_x, traj_y, vel_x,
                                            vel_y)

    return curve_x, curve_y, traj_x, traj_y


def create_plot_arrays(slope_angle, start_pos, approach_len, takeoff_angle,
                       takeoff_curve_x, takeoff_curve_y, flight_traj_x,
                       flight_traj_y):

    # plot approach
    l = np.linspace(start_pos, start_pos + approach_len)
    x = l * np.cos(np.deg2rad(slope_angle))
    y = -l * np.sin(np.deg2rad(slope_angle))
    approach_xy = (x, y)

    # plot takeoff curve
    shifted_takeoff_curve_x = takeoff_curve_x + x[-1]
    shifted_takeoff_curve_y = takeoff_curve_y + y[-1]
    takeoff_xy = (shifted_takeoff_curve_x, shifted_takeoff_curve_y)

    # plot takeoff angle line
    takeoff_line_slope = np.tan(np.deg2rad(takeoff_angle))
    takeoff_line_intercept = (shifted_takeoff_curve_y[-1] - takeoff_line_slope *
                              shifted_takeoff_curve_x[-1])

    x_takeoff = np.linspace(shifted_takeoff_curve_x[0],
                            shifted_takeoff_curve_x[-1] + 5.0)
    y_takeoff = takeoff_line_slope * x_takeoff + takeoff_line_intercept

    # plot flight trajectory
    flight_xy = (flight_traj_x, flight_traj_y)

    return approach_xy, takeoff_xy, flight_xy


def plot_jump(start_pos, approach_len, slope_angle, takeoff_angle,
              takeoff_curve_x, takeoff_curve_y, flight_traj_x,
              flight_traj_y):

    fig, ax = plt.subplots(1, 1)

    # plot approach
    l = np.linspace(start_pos, start_pos + approach_len)
    x = l * np.cos(np.deg2rad(slope_angle))
    y = -l * np.sin(np.deg2rad(slope_angle))
    ax.plot(x[0], y[0], marker='x', markersize=14)

    # plot starting location of skier
    ax.plot(x, y)

    # plot launch curve
    shifted_takeoff_curve_x = takeoff_curve_x + x[-1]
    shifted_takeoff_curve_y = takeoff_curve_y + y[-1]
    ax.plot(shifted_takeoff_curve_x, shifted_takeoff_curve_y)

    # plot takeoff angle line
    takeoff_line_slope = np.tan(np.deg2rad(takeoff_angle))
    takeoff_line_intercept = (shifted_takeoff_curve_y[-1] - takeoff_line_slope *
                              shifted_takeoff_curve_x[-1])

    x_takeoff = np.linspace(shifted_takeoff_curve_x[0],
                            shifted_takeoff_curve_x[-1] + 5.0)
    y_takeoff = takeoff_line_slope * x_takeoff + takeoff_line_intercept
    ax.plot(x_takeoff, y_takeoff, '--')

    # plot flight trajectory
    ax.plot(flight_traj_x, flight_traj_y)

    ax.set_aspect('equal')

    return ax
