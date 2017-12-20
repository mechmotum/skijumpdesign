import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


PARAMETERS = {
              'friction_coeff': 0.03,
              'grav_accel': 9.81,  # m/s**2
              'skier_mass': 75,  # kg
              'drag_coeff_times_area': 0.279,
              'air_density': 0.85,
              'gamma': 0.99, # fraction of circular section in inrun transition
              'tolerableGs_tranIn': 1.5, # max g force we are willing to let jumper feel
              'tolerableGs_tranOut': 3, # use this to find transition point
              }


def compute_approach_exit_speed(parent_slope_angle, start_pos, approach_len):
    """Returns the speed of the skier in meters per second  at the end of the
    approach (entry to approach-takeoff transition).

    Parameters
    ==========
    parent_slope_angle : float
        The angle of the existing parent slope in degrees.
    start_pos : float
        The position in meters along the slope from the beginning of the slope.
    approach_len : float
        The distance in meters along the slope from the skier starting position
        to the beginning of the approach transition.

    Returns
    =======
    exit_speed : float
        The speed the skier is traveling in meters per second at the entrance
        to the approach transition.

    """
    m = PARAMETERS['skier_mass']
    g = PARAMETERS['grav_accel']
    mu = PARAMETERS['friction_coeff']
    CdA = PARAMETERS['drag_coeff_times_area']
    rho = PARAMETERS['air_density']

    eta = (CdA * rho) / (2 * m)

    cos_ang = np.cos(np.deg2rad(parent_slope_angle))
    sin_ang = np.sin(np.deg2rad(parent_slope_angle))

    def rhs(t, x):
        """Right hand side of particle with drag and friction."""

        vel = x[1]

        pos_dot = vel
        vel_dot = (g * sin_ang - eta * vel**2 -
                   mu * g * cos_ang * np.sign(vel))

        return pos_dot, vel_dot

    def reach_approach_transition(t, x):
        """Returns zero when the skier gets to the end of the approach
        length."""
        pos = x[0]
        return pos - start_pos - approach_len

    reach_approach_transition.terminal = True

    sol = solve_ivp(rhs, (0.0, 1E4), (start_pos, 0),
                    events=(reach_approach_transition, ))

    return sol.y[1, -1]


def generate_approach_curve(entry_speed, parent_slope_angle, start_pos,
                            approach_len, max_acc):

    return approach_x, approach_y


def compute_design_speed(entry_speed, parent_slope_angle, launch_curve_x,
                         launch_curve_y):
    """Returns the magnitude of the takeoff velocity out of the approach
    curve.

    Parameters
    ==========
    entry_speed : float
        The magnitude of the skier's speed at the entry of the approach-takeoff
        transition.
    parent_slope_angle : float
        The angle of the existing parent slope in degrees.
    launch_curve_x : ndarray, shape(n,)
        The X coordinates in meters of points on the approach-takeoff
        transition.
    launch_curve_y : ndarray, shape(n,)
        The Y coordinates in meters of points on the approach-takeoff
        transition.

    Returns
    =======
    design_speed : float
        The magnitude of the skier's velocity at takeoff, this is called the
        "design speed."

    """
    m = PARAMETERS['skier_mass']
    g = PARAMETERS['grav_accel']
    mu = PARAMETERS['friction_coeff']
    CdA = PARAMETERS['drag_coeff_times_area']
    rho = PARAMETERS['air_density']

    eta = (CdA * rho) / (2 * m)

    dydx = np.hstack((0, np.diff(launch_curve_y) / np.diff(launch_curve_x)))
    ddydx = np.hstack((0, np.diff(dydx) / np.diff(launch_curve_x)))
    kurvature = ddydx / (1 + dydx**2)**1.5

    slope_interpolator = interp1d(launch_curve_x, dydx)
    kurva_interpolator = interp1d(launch_curve_x, kurvature)

    def rhs(t, y):

        x = y[0]  # horizontal position
        v = y[1]  # velocity tangent to slope

        slope = slope_interpolator(x)
        kurva = kurva_interpolator(x)

        theta = np.arctan(slope)

        xdot = v * np.cos(theta)

        N = m * (g * np.cos(theta) + kurva * v * v)
        vdot = -g * np.sin(theta) - eta*v*v - mu * N * np.sign(v) / m

        return xdot, vdot

    def reach_launch(t, y):
        """Returns zero when the skier gets to the end of the approach
        length."""
        return y[0] - launch_curve_x[-1]

    reach_launch.terminal = True

    sol = solve_ivp(rhs, (0.0, 1E4), (launch_curve_x[0], entry_speed),
                    events=(reach_launch, ))

    return sol.y[1, -1]


def compute_flight_trajectory(launch_angle, launch_speed, initial_x, initial_y):

    return traj_x, traj_y


def find_trajectory_point_where_path_is_parallel_to_parent_slope():

    return


def find_landing_transition_point():
    # beginning of transition curve

    return x, y


def compute_landing_surface():

    return x, y


def calculate_landing_transition_curve():

    return x, y
