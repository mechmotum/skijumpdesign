import numpy as np
import sympy as sm
from sympy.utilities.autowrap import autowrap

EPS = np.finfo(float).eps

# NOTE : These parameters are more associated with an environment, but this
# doesn't warrant making a class for them. Maybe a namedtuple would be useful
# though.
GRAV_ACC = 9.81  # m/s/s
AIR_DENSITY = 0.85  # kg/m/m/m


class InvalidJumpError(Exception):
    """Custom class to signal that a poor combination of parameters have been
    supplied to the surface building functions."""
    pass


def _generate_fast_drag_func():
    v, A, ro, C = sm.symbols('v, A, ro, C')
    drag_expr = -sm.sign(v) / 2 * ro * C * A * v**2
    return autowrap(drag_expr, backend='cython', args=(ro, v, C, A))

try:
    compute_drag = _generate_fast_drag_func()
except:
    compute_drag = None


def _gen_fast_distance_from():
    theta, x, y = sm.symbols('theta, x, y')
    expr = (y - sm.tan(theta) * x) * sm.cos(theta)
    return autowrap(expr, backend='cython', args=(theta, x, y))

try:
    compute_dist_from_flat = _gen_fast_distance_from()
except:
    compute_dist_from_flat = None


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
    vel_y = speed * np.sin(angle)
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
