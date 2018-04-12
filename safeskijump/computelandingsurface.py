#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:02:33 2017

@author: monthubbardMacbookPro
"""

import numpy as np
from numpy import sin, cos
#
def compute_landing_surfac(xTran,yTran):
    
    """ this function calculates the x,y coordinates of the safe landing surface 
    backwards from the landing transition point back up the hill to a point 
    near the critical point under the takeoff by integrating the safe slope 
    differential equation backwards from xTran,yTran
"""

    x=np.linspace(xTran,)

sol = solve_ivp(rhs,
                    (0.0, 1E4),  # time span
                    (takeoff_curve_x[0], entry_speed),  # initial conditions
                    events=(reach_near critical, ))

    design_speed = sol.y[1, -1]












    return xLanding, yLanding  # safe landing surface coordinates



This below is example numerical integration


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
    m = PARAMETERS['skier_mass']
    g = PARAMETERS['grav_acc']
    mu = PARAMETERS['friction_coeff']
    CdA = PARAMETERS['drag_coeff_times_area']
    rho = PARAMETERS['air_density']

    eta = (CdA * rho) / (2 * m)

    dydx = np.hstack((0, np.diff(takeoff_curve_y) / np.diff(takeoff_curve_x)))
    ddydx = np.hstack((0, np.diff(dydx) / np.diff(takeoff_curve_x)))
    kurvature = ddydx / (1 + dydx**2)**1.5

    slope_interpolator = interp1d(takeoff_curve_x, dydx,
                                  fill_value='extrapolate')
    kurva_interpolator = interp1d(takeoff_curve_x, kurvature,
                                  fill_value='extrapolate')

    def rhs(t, state):

        x = state[0]  # horizontal position
        v = state[1]  # velocity tangent to slope

        slope = slope_interpolator(x)
        kurva = kurva_interpolator(x)

        theta = np.arctan(slope)

        xdot = v * np.cos(theta)

        N = m * (g * np.cos(theta) + kurva * v * v)
        vdot = -g * np.sin(theta) - eta*v*v - mu * N * np.sign(v) / m

        return xdot, vdot

    def reach_launch(t, state):
        """Returns zero when the skier gets to the end of the approach
        length."""
        return state[0] - takeoff_curve_x[-1]

    reach_launch.terminal = True

    sol = solve_ivp(rhs,
                    (0.0, 1E4),  # time span
                    (takeoff_curve_x[0], entry_speed),  # initial conditions
                    events=(reach_launch, ))

    design_speed = sol.y[1, -1]

    return design_speed
