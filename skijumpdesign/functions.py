import os
import logging

import numpy as np
from fastcache import clru_cache

# TODO : Might be better to use:
# import matplotlib
# matplotlib.use('Agg')
# so that it doesn't try to use tk on heroku.
if 'ONHEROKU' in os.environ:
    plt = None
else:
    import matplotlib.pyplot as plt

from .skiers import Skier
from .surfaces import (Surface, FlatSurface, HorizontalSurface, TakeoffSurface,
                       LandingTransitionSurface, LandingSurface)
from .utils import InvalidJumpError, vel2speed


def snow_budget(parent_slope, takeoff, landing, landing_trans):
    """Returns the jump's cross sectional snow budget area of the EFH jump.

    Parameters
    ==========
    parent_slope : FlatSurface
        A FlatSurface that spans before and after the jump.
    takeoff : TakeoffSurface
        The clothiod-circle-clothiod-flat takeoff surface.
    landing : LandingSurface
        The EFH landing surface.
    landing_trans: LandingTransitionSurface
        The EFH landing transition surface.

    Returns
    =======
    float
        The cross sectional snow budget (area between the parent slope and jump
        curve) in meters squared.

    """

    # TODO : Make this function more robust, may need to handle jumps that are
    # above the x axis.
    if (np.any(takeoff.y > 0.0) or np.any(landing.y > 0.0) or
            np.any(landing_trans.y > 0.0)):
        logging.warn('Snowbudget invalid since jump about X axis.')

    A = parent_slope.area_under(x_start=takeoff.start[0],
                                x_end=landing_trans.end[0])
    B = takeoff.area_under() + landing.area_under() + landing_trans.area_under()

    return np.abs(A - B)


@clru_cache(maxsize=128)
def make_jump(slope_angle, start_pos, approach_len, takeoff_angle, fall_height,
              plot=False):
    """Returns a set of surfaces and output values that define the equivalent
    fall height jump design and the skier's flight trajectory.

    Parameters
    ==========
    slope_angle : float
        The parent slope angle in degrees. Counter clockwise is positive and
        clockwise is negative.
    start_pos : float
        The distance in meters along the parent slope from the top (x=0, y=0)
        to where the skier starts skiing.
    approach_len : float
        The distance in meters along the parent slope the skier travels before
        entering the takeoff.
    takeoff_angle : float
        The angle in degrees at end of the takeoff ramp. Counter clockwise is
        positive and clockwise is negative.
    fall_height : float
        The desired equivalent fall height of the landing surface in meters.
    plot : boolean
        If True a matplotlib figure showing the jump will appear.

    Returns
    =======
    slope : FlatSurface
        The parent slope starting at (x=0, y=0) until a meter after the jump.
    approach : FlatSurface
        The slope the skier travels on before entering the takeoff.
    takeoff : TakeoffSurface
        The circle-clothoid-circle-flat takeoff ramp.
    landing : LandingSurface
        The equivalent fall height landing surface.
    landing_trans : LandingTransitionSurface
        The minimum exponential landing transition.
    flight : Trajectory
        The maximum velocity flight trajectory.
    outputs : dictionary
        A dictionary of output values.

    """

    # TODO : function is too long!

    outputs = {'Takeoff Speed': None,
               'Flight Time': None,
               'Snow Budget': None}

    logging.info('Calling make_jump({}, {}, {}, {}, {})'.format(
        slope_angle, start_pos, approach_len, takeoff_angle, fall_height))

    skier = Skier()

    if takeoff_angle >= 90.0 or takeoff_angle <= slope_angle:
        msg = 'Invalid takeoff angle. Enter value between {} and 90 degrees'
        raise ValueError(msg.format(slope_angle))

    slope_angle = np.deg2rad(slope_angle)
    takeoff_angle = np.deg2rad(takeoff_angle)

    # The approach is the flat slope that the skier starts from rest on to gain
    # speed before reaching the takeoff ramp.
    init_pos = (start_pos * np.cos(slope_angle),
                start_pos * np.sin(slope_angle))

    approach = FlatSurface(slope_angle, approach_len, init_pos=init_pos)

    # The takeoff surface is the combined circle-clothoid-circle-flat.
    # TODO : If there is not enough speed, then this method will run forever
    # because the skier can't make the jump. Need to raise an error if this is
    # the case.
    takeoff_entry_speed = skier.end_speed_on(approach)
    takeoff = TakeoffSurface(skier, slope_angle, takeoff_angle,
                             takeoff_entry_speed, init_pos=approach.end)

    # The skier becomes airborne after the takeoff surface and the trajectory
    # is computed until the skier contacts the parent slope.
    takeoff_vel = skier.end_vel_on(takeoff, init_speed=takeoff_entry_speed)

    msg = 'Takeoff speed: {:1.3f} [m/s]'
    takeoff_speed = vel2speed(*takeoff_vel)[0]
    outputs['Takeoff Speed'] = takeoff_speed
    logging.info(msg.format(takeoff_speed))

    slope = FlatSurface(slope_angle, 100 * approach_len)

    flight = skier.fly_to(slope, init_pos=takeoff.end, init_vel=takeoff_vel)

    # The landing transition curve transfers the max velocity skier from their
    # landing point smoothly to the parent slope.
    landing_trans = LandingTransitionSurface(slope, flight, fall_height,
                                             skier.tolerable_landing_acc)

    slope = FlatSurface(slope_angle, np.sqrt(landing_trans.end[0]**2 +
                                             landing_trans.end[1]**2) + 1.0)

    land_trans_contact = HorizontalSurface(landing_trans.start[1],
                                           50.0,
                                           start=landing_trans.start[0] - 10.0)

    flight = skier.fly_to(land_trans_contact, init_pos=takeoff.end,
                          init_vel=takeoff_vel)
    outputs['Flight Time'] = flight.duration
    outputs['Flight Distance'] = flight.pos[-1, 0] - flight.pos[0, 0]
    logging.info('Flight time: {:1.3f} [s]'.format(flight.duration))

    # The landing surface ensures an equivalent fall height for any skiers that
    # do not reach maximum velocity.
    landing = LandingSurface(skier, takeoff.end, takeoff_angle,
                             landing_trans.start, fall_height, surf=slope)

    logging.info("Num points in landing surface: {}".format(len(landing.x)))

    if landing.y[0] < slope.interp_y(landing.x[0]):
        raise InvalidJumpError('Fall height is too large.')

    x_at_highest = flight.interp_pos_wrt_slope(0.0)[0]
    y_at_highest = flight.interp_pos_wrt_x(x_at_highest)[1]
    outputs['Flight Height'] = y_at_highest - landing.interp_y(x_at_highest)

    budget = snow_budget(slope, takeoff, landing, landing_trans)
    outputs['Snow Budget'] = budget
    logging.info('Snow budget: {} m^2'.format(budget))

    if plot:
        plot_jump(slope, approach, takeoff, landing, landing_trans, flight)
        plt.show()

    return slope, approach, takeoff, landing, landing_trans, flight, outputs


def plot_jump(slope, approach, takeoff, landing, landing_trans, flight):
    """Returns a matplotlib axes with the jump and flight plotted given the
    surfaces created by ``make_jump()``."""
    ax = slope.plot(linestyle='dashed', color='black', label='Slope')
    ax = approach.plot(ax=ax, linewidth=2, label='Approach')
    ax = takeoff.plot(ax=ax, linewidth=2, label='Takeoff')
    ax = landing.plot(ax=ax, linewidth=2, label='Landing')
    ax = landing_trans.plot(ax=ax, linewidth=2, label='Landing Transition')
    ax = flight.plot(ax=ax, linestyle='dotted', label='Flight')
    ax.grid()
    ax.legend()
    return ax
