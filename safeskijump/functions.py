import numpy as np
import matplotlib.pyplot as plt

from .classes import (Surface, FlatSurface, ClothoidCircleSurface,
                      TakeoffSurface, LandingTransitionSurface, LandingSurface,
                      Skier)


def make_jump(slope_angle, start_pos, approach_len, takeoff_angle, fall_height,
              plot=False):
    """Returns a set of surfaces that define the equivalent fall height jump
    design and the skier's flight trajectory.

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
    flight : Surface
        A "surface" that encodes the maximum velocity flight trajectory.

    """

    # TODO : Move these to skier?
    time_on_ramp = 0.2  # seconds

    skier = Skier()

    slope_angle = np.deg2rad(slope_angle)
    takeoff_angle = np.deg2rad(takeoff_angle)

    # The approach is the flat slope that the skier starts from rest on to gain
    # speed before reaching the takeoff ramp.
    init_pos = (start_pos * np.cos(slope_angle),
                start_pos * np.sin(slope_angle))

    approach = FlatSurface(slope_angle, approach_len, init_pos=init_pos)

    # The takeoff entry surface is the first portion of the ramp that the skier
    # encounters that does not include the flat final portion of the takeoff
    # surface.
    takeoff_entry_speed = skier.end_speed_on(approach)

    takeoff_entry = ClothoidCircleSurface(slope_angle,
                                          takeoff_angle,
                                          takeoff_entry_speed,
                                          skier.tolerable_sliding_acc,
                                          init_pos=approach.end)

    # The takeoff surface is the combined circle-clothoid-circle-flat.
    ramp_entry_speed = skier.end_speed_on(takeoff_entry,
                                          init_speed=takeoff_entry_speed)

    takeoff = TakeoffSurface(takeoff_entry, ramp_entry_speed, time_on_ramp)

    # The skier becomes airborne after the takeoff surface and the trajectory
    # is computed until the skier contacts the parent slope.
    takeoff_vel = skier.end_vel_on(takeoff, init_speed=takeoff_entry_speed)

    slope = FlatSurface(slope_angle, 100 * approach_len)

    flight_time, flight_traj = skier.fly_to(slope, init_pos=takeoff.end,
                                            init_vel=takeoff_vel)

    flight = Surface(x=flight_traj[0], y=flight_traj[1])

    # The landing transition curve transfers the max velocity skier from their
    # landing point smoothly to the parent slope.
    landing_trans = LandingTransitionSurface(slope, flight_traj, fall_height,
                                             skier.tolerable_landing_acc)

    slope = FlatSurface(slope_angle, np.sqrt(landing_trans.end[0]**2 +
                                             landing_trans.end[1]**2) + 1.0)

    # The landing surface ensures an equivalent fall height for any skiers that
    # do not reach maximum velocity.
    landing = LandingSurface(skier, takeoff.end, takeoff_angle,
                             landing_trans.start, fall_height, surf=slope)

    if plot:
        plot_jump(slope, approach, takeoff, landing, landing_trans, flight)
        plt.show()

    return slope, approach, takeoff, landing, landing_trans, flight


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
