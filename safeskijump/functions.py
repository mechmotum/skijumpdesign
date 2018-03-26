import numpy as np
import matplotlib.pyplot as plt

from .classes import (Surface, FlatSurface, ClothoidCircleSurface,
                      TakeoffSurface, LandingTransitionSurface, LandingSurface,
                      Skier)


def make_jump(slope_angle, start_pos, approach_len, takeoff_angle, fall_height,
              plot=False):
    """

    Parameters
    ==========
    slope_angle : float
        The parent slope angle in degrees. Counter clockwise is positive and
        clockwise is negative.
    start_pos : float
        The distance in meters from the top for the parent slope to where the
        skier starts from.
    approach_len : float
        The distance in meters
    takeoff_angle : float
        The angle in degrees at end of the takeoff ramp. Counter clockwise is
        positive and clockwise is negative.
    fall_height : float
        The equivalent fall height in meters.
    plot : boolean
        If True a matplotlib figure showing the jump will appear.

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

    slope = FlatSurface(slope_angle, 4 * approach_len)

    flight_time, flight_traj = skier.fly_to(slope, init_pos=takeoff.end,
                                            init_vel=takeoff_vel)

    flight = Surface(x=flight_traj[0], y=flight_traj[1])

    # The landing transition curve transfers the max velocity skier from their
    # landing point smoothly to the parent slope.
    landing_trans = LandingTransitionSurface(slope, flight_traj, fall_height,
                                             skier.tolerable_landing_acc)

    # The landing surface ensures an equivalent fall height for any skiers that
    # do not reach maximum velocity.
    landing = LandingSurface(skier, takeoff.end, takeoff_angle,
                             landing_trans.start, fall_height, surf=slope)

    if plot:
        plot_jump(slope, approach, takeoff, landing, landing_trans, flight)
        plt.show()

    return slope, approach, takeoff, landing, landing_trans, flight


def plot_jump(slope, approach, takeoff, landing, landing_trans, flight):
    """Returns a matplotlib axes that plots the jump."""
    ax = slope.plot(linestyle='dashed', color='black', label='Slope')
    ax = approach.plot(ax=ax, linewidth=2, label='Approach')
    ax = takeoff.plot(ax=ax, linewidth=2, label='Takeoff')
    ax = landing.plot(ax=ax, linewidth=2, label='Landing')
    ax = landing_trans.plot(ax=ax, linewidth=2, label='Landing Transition')
    ax = flight.plot(ax=ax, linestyle='dotted', label='Flight')
    ax.grid()
    ax.legend()
    return ax


def create_plot_arrays(slope, approach, takeoff, landing, landing_trans,
                        flight):
    return approach.xy, takeoff.xy, flight.xy
