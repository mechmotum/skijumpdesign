=======================
Example EFH Jump Design
=======================

The following page describes how to construct a constant equivalent fall height
ski jump landing surface using the ``skijumpdesign`` API. Make sure to :ref:`install <install>`
the library first.

Approach
========

Start by creating a 25 meter length of an approach path (also called the
in-run) which is flat and has a downward slope angle of 20 degrees. The
resulting surface can be visualized with the ``FlatSurface.plot()`` method.

.. plot::
   :include-source: True
   :context:
   :width: 600px

   from skijumpdesign import FlatSurface

   approach_ang = -np.deg2rad(20)  # radians
   approach_len = 25.0  # meters

   approach = FlatSurface(approach_ang, approach_len)

   approach.plot()

Now that a surface has been created, a skier can be created. The skier can "ski"
along the approach surface using the ``slide_on()`` method which generates a
skiing simulation trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import Skier

   skier = Skier()

   approach_traj = skier.slide_on(approach)

   approach_traj.plot_time_series()

Approach-Takeoff Transition
===========================

The approach-takeoff transition is constructed with a clothoid-circle-clothoid-flat surface to
transition from the parent slope angle to the desired takeoff angle, in this case 15 degrees.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import TakeoffSurface

   takeoff_entry_speed = skier.end_speed_on(approach)

   takeoff_ang = np.deg2rad(15)

   takeoff = TakeoffSurface(skier, approach_ang, takeoff_ang,
                            takeoff_entry_speed, init_pos=approach.end)

   ax = approach.plot()
   takeoff.plot(ax=ax)

The trajectory of the skier on the takeoff can be examined also.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   takeoff_traj = skier.slide_on(takeoff, takeoff_entry_speed)

   takeoff_traj.plot_time_series()

Flight
======

Once the skier leaves the takeoff ramp at the maximum (design) speed they will be in flight. The
``Skier.fly_to()`` method can be used to simulate this longest flight trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   takeoff_vel = skier.end_vel_on(takeoff, init_speed=takeoff_entry_speed)

   flight = skier.fly_to(approach, init_pos=takeoff.end,
                         init_vel=takeoff_vel)

   flight.plot_time_series()

The design speed flight trajectory can be plotted as an extension of the approach and takeoff
surfaces.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   flight.plot(ax=ax, color='#9467bd')

Landing Transition
==================

There is a single infinity of landing surfaces that satisfy the efh differential 
equation and provide the desired equivalent fall height. The algorithm selects 
the one of these that is closest to the parent slope, and hence is least expensive 
to build, but which still is able to transition back to the parent slope with 
slope continuity and simultaneously is constrained to experience limited normal acceleration. 
The final part of this step is to determine the landing transition curve (shown in red below)
which connects the optimum (cheapest) constant efh landing surface to the parent slope.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import LandingTransitionSurface

   fall_height = 0.5

   landing_trans = LandingTransitionSurface(approach,
       flight, fall_height, skier.tolerable_landing_acc)

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   ax = flight.plot(ax=ax, color='#9467bd')
   landing_trans.plot(ax=ax, color='#d62728')

Constant EFH Landing
====================

Finally, the cheapest equivalent fall height landing surface (shown in green below)
can be calculated. This surface is continuous in slope with the landing transition
surface at the impact point. It accommodates all takeoff speeds below the maximum
takeoff (design) speed above.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import LandingSurface

   slope = FlatSurface(approach_ang, np.sqrt(landing_trans.end[0]**2 +
                                             landing_trans.end[1]**2) + 1.0)


   landing = LandingSurface(skier, takeoff.end, takeoff_ang,
                            landing_trans.start, fall_height,
                            surf=slope)

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   ax = flight.plot(ax=ax, color='#9467bd')
   ax = landing_trans.plot(ax=ax, color='#d62728')
   landing.plot(ax=ax, color='#2ca02c')

The design calculates a landing surface shape that produces a constant equivalent
fall height. This can be verified using the ``landing.calculate_efh()`` function that
calculates the equivalent fall height for the surface that was produced.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign.functions import plot_efh

   dist, efh = landing.calculate_efh(takeoff_ang, takeoff.end, skier, increment=0.2)
   plot_efh(landing, takeoff_ang, takeoff.end)

Entire Jump
===========

There is a convenience function for plotting the jump:

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import plot_jump

   plot_jump(slope, approach, takeoff, landing, landing_trans, flight)
