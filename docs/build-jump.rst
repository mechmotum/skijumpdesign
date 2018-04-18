=======================
Example EFH Jump Design
=======================

The following page describes how to construct a typical equivalent fall height
ski jump using the ``safeskijump`` API.

Start by creating a 20 meter length of an approach surface (also called the
in-run) which is flat and has a downward slope angle of 20 degrees. The
resulting surface can be visualized with the ``FlatSurface.plot()`` method.

.. plot::
   :include-source: True
   :context:
   :width: 600px

   from safeskijump.classes import FlatSurface

   approach_ang = -np.deg2rad(20)  # radians
   approach_len = 20.0  # meters

   approach = FlatSurface(approach_ang, approach_len)

   approach.plot()

Now that a surface has been created a skier can be created. The skier can "ski"
along the approach surface using the ``slide_on()`` method which generates a
skiing simulation trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from safeskijump.classes import Skier

   skier = Skier()

   times, approach_traj = skier.slide_on(approach)

   fig, ax = plt.subplots()
   ax.plot(approach_traj[0], approach_traj[1])
   ax.set_xlabel('Horizontal Position [m]')
   ax.set_ylabel('Horizontal Speed [m/s]')

The takeoff ramp is constructed with a clothoid-circle-clothoid-flat surface to
transition from the approach to the desired takeoff angle, in this case 15
degrees.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from safeskijump.classes import (ClothoidCircleSurface,
                                    TakeoffSurface)

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

   times, takeoff_traj = skier.slide_on(takeoff, takeoff_entry_speed)

   fig, ax = plt.subplots()
   ax.plot(takeoff_traj[0], takeoff_traj[1])
   ax.set_xlabel('Horizontal Position [m]')
   ax.set_ylabel('Horizontal Speed [m/s]')

Once the skier leaves the takeoff ramp they will be in flight. The
``Skier.fly_to()`` method can be used to simulate the flight trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   takeoff_vel = skier.end_vel_on(takeoff, init_speed=takeoff_entry_speed)

   flight = skier.fly_to(approach, init_pos=takeoff.end,
                         init_vel=takeoff_vel)

   flight.plot_time_series()

The flight trajectory can be plotted alongside the surfaces.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   flight.plot(ax=ax)

The next step is to determine a landing transition curve.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from safeskijump.classes import LandingTransitionSurface

   fall_height = 0.5

   landing_trans = LandingTransitionSurface(approach,
       flight, fall_height, skier.tolerable_landing_acc)

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   ax = flight.plot(ax=ax)
   landing_trans.plot(ax=ax)

Finally, the equivalent fall height landing surface can be generated to
accommodate all takeoff speeds below the maximum takeoff speed above.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from safeskijump.classes import LandingSurface

   slope = FlatSurface(approach_ang, np.sqrt(landing_trans.end[0]**2 +
                                             landing_trans.end[1]**2) + 1.0)


   landing = LandingSurface(skier, takeoff.end, takeoff_ang,
                            landing_trans.start, fall_height,
                            surf=slope)

   ax = approach.plot()
   ax = takeoff.plot(ax=ax)
   ax = flight.plot(ax=ax)
   ax = landing_trans.plot(ax=ax)
   landing.plot(ax=ax)

There is a convenience function for plotting the jump:

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from safeskijump.functions import plot_jump

   plot_jump(slope, approach, takeoff, landing, landing_trans, flight)
