========================
Example Analyze EFH Jump
========================

The following page describes how to analyze the equivalent fall height of
ski jump landing surface using the ``skijumpdesign`` API. Make sure to
:ref:`install <install>` the library first.

Load Data
=========

Start by loading data of a ski jump surface's horizontal and vertical
components measured in meters. The resulting surface can be visualized
with the ``Surface.plot()`` method. Create a tuple for the takeoff point
coordinates, and a variable for the takeoff angle. Data for this example
is taken from a jump measured with a level and tape measure and translated
into horizontal (x) and vertical (y) components.

.. plot::
   :include-source: True
   :context:
   :width: 600px

   from skijumpdesign import Surface

   takeoff_ang = np.deg2rad(10)  # radians
   takeoff_point = (0,0)  # meters

   x_ft = [-289.1,-260.9,-232.3,-203.7,-175.0,-146.3,-117.0,-107.4,-97.7,
           -88.0,-78.2,-68.5,-58.8,-49.1,-39.4,-34.5,-29.7,-24.8,-19.8,-17.8,
           -15.8,-13.8,-11.8,-9.8,-7.8,-5.9,-3.9,-2.0,0.0,0.0,0.0,2.0,3.9,5.9,
           7.9,9.9,11.9,13.9,15.9,17.9,19.9,21.9,23.9,25.8,27.8,29.7,31.5,33.4,
           35.2, 37.0,38.8,43.3,47.8,52.3,56.8,61.5,66.2,70.9,75.7,80.6,85.5,
           88.4,88.4] # feet

   y_ft = [74.8,64.4,55.5,46.4,37.7,29.1,22.2,19.7,17.2,14.8,12.5,10.2,7.7,5.2,
           2.9,1.8,0.7,-0.2,-1.0,-1.2,-1.4,-1.6,-1.7,-1.6,-1.5,-1.3,-1.0,-0.4,
           0.0,0.0,0.0,-0.3,-0.8,-1.0,-1.4,-1.4,-1.5,-1.5,-1.5,-1.5,-1.6,-1.8,
           -2.0,-2.4,-2.9,-3.5,-4.2,-5.0,-5.8,-6.7,-7.5,-9.8,-12.0,-14.2,-16.2,
           -18.1,-19.8,-21.4,-22.9,-24.0,-25.0,-25.6,-25.6] # feet

   x = np.asarray(x_ft)*0.3048 # convert to meters
   y = np.asarray(y_ft)*0.3048 # convert to meters

   measured_surf = Surface(x, y)

   measured_surf.plot()

Now that a surface has been created, a skier can be created. The skier can "ski"
along the surface before takeoff by slicing the array and using the ``slide_on()``
method which generates a skiing simulation trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import Skier

   x_beforetakeoff = x[x<=takeoff_point[0]]
   y_beforetakeoff = y[x<=takeoff_point[0]]

   before_takeoff = Surface(x_beforetakeoff, y_beforetakeoff)

   skier = Skier()

   beforetakeoff_traj = skier.slide_on(before_takeoff)

   beforetakeoff_traj.plot_time_series()

Flight
======

Once the skier leaves the takeoff ramp at the maximum speed they will be in flight. The
``Skier.fly_to()`` method can be used to simulate this longest flight trajectory.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   takeoff_vel = skier.end_vel_on(before_takeoff)

   flight = skier.fly_to(measured_surf, init_pos=before_takeoff.end,
                         init_vel=takeoff_vel)

   flight.plot_time_series()

The speed flight trajectory can be plotted alongside the surface.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   ax = measured_surf.plot()
   flight.plot(ax=ax, color='#9467bd')

Because the maximum flight trajectory is farther than the measured surface,
create a surface under the measured surface that the skier will impact when
they pass over ``measured_surf``.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import HorizontalSurface

   catch_surf = HorizontalSurface(np.min(measured_surf.y) - 0.1,
                               flight.pos[-1,0] + 2.0,
                               start=takeoff_point[0])
   ax = measured_surf.plot()
   ax = catch_surf.plot(ax=ax)
   flight.plot(ax=ax, color='#9467bd')


Calculate Equivalent Fall Height
================================

The equivalent fall height of the landing surface can be recalculated at constant
intervals relative to the provided takeoff point or start of the surface.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   dist, efh = measured_surf.calculate_efh(takeoff_ang, takeoff_point,
                                           skier, increment=0.2)

There is a convenience function for plotting the calculated efh.

.. plot::
   :include-source: True
   :context: close-figs
   :width: 600px

   from skijumpdesign import plot_efh

   plot_efh(dist,efh)

