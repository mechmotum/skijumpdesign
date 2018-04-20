import time
import logging
from math import isclose

import numpy as np
from scipy.integrate import solve_ivp

from .trajectories import Trajectory
from .utils import GRAV_ACC, AIR_DENSITY
from .utils import InvalidJumpError
from .utils import compute_drag


class Skier(object):
    """Class that represents a skier which can slide on surfaces and fly in the
    air."""

    samples_per_sec = 360
    max_flight_time = 30.0  # seconds

    def __init__(self, mass=75.0, area=0.34, drag_coeff=0.821,
                 friction_coeff=0.03, tolerable_sliding_acc=1.5,
                 tolerable_landing_acc=3.0):
        """Instantiates a skier with default properties.

        Parameters
        ==========
        mass : float
            The mass of the skier.
        area : float
            The frontal area of the skier.
        drag_coeff : float
            The air drag coefficient of the skier.
        friction_coeff : float
            The sliding friction coefficient between the skis and the slope.
        tolerable_sliding_acc : float
            The maximum normal acceleration in G's that a skier can withstand
            while sliding.
        tolerable_landing_acc : float
            The maximum normal acceleration in G's that a skier can withstand
            when landing.

        """

        self.mass = mass
        self.area = area
        self.drag_coeff = drag_coeff
        self.friction_coeff = friction_coeff
        self.tolerable_sliding_acc = tolerable_sliding_acc
        self.tolerable_landing_acc = tolerable_landing_acc

    def drag_force(self, speed):
        """Returns the drag force in Newtons opposing the speed of the
        skier."""

        if compute_drag is None:
            return (-np.sign(speed) / 2 * AIR_DENSITY * self.drag_coeff *
                    self.area * speed**2)
        else:
            return compute_drag(AIR_DENSITY, speed, self.drag_coeff,
                                self.area)

    def friction_force(self, speed, slope=0.0, curvature=0.0):
        """Returns the friction force in Newtons opposing the speed of the
        skier.

        Parameters
        ==========
        speed : float
            The tangential speed of the skier in meters per second.
        slope : float
            The slope of the surface at the point of contact.
        curvature : float
            The curvature of the surface at the point of contact.

        """

        theta = np.tan(slope)

        normal_force = self.mass * (GRAV_ACC * np.cos(theta) + curvature *
                                    speed**2)

        return -np.sign(speed) * self.friction_coeff * normal_force

    def fly_to(self, surface, init_pos, init_vel, fine=True,
               logging_type='info'):
        """Returns the flight trajectory of the skier given the initial
        conditions and a surface which the skier contacts at the end of the
        flight trajectory.

        Parameters
        ==========
        surface : Surface
            A landing surface. This surface must intersect the flight path.
        init_pos : 2-tuple of floats
            The X and Y coordinates of the starting point of the flight.
        init_vel : 2-tuple of floats
            The X and Y components of the skier's velocity at the start of the
            flight.
        fine : boolean
            If True two integrations occur. The first finds the landing time
            with coarse time steps and the second integrates over a finer
            equally spaced time steps. False will skip the second integration.
        logging_type : string
            The logging level desired for the non-debug logging calls in this
            function. Useful for suppressing too much information since this
            runs a lot.

        Returns
        =======
        trajectory : Trajectory
        times : ndarray, shape(n,)
            The values of time corresponding to each state instance.
        states : ndarray, shape(n, 4)
            The states: (x, y, vx, vy) for each instance of time. The last
            value of the state corresponds to the skier touching the surface.

        Raises
        ======
        InvalidJumpError if the skier does not contact a surface within
        Skier.max_flight_time.

        """

        def rhs(t, state):

            xdot = state[2]
            ydot = state[3]

            vxdot = self.drag_force(xdot) / self.mass
            vydot = -GRAV_ACC + self.drag_force(ydot) / self.mass

            return xdot, ydot, vxdot, vydot

        def touch_surface(t, state):

            x = state[0]
            y = state[1]

            return surface.distance_from(x, y)

        touch_surface.terminal = True
        # NOTE: always from above surface, positive to negative crossing
        touch_surface.direction = -1

        logging_call = getattr(logging, logging_type)

        # NOTE : For a more accurate event time, the error tolerances on the
        # states need to be lower.
        logging_call('Integrating skier flight.')
        start_time = time.time()

        # integrate to find the final time point
        sol = solve_ivp(rhs,
                        (0.0, self.max_flight_time),
                        init_pos + init_vel,
                        events=(touch_surface, ),
                        rtol=1e-6, atol=1e-9)

        if isclose(sol.t[-1], self.max_flight_time):
            msg = ('Flying skier did not contact ground within {:1.3f} '
                   'seconds, integration aborted.')
            raise InvalidJumpError(msg.format(self.max_flight_time))

        msg = 'Flight integration terminated at {:1.3f} s'
        logging_call(msg.format(sol.t[-1]))
        msg = 'Flight contact event occurred at {:1.3f} s'
        logging_call(msg.format(float(sol.t_events[0])))

        logging.debug(sol.t[-1] - sol.t_events[0])
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(sol.t[-1], sol.y[:, -1]))

        te = sol.t_events[0]

        if fine:
            # integrate at desired resolution
            times = np.linspace(0.0, sol.t[-1],
                                num=int(self.samples_per_sec * sol.t[-1]))
            sol = solve_ivp(rhs, (0.0, sol.t[-1]), init_pos + init_vel,
                            t_eval=times, rtol=1e-6, atol=1e-9)

        msg = 'Flight integration finished in {:1.3f} seconds.'
        logging_call(msg.format(time.time() - start_time))

        logging.debug(sol.t[-1])
        logging.debug(sol.t[-1] - te)
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(sol.t[-1], sol.y[:, -1]))

        return Trajectory(sol.t, sol.y[:2].T, vel=sol.y[2:].T)

    def slide_on(self, surface, init_speed=0.0, fine=True):
        """Returns the trajectory of the skier sliding over a surface.

        Parameters
        ==========
        surface : Surface
            A surface that the skier will slide on.
        init_speed : float
            The magnitude of the velocity of the skier at the start of the
            surface which is directed tangent to the surface.
        fine : boolean
            If True two integrations occur. The first finds the exit time with
            coarse time steps and the second integrates over a finer equally
            spaced time steps. False will skip the second integration.

        Returns
        =======
        times : ndarray(n,)
            The time values from the simulation.
        states : ndarray(n, 4)
            The [x, y, vx, vy] states.

        Raises
        ======
        InvalidJumpError if skier can't reach the end of the surface within
        1000 seconds.

        """

        def rhs(t, state):

            x = state[0]  # horizontal position
            v = state[1]  # velocity tangent to slope

            slope = surface.interp_slope(x)
            kurva = surface.interp_curvature(x)

            theta = np.arctan(slope)

            xdot = v * np.cos(theta)
            vdot = -GRAV_ACC * np.sin(theta) + (
                (self.drag_force(v) + self.friction_force(v, slope, kurva)) /
                self.mass)

            return xdot, vdot

        def reach_end(t, state):
            """Returns zero when the skier gets to the end of the approach
            length."""
            return state[0] - surface.x[-1]

        reach_end.terminal = True

        logging.info('Integrating skier sliding.')
        start_time = time.time()

        sol = solve_ivp(rhs,
                        (0.0, 1000.0),  # time span
                        (surface.x[0], init_speed),  # initial conditions
                        events=(reach_end, ))

        if fine:
            times = np.linspace(0.0, sol.t[-1],
                                num=int(self.samples_per_sec * sol.t[-1]))
            sol = solve_ivp(rhs, (0.0, sol.t[-1]), (surface.x[0], init_speed),
                            t_eval=times)

        msg = 'Sliding integration finished in {} seconds.'
        logging.info(msg.format(time.time() - start_time))

        logging.info('Skier slid for {} seconds.'.format(sol.t[-1]))

        if np.any(sol.y[1] < 0.0):  # if tangential velocity is ever negative
            msg = ('Skier does not have a high enough velocity to make it to '
                   'the end of the surface.')
            raise InvalidJumpError(msg)

        return sol.t, sol.y

    def end_speed_on(self, surface, **kwargs):
        """Returns the ending speed after sliding on the provided surface.
        Keyword args are passed to Skier.slide_on()."""

        _, traj = self.slide_on(surface, **kwargs)

        return traj[1, -1]

    def end_vel_on(self, surface, **kwargs):
        """Returns the ending velocity (vx, vy) after sliding on the provided
        surface.  Keyword args are passed to Skier.slide_on()."""

        _, traj = self.slide_on(surface, **kwargs)
        end_angle = np.tan(surface.slope[-1])
        speed_x = traj[1, -1] * np.cos(end_angle)
        speed_y = traj[1, -1] * np.sin(end_angle)
        return speed_x, speed_y

    def speed_to_land_at(self, landing_point, takeoff_point, takeoff_angle,
                         surf=None):
        """Returns the magnitude of the velocity required to land at a specific
        point given launch position and angle.

        Parameters
        ==========
        landing_point : 2-tuple of floats
            The (x, y) coordinates of the desired landing point in meters.
        takeoff_point : 2-tuple of floats
            The (x, y) coordinates of the takeoff point in meters.
        takeoff_angle : float
            The takeoff angle in radians.
        surf : Surface
            This should most likely be the parent slope but needs to be
            something that ensures the skier flys past the landing point.

        Returns
        =======
        takeoff_speed : float
            The magnitude of the takeoff velocity.

        Notes
        =====
        This method corresponds to Mont's Matlab function findVoWithDrag.m.

        """

        # NOTE : This may only work if the landing surface is below the takeoff
        # point.

        # TODO : Is it possible to solve a boundary value problem here instead
        # using this iterative approach with an initial value problem?

        x, y = landing_point

        if isclose(landing_point[0] - takeoff_point[0], 0.0):
            return 0.0, (0.0, 0.0)

        theta = takeoff_angle
        cto = np.cos(takeoff_angle)
        sto = np.sin(takeoff_angle)
        tto = np.tan(takeoff_angle)

        # guess init. velocity for impact at x,y based on explicit solution
        # for the no drag case
        delx = landing_point[0] - takeoff_point[0]
        dely = landing_point[1] - takeoff_point[1]
        logging.debug('delx = {}, dely = {}'.format(delx, dely))
        logging.debug(delx**2 * GRAV_ACC / (2*cto**2 * (delx*tto - dely)))
        logging.debug(2*cto**2 * (delx*tto - dely))
        vo = np.sqrt(delx**2 * GRAV_ACC / (2*cto**2 * (delx*tto - dely)))
        logging.debug('vo = {}'.format(vo))
        # dvody is calculated from the explicit solution without drag @ (x,y)
        dvody = ((delx**2 * GRAV_ACC / 2 / cto**2)**0.5 *
                 ((delx*tto-dely)**(-3/2)) / 2)
        # TODO : This gets a negative under the sqrt for some cases, e.g.
        # make_jump(-10.0, 0.0, 30.0, 20.0, 0.1)
        dvody = (np.sqrt(GRAV_ACC*(delx)**2/((delx)*np.sin(2*theta) -
                                             2*(dely)*cto**2))*cto**2 /
                 ((delx)*np.sin(2*theta) - 2*(dely)*cto**2))

        # TODO : Make this should take in the parent slope and use it.
        # creates a flat landing surface that starts at the landing point x
        # position and 1 meter below the y position, this ensures we get a
        # flight trajectory that passes through a horizontal line through the
        # landing position
        if surf is None:
            #surf = FlatSurface(np.deg2rad(45.0), 10.0, init_pos=(x + 6, y),
                            #num_points=100)
            surf = FlatSurface(np.deg2rad(-20.0), 40.0)

        deltay = np.inf

        #ax = surf.plot()

        while abs(deltay) > 0.001:
            vox = vo*cto
            voy = vo*sto

            flight_traj = self.fly_to(surf, init_pos=takeoff_point,
                                      init_vel=(vox, voy),
                                      logging_type='debug')

            #ax.plot(*flight_traj[:2])

            traj_at_impact = flight_traj.interp_wrt_x(x)

            ypred = traj_at_impact[2]
            logging.debug('ypred = {}'.format(ypred))

            deltay = ypred - y
            logging.debug('deltay = {}'.format(deltay))
            dvo = -deltay * dvody
            logging.debug('dvo = {}'.format(dvo))
            vo = vo + dvo
            logging.debug('vo = {}'.format(vo))

        #ax.plot(*takeoff_point, 'o')
        #ax.plot(*landing_point, 'o')
        #ax.set_xlim((10.0, 35.0))
        #plt.show()

        # the takeoff velocity is adjsted by dvo before the while loop ends
        vo = vo - dvo

        takeoff_speed = vo

        impact_vel = (traj_at_impact[3], traj_at_impact[4])

        return takeoff_speed, impact_vel