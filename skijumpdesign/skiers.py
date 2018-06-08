import time
import logging
from math import isclose

import numpy as np
from scipy.integrate import solve_ivp
try:
    import pycvodes
except ImportError:
    pycvodes = None
else:
    from pycvodes import integrate_adaptive, integrate_predefined

from .trajectories import Trajectory
from .utils import GRAV_ACC, AIR_DENSITY
from .utils import InvalidJumpError
from .utils import compute_drag


class Skier(object):
    """Class that represents a two dimensional skier who can slide on surfaces
    and fly in the air."""

    # All trajectories are resampled at this rate.
    samples_per_sec = 360  # Hz
    # If the skier flies too long the integration will be stopped.
    max_flight_time = 30.0  # seconds

    def __init__(self, mass=75.0, area=0.34, drag_coeff=0.821,
                 friction_coeff=0.03, tolerable_sliding_acc=1.5,
                 tolerable_landing_acc=3.0):
        """Instantiates a skier with default properties.

        Parameters
        ==========
        mass : float, optional
            The mass of the skier in kilograms.
        area : float, optional
            The frontal area of the skier in squared meters.
        drag_coeff : float, optional
            The air drag coefficient of the skier.
        friction_coeff : float, optional
            The sliding friction coefficient between the skis and the slope.
        tolerable_sliding_acc : float, optional
            The maximum normal acceleration in G's that a skier can withstand
            while sliding.
        tolerable_landing_acc : float, optional
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
        """Returns the drag force in Newtons opposing the speed in meters per
        second of the skier."""

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
        slope : float, optional
            The slope of the surface at the skier's point of contact.
        curvature : float, optional
            The curvature of the surface at the skier's point of contact.

        """

        theta = np.tan(slope)

        normal_force = self.mass * (GRAV_ACC * np.cos(theta) + curvature *
                                    speed**2)

        return -np.sign(speed) * self.friction_coeff * normal_force

    def _flight_rhs(self, t, state):
        """Returns the time derivative of the skier's state during flight.

        Parameters
        ==========
        t : float
            The value of time in seconds.
        state : array_like, shape(4,)
            The values of the states: [x, y, vx, vy].

        Returns
        =======
        4-tuple of floats
            The values of the derivatives of the states.

        """

        xdot = state[2]
        ydot = state[3]
        # TODO : sympy autowrap these entire expressions instead of just
        # drag_force().
        vxdot = self.drag_force(xdot) / self.mass
        vydot = -GRAV_ACC + self.drag_force(ydot) / self.mass

        return xdot, ydot, vxdot, vydot

    def _flight_rhs_sundials(self, t, s, dsdt):
        """Populates the time derivative of the skier's state during flight.

        Parameters
        ==========
        t : float
            The value of time in seconds.
        s : array_like, shape(4,)
            The values of the states: [x, y, vx, vy].
        dsdt : array_like, shape(4,)
            The values of the time derivatives of the states.

        """

        xdot = s[2]
        ydot = s[3]
        vxdot = self.drag_force(xdot) / self.mass
        vydot = -GRAV_ACC + self.drag_force(ydot) / self.mass

        dsdt[0] = xdot
        dsdt[1] = ydot
        dsdt[2] = vxdot
        dsdt[3] = vydot

    def fly_to(self, surface, init_pos, init_vel, fine=True, compute_acc=True,
               logging_type='info'):
        """Returns the flight trajectory of the skier given the initial
        conditions and a surface which the skier contacts at the end of the
        flight trajectory.

        Parameters
        ==========
        surface : Surface
            A landing surface. This surface must intersect the flight path.
        init_pos : 2-tuple of floats
            The x and y coordinates of the starting point of the flight in
            meters.
        init_vel : 2-tuple of floats
            The x and y components of the skier's velocity at the start of the
            flight in meters per second.
        fine : boolean
            If True two integrations occur. The first finds the landing time
            with coarse time steps and the second integrates over a finer
            equally spaced time steps. False will skip the second integration.
        compute_acc : boolean, optional
            If true acceleration will be calculated. If false acceleration is
            set to zero.
        logging_type : string
            The logging level desired for the non-debug logging calls in this
            function. Useful for suppressing too much information since this
            runs a lot.

        Returns
        =======
        trajectory : Trajectory
            A trajectory instance that contains the time, position, velocity,
            acceleration, speed, and slope of the flight.

        Raises
        ======
        InvalidJumpError if the skier does not contact a surface within
        Skier.max_flight_time.

        """
        logging_call = getattr(logging, logging_type)

        if pycvodes is not None:
            logging_call('Using pycvodes for flight integration.')
            return self._fly_to_sundials(surface, init_pos, init_vel,
                                         fine=fine, compute_acc=compute_acc,
                                         logging_type=logging_type)
        else:
            logging_call('Using scipy for flight integration.')
            return self._fly_to_scipy(surface, init_pos, init_vel, fine=fine,
                                      compute_acc=compute_acc,
                                      logging_type=logging_type)

    def _fly_to_scipy(self, surface, init_pos, init_vel, fine=True,
                      compute_acc=True, logging_type='info'):

        def touch_surface(t, state):

            x = state[0]
            y = state[1]

            return surface.distance_from(x, y)

        touch_surface.terminal = True
        # NOTE: always from above surface, positive to negative crossing
        touch_surface.direction = -1

        logging_call = getattr(logging, logging_type)

        logging_call('Integrating skier flight.')
        start_time = time.time()

        # integrate to find the impact time
        # NOTE : For a more accurate event time, the error tolerances on the
        # states need to be lower.
        sol = solve_ivp(self._flight_rhs,
                        (0.0, self.max_flight_time),
                        init_pos + init_vel,
                        events=(touch_surface, ),
                        rtol=1e-6, atol=1e-9)

        impact_time = sol.t[-1]

        te = sol.t_events[0]

        if (isclose(impact_time, self.max_flight_time) or impact_time >
                self.max_flight_time):
            msg = ('Flying skier did not contact ground within {:1.3f} '
                   'seconds, integration aborted.')
            raise InvalidJumpError(msg.format(self.max_flight_time))

        msg = 'Flight integration terminated at {:1.3f} s'
        logging_call(msg.format(impact_time))
        msg = 'Flight impact event occurred at {:1.3f} s'
        logging_call(msg.format(float(te)))

        logging.debug(impact_time)
        logging.debug(impact_time - te)
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(impact_time, sol.y[:, -1]))

        if fine:  # integrate at desired resolution
            times = np.linspace(0.0, impact_time,
                                num=int(self.samples_per_sec * impact_time))
            sol = solve_ivp(self._flight_rhs,
                            (0.0, impact_time),
                            init_pos + init_vel,
                            t_eval=times,
                            rtol=1e-6,
                            atol=1e-9)

        msg = 'Flight integration finished in {:1.3f} seconds.'
        logging_call(msg.format(time.time() - start_time))

        impact_time = sol.t[-1]

        logging.debug(impact_time)
        logging.debug(impact_time - te)
        logging.debug(sol.y[:, -1])
        logging.debug(touch_surface(impact_time, sol.y[:, -1]))

        # NOTE : This prevents Trajectory from running the acceleration
        # gradient if not needed.
        if compute_acc:
            acc = None
        else:
            acc = np.zeros_like(sol.y[:2].T)

        return Trajectory(sol.t, sol.y[:2].T, vel=sol.y[2:].T, acc=acc)

    def _fly_to_sundials(self, surface, init_pos, init_vel, fine=True,
                         compute_acc=True, logging_type='info'):

        def touch_surface_sundials(t, state, out):
            x = state[0]
            y = state[1]
            out[0] = surface.distance_from(x, y)

        logging_call = getattr(logging, logging_type)

        logging_call('Integrating skier flight.')
        start_time = time.time()

        times, states, info = integrate_adaptive(rhs=self._flight_rhs_sundials,
                                                 jac=None,
                                                 y0=init_pos + init_vel,
                                                 x0=0.0,
                                                 xend=self.max_flight_time,
                                                 atol=1e-9,
                                                 rtol=1e-6,
                                                 roots=touch_surface_sundials,
                                                 nroots=1,
                                                 return_on_root=True)
        impact_time = times[-1]

        if (isclose(impact_time, self.max_flight_time) or impact_time >
                self.max_flight_time):
            msg = ('Flying skier did not contact ground within {:1.3f} '
                   'seconds, integration aborted.')
            raise InvalidJumpError(msg.format(self.max_flight_time))

        msg = 'Flight integration terminated at {:1.3f} s'
        logging_call(msg.format(impact_time))
        msg = 'Flight impact event occurred at {:1.3f} s'
        #logging_call(msg.format(float(te)))

        if fine:  # integrate at desired resolution
            times = np.linspace(0.0, impact_time,
                                num=int(self.samples_per_sec * impact_time))
            states, info = integrate_predefined(self._flight_rhs_sundials,
                                                None,
                                                init_pos + init_vel,
                                                times,
                                                1e-9,
                                                1e-6,
                                                1e-8)

        msg = 'Flight integration finished in {:1.3f} seconds.'
        logging_call(msg.format(time.time() - start_time))

        # NOTE : This prevents Trajectory from running the acceleration
        # gradient if not needed.
        if compute_acc:
            acc = None
        else:
            acc = np.zeros_like(states[:, :2])

        return Trajectory(times, states[:, :2], vel=states[:, 2:], acc=acc)

    def slide_on(self, surface, init_speed=0.0, fine=True):
        """Returns the trajectory of the skier sliding over a surface.

        Parameters
        ==========
        surface : Surface
            A surface that the skier will slide on.
        init_speed : float, optional
            The magnitude of the velocity of the skier at the start of the
            surface which is directed tangent to the surface.
        fine : boolean
            If True two integrations occur. The first finds the exit time with
            coarse time steps and the second integrates over a finer equally
            spaced time steps. False will skip the second integration.

        Returns
        =======
        trajectory : Trajectory
            A trajectory instance that contains the time, position, velocity,
            acceleration, speed, and slope of the slide,

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

        y = surface.interp_y(sol.y[0])
        slope = surface.interp_slope(sol.y[0])
        angle = np.arctan(slope)
        vx = sol.y[1] * np.cos(angle)
        vy = sol.y[1] * np.sin(angle)

        return Trajectory(sol.t, np.vstack((sol.y[0], y)).T,
                          vel=np.vstack((vx, vy)).T, speed=sol.y[1])

    def end_speed_on(self, surface, **kwargs):
        """Returns the ending speed after sliding on the provided surface.
        Keyword args are passed to Skier.slide_on()."""

        traj = self.slide_on(surface, **kwargs)

        return traj.speed[-1]

    def end_vel_on(self, surface, **kwargs):
        """Returns the ending velocity (vx, vy) after sliding on the provided
        surface. Keyword args are passed to Skier.slide_on()."""

        traj = self.slide_on(surface, **kwargs)
        return tuple(traj.vel[-1])

    def speed_to_land_at(self, landing_point, takeoff_point, takeoff_angle,
                         surf):
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
            something that ensures the skier flies past the landing point.

        Returns
        =======
        takeoff_speed : float
            The magnitude of the takeoff velocity.

        """

        # NOTE : This method corresponds to Mont's Matlab function
        # findVoWithDrag.m.

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

        deltay = np.inf

        while abs(deltay) > 0.001:
            vox = vo*cto
            voy = vo*sto

            flight_traj = self.fly_to(surf, init_pos=takeoff_point,
                                      init_vel=(vox, voy),
                                      compute_acc=False,
                                      logging_type='debug')

            traj_at_impact = flight_traj.interp_wrt_x(x)

            ypred = traj_at_impact[2]
            logging.debug('ypred = {}'.format(ypred))

            deltay = ypred - y
            logging.debug('deltay = {}'.format(deltay))
            dvo = -deltay * dvody
            logging.debug('dvo = {}'.format(dvo))
            vo = vo + dvo
            logging.debug('vo = {}'.format(vo))

        # the takeoff velocity is adjsted by dvo before the while loop ends
        vo = vo - dvo

        takeoff_speed = vo

        impact_vel = (traj_at_impact[3], traj_at_impact[4])

        return takeoff_speed, impact_vel
