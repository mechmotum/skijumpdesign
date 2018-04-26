.. skijumpdesign documentation master file, created by
   sphinx-quickstart on Fri Apr 13 15:20:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to skijumpdesign's documentation!
=========================================

This is the documentation for "skijumpdesign: A Ski Jump Design Tool for
Equivalent Fall Height" based on the work presented in [1]_. The software
includes a library for two dimensional skiing simulations and a graphical web
application for designing basic ski jumps.

.. plot::
   :width: 600px

   from skijumpdesign import make_jump

   make_jump(-15.0, 0.0, 40.0, 25.0, 0.5, plot=True)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   web-app.rst
   build-jump.rst
   api.rst

References
==========

.. [1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. “A
   Design Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall
   Height.” Sports Engineering 18, no. 4 (December 2015): 227–39.
   https://doi.org/10.1007/s12283-015-0182-6.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
