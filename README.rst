.. image:: http://heroku-badge.herokuapp.com/?app=skijumpdesign&svg=1
   :target: https://skijumpdesign.herokuapp.com/
   :alt: Heroku Application

.. image:: https://readthedocs.org/projects/skijumpdesign/badge/?version=latest
   :target: http://skijumpdesign.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://gitlab.com/moorepants/skijumpdesign/badges/master/pipeline.svg
   :target: https://gitlab.com/moorepants/skijumpdesign/commits/master
   :alt: pipeline status

Introduction
============

A ski jump design tool for equivalent fall height based on the work presented
in [1]_. Includes a library for 2D skiing simulations and a graphical web
application for designing ski jumps. It is written in Python backed by NumPy,
SciPy, SymPy, Cython, matplotlib, Plotly, and Dash.

The design tool web application can be accessed via our Heroku deployment at:

https://skijumpdesign.herokuapp.com/

License
=======

The skijumpdesign source code is released under the MIT license. If you make
use of the software we ask that you cite the relevant papers or the software
itself. See the included ``LICENSE`` file for details.

Installation
============

See ``docs/install.rst`` or http://skijumpdesign.readthedocs.io/en/latest/install.html.

References
==========

.. [1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. “A
   Design Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall
   Height.” Sports Engineering 18, no. 4 (December 2015): 227–39.
   https://doi.org/10.1007/s12283-015-0182-6.
