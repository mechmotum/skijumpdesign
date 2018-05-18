=========================================================
Ski Jump Design Tool For Specified Equivalent Fall Height
=========================================================

================= ========
Launch App        |heroku|
PyPi Download     |pypi|
Anaconda Download |conda|
Documentation     |rtd|
Conda Forge       |forge|
Automated Tests   |ci|
================= ========

Introduction
============

A ski jump design tool for equivalent fall height based on the work presented
in [1]_. Includes a library for 2D skiing simulations and a graphical web
application for designing ski jumps. It is written in Python backed by NumPy,
SciPy, SymPy, Cython, matplotlib, fastcache, pycvodes, Plotly, and Dash.

The design tool web application can be accessed via our Heroku deployment at:

https://skijumpdesign.herokuapp.com/

License
=======

The skijumpdesign source code is released under the MIT license. If you make
use of the software we ask that you cite the relevant papers and the software
itself. See the included ``LICENSE`` file for details.

Installation
============

See ``docs/install.rst`` or http://skijumpdesign.readthedocs.io/en/latest/install.html.

References
==========

.. [1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A
   Design Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall
   Height." Sports Engineering 18, no. 4 (December 2015): 227–39.
   https://doi.org/10.1007/s12283-015-0182-6.

.. |pypi| image:: https://badge.fury.io/py/skijumpdesign.svg
   :target: https://badge.fury.io/py/skijumpdesign

.. |conda| image:: https://anaconda.org/conda-forge/skijumpdesign/badges/version.svg
   :target: https://anaconda.org/conda-forge/skijumpdesign

.. |heroku| image:: http://heroku-badge.herokuapp.com/?app=skijumpdesign&svg=1
   :target: https://skijumpdesign.herokuapp.com/
   :alt: Heroku Application

.. |rtd| image:: https://readthedocs.org/projects/skijumpdesign/badge/?version=latest
   :target: http://skijumpdesign.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |forge| image:: https://img.shields.io/conda/vn/conda-forge/skijumpdesign.svg
   :target: https://github.com/conda-forge/skijumpdesign-feedstock

.. |ci| image:: https://gitlab.com/moorepants/skijumpdesign/badges/master/pipeline.svg
   :target: https://gitlab.com/moorepants/skijumpdesign/commits/master
   :alt: pipeline status
