=========================================================
Ski Jump Design Tool For Specified Equivalent Fall Height
=========================================================

================= ========
Launch App        |heroku|
JOSS Paper        |joss|
PyPi Download     |pypi|
Anaconda Download |conda|
Documentation     |rtd|
Conda Forge       |forge|
Automated Tests   |ci|
================= ========

Introduction
============

A ski jump design tool for specified equivalent fall height based on the work
presented in [1]_. Includes a library for 2D skiing simulations and a graphical
web application for designing ski jumps. It is written in Python backed by
NumPy, SciPy, SymPy, Cython, matplotlib, pycvodes, Plotly, and Dash.

The design tool web application can be accessed at http://www.skijumpdesign.info.

License
=======

The skijumpdesign source code is released under the MIT license. If you make
use of the software we ask that you cite this paper along with including the
license:

   Moore, Jason K. and Mont Hubbard, (2018). skijumpdesign: A Ski Jump Design
   Tool for Specified Equivalent Fall Height. Journal of Open Source Software,
   3(28), 818, https://doi.org/10.21105/joss.00818

The gtag.js file is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

See the included ``LICENSE`` file for more details.

Installation
============

See ``docs/install.rst`` or http://skijumpdesign.readthedocs.io/en/stable/install.html.

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
   :target: http://www.skijumpdesign.info
   :alt: Heroku Application

.. |rtd| image:: https://readthedocs.org/projects/skijumpdesign/badge/?version=stable
   :target: http://skijumpdesign.readthedocs.io/en/stable/?badge=stable
   :alt: Documentation Status

.. |forge| image:: https://img.shields.io/conda/vn/conda-forge/skijumpdesign.svg
   :target: https://github.com/conda-forge/skijumpdesign-feedstock

.. |ci| image:: https://gitlab.com/moorepants/skijumpdesign/badges/master/pipeline.svg
   :target: https://gitlab.com/moorepants/skijumpdesign/commits/master
   :alt: pipeline status

.. |joss| image:: http://joss.theoj.org/papers/10.21105/joss.00818/status.svg
   :target: https://doi.org/10.21105/joss.00818
