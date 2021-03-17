Welcome to skijumpdesign's documentation!
=========================================

This is the documentation for "skijumpdesign: A Ski Jump Design and Analysis
Tool for Equivalent Fall Height" based on the work presented in [2]_. The
software includes a library for two dimensional skiing simulations and a
graphical web application for designing and analyzing basic ski jumps. The
primary purpose of the software is to provide an open source,
layperson-friendly web application for designing and assessing ski jumps for
the purposes of minimizing equivalent fall height (EFH). Ski jumps that are
designed with low equivalent fall heights will likely reduce injuries. See the
references below for a more thorough discussion of the reasons. A current
version of the web application can be accessed at
http://www.skijumpdesign.info.

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
   analyze-jump.rst
   real-jumps.rst
   api.rst

References
==========

The following references provide background information on the theory and
rationale of the software implementation.

A paper on this software implementation:

.. [1] Moore, Jason K. and Mont Hubbard, (2018). skijumpdesign: A Ski Jump
   Design Tool for Specified Equivalent Fall Height. Journal of Open Source
   Software, 3(28), 818, https://doi.org/10.21105/joss.00818

Which is based on this primary reference:

.. [2] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A
   Design Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall
   Height." Sports Engineering 18, no. 4 (December 2015): 227–39.
   https://doi.org/10.1007/s12283-015-0182-6.

The following are also useful for more in-depth study (in chronological order):

.. [3] Hubbard, Mont. "Safer Ski Jump Landing Surface Design Limits Normal
   Impact Velocity." Journal of ASTM International 6, no. 1 (2009): 10.
   https://doi.org/10.1520/STP47480S.
.. [4] McNeil, James A., and James B. McNeil. "Dynamical Analysis of Winter
   Terrain Park Jumps." Sports Engineering 11, no. 3 (June 2009): 159–64.
   https://doi.org/10.1007/s12283-009-0013-8.
.. [5] Swedberg, Andrew Davis. "Safer Ski Jumps: Design of Landing Surfaces and
   Clothoidal in-Run Transitions." Master of Science in Applied Mathematics,
   Naval Postgraduate School, 2010.
.. [6] Hubbard, Mont, and Andrew D. Swedberg. "Design of Terrain Park Jump
   Landing Surfaces for Constant Equivalent Fall Height Is Robust to
   'Uncontrollable' Factors." In Skiing Trauma and Safety: 19th Volume, edited
   by Robert J. Johnson, Jasper E. Shealy, Richard M. Greenwald, and Irving S.
   Scher, 75–94. 100 Barr Harbor Drive, PO Box C700, West Conshohocken, PA
   19428-2959: ASTM International, 2012. https://doi.org/10.1520/STP104515.
.. [7] Swedberg, Andrew D., and Mont Hubbard. "Modeling Terrain Park Jumps:
   Linear Tabletop Geometry May Not Limit Equivalent Fall Height." In Skiing
   Trauma and Safety: 19th Volume, edited by Robert J. Johnson, Jasper E.
   Shealy, Richard M. Greenwald, and Irving S. Scher, 120–35. 100 Barr Harbor
   Drive, PO Box C700, West Conshohocken, PA 19428-2959: ASTM International,
   2012.  https://doi.org/10.1520/STP104335.
.. [8] McNeil, James A., Mont Hubbard, and Andrew D. Swedberg. "Designing
   Tomorrow's Snow Park Jump." Sports Engineering 15, no. 1 (March 2012): 1–20.
   https://doi.org/10.1007/s12283-012-0083-x.
.. [9] Hubbard, Mont, James A. McNeil, Nicola Petrone, and Matteo Cognolato.
   "Impact Performance of Standard Tabletop and Constant Equivalent Fall Height
   Snow Park Jumps." In Skiing Trauma and Safety: 20th Volume, edited by Robert
   J.  Johnson, Jasper E. Shealy, and Richard M. Greenwald, 51–71. 100 Barr
   Harbor Drive, PO Box C700, West Conshohocken, PA 19428-2959: ASTM
   International, 2015. https://doi.org/10.1520/STP158220140027.
.. [10] Petrone, Nicola, Matteo Cognolato, James A. McNeil, and Mont Hubbard.
   "Designing, Building, Measuring, and Testing a Constant Equivalent Fall Height
   Terrain Park Jump." Sports Engineering 20, no. 4 (December 2017): 283–92.
   https://doi.org/10.1007/s12283-017-0253-y.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
