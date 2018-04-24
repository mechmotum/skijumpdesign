.. image:: https://readthedocs.org/projects/skijumpdesign/badge/?version=latest
   :target: http://skijumpdesign.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Introduction
============

A ski jump design tool for equivalent fall height based on the work presented
in [1]_. Includes a library for 2D skiing simulations and a graphical web
application for designing ski jumps. It is written in Python backed by NumPy,
SciPy, SymPy, Cython, matplotlib, Plotly, and Dash.

License
=======

The skijumpdesign source code is released under the MIT license. If you make
use of the software we ask that you cite the relevant papers or the software
itself.

Installation
============

Download and unpack the source code or git clone to a local directory, e.g.
``/path/to/skijumpdesign``.

Open a terminal. Navigate to the ``skijumpdesign`` directory::

   $ cd /path/to/skijumpdesign

Setup the custom development Conda environment to ensure it has all of the
correct software dependencies. To create the environment type::

   $ conda env create -f environment.yml

This environment will also show up in the Anaconda Navigator program.

Running the Web Application
===========================

In Spyder
---------

Open Anaconda Navigator, switch to the ``skijumpdesign`` environment, and then
launch Spyder.

Set the working directory to the ``skijumpdesign`` directory.

In the Spyder IPython console execute::

   In [1]: run dash_app.py

If successful, you will see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.

To shutdown the web app, close the tab in your web browser. Go back to Spyder
and execute ``<CTRL>+C`` to shutdown the web server.

In a terminal
-------------

Navigate to the ``skijumpdesign`` directory::

   $ cd /path/to/skijumpdesign

Activate the custom Conda environment with::

   $ source activate skijumpdesign

Now run the application with::

   (skijumpdesign)$ python dash_app.py

You should see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.
Type ``<CTRL>+C`` in the terminal to shutdown the web server.

Using the Library
=================

In Spyder
---------

Open Anaconda Navigator, switch to the ``skijumpdesign`` environment, and then
launch Spyder.

In Spyder, set the working directory to the ``skijumpdesign`` directory.

In the IPython console execute::

   In [1]: from skijumpdesign import *

This will import all of the main functions and classes in the library.  For
example you can now use the ``make_jump()`` function::

   In [2]: surfaces = make_jump(-30.0, 0.0, 60.0, 10.0, 1.0, plot=True)

References
==========

.. [1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. “A
   Design Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall
   Height.” Sports Engineering 18, no. 4 (December 2015): 227–39.
   https://doi.org/10.1007/s12283-015-0182-6.

