A ski jump design tool for the web written in Python backed by Dash, NumPy and
SciPy.

Installation
============

Open a terminal. Navigate to the ``safe-ski-jump`` directory::

   $ cd /path/to/safe-ski-jump

The application must be run in a custom Conda environment to ensure it has all
of the correct software dependencies. To create the environment type::

   $ conda env create -f environment.yml

This environment will also show up in the Anaconda Navigator program.

Running the Web Application
===========================

In Spyder
---------

Open Anaconda Navigator, switch to the ``safeskijump`` environment, and then
launch Spyder.

Set the working directory to the ``safe-ski-jump`` directory.

In the IPython console execute::

   In [1]: run dash_app.py

If successful, you will see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.

To shutdown the web app, close the tab in your web browser. Go back to Spyder
and type and execute ``<CTRL>+C`` to shutdown the web server.

In the terminal
---------------

Navigate to the ``safe-ski-jump`` directory::

   $ cd /path/to/safe-ski-jump

Activate the custom Conda environment with::

   $ source activate safeskijump

Now run the application with::

   (safeskijump)$ python dash_app.py

You should see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.
Type ``<CTRL>+C`` in the terminal to shutdown the web server.

Running the functions
=====================

In Spyder
---------

Open Anaconda Navigator, switch to the ``safeskijump`` environment, and then
launch Spyder.

In Spyder, set the working directory to the ``safe-ski-jump`` directory.

In the IPython console execute::

   In [1]: from safeskijump.functions import *

This will import all of the functions defined in ``safeskijump/functions.py``.
For example you can now use the ``compute_approach_exit_speed()`` function::

   In [2]: compute_approach_exit_speed(10.0, 5.0, 75.0)
   Out[2]: 13.739985254238626

To run the test functions::

   In [3]: from safeskijump.tests.test_functions import *

This will import all of the functions defined in
``safeskijump/tests/test_functions.py``.  For example you can now use the
``test_all()`` function::

   In [4]: test_all(plot=True)
