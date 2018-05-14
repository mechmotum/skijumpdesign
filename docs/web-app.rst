===========================
Running the Web Application
===========================

User
====

After :ref:`installing <install>` skijumpdesign type::

   $ skijumpdesign

to run the web application. This should launch the application on your
computer's port 8050. You can view the application by visiting
``localhost:8050`` in your preferred web browser. ``<CTRL + C>`` will stop the
server.

Developer
=========

The ``bin/skijumpdesign`` entry point is not available unless you install the
software. The following shows how to launch the app from the Python file.

In a terminal
-------------

Navigate to the ``skijumpdesign`` directory on your computer::

   $ cd /path/to/skijumpdesign

Activate the custom Conda environment with::

   $ conda activate skijumpdesign-dev

Now run the application with::

   (skijumpdesign-dev)$ python skijumpdesignapp.py

You should see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.
Type ``<CTRL>+C`` in the terminal to shutdown the web server.

In Spyder
---------

Open Anaconda Navigator, switch to the ``skijumpdesign-dev`` environment, and
then launch Spyder. Set the working directory to the ``/path/to/skijumpdesign``
directory. In the Spyder IPython console execute::

   In [1]: run skijumpdesignapp.py

If successful, you will see something like::

    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Open your web browser and enter the displayed URL to interact with the web app.

To shutdown the web app, close the tab in your web browser. Go back to Spyder
and execute ``<CTRL>+C`` to shutdown the web server.
