.. _install:

============
Installation
============

skijumpdesign can be installed using several tools. Below are recommended
options, in order of the developers' preference.

conda
=====

The library and web application can be installed into the root conda_
environment from the `Conda Forge channel`_ at anaconda.org. This requires
installing either miniconda_ or Anaconda_ first. Once conda is available run::

   $ conda install -c conda-forge skijumpdesign

The Anaconda Navigator graphical installer can also be used to accomplish the
same result.

.. _conda: http://conda.io
.. _Conda Forge channel: https://anaconda.org/conda-forge
.. _miniconda: https://conda.io/miniconda.html
.. _anaconda: https://www.anaconda.com/download

pip
===

The library and web application can be installed from PyPi_ using pip_ [1]_::

   $ pip install skijumpdesign

If you want to run the unit tests and/or build the documentation use::

   $ pip install skijumpdesign[dev]

to also install the development dependencies.

.. _PyPi: http://pypi.org
.. _pip: http://pip.pypa.io

setuptools
==========

Download and unpack the source code to a local directory, e.g.
``/path/to/skijumpdesign``.

Open a terminal. Navigate to the ``skijumpdesign`` directory::

   $ cd /path/to/skijumpdesign

Install with [1]_::

   $ python setup.py install

.. [1] Note that you likely want to install into a user directory with
   pip/setuptools. See the pip and setuptools documentation on how to do this.

Optional dependencies
=====================

If pycvodes_ is installed it will be used to speed up the flight simulation and
the landing surface calculation significantly. This library is not trivial to
install on all operating systems, so you will need to refer its documentation
for installation instructions. If you are using conda Linux or OSX, this
package can be installed using conda with::

   $ conda install -c conda-forge pycvodes

.. _pycvodes: https://github.com/bjodah/pycvodes

Development Installation
========================

Clone the repository with git::

   $ git clone https://gitlab.com/moorepants/skijumpdesign

Navigate to the cloned ``skijumpdesign`` repository::

   $ cd skijumpdesign/

Setup the custom development conda environment named ``skijumpdesign`` to
ensure it has all of the correct software dependencies. To create the
environment type::

   $ conda env create -f conda/skijumpdesign-lib-dev.yml

To activate the environment type [2]_::

   $ conda activate skijumpdesign-lib-dev
   (skijumpdesign-lib-dev)$

Optionally, install in development mode using setuptools for use from any
directory::

   (skijumpdesign-lib-dev)$ python setup.py develop

There are several conda environment files provided in the source code that may
be of use:

- ``skijumpdesign-app.yml``: Installs the versions of the required dependencies
  to run the library and the web app pinned to specific versions for the app.
  These are the versions we use to run the official web app.
- ``skijumpdesign-app-opt.yml``: Installs the versions of the required and
  optional dependencies to run the library and the web app pinned to specific
  versions for the app. These are the versions we use to run the official web
  app.
- ``skijumpdesign-app-dev.yml``: Installs the versions of the required
  dependencies to run the library and the web app pinned to specific versions
  for the app plus tools for development. These are the versions we use to run
  the official web app.
- ``skijumpdesign-app-opt-dev.yml``: Installs the versions of the required and
  optional dependencies to run the library and the web app pinned to specific
  versions for the app plus tools for development. These are the versions we
  use to run the official web app.
- ``skijumpdesign-lib.yml``: Installs the latest version of the required
  dependencies to run the library and the web app.
- ``skijumpdesign-lib-opt.yml``: Installs the latest version of the required
  and optional dependencies to run the library and the web app.
- ``skijumpdesign-lib-dev.yml``: Installs the latest version of the required
  dependencies to run the library and the web app, test the code, and build the
  documentation.
- ``skijumpdesign-lib-opt-dev.yml``: Installs the latest version of the
  required and optional dependencies to run the library and the web app, test
  the code, and build the documentation.

.. [2] This environment will also show up in the Anaconda Navigator program.

Heroku Installation
===================

When installing into a Heroku instance, the application will make use of the
``requirements.txt`` file included in the source code which installs all of the
dependencies needed to run the software on a live Heroku instance. You need to
set some environment variables for the Heroku app:

- ``ONHEROKU=true``: Lets the app know if it is running on Heroku.
- ``GATRACKINGID``: Set the value as a string with your Google Analytics
  tracking id.
