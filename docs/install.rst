.. _install:

============
Installation
============

skijumpdesign can be installed via several tools. Below are recommended
options, in order of the developers' preference.

conda
=====

The library and web application can be installed into the root conda_
environment from the `Conda Forge channel`_ at anaconda.org. This requires
installing either miniconda_ or Anaconda_::

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

   $ conda env create -f environment-dev.yml

To activate the environment type [2]_::

   $ conda activate skijumpdesign-dev
   (skijumpdesign-dev)$

Optionally, install in development mode using setuptools for use from any
directory::

   (skijumpdesign-dev)$ python setup.py develop

Heroku Installation
===================

When installing into a Heroku instance, the application will make use of the
``requirements.txt`` file included in the source code which installs all of the
dependencies needed to run the software on a live Heroku instance. Note that
this currently only runs on the deprecated cedar-14 stack.

.. [1] Note that you likely want to install into a user directory with
   pip/setuptools. See the pip and setuptools documentation on how to do this.
.. [2] This environment will also show up in the Anaconda Navigator program.
