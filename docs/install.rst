.. _install:

============
Installation
============

conda
=====

The library and web application can be installed into the root conda
environment from the `Conda Forge channel`_ at Anaconda.org::

   $ conda install -c conda-forge skijumpdesign

.. _Conda Forge channel: https://anaconda.org/conda-forge/

pip
===

The library and web application can be installed from PyPi using pip [1]_::

   $ pip install skijumpdesign

If you want to run the unit tests and/or build the documentation use::

   $ pip install skijumpdesign[dev]

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
for installation instructions. If you are using conda and 64 bit Linux, this
package can be installed using::

   $ conda install -c conda-forge -c bjodah pycvodes

.. _pycvodes: https://github.com/bjodah/pycvodes

Development Installation
========================

Clone the repository with git::

   git clone https://gitlab.com/moorepants/skijumpdesign

Navigate to the cloned ``skijumpdesign`` repository::

   $ cd skijumpdesign/

Setup the custom development conda environment named ``skijumpdesign`` to
ensure it has all of the correct software dependencies. To create the
environment type::

   $ conda env create -f environment-dev.yml

To activate the environment type [2]_::

   $ conda activate skijumpdesign-dev
   (skijumpdesign-dev)$

Heroku Installation
===================

When installing into a Heroku instance, the application will make use of the
``requirements.txt`` file included in the source code which installs all of the
dependencies needed to run the software on a live Heroku instance.

.. [1] Note that you likely want to install into a user directory with
   pip/setuptools. See the pip and setuptools documentation on how to do this.
.. [2] This environment will also show up in the Anaconda Navigator program.
