.. _install:

============
Installation
============

conda
=====

The library can be installed into the root conda environment from the `Conda
Forge channel`_ at Anaconda.org::

   $ conda install -c conda-forge skijumpdesign

.. _Conda Forge channel: https://anaconda.org/conda-forge/

pip
===

The library can be installed from PyPi using pip [1]_::

   $ pip install skijumpdesign

If you want to use the web application install both the library and web
application with::

   $ pip install skijumpdesign[web]

If you want to run the unit tests use::

   $ pip install skijumpdesign[dev]

setuptools
==========

Download and unpack the source code to a local directory, e.g.
``/path/to/skijumpdesign``.

Open a terminal. Navigate to the ``skijumpdesign`` directory::

   $ cd /path/to/skijumpdesign

Install with [1]_::

   $ python setup.py install

Development Installation
========================

Clone the repository with git::

   git clone https://gitlab.com/moorepants/skijumpdesign

Navigate to the cloned ``skijumpdesign`` repository::

   $ cd skijumpdesign/

Setup the custom development conda environment named ``skijumpdesign`` to
ensure it has all of the correct software dependencies. To create the
environment type::

   $ conda env create -f environment.yml

To activate the environment type [2]_::

   $ conda activate skijumpdesign
   (skijumpdesign)$

Heroku Installation
===================

When installing into a Heroku instance, the application will make use of the
``requirements.txt`` file included in the source code which installs all of the
dependencies needed to run the software on a live Heroku instance.

.. [1] Note that you likely want to install into a user directory with
   pip/setuptools. See the pip and setuptools documentation on how to do this.
.. [2] This environment will also show up in the Anaconda Navigator program.