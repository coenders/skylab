.. Coenders documentation master file, created by
   sphinx-quickstart on Mon Jul  7 04:59:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation and Requirements
=================================================================

.. contents::

To use skylab simply call

.. code-block:: bash

    pip install [--user] git+http://github.com/coenders/skylab

Make sure that the directory where skylab is installed is contained in your
PYTHONPATH (e.g. ~/.local/lib/python2.7/site-packages)

Requirements
#############

The core of SkyLab works with recent versions of python, numpy, scipy.

- *SciPy* (12.0 and older) gives logging information even in quiet
  mode [#scipy]_ . This is fixed in later versions. This could get annoying
  when fitting thousands of times, while physics / statistics remain unaffected
  by this.
- For using splines in more than 2 dimensionen, *SciPy*>14.0 is needed.

You can install the libraries via pip.

.. code-block:: bash

    pip install --user scipy numpy healpy [...]

Make sure, you set your path so that python finds these libraries.
The default is for LINUX systems:

.. code-block:: bash

    export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

In *docs/examples*, various testing scripts are located that show how to use some of the
features.

.. rubric:: Footnotes

.. [#scipy] `Bug report <http://mail.scipy.org/pipermail/scipy-tickets/2012-October/005693.html>`_


