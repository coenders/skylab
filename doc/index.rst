.. Coenders documentation master file, created by
   sphinx-quickstart on Mon Jul  7 04:59:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SkyLab - Python based Likelihood analysis in PointSource Searches
=================================================================

.. contents::

.. toctree::
   :maxdepth: 2

   tutorial
   installation
   method
   sensitivity
   modules/modules

SkyLab is a python based tool to perform unbinned Likelihood minimisations
on data similar to the one used at IceCube Point Source searches. The code
relies heavily on NumPy [#numpy]_ , SciPy [#scipy]_ , and HealPy [#healpy]_ .
The SkyLab package is divided into sub-packages each with different purposes:

psLLH.py
    Core of the Likelihood-Calculation. Data handling, minimisation of the
    likelihood model and calculation of sensitivities, etc..

ps_injector.py
    Collection of source injectors for the calculation of sensitivities and
    discovery potentials.

ps_model.py
    Collection of llh models, classic (spatial) point source likelihood,
    additional energy weights, source extension, time dependence, etc.

utils.py
    Helper functions used in parts of the code.

.. rubric:: Footnotes

.. [#numpy] www.numpy.org
.. [#scipy] www.scipy.org
.. [#healpy] www.healpy.readthedocs.org

