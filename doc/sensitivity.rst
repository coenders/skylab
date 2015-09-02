.. Coenders documentation master file, created by
   sphinx-quickstart on Mon Jul  7 04:59:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sensitivity Calculation
========================

.. contents::

When estimating the sensitivity of an unbinned likelihood analysis,
we are interested in the number of trials with injected sources
(percentage :math:`\beta`) that are above a threshold :math:`\hat{\mathcal{TS}}`
of a percentage :math:`\alpha` of background scrambles.

.. _twoside_chi2:

Background Estimation
----------------------

The test statistic :eq:`logLambda` is separated into over and under-fluctuations
that can be described individually by :math:`\chi^2` distributions with
:math:`n_{\rm dof}` degrees of freedom. Combining the two distributions, the
percentage of over-fluctuations :math:`\eta` with respect to all trials is important.

Thus, the complete survival probability of background scrambles is described by the
composite distribution

.. math::

    \chi^2\left(\mathcal{TS}\right) =
    \begin{cases}
        \eta\,\chi^2\left(\mathcal{TS};n_{\rm dof}\right)&,\mathcal{TS}\geq 0\\
        \left(1-\eta\right)\,\chi^2\left(\mathcal{TS};n'_{\rm dof}\right)&,\mathcal{TS}<0\\
    \end{cases}


