.. Coenders documentation master file, created by
   sphinx-quickstart on Mon Jul  7 04:59:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _skylab:

Likelihood Method and Implementation
=================================================================

.. contents::

Point source searches in IceCube are based on an unbinned Likelihood
approach using information of the incoming direction of every event, plus
additional information as for example the reconstruceted energy. The
unbinned likelihood equation

.. math::
    :label: unbinned_llh

    \mathcal{L}=\prod_i\left(\frac{n_s}{N}\mathcal{S}_i
                           \left(\mathbf{x},\mathbf{x_S};
                                 \sigma,E,\delta,\gamma\right)
                           +\left(1-\frac{n_s}{N}\right)\mathcal{B}_i
                           \left(\delta;E\right)\right)\;,

where :math:`n_s,\mathbf{x},\mathbf{x_S},\sigma,E,\delta,\gamma` denote the
number of associated signal neutrinos, the event and source location,
reconstruction accuracy, energy,
declination, and spectral index of the source, respectively, for each event *i*.
Each event is associated with a probability to be signal- :math:`\mathcal{S}` or
background-like :math:`\mathcal{B}`, dependend on the location, accuracy,
energy, etc, where the signal hypothesis can depend on parameters like the
spectral index :math:`\gamma`. More information can be found at [#ps-paper]_


Likelihood Ratio and Test Statistic
####################################

For estimating the significance of an observation, the likelihood ratio with respect
to a null hypothesis of no observation :eq:`unbinned_llh` at :math:`n_s=0` [#null_hyp]_
is tested. This yields the *test statistic* :math:`\mathcal{TS}`

.. math::
    :label: logLambda

    \log\Lambda
        =&\sum_i\log\left(
                1+\frac{n_s}{N}\left(\frac{\mathcal{S}_i}{\mathcal{B}_i}\mathcal{W}_i-1\right)\right)\\
        \equiv&\sum_i\log\left(1+n_s\mathcal{X}_i\right)\\
    \mathcal{TS}
        =&\mathrm{sgn}\left(n_s\right)\log\Lambda

with separation of over- (:math:`n_s>0`) and under-fluctiations (:math:`n_s<0`). Using this
likelihood ratio, the evaluation reduces into the evaluation of signal- over
background-probabilities, :math:`\mathcal{S},\mathcal{B}`, respectively. In this sense,
the ratio will be split up into a spatial- and time-like part containing information about
space and time separation of source and each event, and additional weightings :math:`\mathcal{W}`
that depend on other parameters, e.g. the reconstruced event energy.

For notation, the likelihood function splits up into the unbinned likelihood part of the
sum of logarithms described by the signal-ness parameter :math:`n_s`, and event weights
:math:`\mathcal{X}_i` that can depend on additional minimisation parameters.

For clustering analysis on the two-dimensional sphere, angular differences can be expressed
with the *von Mises* distribution. However, for small angular uncertainties, boundaries
at :math:`180^\circ` are negligible and the expression reduces to a Gaussian normal
distribution.

.. math::
    :label: gaussian_S

    \mathcal{S}_i=\frac{1}{2\pi\sigma^2_i}{\rm e}^{-\frac{\Delta\Psi^2}{2\sigma_i^2}}

Many of the events in the sample will not be located at the designated search position,
formin an off-source region and havin a very low, close to zero signal probability
:eq:`gaussian_S`. To reduce amount of calculation, only signal-like events are
evaluated in detail, while other events are set to be pure background. Selecting
:math:`N'` events out of a sample of :math:`N` events, the likelihood then
simplifies to

.. math::
    :label: selected_llh

    \log\Lambda=&\log\Lambda^{N'}_\mathcal{S}
        +\left(N-N'\right)\log\left(1-\frac{n_s}{N}\right)

For the background distribution, the probaility to observe an event at direction
:math:`\theta,\phi` is mapped from the data. In the case of IceCube, the effect
of :math:`\phi` is smeared out due to rotation each day and decouples from the
declination. Thus, only the declination distribution is mapped from data.

.. math::
    :label: dist_B

    P_{\rm backg}=\frac{1}{2\pi}P_{\rm exp}\left(\delta\right)

Additional weights :math:`\mathcal{W}` are constructed using experimental
and simulated data to obtain the distributions. Spatial parts are already
used in :math:`\mathcal{S},\mathcal{B}` so the distributions have to be
normalized along the observed direction (declination for IceCube). The signal
part of the weight is done from signal simulation, taking into account model
parameters like the spectral index, cut-offs, etc..


Implementation of the LLH
###########################

.. image:: figures/skylab-structure.png
    :width: 75 %
    :align: center

SkyLab is mainly divided into three important parts of the likelihood construction,
together with helper-functions giving more functionality. The core of the project
covers the calculation and minimisation of the likelihood, the llh-model gives the
statistical treatment of the data assuming a physics model, and the injector adds
signal like data into the sample for estimating sensitivities and discovery
potentials.

This three parts work together in the project doing the analysis of the unbinned
likelihood, talking to each other in a slim interface of a few methods to allow
flexibility in using different models.


Core
-----

The core of the project with the class PointSourceLikelihood is the heart of
the unbinned point source likelihood, hosting all the data, selecting the interesting
events for source location(s) and knowing about the parameters needed in the fit.

Minimisation and gradients
***************************

Minimisation is done here, using the L-BFGS-B algorithm [#BFGS]_ implemented
in SciPy using gradients of the likelihood function. One parameter always used
in the fitting is the signalness parameter :math:`n_s`, additional parameteres
will be defined in the likelihood model. Following :eq:`logLambda` the gradients
for :math:`n_s` and the additional parameters :math:`\sigma_j` are calculated as

.. math::
    :label: gradients

    \frac{\partial\log\Lambda}{\partial n_s}=&
        \sum_i\frac{\mathcal{X}_i}{1+n_s\mathcal{X}_i}
        -\frac{N-N'}{N-n_s}\\
    \frac{\partial\log\Lambda}{\partial \sigma_j}=&
        \sum_i\frac{n_s}{1+n_s\mathcal{X}_i}\frac{\partial \mathcal{X}_i}{\partial \sigma_j}

where the gradient in the weights is calculated by the llh-model class.

Under-Fluctuations
*******************

From the likelihood description :eq:`unbinned_llh`, negative values for
:math:`n_s` are possible and can be interpreted as under-fluctuations of the data.
This is interesting for estimating the underlying background statistics in the region
of the null-hypothesis, and setting harder limits when observing underflutuations.

However, the minimiser doesn't know about logrithms being part of the likelihood function
and the resulting divergence, if

.. math::

    n_s = -1\,/\,{\rm max}_{\{i\}}\mathcal{X}_i

which can happen both for positive and negative values, if the event distribution
is in a specific case, however, positive overfluctuations should never happen.
A good summary of this can be found on the wiki [#neg-ns]_.

Choosing a lower / upper boundary cannot solve the problem because :math:`\mathcal{X}` depends
on additional fit-parameters defined in the likelihood class, making the point of divergence
a n-1 dimensional hyperplane that cannot be chosen in the common minimisers.

To avoid running into the divergence of the likelihood, the code turns the logarithms
into parabola very close to the divergence. The true minimum should always be located
closer to zero than the divergence (by logical reasons), so choosing a transition value
:math:`n_s\,\mathcal{X}_i<\alpha_i` (configured in the class, default :math:`1-10^{-5}`)
close to the divergence will not change the likelihood landscape in the interesting region.
Values below the configured value will be evaluated as parabola

.. math::
    :label: taylor

    \log\Lambda=&
        \log\left(1+\alpha\right)
        + \frac{1}{1+\alpha}\left(\alpha_i-\alpha\right)
        - \frac{1}{2}\frac{1}{\left(1+\alpha\right)^2}\left(\alpha_i-\alpha\right)^2
        + \mathcal{O}\left(\left(\alpha_i-\alpha\right)^3\right)\\
    \frac{\partial\log\Lambda}{\partial\sigma_j}=&
        \frac{1}{1+\alpha}\frac{\partial\alpha_i}{\partial\sigma_j}
        +\frac{1}{\left(1+\alpha\right)^2}\left(\alpha_i-\alpha\right)\frac{\partial\alpha_i}{\partial\sigma_j}
        + \mathcal{O}\left(\left(\alpha_i-\alpha\right)^2\right)

with :math:`\sigma_j` including :math:`n_s`.

Stacking
*********

In point source searches, stacking of many weak sources can boost sensitivity
for a test of anisotropy. Stacking means evaluating the point source probability
at several locations

.. math::
    :label: stacking

    \mathcal{S}_i\rightarrow\frac{\sum_i\omega_i\mathcal{S}_i\left(\mathbf{x}_i\right)}{\sum_i\omega_i}

while the background probability and (spatial-independent) weights stay constant.
Using weights :math:`\omega_i`, different locations can be given more strength than others,
while the total normalization adds up to 1.

Giving multiple locations for the sources to *PointSourceLLH*, it will select all events which
fall into any of the sources and calculate the probability. The rest of the likelihood remains
unchanged leaving the rest of the code as before.


LLH Model
----------

The likelihood model classes in SkyLab define the physics part of the
analysis by weighting data by a specific hypothesis. Commonly used weights
in IceCube are energy information, or time information.

In that sense, LLH models calculate the quantities
:math:`\mathcal{S,B,W}` in the unbinned likelihood. For each of this quantity,
the llh model class defines a method that is called in the core of the
SkyLab code if needed.

The Likelihood model defines new parameters in addition to :math:`n_s` and
gives them to the core for minimisation. If one of the methods is called,
the parameters of the evaluation point are passed to be evaluated in the
likelihood model.

As of now, the classic likelihood and one implementing energy weights are
part of SkyLab.

Signal
*******

This part calculates the spatial signal part of the likelihood, assuming
the gaussian distribution. This function is called once after selecting events
for a given source location, i.e., it does not change with the fitted parameters.

Background
***********

Like signal, this is only calculated once for a fit. In fact, the experimental data
does not change the background probability (in IceCube) even for scrambling, so
at initialisation, a spline for the background probability is calculated to
parametrize the background pdf and evaluate :math:`B`.

.. image:: figures/bckg_dec_spline.pdf
    :width: 50 %
    :align: center

For other analyses that e.g. include time dependence or if at all the
scrambling is not a simple rotation in azimuth, this description has to be
more sophisticated.

Weight
*******

Weights are evaluated at every call of the likelihood function, i.e., this
are values that change with change of fitting parameters and will not stay
constant. The weights are then returned together with gradients for all
parameters.

In general, like in the case for energy weights, or Monte-Carlo & Data based
pdf-estimation of weights, is a ratio of sinal- over background-pdfs,
normalized along the declination, which is handled in the other two parts
of the likelihood model.

Using the binning for signal- and background-distributions when creating
splines from histograms, one does not need the correct normalization to the
bin hypervolume of each bin, making evaluation more efficient:

.. math::
    :label: bin_norm

    p_i = \frac{n_i}{V_i\sum_jn_j}

When building the ratio, the bin volume :math:`V_i` cancels and the simple
normalisation to the total number results in a correct ratio (normalisation
along declination).

If injecting events from Monte Carlo, there is a high probability to inject
events that are not covered by the experimental parameter space, and can therefore
not be evaluated. Assuming that this is due to experimental data being sparse in this
spot of the parameter-space, those points are assigned to get the maximal observed
weight in the declination regime. This results in a fast way of evaluating injected trials,
while not changing analysis of pure data.

In current implementations, fit parameters like :math:`\gamma` are evaluated on
a grid, interpolating with an parabola in between weights for each event.
Using this parabola created from the next and neighbouring grid points, the
value and gradient for any parameter-value can be evaluated.

Source Hypothesis
------------------

Like the llh model, source injection can be easily implementing by a small number
of functions talking to the core of the module. The needed functions are

fill
    Fill the injector-object with MC data to sample events from

get_weights
    Get event weights for a physics scenario specified with the injector

sample
    Use the weights to sample events continously assuming a mean Poissonian number
    :math:`\mu`. This returns a generator object that has a *next* method to use
    in loops. In the sampling, events will be rotated to the wanted location.

flux2mu
    Convert a flux to a mean number of expected events

mu2flux
    Convert a number of observed events to a steady flux.


.. rubric:: Footnotes

.. [#ps-paper] `Braun et al. <http://arxiv.org/abs/0801.1604>`_
.. [#null_hyp] At :math:`n_s=0`, signal related parameters like :math:`\gamma` do
               not contribute to the likelihood value, i.e. are degenerate.
.. [#BFGS] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
           Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical
           Computing, 16, 5, pp. 1190-1208.
.. [#neg-ns] `Discussion <https://wiki.icecube.wisc.edu/index.php/Negative_ns_fits>`_

