.. Coenders documentation master file, created by
   sphinx-quickstart on Mon Jul  7 04:59:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorial and Quick Start Guide
=================================================================

.. contents::

So you want to use SkyLab, here is a short set-up guide for the impatient.
Some notes before you start:

- Angles are always in radians, unless stated otherwise
- Energies and other units follow the particle physics standard

  - your data should be in this format
  - i.e., GeV, s, m, [...]

Quick Start Guide
------------------

Data Formatting
################

First thing you need for your calculation is data, this is stored
in numpy.recarray format, containing all needed information. Dependent
on the complexity of the Likelihood you chose this can include more or less
information.

Essential information needed are the reconstructed event coordinates:

- *ra*, rightascension of the event
- *sinDec*, sine of the declination, this is used in the code all the time, however, you can pass the declination *dec* as well and the code will calculate the sine right at the beginning
- *sigma*, uncertainty of the event, usually the pull-corrected paraboloid.

This is it, but if you use for example energy weights of ps_model.EnergyLLH, additional
keys are required, in this case

- *logE*, a quantity used for the weighting, usually the reconstructed energy, but this can be anything that gives separation power of signal vs. background

.. note:

  At the moment, no checks for complete data are done, and no checks for sane data. So if your data is missing
  keys like energy or has NaN values, somewhere it will crash...

Monte Carlo files have the exact same keys as data, with addition of some true information:

- *trueRa*, true rightascension
- *trueDec*, true declination
- *trueSinDec*, true sine of declination, calculated if not available
- *trueE*, true energy, not in log and in GeV
- *ow*, one weight, should be normalized to the number of files read in to correctly calculate fluxes and sensitivities, for the likelihood estimation itself, this is not needed. In the code this is multiplied by the livetime, so one weight should used in seconds

Setting up the LLH modules
###########################

After preparing the data, the likelihood model has to be specified and passed to the Likelihood class together with the data.

In the likelihood model, splines are created to evaluate the pdf's of background (and signal).
When creating the likelihood model, the binning and spline evaluation of the data can be
specified.

Having specified the likelihood model, everything is passed to the Likelihood class,
that takes care of the data, minimisation, etc.. As for the likelihood model, various
parameters can be set to change the behaviour of the model.

.. code-block:: python
  :linenos:

  from skylab.psLLH import PointSourceLLH
  from skylab.ps_model import ClassicLLH

  # implement the likelihood to use to bins for the entire sky
  llh = ClassicLLH(sinDec_range=(-1., 1.), sinDec_bins=2)

  # pass everything to the likelihood model
  psllh = PointSourceLLH(exp, mc, # data used for the analysis
                         365., # livetime in days
                         scramble=False, # unblinded data (default True)
                         llh_model=llh_model, # let's use our defined model
                         delta_ang=np.radians(10.), # use events within 10 deg for detailed minimisation
                         mode="box", # select events in a box around the fitted source
                         nside=128, # resolution starting point for an all-sky-scan
                         )

  # fit a source at the center, just for fun
  TS, Xmin = psllh.fit_source(0., 0.)

  print("Result: ", TS, Xmin)

.. code-block:: python

  (Result:, 2.759221, {'nsources': 2.85})

After fitting a source, the value of the test statistic together with
the fit parameters at the minimum is returned. Depending on the likelihood model,
the fit parameters can be a dictionary of just the number of sources *nsources*,
or additional parameter like the spectral index *gamma* in the energy likelihood.

Scanning the entire sky
########################

To do a scan of the entire sky, the method *all_sky_scan* is used. Basically, this
is a python iterator, starting at a finite resolution map, scanning the entire sky,
and doing follow-up scans on the most interesting regions with higher resolution until
we break the loop.

.. code-block:: python
  :linenos:

  for i, scan in enumerate(psllh.all_sky_scan(
                            decRange=np.radians([-85., 85.]), # exclude Pole region
                            threshold=10., # require TS for follow up to be at least 10
                            follow_up_factor=2, # increase resolution by 2 every iteration
                            )):
    # do something nice here, we just stop after on follow up
    if i > 0:
        break

    # scan is a dictionary with all fitted points, their fitted TS, parameters and information of the hottest spots
    print(scan)

Calculating upper limits
#########################

So let's say we fitted some sources, got a test statistic value of 15 and now we want
to know the :math:`90\%` confidence level for a point source flux. That is, we need to
inject neutrinos at this location.

.. code-block:: python
  :linenos:

  from skylab.ps_injector import PointSourceInjector

  # define our physics hypothesis of a source
  inj = PointSourceInjector(2, # gamma index
                            GeV=1000., # convert GeV to TeV
                            E0=100., # normalize flux at 100*GeV == 100 TeV
                            e_range=(3., 6.), # only inject events from PeV to EeV (for fun),
                            )

  # estimate sensitivity
  result = psllh.weighted_sensitivity(0., # again at the center, rightascension not needed since icecube is invariant under rotation
                                      [0.5], [0.9], # sensitivity: Median of bckg.-fluctuation and 90% signal over threshold
                                      n_iter=1000, # use 1000 trials per estimation
                                      eps=1.e-2, # stop if known at 1% level
                                      )

  # all used trials together with flux needed to reach 90% level
  print(result)

.. code-block:: python

  Estimate Sensitivity for declination   0.0
          Do background scrambles for estimation of TS value for alpha =  50.00%
            250 Background scrambles finished after   0h  0' 5.41''
  Fit background function to scrambles
  Fit delta chi2 to background scrambles
  Delta Distribution plus chi-square <skylab.statistics.delta_chi2 object at 0x1123f13d0>
        Separation factor =  29.600% +/-   2.887%
                NDoF  =   1.65
                Mean  =   0.41
                Scale =   1.34
                KS    =  29.73%
        TS    =   0.00
        alpha =  50.00%
        beta  =  90.00%

  Quick estimate of active region, inject increasing number of events ...
        Active region:   8.0

  Estimate sens. in region up to  16.9
        Best estimate:   6.27, ( 90.00% +/-   3.913%)
  Do    250 trials with mu =   6.27 events
  Estimate sens. in region up to  15.0
        Best estimate:   5.88, ( 90.00% +/-   1.735%)
  Do    250 trials with mu =   5.88 events
  Estimate sens. in region up to  15.0
        Best estimate:   6.31, ( 90.00% +/-   1.151%)
  Do    250 trials with mu =   6.31 events
  Estimate sens. in region up to  15.0
        Best estimate:   6.40, ( 90.00% +/-   0.975%)
  Finished after   0h  0' 40.15''
  Injected:   6.40
  Flux    : 3.01e-12

