# -*-coding:utf8-*-

from __future__ import print_function

r"""This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

psLLH
=====

Core class of the Point Source Likelihood calculation

"""

# python packages
from itertools import repeat
import logging
import sys
import time

# scipy-project imports
import healpy as hp
import numpy as np
import numpy.lib.recfunctions
import scipy.interpolate
import scipy.optimize
from scipy.signal import convolve2d

# local package imports
from . import set_pars
from . import ps_model
from . import utils

# get module logger
def trace(self, message, *args, **kwargs):
    r"""Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# variable defaults
_aval = 1.e-3
_B = np.nan
_b_eps = 0.9
_beta_val = 0.5
_delta_ang = np.radians(10.)
_eps = 5.e-3
_ev = None
_ev_B = np.nan
_ev_S = np.nan
_follow_up_factor = 2
_gamma_bins = np.linspace(1., 4., 50 + 1)
_gamma_def = 2.
_hemisphere_dec = np.radians(-5.)
_livetime = np.nan
_log_level = logging.root.getEffectiveLevel()
_max_iter = int(1.e5)
_max_trial = int(1.e3)
_min_iter = int(2.5e3)
_min_ns = 1.
_mode = "box"
_n = 0
_n_iter = 1000
_n_sig = 5.
_nside = 128
_out_print = 0.1
_pgtol = 1.e-3
_pVal = lambda TS, sinDec: TS
_rho_max = 0.95
_rho_nsource = 0.01
_rho_nsource_bounds = (0., 0.9)
_src_dec = np.nan
_src_ra = np.nan
_seed = None
_sindec_bins = np.linspace(-1., 1., 100. + 1)
_thresh_S = 0.
_TSval = None
_ub_perc = 1.
_win_points = 50


class PointSourceLLH(object):
    r"""Basic Point Source Likelihood class

    `PointSouceLLH` handles the data for one event sample, calculating
    the unbinned Point Source Likelihood

    .. math::    \mathcal{L}=\prod_i\left(
                        \frac{n_s}{N}\mathcal{S}
                       +\left(1-\frac{n_s}{N}\right)\mathcal{B}\right)

    Attributes
    ----------
    dec_bandwidth : float
        Declination bandwidth in radians for calculation of expected signal
        distribution.
    delta_ang : float
        Angular separation between source and event candidates used for
        calculation, value has to be within :math:`\left[0,\pi\right]`.
    hemisphere_dec : float
        Declination angle separating the two selection hemisphere's of up-
        and down-going events.
        Value has to be within :math:`\left[0,\pi\right]`.
    livetime : float
        Livetime of the event sample in days.
    llh_model : `ps_model.NullModel`-like class
        Likelihood model used for the minimisation.
    log_level : int
        Logging level for output information.
    mode : str
        Event selection mode for minimisation ("all", "band", "box").
    nside : int
        N_Side value for pixelisation of SkyMap in `HealPix`. Value has to be
        valid power of 2.
    seed : int
        Global seed for NumPy's random mode.
    threshold : float
        Percentage of events in a scan to be scanned again with finer binning.

    Methods
    -------
    all_sky_scan(**kwargs)
        Perform likelihood minimisation on the entire sphere.
    do_trials(src_dec, **kwargs)
        Analyse scrambled trials on declination band `src_dec`.
    llh(**fit_pars)
        Calculate Likelihood.
    fit_source(src_ra, src_dec, **kwargs)
        Minimize the likelihood at the position `src_ra`, `src_dec`.
    reset()
        Delete all cached values.
    sensitivity(src_dec, alpha, beta, inj, **kwargs)
        Calculate the sensitivity for `beta` percent of scrambles being above
        threshold.

    """

    # default values for psLLH class. Can be overwritten in constructor
    # by setting the attribute as keyword argument

    _log_level = _log_level
    _out_print = _out_print

    # LLH model
    _llh_model = ps_model.ClassicLLH()

    # settings for fitting
    _rho_nsource = _rho_nsource
    _rho_nsource_bounds = _rho_nsource_bounds

    # settings for all-sky scan
    _follow_up_factor = _follow_up_factor
    _hemisphere_dec = _hemisphere_dec
    _nside = _nside
    _random = np.random.RandomState()
    _seed = _seed

    # Data sample
    _livetime = _livetime

    # event selection
    _delta_ang = _delta_ang
    _mode = _mode
    _thresh_S = _thresh_S

    # cached values used to determine if recalculation of llh-weights is needed

    # source hypothesis related
    _B = _B
    _n = _n

    # events in current selection for llh evaluation
    _ev = _ev
    _ev_B = _ev_B
    _ev_S = _ev_S
    _src_ra = _src_ra
    _src_dec = _src_dec

    def __init__(self, exp, mc, livetime,
                 scramble=True, upscale=False, **kwargs):
        r"""Constructor of `PointSourceLikelihood`.

        Fill the class with data and all necessary configuration.

        Parameters
        ----------
        exp : NumPy structured array
            Experimental data with all information needed in the likelihood
            model. Essential values are `ra`, `sinDec`, `sigma`.

        mc : NumPy structured array
            Monte Carlo data similar to `exp`, with additional Monte Carlo
            information `trueRa`, `trueDec`, `trueE`, `ow`.

        livetime : float
            Livetime of experimental data.

        upscale : bool or float
            If float, scale data to match livetime *upscale*

        Other Parameters
        ----------------
        scramble : bool
            Scramble data rightaway.

        kwargs
            Configuration parameters to assign values to class attributes.

        """

        if upscale and not scramble:
            raise ValueError("Cannot upscale UNBLINDED data, "+
                             "turn on scrambling!")

        # UPSCALING
        if upscale:
            print("Upscale exp. data distribution!")
            if isinstance(upscale, tuple):
                try:
                    up_seed, up_livetime = upscale
                    up_livetime = float(up_livetime)
                except ValueError as ve:
                    print("For Upscaling, seed and livetime have to be given "
                          "as a tuple (seed, livetime).")
                    raise ve

            elif isinstance(upscale, bool):
                raise ValueError("Upscale cannot be set to true.")
            else:
                try:
                    up_livetime = float(upscale)
                except ValueError as ve:
                    print("Need a float for livetime in days.")
                    raise ve
                up_seed = None

            # create random generator
            RS = np.random.RandomState(up_seed)

            fact = up_livetime / livetime

            # get number of events needed for sampling
            mu = RS.poisson(fact * len(exp))

            if up_seed is not None:
                print(("\tSample {0:6d} events from exp. data "+
                       "from {1:6.1f} to {2:6.1f} days (x {3:4.1f}), "+
                       "seed {4:7d}").format(
                        mu, livetime, up_livetime, fact, up_seed))
            else:
                print(("\tSample {0:6d} events from exp. data "+
                       "from {1:6.1f} to {2:6.1f} days (x {3:4.1f})").format(
                        mu, livetime, up_livetime, fact))

            # update livetime
            livetime = up_livetime

            exp = RS.choice(exp, size=mu)

        for name, val in zip(["exp", "mc"], [exp, mc]):
            fields = val.dtype.fields.keys()
            if not "sinDec" in fields:
                val = numpy.lib.recfunctions.append_fields(
                        val, "sinDec", np.sin(val["dec"]),
                        dtypes=np.float, usemask=False)
            setattr(self, name, val)

        # weight Monte Carlo weights by livetime in seconds
        self.mc["ow"] *=  3600. * 24. * livetime

        # Experimental data values
        self.livetime = livetime

        # set all other parameters
        set_pars(self, **kwargs)

        # scramble data if not unblinded. Do this after seed has been set
        if scramble:
            self.exp["ra"] = self.random.uniform(0., 2. * np.pi, len(self.exp))
        else:
            print("\t####################################\n"+
                  "\t# Working on >> UNBLINDED << data! #\n"+
                  "\t####################################\n")

        # background will not change, calculate right-away
        self._B = self.llh_model.background(self.exp)

        return

    def __str__(self):
        r"""String representation of PointSourceLLH

        """
        # Data information
        sout = ("{0:s}\n"+
                67*"-"+"\n"+
                "Number of Data Events: {1:7d}\n"+
                "\tZenith Range       : {2:6.1f} - {3:6.1f} deg\n"+
                "\tlog10 Energy Range : {4:6.1f} - {5:6.1f}\n"+
                "\tLivetime of sample : {6:7.2f} days\n").format(
                         self.__repr__(),
                         len(self.exp),
                         np.degrees(np.arcsin(np.amin(self.exp["sinDec"]))),
                         np.degrees(np.arcsin(np.amax(self.exp["sinDec"]))),
                         np.amin(self.exp["logE"]), np.amax(self.exp["logE"]),
                         self.livetime)
        # Monte Carlo information
        sout += (67*"-"+"\n"+
                 "Number of MC Events  : {0:7d}\n"+
                 "\tZenith Range       : {1:6.1f} - {2:6.1f} deg\n"+
                 "\tlog10 Energy Range : {3:6.1f} - {4:6.1f}\n").format(
                         len(self.mc),
                         np.degrees(np.arcsin(np.amin(self.mc["sinDec"]))),
                         np.degrees(np.arcsin(np.amax(self.mc["sinDec"]))),
                         np.amin(self.mc["logE"]), np.amax(self.mc["logE"]))

        # Selection
        sout += (67*"-"+"\n"+
                 "Selected Events      : {0:7d}\n".format(self._n))

        # LLH information
        sout += 67*"-"+"\n"
        sout += "Likelihood model:\n"
        sout += "{0:s}\n".format("\n\t".join(
                        [i if len(set(i)) > 2
                           else i[:-len("\t".expandtabs())]
                         for i in str(self._llh_model).splitlines()]))
        sout += "Fit Parameter\tSeed\tBounds\n"
        pars = self.params
        seed = self.par_seeds
        bounds = self.par_bounds
        for p, s, b in zip(pars, seed, bounds):
            sout += "{0:15s}\t{1:.2f}\t{2:.2f} to {3:.2f}\n".format(
                    p, s, *b)
        sout += 67*"-"

        return sout

    # INTERNAL METHODS

    def _select_events(self, src_ra, src_dec, **kwargs):
        r"""Select events around source location(s) used in llh calculation.

        Parameters
        ----------
        src_ra src_dec : float, array_like
            Rightascension and Declination of source(s)

        Other parameters
        ----------------
        scramble : bool
            Scramble rightascension prior to selection.

        inject : numpy_structured_array
            Events to add to the selected events, fields equal to experimental
            data.

        """

        scramble = kwargs.pop("scramble", False)
        inject = kwargs.pop("inject", None)

        # reset
        self.reset()

        # get the zenith band with correct boundaries
        dec = (np.pi - 2. * self.delta_ang) / np.pi * src_dec
        min_dec = max(-np.pi / 2. , dec - self.delta_ang)
        max_dec = min(np.pi / 2., dec + self.delta_ang)

        dPhi = 2. * np.pi

        # number of total events
        self._N = len(self.exp)

        if self.mode == "all" :
            # all events are selected
            exp_mask = np.ones_like(self.exp["sinDec"], dtype=np.bool)

        elif self.mode in ["band", "box"]:
            # get events that are within the declination band
            exp_mask = ((self.exp["sinDec"] > np.sin(min_dec))
                        &(self.exp["sinDec"] < np.sin(max_dec)))

        else:
            raise ValueError("Not supported mode: {0:s}".format(self.mode))

        # update the zenith selection and background probability
        self._ev = self.exp[exp_mask]
        self._ev_B = self._B[exp_mask]

        # update rightascension information for scrambled events
        if scramble:
            self._ev["ra"] = self.random.uniform(0., 2. * np.pi,
                                                 size=len(self._ev))

        # selection in rightascension
        if self.mode == "box":
            # the solid angle dOmega = dRA * dSinDec = dRA * dDec * cos(dec)
            # is a function of declination, i.e., for a constant dOmega,
            # the rightascension value has to change with declination
            cosFact = np.amin(np.cos([min_dec, max_dec]))
            dPhi = np.amin([2. * np.pi, 2. * self.delta_ang / cosFact])
            ra_dist = np.fabs((self._ev["ra"] - src_ra + np.pi) % (2. * np.pi)
                              - np.pi)
            mask = ra_dist < dPhi/2.

            self._ev = self._ev[mask]

            self._ev_B = self._ev_B[mask]

        self._src_ra = src_ra
        self._src_dec = src_dec

        if inject is not None:
            '''
            # how many events are randomly inside of the selection
            m = self.random.poisson(float(len(inject)) * dPhi / (2.*np.pi))
            ind = np.random.choice(len(self._ev), size=m)
            ind = np.array([i for i in range(len(self._ev)) if not i in ind])

            self._ev = self._ev[ind]
            '''

            self._ev = np.append(self._ev, inject)

            # add background probabilities to injected events
            self._ev_B = np.append(self._ev_B,
                                   self.llh_model.background(inject))

        # calculate signal term
        self._ev_S = self.llh_model.signal(src_ra, src_dec, self._ev)

        # do not calculate values with signal below threshold
        ev_mask = self._ev_S > self.thresh_S
        self._ev = self._ev[ev_mask]
        self._ev_S = self._ev_S[ev_mask]
        self._ev_B = self._ev_B[ev_mask]

        # set number of selected events
        self._n = len(self._ev)

        if (self._n < 1
            and (np.sin(self._src_dec) < self.sinDec_range[0]
                 and np.sin(self._src_dec) > self.sinDec_range[-1])):
            logger.error("No event was selected, fit will go to -infinity")

        logger.info("Select new events for mode {0:s}\n".format(self.mode) +
                    ("For point at ra = {0:6.2f} deg, dec = {1:-6.2f} deg, " +
                     "{2:6d} events were selected").format(
                          np.degrees(src_ra), np.degrees(src_dec),
                          self._n))

        return

    # PROPERTIES for public variables using getters and setters

    @property
    def delta_ang(self):
        return self._delta_ang

    @delta_ang.setter
    def delta_ang(self, val):
        if val < 0. or val > 2.*np.pi:
            logger.warn("Separation angle {0:2e} not in 2pi range".format(
                            val))
            val = np.fabs(np.fmod(val, 2.*np.pi))
        self._delta_ang = float(val)

        return

    @property
    def follow_up_factor(self):
        return self._follow_up_factor

    @follow_up_factor.setter
    def follow_up_factor(self, val):
        if not type(val) is int:
            logger.warn("Follow up factor has to be positive integer")
            val = int(val)
        if val < 1:
            raise ValueError("Follow up factor has to be bigger than 0!")

        self._follow_up_factor = val

        return

    @property
    def hemisphere_dec(self):
        return self._hemisphere_dec

    @hemisphere_dec.setter
    def hemisphere_dec(self, val):
        if np.any(np.fabs(np.asarray(val)) > np.pi/2):
            logger.warn("Declination has to be within +/-90deg")
        self._hemisphere_dec = float(np.fmod(val, np.pi/2.))

        return

    @property
    def livetime(self):
        return self._livetime

    @livetime.setter
    def livetime(self, val):
        if not val > 0.:
            raise ValueError("Livetime has to be positive!")
        self._livetime = float(val)

        return

    @property
    def llh_model(self):
        return self._llh_model

    @llh_model.setter
    def llh_model(self, val):
        if not isinstance(val, ps_model.NullModel):
            raise TypeError("LLH model should inherit from "
                            +"ps_model.NullModel")

        # set likelihood module to variable and fill it with data
        self._llh_model = val

        self._llh_model(self.exp, self.mc)

        return

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        self._log_level = value
        logger.setLevel(value)

        return

    @property
    def params(self):
        return ["nsources"] + self.llh_model.params.keys()

    @property
    def par_bounds(self):
        return np.array([self._n * np.array(self._rho_nsource_bounds)] +
                        [self.llh_model.params[par][1]
                            for par in self.params[1:]])

    @property
    def par_seeds(self):
        return np.array([self._n * self._rho_nsource] +
                        [self.llh_model.params[par][0]
                            for par in self.params[1:]])

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if val == self._mode:
            return
        logger.info("Changing mode from {0:s} to {1:s}".format(self._mode,
                                                               val))
        self._mode = val
        logger.debug("Need to rGeset all cached values...")
        self.reset()

        return

    @property
    def nside(self):
        return self._nside

    @nside.setter
    def nside(self, val):
        if not hp.isnsideok(val):
            logger.warn("Trying to set non-valid nside of {0:d}. ".format(val)+
                        "Choosing the lower nside next to the given value")
            val = np.power(2, int(np.log2(val)))
        self._nside = val

        return

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, value):
        self._random = value

        return

    @property
    def rho_nsource_bounds(self):
        return self._rho_nsource_bounds

    @rho_nsource_bounds.setter
    def rho_nsource_bounds(self, val):
        if not len(val) == 2:
            raise ValueError("Bounds have to be of length 2!")

        self._rho_nsource_bounds = val

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        logger.info("Setting global seed to {0:d}".format(int(val)))
        self._seed = int(val)
        self.random = np.random.RandomState(self.seed)

        return

    @property
    def sinDec_range(self):
        return self.llh_model.sinDec_range

    @property
    def thresh_S(self):
        return self._thresh_S

    @thresh_S.setter
    def thresh_S(self, val):
        if val < 0.:
            raise ValueError("Threshold has to be higher than or equal to zero")

        self._thresh_S = float(val)

        return

    # PUBLIC methods

    def all_sky_scan(self, **kwargs):
        r"""Scan the entire sky for single point sources.

        Perform an all-sky scan. First calculation is done on a coarse grid
        with `nside`, follow up scans are done with a finer binning, while
        conserving the number of scan points by only evaluating the most
        promising grid points.

        Returns
        -------
        scan : iterator
            Iterator of scans, the first element is the scan in nside, the
            following ones are re-scanning the most interesting regions with
            finer binning.

        Other Parameters
        ----------------
        decRange : tuple
            Declination range to scan.

        nside : int
            NSide value for starting healpy map, has to be power of 2.

        follow_up_factor : int
            Power of 2 for grid size in secondary scans.
            N_Side[n+1] = NSide[N] * 2**fuf.

        pVal : lambda function
            Function to convert from test statistic and sin(decl.) to a p-value
            and return all output in form of p-values

                Example: pVal = lambda TS, sinDec: -np.log10(0.5 * scipy.stats.chi2(2.).sf(TS))

            The conversion must be monotonic increasing, because follow up
            scans focus on high values.

        """

        def do_scan(ra, dec, TSs, xmins, mask, **kwargs):
            r"""Scan all events in the given event list.

            Iterate over all the given locations, and get the output.
            Other keyword arguments are passed to the source fitting
            """

            nside = hp.pixelfunc.npix2nside(len(ra))

            out_print = self._out_print
            start = time.clock()
            n_iters = np.count_nonzero(mask)
            n = 0
            for i, (ra_i, dec_i, m) in enumerate(zip(ra, dec, mask)):
                if not m:
                    continue
                logger.trace(("Point {0:7d}, ra = {1:6.1f} deg, "+
                              "dec = {1:6.1f} deg").format(i, np.degrees(ra_i),
                                                           np.degrees(dec_i)))
                # seed the data with the hottest spot around the scanned point
                # if no interesting point around, seed with default value
                neighbours = hp.pixelfunc.get_all_neighbours(nside, i)
                hottest_nb = neighbours[np.argmax(TSs[neighbours])]
                x_seed = xmins[hottest_nb]
                x_dict = [(par, x_seed[par])
                          for par in self.params
                          if x_seed["nsources"] > _min_ns
                          and seed_bounds[par][0]
                                < x_seed[par] < seed_bounds[par][1]]

                TS_i, xmin_i = self.fit_source(ra_i, dec_i,
                                               **dict(x_dict, **kwargs))

                TSs[i] = TS_i

                for key, val in xmin_i.iteritems():
                    xmins[key][i] = val

                # report output
                if float(n)/n_iters > out_print:
                    stop = time.clock()
                    mins, secs = divmod(stop - start, 60)
                    print(("\t{0:7.2%} after {1:2.0f}' {2:4.1f}'' "+
                           "({3:8d} of {4:8d})").format(
                            float(n)/n_iters, mins, secs, n, n_iters))
                    out_print += 0.1

                    sys.stdout.flush()

                n += 1

            return TSs, xmins

        def hotspot(arr):
            r"""Get information of hottest spots in the hemisphere's.

            """
            # get hottest spot with lowest p-value for background hypothesis
            north = ((arr["dec"] > self.hemisphere_dec)
                     & (arr["dec"] > decRange[0]) & (arr["dec"] < decRange[1]))
            result = dict()
            for hem, mask in zip(["North", "South"], [north, ~north]):
                if not np.any(mask):
                    print("{0:s}: No events here".format(hem))
                    continue
                if not np.any(arr[mask]["nsources"] > 0):
                    print("{0:s}: No overfluctuation seen".format(hem))
                    continue
                hot = np.sort(arr[mask], order=["pVal", "TS"])[-1]

                fmin, xmin = self.fit_source_loc(
                                hot["ra"], hot["dec"],
                                hp.pixelfunc.nside2resol(nside),
                                dict([(par, hot[par])
                                      for par in self.params]))

                pV = pVal(fmin, np.sin(xmin["dec"]))

                print(hem)
                print(("Hottest Grid at ra = {0:6.1f}deg, dec = {1:6.1f}deg\n"+
                       "\twith pVal  = {2:4.2f}\n"+
                       "\tand TS     = {3:4.2f} at\n").format(
                            np.degrees(hot["ra"]), np.degrees(hot["dec"]),
                            hot["pVal"], hot["TS"])+
                       "\n".join(["\t{0:10s} = {1:6.2f}".format(par, hot[par])
                                  for par in self.params]))
                print(("Refit location: ra = {0:6.1f}deg, dec = {1:6.1f}deg\n"+
                       "\twith pVal  = {2:4.2f}\n"+
                       "\tand TS     = {3:4.2f} at\n").format(
                            np.degrees(xmin["ra"]), np.degrees(xmin["dec"]),
                            pV, fmin)+
                       "\n".join(["\t{0:10s} = {1:6.2f}".format(par, xmin[par])
                                  for par in self.params]))

                result[hem] = dict(grid=dict([(par, hot[par])
                                              for par in self.params],
                                             ra=hot["ra"], dec=hot["dec"],
                                             nside=hp.npix2nside(len(ra)),
                                             pix=hp.pixelfunc.ang2pix(
                                                    hp.pixelfunc.npix2nside(len(ra)),
                                                    np.pi/2 - hot["dec"], hot["ra"]),
                                             TS=hot["TS"], pVal=hot["pVal"]),
                                   fit=dict([(par, val)
                                                for par, val in xmin.iteritems()],
                                            TS=fmin, pVal=pV))

                sys.stdout.flush()

            return result

        # bounds for seeding values
        seed_bounds = dict([(par, (np.mean(bound) - _b_eps*np.diff(bound)/2.,
                                   np.mean(bound) + _b_eps*np.diff(bound)/2.))
                                if not par == "nsources"
                                else (par, np.array([0., np.inf]))
                            for par, bound in zip(self.params,
                                                  self.par_bounds)])
        par_dtypes = [(par, np.float) for par in self.params]

        decRange = kwargs.pop("decRange", np.arcsin(self.sinDec_range))
        nside = kwargs.pop("nside", self.nside)
        follow_up = kwargs.pop("follow_up_factor", self.follow_up_factor)
        pVal = kwargs.pop("pVal", _pVal)

        # initialize empty maps
        N_points = hp.nside2npix(nside)
        TSs = np.zeros(N_points, dtype=np.float)
        xmins = np.zeros_like(TSs, dtype=par_dtypes)
        mask = np.ones_like(TSs, dtype=np.bool)

        # Generator object
        i = 1
        while True:
            # use healpy to create an isotropic grid
            print("Iteration {0:2d}".format(i))
            print("\tGenerating equal distant points on skymap...")
            print("\t\tNside = {0:4d}, resolution {1:4.2f} deg".format(
                    nside, np.degrees(hp.nside2resol(nside))))

            pix = hp.pixelfunc.nside2npix(nside)

            # create coordinate grid
            # NOTE: theta is NOT in local coordinates, but equatorial
            theta, ra = hp.pixelfunc.pix2ang(nside, np.arange(pix))

            # turn from zenith to declination
            dec = np.pi/2 - theta

            # get previous scans and interpolate to nside
            TSs = hp.pixelfunc.get_interp_val(TSs, theta, ra)
            pVals = pVal(TSs, np.sin(dec))
            _xmins = np.empty_like(TSs, dtype=par_dtypes)
            for par in self.params:
                _xmins[par] = hp.pixelfunc.get_interp_val(xmins[par], theta, ra)
            xmins = _xmins

            # get percentage of points to do of map and the threshold
            perc_points = float(N_points) / len(TSs)

            print("Analysing {0:7.2%} of the scan...".format(perc_points))

            mask = np.isfinite(TSs) & (dec > decRange[0]) & (dec < decRange[1])
            dec_bound = np.unique(np.concatenate([decRange,
                                                  [self.hemisphere_dec]]))
            dec_bound = dec_bound[(dec_bound >= decRange[0])
                                  &(dec_bound <= decRange[1])]
            for ldec, udec in zip(dec_bound[:-1], dec_bound[1:]):
                print("\tDec. {0:-5.1f} to {1:-5.1f} deg".format(
                        *np.degrees([ldec, udec])),
                        end=": ")
                dmask = (dec < ldec)|(dec > udec)
                if np.all(dmask):
                    print("No scan points here")
                    continue
                threshold = np.percentile(pVals[~dmask],
                                          100.*(1. - perc_points))
                tmask = pVals >= threshold
                print("{0:7.2%} above threshold pVal = {1:.2f}".format(
                        np.sum(tmask&~dmask, dtype=np.float)
                            / (len(tmask) - np.sum(dmask)),
                        threshold))

                mask &= (dmask|tmask)

            print("Scanning area of ~{0:4.2f}pi sr ({1:7.2%})".format(
                        np.sum(mask, dtype=np.float)
                            * hp.pixelfunc.nside2pixarea(nside) / np.pi,
                        np.sum(mask, dtype=np.float) / len(mask)))

            print("\tStart all-sky scan")
            start = time.clock()

            TSs, xmins = do_scan(ra, dec, TSs, xmins, mask, **kwargs)

            stop = time.clock()

            mins, secs = divmod(stop - start, 60)
            hours, mins = divmod(mins, 60)
            print()
            print("\tScan finished after {0:3d}h {1:2d}' {2:4.2f}''".format(
                    int(hours), int(mins), secs))

            res_arr = np.zeros((len(ra), ), dtype=[("ra", np.float),
                                                   ("dec", np.float),
                                                   ("theta", np.float),
                                                   ("TS", np.float),
                                                   ("pVal", np.float)])
            res_arr["ra"] = ra
            res_arr["dec"] = dec
            res_arr["theta"] = theta
            res_arr["TS"] = TSs
            res_arr["pVal"] = pVal(TSs, np.sin(dec))

            res_arr = numpy.lib.recfunctions.append_fields(
                        res_arr, self.params,
                        [xmins[par] for par in self.params],
                        [np.float for par in self.params], usemask=False)

            # Return result
            yield res_arr, hotspot(res_arr)

            i += 1

            print(67*"-")
            print("\tNext follow up: nside = {0:d} * 2**{1:d} = {2:d}".format(
                    nside, follow_up, nside * 2**follow_up))

            nside *= 2**follow_up

            sys.stdout.flush()

        # inifinite loop end

        return

    def do_trials(self, src_dec, **kwargs):
        r"""Calculation of scrambled trials.

        Perform trials on scrambled event maps to estimate the event
        distribution.

        Parameters
        ----------
        src_dec : float
            Declination of the interesting point for scrambling.

        Returns
        -------
        dict
            Dictionary with fit results of each scan, the `TS` quantile at
            `beta`, `beta` and its error, and the number of injected sources
            per trial `n_inj`.

        Other parameters
        ----------------
        mu_gen : iterator
            Iterator yielding injected events. Stored at ps_injector.

        miniter maxiter : int
            Minimium and Maximum amount of iterations before stopping.

        TSval : float
            TS value at which to calculate the overlap `beta`.

        n_iter : int
            If given, do exactly that many iterations

        eps : float
            Accuracy < 1 of the overlap `beta` before breaking the scrambling.

        beta_val : float
            Targeted `beta`. Break, if scrambling is too far away of this
            value.

        n_sig : float
            Break, if `beta` differs by more than `n_sig` with respect to the
            targeted value.

        kwargs
            Other keyword arguments are passed to the source fitting.

        .. note:: Possible Modes
                  Background scrambles:
                    Calculate the TS value at a given percentile *beta_val*.
                  Signal scrambles:
                    Calculate the precentile *beta* for a given *TSval*.
                    Break, if the precision of *beta* is good enough or the
                    obtained value is too far away from the wanted *beta_val*

        """
        start  = time.clock()

        mu_gen = kwargs.pop("mu", repeat((0, None)))

        # values for iteration procedure
        maxiter = int(kwargs.pop("maxiter", _max_iter))
        miniter = int(kwargs.pop("miniter", _min_iter))
        TSval = kwargs.pop("TSval", _TSval)
        eps = kwargs.pop("eps", _eps)
        beta_val = kwargs.pop("beta_val", _beta_val)
        n_sig = kwargs.pop("n_sig", _n_sig)

        if "n_iter" in kwargs:
            maxiter = miniter = kwargs.pop("n_iter")

        if n_sig <= 1.:
            raise ValueError(
                "n_sig has to be bigger than one due to breaking conditions.")

        # start looping over scrambles
        mus = list()
        TSs = list()
        xmins = dict([(par, []) for par in self.params])
        for i in xrange(1, int(maxiter) + 1):
            # inject events if injector is given and initialized
            n_inj, inj_sample = mu_gen.next()

            TS_i, xmin_i = self.fit_source(np.pi, src_dec,
                                           inject=inj_sample, scramble=True,
                                           **kwargs)

            TSs.append(TS_i)
            mus.append(n_inj)
            for key, value in xmin_i.iteritems():
                xmins[key].append(value)

            # get the value and error of the test-statistic at a given beta
            TS_beta = np.percentile(TSs, 100.*(1. - beta_val))

            # calculate the percentile of values bigger than given TS,
            # if no TSval is given, use percentile value for beta cross-check
            m = np.count_nonzero(
                  np.asarray(TSs) > (TSval if TSval is not None else TS_beta))
            beta = float(m) / float(i)
            beta_err = (np.sqrt(beta * (1. - beta) / float(i)) if 0 < beta < 1
                                                               else 1.)

            # calculate the relative difference of beta to beta_val
            dsig_beta = np.fabs(beta - beta_val) / beta_err

            # check if any of the convergence criterions is met
            if (i > miniter and (beta_err < eps or dsig_beta > n_sig)):
                logger.debug("Finished after {0:d} iterations".format(i))
                break

            # As long as precision is not reached, give recent output
            if i % (max(1, int(self._out_print * maxiter))) == 0:
                stop = time.clock()
                mins, secs = divmod(stop - start, 60)
                logger.debug(
                    ("{0:6d} of max {1:6d} scrambles completed after "+
                     "{2:2.0f}' {3:4.1f}'', precision: {4:.2f}").format(
                           i, maxiter, mins, secs, beta_err))

        else:
            if miniter < maxiter:
                logger.warn("Iterations ended before converge "+
                            "criterion was met.")
                logger.info("\tTrials  = {0:6d}\n".format(i)+
                            "\tBeta    = {0:7.2%} +/- {1:7.2%}\n".format(
                                        beta, beta_err)+
                            "\tOverfl. = {0:6d}".format(
                                        np.count_nonzero(np.asarray(TSs) > 0)))

        TSs = np.asarray(TSs)
        xmins = dict([(key, np.asarray(value))
                      for key, value in xmins.iteritems()])

        # return each trial result with its fit parameters,
        # the test statistic value matching beta, beta and its error as well as
        result = dict(TS=TSs, TS_beta=TS_beta, beta=beta, beta_err=beta_err,
                      n_inj=mus, **xmins)

        # reset event selection, because trials are run on scrambles
        self.reset()

        return result

    def llh(self, **fit_pars):
        r"""Calculate the likelihood ratio for the selected events.

        Evaluate pointsource likelihood using cached values. For new input,
        values are re-evaluated and cached.

        .. math:: \log\Lambda=\sum_i\log\left(
                  \frac{n_s}{N}\left(\frac{\mathcal{S}}{\mathcal{B}}w-1\right)
                                     +1\right)

        Parameters
        ----------
        fit_pars : dict
            Dictionary with all fit parameters, nsources and all defined by
            `llh_model`.

        Returns
        -------
        funval : float
            Function value
        grad : array_like
            Gradient at the point.
        """

        nsources = fit_pars.pop("nsources")

        N = self._N
        n = self._n

        assert(n == len(self._ev))

        SoB = self._ev_S / self._ev_B

        w, grad_w = self.llh_model.weight(self._ev, **fit_pars)

        x = (SoB * w - 1.) / N

        # check which sums of the likelihood are close to the divergence
        aval = -1. + _aval
        alpha = nsources * x

        # select events close to divergence
        xmask = alpha > aval

        # function value, log1p for OK, otherwise quadratic taylor
        funval = np.empty_like(alpha, dtype=np.float)
        funval[xmask] = np.log1p(alpha[xmask])
        funval[~xmask] = (np.log1p(aval)
                      + 1. / (1.+aval) * (alpha[~xmask] - aval)
                      - 1./2./(1.+aval)**2 * (alpha[~xmask]-aval)**2)
        funval = funval.sum() + (N - n) * np.log1p(-nsources / N)

        # gradients

        # in likelihood function
        ns_grad = np.empty_like(alpha, dtype=np.float)
        ns_grad[xmask] = x[xmask] / (1. + alpha[xmask])
        ns_grad[~xmask] = (x[~xmask] / (1. + aval)
                       - x[~xmask] * (alpha[~xmask] - aval) / (1. + aval)**2)
        ns_grad = ns_grad.sum() - (N - self._n) / (N - nsources)

        # in weights
        if grad_w is not None:
            par_grad = 1. / N * SoB * grad_w

            par_grad[:, xmask] *= nsources / (1. + alpha[xmask])
            par_grad[:, ~xmask] *= (nsources / (1. + aval)
                                    - nsources * (alpha[~xmask] - aval)
                                        / (1. + aval)**2)

            par_grad = par_grad.sum(axis=-1)

        else:
            par_grad = np.zeros((0,))

        grad = np.append(ns_grad, par_grad)

        # multiply by two for chi2 distributed test-statistic
        LogLambda = 2. * funval
        grad = 2. * grad

        return LogLambda, grad

    def fit_source(self, src_ra, src_dec, **kwargs):
        """Minimize the negative log-Likelihood at source position(s).

        Parameters
        ----------
        src_ra src_dec : array_like
            Source position(s).

        Returns
        -------
        fmin : float
            Minimal function value turned into test statistic
            -sign(ns)*logLambda
        xmin : dict
            Parameters minimising the likelihood ratio.

        Other parameters
        ----------------
        scramble : bool
            Scramble events prior to selection.

        inject
            Source injector

        kwargs
            Parameters passed to the L-BFGS-B minimiser.

        """

        # wrap llh function to work with arrays
        def _llh(x, *args):
            """Scale likelihood variables so that they are both normalized.
            Returns -logLambda which is the test statistic and should
            be distributed with a chi2 distribution assuming the null
            hypothesis is true.

            """

            fit_pars = dict([(par, xi) for par, xi in zip(self.params, x)])

            fun, grad = self.llh(**fit_pars)

            # return negative value needed for minimization
            return -fun, -grad

        scramble = kwargs.pop("scramble", False)
        inject = kwargs.pop("inject", None)
        kwargs.setdefault("pgtol", _pgtol)

        # Set all weights once for this src location, if not already cached
        self._select_events(src_ra, src_dec, inject=inject, scramble=scramble)

        # get seeds
        pars = self.par_seeds
        inds = [i for i, par in enumerate(self.params) if par in kwargs]
        pars[inds] = np.array([kwargs.pop(par) for par in self.params
                                               if par in kwargs])

        # minimizer setup
        xmin, fmin, min_dict = scipy.optimize.fmin_l_bfgs_b(
                                _llh, pars,
                                bounds=self.par_bounds,
                                **kwargs)

        if abs(xmin[0]) > _rho_max * self._n:
            logger.error(("nsources > {0:7.2%} * {1:6d} selected events, "+
                          "fit-value nsources = {2:8.1f}").format(
                              _rho_max, self._n, xmin[0]))

        xmin = dict([(par, xi) for par, xi in zip(self.params, xmin)])

        # Separate over and underfluctuations
        fmin *= -np.sign(xmin["nsources"])

        return fmin, xmin

    def fit_source_loc(self, src_ra, src_dec, size, seed, **kwargs):
        """Minimize the negative log-Likelihood around interesting position.

        Parameters
        ----------
        src_ra src_dec : array_like
            Source position(s).

        size : float
            Size of the box for minimisation

        seed : dictionary
            Best seed for region

        Returns
        -------
        fmin : float
            Minimal function value turned into test statistic
            -sign(ns)*logLambda
        xmin : dict
            Parameters minimising the likelihood ratio.

        Other parameters
        ----------------
        kwargs
            Parameters passed to the L-BFGS-B minimiser.

        """

        # wrap llh function to work with arrays
        def _llh(x, *args):
            """Scale likelihood variables so that they are both normalized.
            Returns -logLambda which is the test statistic and should
            be distributed with a chi2 distribution assuming the null
            hypothesis is true.

            """

            # check if new source selection has to be done
            if not (x[0] == self._src_ra and x[1] == self._src_dec):
                self._select_events(x[0], x[1])

            # forget about source position
            x = x[2:]

            fit_pars = dict([(par, xi) for par, xi in zip(self.params, x)])

            fun, grad = self.llh(**fit_pars)

            # return negative value needed for minimization
            return -fun

        if "scramble" in kwargs:
            raise ValueError("No scrambling of events allowed fit_source_loc")
        if "approx_grad" in kwargs and not kwargs["approx_grad"]:
            raise ValueError("Cannot use gradients for location scan")

        kwargs.pop("approx_grad", None)

        kwargs.setdefault("pgtol", _pgtol)

        loc_bound = [[max(0., src_ra - size / np.cos(src_dec)),
                      min(2. * np.pi, src_ra + size / np.cos(src_dec))],
                     [src_dec - size, src_dec + size]]
        pars = [src_ra, src_dec] + [seed[par] for par in self.params]
        bounds = np.vstack([loc_bound, self.par_bounds])

        xmin, fmin, min_dict = scipy.optimize.fmin_l_bfgs_b(
                                _llh, pars, bounds=bounds,
                                approx_grad=True, **kwargs)

        if abs(xmin[0]) > _rho_max * self._n:
            logger.error(("nsources > {0:7.2%} * {1:6d} selected events, "+
                          "fit-value nsources = {2:8.1f}").format(
                              _rho_max, self._n, xmin[0]))

        xmin = dict([("ra", xmin[0]), ("dec", xmin[1])]
                    + [(par, xi) for par, xi in zip(self.params, xmin[2:])])

        # Separate over and underfluctuations
        fmin *= -np.sign(xmin["nsources"])

        return fmin, xmin

    def reset(self):
        r"""Reset all cached values.

        """
        self._N = _n

        self._src_ra = _src_ra
        self._src_dec = _src_dec

        self._ev = _ev
        self._ev_B = _ev_B
        self._ev_S = _ev_S

        self.llh_model.reset()

        return

    def weighted_sensitivity(self, src_dec, alpha, beta, inj, **kwargs):
        """Calculate the point source sensitivity for a given source
        hypothesis using weights.

        All trials calculated are used at each step and weighted using the
        Poissonian probability.

        Credits for this idea goes to Asen.

        Parameters
        ----------
        src_dec : float
            Source position(s)

        alpha : float
            Error of first kind

        beta : float
            Error of second kind

        inj : skylab.BaseInjector instance
            Injection module

        Returns
        -------
        flux : array_like
            Flux needed to reach sensitivity of *alpha*, *beta*

        TSval : array-like
            All trials calculated

        w : array-like
            Weights for each trial for corresponding flux-limit.

        """

        def do_estimation(TSval, beta, mu, mu_TS, mu_xmin):
            r"""Perform sensitivity estimation by varying the injected source
            strength until the scrambling yields a test statistic with the
            wanted value of *beta*.

            """

            print("\tTS    = {0:6.2f}\n".format(TSval) +
                  "\tbeta  = {0:7.2%}".format(beta))
            print()

            if (len(mu) < 1 or (not np.any(mu > 0))
                    or (not np.any(mu_TS[mu > 0] > TSval))):
                # if no events have been injected, do quick estimation
                # of active region by doing a few trials

                print("Quick estimate of active region, " +
                      "inject increasing number of events ...")

                n_inj = int(np.mean(mu)) if len(mu) > 0 else 0
                while True:
                    # break if *beta* percent of injected trials are above the
                    # wanted threshold
                    n_sig = np.count_nonzero(mu > 0)
                    p_up = (float(np.count_nonzero(
                                        mu_TS[mu > 0] > TSval))
                                / n_sig if n_sig > 0 else 0.)

                    if n_sig > 1 and p_up > beta:
                        break

                    n_inj += 1
                    n_inj, sample = inj.sample(n_inj, poisson=False).next()

                    TS_i, xmin_i = self.fit_source(np.pi, src_dec,
                                                   inject=sample,
                                                   scramble=True)

                    mu = np.append(mu, n_inj)
                    mu_TS = np.append(mu_TS, TS_i)
                    for par, val in xmin_i.iteritems():
                        mu_xmin[par] = np.append(mu_xmin[par], val)

                    mu_eff = np.sqrt(n_inj)

                print("\tActive region: {0:5.1f}".format(mu_eff))
                print()

                # do trials around active region
                trial = self.do_trials(src_dec, TSval=TSval, beta_val=beta,
                                       mu=inj.sample(mu_eff),
                                       n_iter=n_iter, **kwargs)

                mu = np.append(mu, trial["n_inj"])
                mu_TS = np.append(mu_TS, trial["TS"])
                for par, val in mu_xmin.iteritems():
                    mu_xmin[par] = np.append(val, trial[par])

            # start estimation
            for i in xrange(1, maxtrial + 1):
                # use existing scrambles to determine best starting point
                fun = lambda n: np.log10(
                                    (utils.poisson_percentile(n, mu, mu_TS,
                                                              TSval)[0]
                                     - beta)**2)

                # do not fit values too high for sampled distributions
                ub = np.percentile(mu[mu > 0], _ub_perc)

                # fit values in region where sampled before
                bounds = np.percentile(mu[mu > 0],
                                       [_ub_perc, 100. - _ub_perc])

                print("\tEstimate sens. in region {0:5.1f} to {1:5.1f}".format(
                            *bounds))

                # get best starting point
                ind = np.argmin([fun(n_i) for n_i in np.arange(0., bounds[-1])])

                # fit closest point to beta value
                x, f, info = scipy.optimize.fmin_l_bfgs_b(
                                    fun, [ind], bounds=[bounds],#[(0., ub)],
                                    approx_grad=True)

                mu_eff = np.asscalar(x)

                # get the statistical uncertainty of the quantile
                b, b_err = utils.poisson_percentile(mu_eff, mu, mu_TS, TSval)

                print("\t\tBest estimate: {0:6.2f}, ({1:7.2%} +/- {2:8.3%})".format(
                            mu_eff, b, b_err))

                # if precision is high enough and fit did converge,
                # the wanted values is reached, stop trial computation
                if (i > 1 and b_err < kwargs["eps"]
                        and mu_eff > bounds[0] and mu_eff < bounds[-1]
                        and np.fabs(b - beta) < kwargs["eps"]):
                    break

                # to avoid a spiral with too few events we want only half
                # of all events to be background scrambles
                p_bckg = np.sum(mu == 0, dtype=np.float) / len(mu)
                mu_eff_min = np.log(1. / (1. - p_bckg))
                mu_eff = np.amax([mu_eff, mu_eff_min])

                print("\tDo {0:6d} trials with mu = {1:6.2f} events".format(
                            n_iter, mu_eff))

                # do trials with best estimate
                trial = self.do_trials(src_dec, TSval=TSval, beta_val=beta,
                                       mu=inj.sample(mu_eff),
                                       n_iter=n_iter, **kwargs)

                mu = np.append(mu, trial["n_inj"])
                mu_TS = np.append(mu_TS, trial["TS"])
                for par, val in mu_xmin.iteritems():
                    mu_xmin[par] = np.append(val, trial[par])

                sys.stdout.flush()

            else:
                logger.warn("Trials ended before converge "+
                            "criterion was met.")

            # save all trials

            return mu_eff, mu, mu_TS, mu_xmin

        start = time.clock()

        # configuration
        maxtrial = int(kwargs.pop("maxtrial", _max_trial))
        n_iter = int(kwargs.pop("n_iter", _n_iter))
        kwargs.setdefault("eps", _eps)
        fit = kwargs.pop("fit", None)

        if fit is not None and not hasattr(fit, "isf"):
            raise AttributeError("fit must have attribute 'isf(alpha)'!")

        # all input values as lists
        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        TSval = np.atleast_1d(kwargs.pop("TSval", [_TSval for i in alpha]))
        if not (len(alpha) == len(beta) == len(TSval)):
            raise ValueError("alpha, beta, and (if given) TSval must have "+
                             " same length!")

        # setup source injector
        inj.fill(src_dec, self.mc)

        print("Estimate Sensitivity for declination {0:5.1f} deg".format(
                np.degrees(src_dec)))

        # storage of all trials
        mu = np.array([], dtype=np.int)
        mu_TS = np.array([], dtype=np.float)
        mu_xmin = dict([(par, np.array([], dtype=np.float))
                        for par in self.params])

        # result list
        TS = list()
        mu_flux = list()
        flux = list()

        for i, (TSval_i, alpha_i, beta_i) in enumerate(zip(TSval, alpha, beta)):

            if TSval_i is None:
                # Need to calculate TS value for given alpha values
                if fit == None:
                    # No parametrization of background given, do scrambles
                    print("\tDo background scrambles for estimation of "+
                          "TS value for alpha = {0:7.2%}".format(alpha_i))

                    bckg_trials = self.do_trials(src_dec, beta_val=alpha_i,
                                                 **kwargs)

                    # use background scrambles for sensitivity estimation
                    mu = np.append(mu, bckg_trials["n_inj"])
                    mu_TS = np.append(mu_TS, bckg_trials["TS"])
                    for par in mu_xmin.iterkeys():
                        mu_xmin[par] = np.append(mu_xmin[par],
                                                 bckg_trials[par])

                    stop = time.clock()
                    mins, secs = divmod(stop - start, 60)
                    hours, mins = divmod(mins, 60)

                    print("\t{0:6d} Background scrambles finished ".format(
                                len(bckg_trials["TS"]))+
                          "after {0:3d}h {1:2d}' {2:4.2f}''".format(
                              int(hours), int(mins), secs))

                    print("Fit background function to scrambles")
                    if self.rho_nsource_bounds[0] < 0:
                        print("Fit two sided chi2 to background scrambles")
                        fitfun = utils.twoside_chi2
                    else:
                        print("Fit delta chi2 to background scrambles")
                        fitfun = utils.delta_chi2
                    fit = fitfun(bckg_trials["TS"], df=2.,
                                 floc=0., fscale=1.)

                    # give information about the fit
                    print(fit)

                else:
                    bckg_trials = None

                # use fitted function to calculate needed TS-value
                TSval_i = np.asscalar(fit.isf(alpha_i))

            # calculate sensitivity
            mu_i, mu, mu_TS, mu_xmin = do_estimation(TSval_i, beta_i,
                                                     mu, mu_TS, mu_xmin)

            TS.append(TSval_i)
            mu_flux.append(mu_i)
            flux.append(inj.mu2flux(mu_i))

            stop = time.clock()

            mins, secs = divmod(stop - start, 60)
            hours, mins = divmod(mins, 60)
            print("\tFinished after "+
                  "{0:3d}h {1:2d}' {2:4.2f}''".format(int(hours), int(mins),
                                                      secs))
            print("\t\tInjected: {0:6.2f}".format(mu_i))
            print("\t\tFlux    : {0:.2e}".format(flux[-1]))
            print("\t\tTrials  : {0:6d}".format(len(mu)))
            print("\t\tTime    : {0:6.2f} trial(s) / sec".format(
                                            float(len(mu)) / (stop - start)))
            print()

            sys.stdout.flush()

        trials = np.empty((len(mu), ),
                          dtype=[("n_inj", np.int), ("TS", np.float)]
                                + [(xmin_i, np.float)
                                   for xmin_i in mu_xmin.iterkeys()])
        trials["n_inj"] = np.asarray(mu)
        trials["TS"] = np.asarray(mu_TS)
        for key, arr in mu_xmin.iteritems():
            trials[key] = arr

        result = dict(flux=flux, mu=mu_flux, TSval=TS, alpha=alpha, beta=beta,
                      fit=fit, trials=trials)

        return result

    def window_scan(self, src_ra, src_dec, width, **kwargs):
        r"""Do a rectangular scan around a position with fine binning.

        Parameters
        -----------
        src_ra, src_dec : float
            Rightascension and declination position of window center in rad.

        width : float
            Window size in rad.

        Returns
        --------
        result : ndarray
            Array of all gridpoints with source result.

        Other Parameters
        -----------------

        npoints : int
            Number of points to scan per dimension.

        xmin : np.recarray
            One seed or a healpy map with all seeds that is used for
            interpolation.

        """

        npoints = kwargs.pop("npoints", _win_points)
        pVal = kwargs.pop("pVal", _pVal)
        xmin = kwargs.pop("xmin", None)

        # get only interesting seeds not close to boundaries
        seed_bounds = dict([(par, (np.mean(bound) - _b_eps*np.diff(bound)/2.,
                                   np.mean(bound) + _b_eps*np.diff(bound)/2.))
                                if not par == "nsources"
                                else (par, np.array([0., np.inf]))
                            for par, bound in zip(self.params,
                                                  self.par_bounds)])

        out_print = self._out_print

        # create grid
        d_ra, dec = np.meshgrid(np.linspace(-width/2., width/2., npoints),
                                np.linspace(-width/2., width/2., npoints))

        # shift window to center location
        dec += src_dec

        # adjust for curvature
        d_ra /= np.cos(dec)

        ra = src_ra + d_ra
        # adjust for periodicity
        mlow = ra < 0.
        mhigh = ra > 2. * np.pi
        ra[mlow] += 2. * np.pi
        ra[mhigh] -= 2. * np.pi

        MIN = np.empty_like(ra.ravel(),
                            dtype=[("TS", np.float), ("pVal", np.float)]
                                  + [(par, np.float) for par in self.params])

        SEED = np.empty_like(ra.ravel(),
                             dtype=[(par, np.float) for par in self.params])

        # create grid with seeds for all points
        if xmin is None:
            # zero seed will be ignored
            SEED = np.zeros_like(SEED)
        elif hasattr(xmin, "__getitem__") and hp.pixelfunc.isnpixok(len(xmin)):
            # healpy map with seeds
            for par in self.params:
                SEED[par] = hp.pixelfunc.get_interp_val(xmin[par],
                                                        np.pi/2. - dec.ravel(),
                                                        ra.ravel())
        else:
            for par in self.params:
                SEED[par] = xmin[par] * np.ones_like(ra.ravel(),
                                                     dtype=np.float)

        start = time.clock()
        n = 0
        n_iters = len(ra.ravel())
        for i, (r, d, s) in enumerate(zip(ra.ravel(), dec.ravel(), SEED)):
            if d < -np.pi/2. or d > np.pi/2.:
                for key in MIN.dtype.names:
                    MIN[key][i] = np.nan
                continue

            # collect all neighbouring completed fits in 2d-grid
            inds = []
            if i % npoints > 0:
                inds.append(i - 1)
            if i > npoints - 1:
                inds.append(i - npoints)
                inds.append(i - npoints + 1)
            if i % npoints > 0 and i > npoints - 1:
                inds.append(i - npoints - 1)
            inds = np.array(np.unique(inds), dtype=np.int)
            if len(inds) > 0 and np.any(MIN[inds]["nsources"] > _min_ns):
                best_ind = inds[np.argmax(MIN[inds]["pVal"])]
                s = numpy.lib.recfunctions.drop_fields(MIN, ["TS", "pVal"],
                                                       usemask=False)[best_ind]

            seed = dict([(key, s[key]) for key in s.dtype.names
                                       if s["nsources"] > _min_ns
                                       and seed_bounds[key][0]
                                             < s[key] < seed_bounds[key][1]]
                                       )

            MIN["TS"][i], xmin = self.fit_source(r, d, **seed)
            MIN["pVal"][i] = pVal(MIN["TS"][i], np.sin(d))
            for key, val in xmin.iteritems():
                MIN[key][i] = val

            # report output
            if float(n)/n_iters > out_print:
                stop = time.clock()
                mins, secs = divmod(stop - start, 60)
                print(("\t{0:7.2%} after {1:2.0f}' {2:4.1f}'' "+
                       "({3:8d} of {4:8d})").format(
                        float(n)/n_iters, mins, secs, n, n_iters))
                out_print += 0.1

            n += 1

        ra[mlow] -= 2. * np.pi
        ra[mhigh] += 2. * np.pi

        MIN = numpy.lib.recfunctions.append_fields(MIN, ["ra", "dec", "x"],
                                                   [ra, dec,
                                                    ra-(1.-np.cos(dec))*d_ra],
                                                   usemask=False)
        MIN = MIN.reshape(ra.shape)

        return MIN


class MultiPointSourceLLH(PointSourceLLH):
    r"""Class to handle multiple event samples that are distinct of each other.

    Different samples have different effective areas that have to be taken into
    account for parting the number of expected neutrinos in between the
    diffenrent samples.

    Each sample is added as an object of PointSourceLLH

    """

    # histograms for signal expectation
    _gamma_bins = _gamma_bins
    _gamma_binmids = (_gamma_bins[1:] + _gamma_bins[:-1]) / 2.
    _gamma_def = _gamma_def
    _sindec_bins = _sindec_bins
    _sindec_binmids = (_sindec_bins[1:] + _sindec_bins[:-1]) / 2.

    # caching values
    _src_dec = np.nan

    def __init__(self, *args, **kwargs):
        r"""Constructor, set all parameters passed

        """

        set_pars(self, **kwargs)

        # init empty dictionary containers
        self._enum = dict()
        self._sams = dict()
        self._nuhist = dict()
        self._nuspline = dict()
        self.mc = dict()

        return

    def __str__(self):
        r"""String representation of MultiPointSourceLLH.

        """

        out_str = "{0:s}\n".format(self.__repr__())
        out_str += 67*"=" + "\n"
        out_str += "Number of samples: {0:2d}\n".format(self.N)
        # loop over all samples
        if self.N > 0:
            out_str += "\t{0:>2s} {1:>10s} {2:>8s} {3:>6s}\n".format(
                            "#", "Name", "Livetime", "Events")
        N_tot = 0
        d_tot = 0
        for num in sorted(self._enum.keys()):
            N_i = len(self._sams[num].exp)
            d_i = self._sams[num].livetime
            N_tot += N_i
            d_tot += d_i
            out_str += "\t{0:2d} {1:>10s} {2:8.2f} {3:6d}\n".format(
                            num, self._enum[num], d_i, N_i)
        out_str += "Number of events: {0:6d}\n".format(N_tot)
        out_str += "Total livetime  : {0:9.2f}\n".format(d_tot)

        out_str += 67*"-"+"\n"
        # loop over all samples
        for num in sorted(self._sams.keys()):
            out_str += "Dataset {0:2d}\n".format(num)
            out_str += 67*"-"+"\n{0:s}\n".format(
                            "\n\t".join(
                                [i if len(set(i)) > 2
                                   else i[:-len("\t".expandtabs())]
                                 for i in str(self._sams[num]).splitlines()]))
            out_str += "\n" + 67 * "=" + "\n"

        return out_str

    def _select_events(self, src_ra, src_dec, **kwargs):
        r"""Select events around source location(s) used in llh calculation.

        Parameters
        ----------
        src_ra src_dec : float, array_like
            Rightascension and Declination of source(s)

        Other parameters
        ----------------
        scramble : bool
            Scramble rightascension prior to selection.

        inject : numpy_structured_array
            Events to add to the selected events, fields equal to experimental
            data.

        """

        # cache declination for weighting calculation
        self._src_ra = src_ra
        self._src_dec = src_dec

        inject = kwargs.pop("inject", None)

        # inject events according to their parent sample
        for enum, sam in self._sams.iteritems():

            if isinstance(inject, dict):
                inj_i = inject.pop(enum, None)
            else:
                inj_i = inject

            sam._select_events(src_ra, src_dec, inject=inj_i, **kwargs)

        self._n = sum([sam._n for sam in self._sams.itervalues()])

        return

    @property
    def N(self):
        return len(self._enum)

    @property
    def enum(self):
        return self._enum

    @property
    def gamma_bins(self):
        return self._gamma_bins

    @gamma_bins.setter
    def gamma_bins(self, value):
        value = np.atleast_1d(value)
        if len(value) < 2:
            raise ValueError("Need exact bin-edges!")

        self._gamma_bins = value

        self._gamma_binmids = (value[1:] + value[:-1]) / 2.

        return

    @property
    def params(self):
        # we need to minimize over all parameters given by any likelihood model
        # gamma will be always minimised over, it is used in the weighting
        return ["nsources"] + list(set([j for i in self._sams.itervalues()
                                        for j in i.llh_model.params.keys()]
                                        + ["gamma"]))

    @property
    def par_bounds(self):
        # get tightest parameter bounds
        par_bounds = [np.array([sam.llh_model.params[par][1]
                                for sam in self._sams.itervalues()
                                if par in sam.llh_model.params]
                                + ([self.gamma_bins[[0, -1]]]
                                    if par is "gamma" else []))
                      for par in self.params[1:]]
        par_bounds = [(np.amax(pb[:, 0]), np.amin(pb[:, 1]))
                      for pb in par_bounds]

        ns = sum([sam._n for sam in self._sams.itervalues()])

        return np.array([ns * np.array(self._rho_nsource_bounds)]
                        + par_bounds)

    @property
    def par_seeds(self):
        # get median seed for all parameters
        par_seeds = [np.median([sam.llh_model.params[par][0]
                                    for sam in self._sams.itervalues()
                                    if par in sam.llh_model.params]
                                    + ([self._gamma_def] if par is "gamma"
                                                         else []))
                     for par in self.params[1:]]

        # get weighted sum of the events for nsources
        gamma = (par_seeds[self.params[1:].index("gamma")]
                    if "gamma" in self.params else self._gamma_def)

        ns = sum([self._sams[enum]._n * w
                  for enum, w in self.powerlaw_weights(
                        self._src_dec, gamma=gamma).iteritems()])
        N = ns * self._rho_nsource

        return np.array([N] + par_seeds)

    @property
    def sindec_bins(self):
        return self._sindec_bins

    @sindec_bins.setter
    def sindec_bins(self, value):
        value = np.atleast_1d(value)
        if len(value) < 2:
            raise ValueError("Need exact bin-edges!")

        self._sindec_bins = value

        self._sindec_binmids = (value[1:] + value[:-1]) / 2.

        return

    @property
    def sinDec_range(self):
        sinDec_range = np.array([sam.llh_model.sinDec_range
                                 for sam in self._sams.itervalues()])

        return np.array([np.amin(sinDec_range[:, 0]),
                         np.amax(sinDec_range[:, 1])])

    def add_sample(self, name, obj):
        r"""Add a PointSourceLLH object to the sample.

        Parameters
        -----------

        obj : PointSourceLLH
            PointSourceLLH instance to be used in the multifit

        """

        if not isinstance(obj, PointSourceLLH):
            raise ValueError("'{0}' is not LLH-style".format(obj))

        enum = max(self._enum) + 1 if self._enum else 0

        self._enum[enum] = name
        self._sams[enum] = obj

        # add mc info for injection
        self.mc[enum] = obj.mc

        # create histogram of signal expectation for this sample
        x = np.sin(obj.mc["trueDec"])
        hist = np.vstack([np.histogram(x, weights=obj.mc["ow"]
                                                    * obj.mc["trueE"]**(-gm),
                                       bins=self.sindec_bins)[0]
                          for gm in self._gamma_binmids])
        hist = hist.T

        # take the mean of the histogram neighbouring bins
        nwin = 5
        filter_window = np.ones((nwin, nwin), dtype=np.float)
        filter_window /= np.sum(filter_window)

        self._nuhist[enum] = (convolve2d(hist, filter_window, mode="same")
                              / convolve2d(np.ones_like(hist),
                                           filter_window, mode="same"))

        return

    def llh(self, **fit_pars):
        r"""LLH for multi-sample is the sum of all Likelihood functions.

        The number of fitted source neutrinos is distributed between
        the samples according to their effective area at the declination.

        Parameters
        -----------
        fit_pars : dict
            Parameters used for the fit

        Returns
        --------
        logLambda : float
            Log Likelihood value at the point *fit_pars*.

        logLambda_grad : array-like
            Gradient at the point *fit_pars*.

        """

        src_dec = self._src_dec
        nsources = fit_pars.pop("nsources")

        w = self.powerlaw_weights(src_dec, **fit_pars)

        # adjust nsources for all fit parameters
        nsw = dict([(enum, wj*nsources) for enum, wj in w.iteritems()])

        # likelihood evaluation on each sample
        LLH_eval = dict([(enum, sam.llh(nsources=nsw[enum], **fit_pars))
                         for enum, sam in self._sams.iteritems()])

        # sum up individual contributions
        logLambda = np.sum([llh[0] for llh in LLH_eval.itervalues()])
        logLambda_grad = np.zeros_like(self.params, dtype=np.float)

        # nsources, always first parameter
        logLambda_grad[0] = np.sum([LLH_eval[enum][-1][0] * w[enum]
                                    for enum in self._enum.iterkeys()])

        # parameters except nsources
        for i, par in enumerate(self.params[1:]):
            if par == "gamma":
                # weights depend on gamma as only parameter
                w_grad = self.powerlaw_weights(src_dec, dgamma=1, **fit_pars)
                logLambda_grad[i + 1] += np.sum([nsources * dwj
                                                 * LLH_eval[enum][-1][0]
                                                 for enum, dwj
                                                 in w_grad.iteritems()])

            # loop over all likelihood models to see if they minmize in gamma
            for enum, sam in self._sams.iteritems():
                if not par in sam.params:
                    continue

                ind = sam.params.index(par)

                logLambda_grad[i + 1] += LLH_eval[enum][-1][ind]

        return logLambda, logLambda_grad

    def powerlaw_weights(self, src_dec, dgamma=0, **fit_pars):
        r"""Calculate the weight of each sample assuming a power-law
        distribution.

        Parameters
        -----------
        src_dec : float
            Declination of point source location.

        dy : int
            Order of gradient in gamma-direction.

        fit_pars : dict
            Fit parameters, important value is gamma, set to *_gamma_def* if
            not present.

        """

        gamma = fit_pars.pop("gamma", self._gamma_def)

        # check if samples and splines are both equal, otherwise re-do spline
        if set(self._nuhist) != set(self._nuspline):
            # delete all old splines
            for key in self._nuspline.iterkeys():
                del self._nuspline[key]

            hist_sum = np.sum([i for i in self._nuhist.itervalues()], axis=0)

            # calculate ratio and spline this
            for key, hist in self._nuhist.iteritems():
                rel_hist = np.log(hist) - np.log(hist_sum)

                self._nuspline[key] = scipy.interpolate.RectBivariateSpline(
                            self._sindec_binmids, self._gamma_binmids,
                            rel_hist, kx=2, ky=2, s=0)

        out_dict = dict([(key, np.exp(val(np.sin(src_dec), gamma, grid=False,
                                      dy=0.)))
                         for key, val in self._nuspline.iteritems()])

        if dgamma == 1.:
            # chain rule d/dy exp(f) = exp(f) *df/dy
            out_dict = dict([(key, val
                                   * self._nuspline[key](np.sin(src_dec),
                                                         gamma,
                                                         grid=False,
                                                         dy=dgamma))
                             for key, val in out_dict.iteritems()])

        return out_dict

    def reset(self):
        r"""Reset all cached values for this class and all stored PS-samples.

        """

        self._src_ra = _src_ra
        self._src_dec = _src_dec

        for obj in self._sams.itervalues():
            obj.reset()

        return

