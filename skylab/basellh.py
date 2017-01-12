# -*- coding: utf-8 -*-

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

"""
from __future__ import division, print_function

import abc
import itertools
import multiprocessing
import sys
import time

import healpy as hp
import numpy as np
import numpy.lib.recfunctions
import scipy.optimize

from . import utils


class BaseLLH(object):
    """Base class for unbinned point source log-likelihood functions

    Derived classes must implement the methods `_select_events`, `llh`,
    and the properties `params`, `par_seeds`, `par_bounds`, and
    `sinDec_range`; check corresponding docs for more details.

    The private attributes `_src_ra` and `_src_dec` can be used to cache
    the tested source position.

    Parameters
    ----------
    seed : Optional[int]
        Random seed initializing the pseudo-random number generator.

    Attributes
    ----------
    nsource : float
        Seed for source strength parameter
    nsource_rho : float
        Use fraction `nsource_rho` of the number of selected events as a
        seed for the source strength parameter.
    nsource_bounds : Tuple[float]
        Lower and upper bound for source strength parameter
    random : RandomState
        Pseudo-random number generator
    ncpu : int
        Number of processing units; a number larger than 1 enables
        multi-processing support.

    """
    __metaclass__ = abc.ABCMeta

    _b_eps = 0.9
    _min_ns = 1.
    _pgtol = 1e-3
    _rho_max = 0.95
    _ub_perc = 1.

    def __init__(self, nsource=15., nsource_rho=0.9, nsource_bounds=(0., 1e3),
                 seed=None, ncpu=1):
        self.nsource = nsource
        self.nsource_rho = nsource_rho
        self.nsource_bounds = nsource_bounds
        self.random = np.random.RandomState(seed)
        self.ncpu = ncpu
        self._nevents = 0
        self._nselected = 0
        self._src_ra = np.inf
        self._src_dec = np.inf

    @abc.abstractmethod
    def _select_events(self, src_ra, src_dec, scramble=False, inject=None):
        """Select events for log-likelihood evaluation.

        This method must set the private attributes number of total
        events `_nevents` and number of selected events `_nselected`.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        scramble : Optional[bool]
            Scramble select events in right ascension.
        inject : Optional[ndarray]
            Structured array containing additional events to append to
            selection

        """
        pass

    @abc.abstractmethod
    def llh(self, nsources, **others):
        """Evaluate log-likelihood function given the source strength
        `nsources` and the parameter values specified in `others`.

        Parameters
        ----------
        nsources : float
            Source strength
        \*\*others
            Other parameters log-likelihood function depends on

        Returns
        -------
        ts : float
            Log-likelihood for the given parameter values
        grad : ndarray
            Gradient for each parameter

        """
        pass

    @abc.abstractproperty
    def params(self):
        """List[str]: Log-likelihood parameter names; the default
        implementation returns ``nsources``.
        """
        return ["nsources"]

    @abc.abstractproperty
    def par_seeds(self):
        """ndarray: Log-likelihood parameter seeds; the default
        implementation returns the seed for the source strength.
        """
        if self._nselected > 0:
            ns = min(self.nsource, self.nsource_rho * self._nselected)
        else:
            ns = self.nsource

        return np.atleast_1d(ns)

    @abc.abstractproperty
    def par_bounds(self):
        """ndarray: Lower and upper log-likelihood parameter bounds; the
        default implementation returns `nsource_bounds`.
        """
        return np.atleast_1d(self.nsource_bounds)

    @abc.abstractproperty
    def sinDec_range(self):
        """ndarray: Lower and upper allowed sine declination; the
        default implementation returns ``(-1., 1.)``.
        """
        return np.array([-1., 1.])

    def all_sky_scan(self, nside=128, follow_up_factor=2, hemispheres=None,
                     pVal=None):
        """Scan the entire sky for single point sources.

        Perform an all-sky scan. First calculation is done on a coarse
        grid with `nside`, follow-up scans are done with a finer
        binning, while conserving the number of scan points by only
        evaluating the most promising grid points.

        Parameters
        ----------
        nside : Optional[int]
            NSide value for initial HEALPy map; must be power of 2.
        follow_up_factor : Optional[int]
            Controls the grid size of following scans,
            ``nside *= 2**follow_up_factor``.
        hemispheres : Optional[Dict[str, Tuple[float]]]
            Declination boundaries in radian of northern and southern
            sky; by default, the horizon is at -5 degrees.
        pVal : Optional[Callable[[ndarray, ndarray], ndarray]
            Calculates the p-value given the test statistic and
            optionally sine declination of the source position; by
            default the p-value is equal the test statistic. The p-value
            must be monotonic increasing, because follow-up scans focus
            on high values.

        Returns
        -------
        Iterator[Tuple]
            Structured array describing the scan result and mapping of
            hemispheres to information about the hottest spot.

        Examples
        --------
        In many cases, the test statistic is chi-square distributed.

        >>> def pVal(ts, sindec):
        ...     return -numpy.log10(0.5 * scipy.stats.chi2(2.).sf(ts))

        """
        if pVal is None:
            def pVal(ts, sindec):
                return ts

        if hemispheres is None:
            hemispheres = dict(
                South=(-np.pi/2., -np.deg2rad(5.)),
                North=(-np.deg2rad(5.), np.pi/2.))

        drange = np.arcsin(self.sinDec_range)

        # NOTE: unique sorts the input list.
        dbound = np.unique(np.hstack([drange] + hemispheres.values()))
        dbound = dbound[(dbound >= drange[0]) & (dbound <= drange[1])]

        npoints = hp.nside2npix(nside)
        ts = np.zeros(npoints, dtype=np.float)
        xmin = np.zeros_like(ts, dtype=[(p, np.float) for p in self.params])

        niterations = 1
        while True:
            print("Iteration {0:2d}\n"
                  "\tGenerating equal distant points on skymap...\n"
                  "\t\tNside = {1:4d}, resolution {2:4.2f} deg".format(
                      niterations, nside, np.rad2deg(hp.nside2resol(nside))))

            # Create grid in declination and right ascension.
            # NOTE: HEALPy returns the zenith angle in equatorial coordinates.
            theta, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
            dec = np.pi/2 - theta

            # Interpolate previous scan results on new grid.
            ts = hp.get_interp_val(ts, theta, ra)
            pvalue = pVal(ts, np.sin(dec))

            xmin = np.array(zip(
                *[hp.get_interp_val(xmin[p], theta, ra) for p in self.params]),
                dtype=xmin.dtype)

            # Scan only points above the p-value threshold per hemisphere. The
            # thresholds depend on what percentage of the sky is evaluated.
            ppoints = npoints / ts.size
            print("Analysing {0:7.2%} of the scan...".format(ppoints))

            mask = np.isfinite(ts) & (dec > drange[0]) & (dec < drange[-1])

            for dlow, dup in zip(dbound[:-1], dbound[1:]):
                print("\tDec. {0:-5.1f} to {1:-5.1f} deg".format(
                      *np.rad2deg([dlow, dup])), end=": ")

                dout = np.logical_or(dec < dlow, dec > dup)

                if np.all(dout):
                    print("No scan points here")
                    continue

                threshold = np.percentile(
                    pvalue[~dout], 100.*(1. - ppoints))

                tabove = pvalue >= threshold

                print("{0:7.2%} above threshold pVal = {1:.2f}".format(
                      np.sum(tabove & ~dout) / (tabove.size - dout.sum()),
                      threshold))

                # Apply threshold mask only to points belonging to the current
                # hemisphere.
                mask &= np.logical_or(dout, tabove)

            nscan = mask.sum()
            area = hp.nside2pixarea(nside) / np.pi

            print("Scanning area of ~{0:4.2f}pi sr ({1:7.2%}, "
                  "{2:d} pix)".format(nscan*area, nscan/mask.size, nscan))

            start = time.time()

            # Here, the actual scan is done.
            ts, xmin = self._scan(ra[mask], dec[mask], ts, xmin, mask)
            pvalue = pVal(ts, np.sin(dec))

            stop = time.time()

            mins, secs = divmod(stop - start, 60)
            hours, mins = divmod(mins, 60)

            print("\tScan finished after {0:3d}h {1:2d}' {2:4.2f}''".format(
                  int(hours), int(mins), secs))

            result = np.array(
                zip(ra, dec, theta, ts, pvalue),
                [(f, np.float) for f in "ra", "dec", "theta", "TS", "pVal"])

            result = numpy.lib.recfunctions.append_fields(
                result, names=self.params, data=[xmin[p] for p in self.params],
                dtypes=[np.float for p in self.params], usemask=False)

            yield result, self._hotspot(
                    result, nside, hemispheres, drange, pVal)

        print(67*"-")
        print("\tNext follow up: nside = {0:d} * 2**{1:d} = {2:d}".format(
              nside, follow_up_factor, nside * 2**follow_up_factor))

        sys.stdout.flush()

        nside *= 2**follow_up_factor
        niterations += 1

    def _scan(self, ra, dec, ts, xmin, mask):
        """Minimize negative log-likelihood function for given source
        locations.

        """
        seeds = [
            {p: s[p] for p in self.params}
            if s["nsources"] > self._min_ns else {} for s in xmin[mask]
            ]

        # Minimize negative log-likelihood function for every source position.
        if (self.ncpu > 1 and ra.size > self.ncpu and
                self.ncpu <= multiprocessing.cpu_count()):
            args = [
                (self, ra[i], dec[i], False, None, seeds[i], None)
                for i in range(ra.size)
                ]

            pool = multiprocessing.Pool(self.ncpu)
            results = pool.map(fit_source, args)

            pool.close()
            pool.join()
        else:
            results = [
                self.fit_source(ra[i], dec[i], **seeds[i])
                for i in range(ra.size)
                ]

        ts[mask] = [r[0] for r in results]

        for field in xmin.dtype.names:
            xmin[field][mask] = [r[1][field] for r in results]

        return ts, xmin

    def _hotspot(self, scan, nside, hemispheres, drange, pVal):
        """Gather information about hottest spots in each hemisphere.

        """
        result = {}
        for key, dbound in hemispheres.iteritems():
            mask = (
                (scan["dec"] >= dbound[0]) & (scan["dec"] <= dbound[1]) &
                (scan["dec"] > drange[0]) & (scan["dec"] < drange[1])
                )

            if not np.any(mask):
                print("{0:s}: No events here".format(key))
                continue

            if not np.any(scan[mask]["nsources"] > 0):
                print("{0:s}: No overfluctuation seen".format(key))
                continue

            hotspot = np.sort(scan[mask], order=["pVal", "TS"])[-1]
            seed = {p: hotspot[p] for p in self.params}

            print(key)
            print("Hottest Grid at ra = {0:6.1f}deg, dec = {1:6.1f}deg\n"
                  "\twith pVal  = {2:4.2f}\n"
                  "\tand TS     = {3:4.2f} at".format(
                          np.rad2deg(hotspot["ra"]),
                          np.rad2deg(hotspot["dec"]),
                          hotspot["pVal"],
                          hotspot["TS"]))

            print("\n".join(
                "\t{0:10s} = {1:6.2f}".format(p, seed[p]) for p in seed))

            result[key] = dict(grid=dict(
                ra=hotspot["ra"],
                dec=hotspot["dec"],
                nside=nside,
                pix=hp.ang2pix(nside, np.pi/2 - hotspot["dec"], hotspot["ra"]),
                TS=hotspot["TS"],
                pVal=hotspot["pVal"]))

            result[key]["grid"].update(seed)

            fmin, xmin = self.fit_source_loc(
                hotspot["ra"], hotspot["dec"], size=hp.nside2resol(nside),
                seed=seed)

            pvalue = np.asscalar(pVal(fmin, np.sin(xmin["dec"])))

            print("Refit location: ra = {0:6.1f}deg, dec = {1:6.1f}deg\n"
                  "\twith pVal  = {2:4.2f}\n"
                  "\tand TS     = {3:4.2f} at".format(
                      np.rad2deg(xmin["ra"]),
                      np.rad2deg(xmin["dec"]),
                      pvalue,
                      fmin))

            print("\n".join(
                "\t{0:10s} = {1:6.2f}".format(p, xmin[p]) for p in seed))

            result[key]["fit"] = dict(TS=fmin, pVal=pvalue)
            result[key]["fit"].update(xmin)

            if result[key]["grid"]["pVal"] > result[key]["fit"]["pVal"]:
                result[key]["best"] = result[key]["grid"]
            else:
                result[key]["best"] = result[key]["fit"]

            sys.stdout.flush()

        return result

    def fit_source(self, src_ra, src_dec, scramble=False, inject=None,
                   **kwargs):
        """Minimize the negative log-likelihood function at source
        position.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        scramble : Optional[bool]
            Scramble events in right ascension prior to selection.
        inject : Optional[ndarray]
            Structured array containing additional events to append to
            selection
        \*\*kwargs
            Seeds for parameters given in `params` and parameters passed
            to the L-BFGS-B minimizer

        Returns
        -------
        fmin : float
            Minimal negative log-likelihood converted into the test
            statistic ``-sign(ns)*llh``
        pbest : Dict[str, float]
            Parameters minimizing the negative log-likelihood function

        Raises
        ------
        RuntimeError
            The minimization is repeated with different random seeds
            for the source strength `nsources` until an accurate result
            is obtained. The error is raised after 100 unsuccessful
            minimizations.

        """
        # Set all weights once for this source location, if not already cached.
        self._select_events(src_ra, src_dec, scramble, inject)

        if self._nevents < 1:
            # No events were selected; return default seeds.
            pbest = dict(zip(self.params, self.par_seeds))
            pbest["nsources"] = 0.

            return 0., pbest

        def llh(x, *args):
            """Wrap log-likelihood to work with arrays and return the
            negative log-likelihood, which will be minimized.

            """
            params = dict(zip(self.params, x))
            func, grad = self.llh(**params)

            return -func, -grad

        # Get parameter seeds; override default values with kwargs.
        params = np.array([
            kwargs.pop(p, s) for p, s in zip(self.params, self.par_seeds)
            ])

        bounds = self.par_bounds

        # Minimize negative log-likelihood; repeat minimization with different
        # seeds for nsources until the fit looks nice. Do not more than 100
        # iterations.
        kwargs.setdefault("pgtol", self._pgtol)

        xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
            llh, params, bounds=bounds, **kwargs)

        niterations = 1
        while success["warnflag"] == 2 and "FACTR" in success["task"]:
            if niterations > 100:
                raise RuntimeError("Did not manage a good fit.")

            params[0] = self.random.uniform(0., 2.*params[0])

            xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
                llh, params, bounds=bounds, **kwargs)

            niterations += 1

        # If the null-hypothesis is part of minimization, the fit should be
        # negative; log only if the distance is too large.
        if fmin > 0. and (bounds[0][0] <= 0. and bounds[0][1] >= 0.):
            if abs(fmin) > kwargs["pgtol"]:
                print("Fitter returned positive value, force to be zero at "
                      "null-hypothesis. Minimum found {0} with fmin "
                      "{1}".format(xmin, fmin))

            fmin = 0.
            xmin[0] = 0.

        if self._nevents > 0 and abs(xmin[0]) > self._rho_max*self._nselected:
            print("nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                  "nsources = {2:8.1f}".format(
                      self._rho_max, self._nselected, xmin[0]))

        pbest = dict(zip(self.params, xmin))

        # Separate over and under fluctuations.
        fmin *= -np.sign(pbest["nsources"])

        return fmin, pbest

    def fit_source_loc(self, src_ra, src_dec, size, seed, **kwargs):
        """Minimize the negative log-likelihood function around source
        position.

        Parameters
        ----------
        src_ra : float
            Right ascension of interesting position
        src_dec : float
            Declination of interesting position
        size : float
            Size of box around source position for minimization
        seed : Dict[str, float]
            Seeds for remaining parameters; e.g. result from a previous
            `fit_source` call.
        \*\*kwargs
            Parameters passed to the L-BFGS-B minimizer

        Returns
        -------
        fmin : float
            Minimal negative log-likelihood converted into the test
            statistic ``-sign(ns)*llh``
        pbest : Dict[str, float]
            Parameters minimizing the negative log-likelihood function

        """
        def llh(x, *args):
            """Wrap log-likelihood to work with arrays and return the
            negative log-likelihood, which will be minimized.

            """
            # If the minimizer is testing a new position, different events have
            # to be selected; cache position.
            if (np.fabs(x[0] - self._src_ra) > 0. or
                    np.fabs(x[1] - self._src_dec) > 0.):
                self._select_events(x[0], x[1])
                self._src_ra = x[0]
                self._src_dec = x[1]

            params = dict(zip(self.params, x[2:]))
            func, grad = self.llh(**params)

            return -func

        dra = size / np.cos(src_dec)

        bounds = [
            (max(0., src_ra - dra), min(2.*np.pi, src_ra + dra)),
            (src_dec - size, src_dec + size)
            ]

        bounds = np.vstack([bounds, self.par_bounds])
        params = [src_ra, src_dec] + [seed[p] for p in self.params]

        kwargs.pop("approx_grad", None)
        kwargs.setdefault("pgtol", self._pgtol)

        xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
            llh, params, bounds=bounds, approx_grad=True, **kwargs)

        if self._nevents > 0 and abs(xmin[0]) > self._rho_max*self._nselected:
            print("nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                  "nsources = {2:8.1f}".format(
                      self._rho_max, self._nselected, xmin[0]))

        pbest = dict(ra=xmin[0], dec=xmin[1])
        pbest.update(dict(zip(self.params, xmin[2:])))

        # Separate over and under fluctuations.
        fmin *= -np.sign(pbest["nsources"])

        # Clear cache.
        self._src_ra = np.inf
        self._src_dec = np.inf

        return fmin, pbest

    def do_trials(self, src_ra, src_dec, n_iter=int(1e5), mu=None,
                  **kwargs):
        """Create trials of scrambled event maps to estimate the test
        statistic distribution.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        n_iter : Optional[int]
            Number of trials to create
        mu : Optional[Injector]
            Inject additional events into the scrambled map.
        \*\*kwargs
            Parameters passed to `fit_source`

        Returns
        -------
        ndarray
            Structured array containing number of injected events
            ``n_inj``, test statistic ``TS`` and best-fit values for
            `params` per trial

        """
        if mu is None:
            mu = itertools.repeat((0, None))

        inject = [mu.next() for i in range(n_iter)]

        # Minimize negative log-likelihood function for every trial. In case of
        # multi-processing, each process needs its own sampling seed.
        if (self.ncpu > 1 and n_iter > self.ncpu and
                self.ncpu <= multiprocessing.cpu_count()):
            args = [(
                self, src_ra, src_dec, True, inject[i][1], kwargs,
                self.random.randint(2**32)) for i in range(n_iter)
                ]

            pool = multiprocessing.Pool(self.ncpu)
            results = pool.map(fit_source, args)

            pool.close()
            pool.join()
        else:
            results = [
                self.fit_source(src_ra, src_dec, True, inject[i][1], **kwargs)
                for i in range(n_iter)
                ]

        dtype = [("n_inj", np.int), ("TS", np.float)]
        dtype.extend((p, np.float) for p in self.params)

        trials = np.empty((n_iter, ), dtype=dtype)

        for i in range(n_iter):
            trials["n_inj"][i] = inject[i][0]
            trials["TS"][i] = results[i][0]

            for key in results[i][1]:
                trials[key][i] = results[i][1][key]

        return trials

    def sensitivity(self, src_ra, src_dec, ts, beta, inj, n_iter=1000,
                    eps=5e-3, trials=None, **kwargs):
        """Calculate sensitivity for a given source hypothesis.

        Generate signal trials by injecting events arriving from the
        position of the source until a fraction `beta` of trials have a
        test statistic larger than `ts`.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        ts : array_like
            Test statistic that corresponds to the type I error with
            respect to the background test statistic distribution
        beta : array_like
            Fraction of signal trials with a test static larger than
            `ts`
        inj : Injector
            Inject events arriving from the position of the source into
            the scrambled map.
        n_iter : Optional[int]
            Number of trials to per iteration
        eps : Optional[float]
            Precision in `beta` for execution to break
        trials : Optional[ndarray]
            Structured array describing already performed trials,
            containing number of injected events ``n_inj``, test
            statistic ``TS`` and best-fit values for `params` per trial
        \*\*kwargs
            Parameters passed to `do_trials`

        Returns
        -------
        mu : ndarray
            Number of injected signal events to fulfill sensitivity
            criterion
        trials : ndarray
            Structured array containing previous and all newly generated
            trials

        """
        # Let NumPy handle the broadcasting of ts and beta.
        broadcast = np.broadcast(ts, beta)

        if trials is None:
            dtype = [("n_inj", np.int), ("TS", np.float)]
            dtype.extend((p, np.float) for p in self.params)
            trials = np.empty((0, ), dtype=dtype)

        print("Estimate sensitivity for declination {0:5.2f}deg.".format(
              np.rad2deg(src_dec)))

        values = []
        for ts, beta in broadcast:
            mu, trials = self._sensitivity(
                src_ra, src_dec, ts, beta, inj, n_iter, eps, trials, **kwargs)

            values.append(mu)

        mu = np.empty(broadcast.shape)
        mu.flat = values

        return mu, trials

    def _sensitivity(self, src_ra, src_dec, ts, beta, inj, n_iter, eps, trials,
                     **kwargs):
        start = time.time()

        print("\tTS    = {0:6.2f}\n"
              "\tbeta  = {1:7.2%}\n".format(ts, beta))

        # If no events have been injected, do a quick an estimation of active
        # region by doing a few trials.
        if (len(trials) < 1 or
                not np.any(trials["n_inj"] > 0) or
                not np.any(trials["TS"][trials["n_inj"] > 0] > 2.*ts)):

            trials = self._active_region(
                src_ra, src_dec, ts, beta, inj, n_iter, trials, **kwargs)

        # Calculate number of injected events needed so that beta percent of
        # the trials have a test statistic larger than ts. Fit closest point
        # to beta value; use existing scrambles to determine a seed for the
        # minimization; restrict fit to region where sampled before.
        stop = False
        niterations = 1
        while not stop:
            bounds = np.percentile(
                trials["n_inj"][trials["n_inj"] > 0],
                q=[self._ub_perc, 100. - self._ub_perc])

            if bounds[0] == 1:
                bounds[0] = np.count_nonzero(trials["n_inj"] == 1) /\
                    np.sum(trials["n_inj"] < 2)

            print("\tEstimate sensitivity in region {0:5.1f} to "
                  "{1:5.1f}.".format(*bounds))

            def residual(n):
                return np.log10((utils.poisson_percentile(
                    n, trials["n_inj"], trials["TS"], ts)[0] - beta)**2)

            seed = np.argmin([residual(n) for n in np.arange(bounds[-1])])

            xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
                residual, [seed], bounds=[bounds], approx_grad=True)

            mu = np.asscalar(xmin)

            # Get the statistical uncertainty of the quantile.
            b, b_err = utils.poisson_percentile(
                mu, trials["n_inj"], trials["TS"], ts)

            print("\t\tBest estimate: {0:6.2f}, "
                  "({1:7.2%} +/- {2:8.3%})".format(mu, b, b_err))

            # If precision is high enough and fit did converge, the wanted
            # value is reached and we can stop the trial computation after this
            # iteration. Otherwise, do more trials with best estimate for mu.
            stop = (
                b_err < eps and
                mu > bounds[0] and mu < bounds[-1] and
                np.fabs(b - beta) < eps
                )

            if not stop or niterations == 1:
                # To avoid a spiral with too few events, we want only half of
                # all events to be background scrambles after iterations.
                p_bckg = np.sum(trials["n_inj"] == 0) / len(trials)
                mu_min = np.log(1. / (1. - p_bckg))
                mu = np.amax([mu, mu_min])

                print("\tDo {0:6d} trials with mu = {1:6.2f} events.".format(
                      n_iter, mu))

                trials = np.append(
                    trials, self.do_trials(
                        src_ra, src_dec, n_iter,
                        mu=inj.sample(src_ra, mu), **kwargs))

            niterations += 1

            sys.stdout.flush()

        stop = time.time()
        mins, secs = divmod(stop - start, 60)
        hours, mins = divmod(mins, 60)

        print("\tFinished after {0:3.0f}h {1:2.0f}' {2:4.2f}''.\n"
              "\t\tInjected: {3:6.2f}\n"
              "\t\tFlux    : {4:.2e}\n"
              "\t\tTrials  : {5:6d}\n"
              "\t\tTime    : {6:6.2f} trial(s) / sec\n".format(
                  hours, mins, secs, mu, inj.mu2flux(mu), len(trials),
                  len(trials) / (stop - start)))

        sys.stdout.flush()

        return mu, trials

    def _active_region(self, src_ra, src_dec, ts, beta, inj, n_iter, trials,
                       **kwargs):
        if len(trials) > 0:
            n_inj = int(np.mean(trials["n_inj"]))
        else:
            n_inj = 0

        print("Quick estimate of active region, inject increasing "
              "number of events, starting with {0:d} "
              "events...".format(n_inj + 1))

        stop = False
        while not stop:
            n_inj, inject = inj.sample(
                src_ra, n_inj + 1, poisson=False).next()

            fmin, pbest = self.fit_source(
                src_ra, src_dec, scramble=True, inject=inject)

            trial = np.empty((1, ), dtype=trials.dtype)
            trial["n_inj"] = n_inj
            trial["TS"] = fmin

            for key in self.params:
                trial[key] = pbest[key]

            trials = np.append(trials, trial)

            mts = np.bincount(trials["n_inj"], weights=trials["TS"])
            mw = np.bincount(trials["n_inj"])
            mts[mw > 0] /= mw[mw > 0]

            residuals = mts - ts

            stop = (
                np.count_nonzero(residuals > 0.)/len(residuals) > beta or
                np.all(residuals > 0.)
                )

        mu = len(trials) * beta
        print("\tActive region: {0:5.1f}\n".format(mu))

        # Do trials around active region.
        trials = np.append(
            trials, self.do_trials(
                src_ra, src_dec, n_iter,
                mu=inj.sample(src_ra, mu), **kwargs))

        return trials

    def window_scan(self, src_ra, src_dec, width, npoints=50, xmin=None,
                    pVal=None):
        r"""Do a rectangular scan around source position.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        width : float
            Window size
        npoints : Optional[int]
            Number of scan points per dimension
        xmin : Optional[Dict[str, float]]
            Seeds for parameters given in `params`; either one value or
            a HEALPy map per parameter.
        pVal : Optional[Callable[[ndarray, ndarray], ndarray]]
            Calculates the p-value given the test statistic and
            optionally sine declination of the source position; by
            default the p-value is equal the test statistic.

        Returns
        -------
        ndarray
            Structured array describing effective right ascension ``ra``
            corrected for curvature, right ascension ``x``, declination
            ``dec``, test statistic ``TS``, p-value ``pVal``, and
            best-fit values for `params` on the two-dimensional grid
            declination versus right ascension

        Note
        ----
        The periodicity in right ascension is only taken into account
        internally.

        """
        if pVal is None:
            def pVal(ts, sindec):
                return ts

        # Create rectangular window.
        ra = np.linspace(-width/2., width/2., npoints)
        dra, dec = np.meshgrid(ra, ra)

        # Shift window to source location; adjust right ascension for curvature
        # and periodicity.
        dec = np.ravel(src_dec + dec)
        dra = np.ravel(dra / np.cos(dec))
        ra = src_ra + dra
        mra = np.mod(ra - 2.*np.pi, 2.*np.pi)

        # Create seeds for all scan points and tighten seed boundaries. If
        # xmin consists of HEALPy maps, interpolate them.
        dtype = [(p, np.float) for p in ["TS", "pVal"] + self.params]
        seeds = np.empty_like(mra, dtype=dtype[2:])

        if hasattr(xmin, "__getitem__"):
            if hp.pixelfunc.isnpixok(len(xmin)):
                zen = np.pi/2. - dec
                for p in self.params:
                    seeds[p] = hp.pixelfunc.get_interp_val(xmin[p], zen, mra)
            else:
                for p in self.params:
                    seeds[p] = xmin[p] * np.ones_like(mra, dtype=np.float)
        else:
            seeds = np.zeros_like(seeds)

        bounds = {
            p: (
                np.mean(b) - self._b_eps*np.diff(b)/2.,
                np.mean(b) + self._b_eps*np.diff(b)/2.
                )
            if p != "nsources" else (0., np.inf)
            for p, b in zip(self.params, self.par_bounds)
            }

        # Minimize negative log-likelihood function for scan point.
        results = np.empty_like(seeds, dtype=dtype)
        start = time.time()

        for i in range(mra.size):
            if dec[i] < -np.pi/2. or dec[i] > np.pi/2.:
                for field in results.dtype.names:
                    results[field][i] = np.nan

                continue

            # We can use neighboring completed scan points to select better
            # seeds based on the maximum obtained p-value.
            neighbors = []
            if i % npoints > 0:
                neighbors.append(i - 1)

            if i > npoints - 1:
                neighbors.append(i - npoints)
                neighbors.append(i - npoints + 1)

            if i % npoints > 0 and i > npoints - 1:
                neighbors.append(i - npoints - 1)

            neighbors = np.array(np.unique(neighbors), dtype=np.int)

            if (len(neighbors) > 0 and
                    np.any(results["nsources"][neighbors] > self._min_ns)):
                index = neighbors[np.argmax(results["pVal"][neighbors])]
                seed = results[self.params][index]
            else:
                seed = seeds[i]

            if seed["nsources"] > self._min_ns:
                seed = {
                    p: seed[p] for p in seed.dtype.names
                    if bounds[p][0] < seed[p] < bounds[p][1]
                    }
            else:
                seed = {}

            results["TS"][i], pbest = self.fit_source(mra[i], dec[i], **seed)
            results["pVal"][i] = pVal(results["TS"][i], np.sin(dec[i]))

            for key in pbest:
                results[key][i] = pbest[key]

            stop = time.time()
            mins, secs = divmod(stop - start, 60)

            print("\t{0:7.2%} after {1:2.0f}' {2:4.1f}'' ({3:8d} of "
                  "{4:8d})".format((i+1)/mra.size, mins, secs, i+1, mra.size))

        results = numpy.lib.recfunctions.append_fields(
            results, names=["ra", "dec", "x"],
            data=[ra, dec, ra - dra*(1. - np.cos(dec))], usemask=False)

        return results.reshape((npoints, npoints))


def fit_source((llh, src_ra, src_dec, scramble, inject, kwargs, seed)):
    """Wraps `BaseLLH.fit_source` to enable multi-processing support.

    """
    if scramble:
        llh.random = np.random.RandomState(seed)

    return llh.fit_source(src_ra, src_dec, scramble, inject, **kwargs)
