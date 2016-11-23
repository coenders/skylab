from __future__ import division, print_function

import abc
import itertools
import multiprocessing
import sys
import time

import numpy as np
import numpy.lib.recfunctions
import scipy.optimize

from . import utils


def fs(args):
    llh, src_ra, src_dec, scramble, inject, kwargs, seed = args

    if scramble:
        llh.random = np.random.RandomState(seed)

    return llh.fit_source(src_ra, src_dec, scramble, inject, **kwargs)


class BaseLLH(object):
    """Base class for unbinned point source log-likelihood functions

    Derived classes must implement the methods `_select_events`, `llh`
    and the properties `params`, `par_seeds`, `par_bounds` and `size`.

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

    """
    __metaclass__ = abc.ABCMeta

    _pgtol = 1e-3
    _rho_max = 0.95
    _ub_perc = 1.

    def __init__(self, nsource=15., nsource_rho=0.9, nsource_bounds=(0., 1e3),
                 seed=None):
        self.nsource = nsource
        self.nsource_rho = nsource_rho
        self.nsource_bounds = nsource_bounds
        self.random = np.random.RandomState(seed)
        self._nselected = 0
        self._src_ra = np.nan
        self._src_dec = np.nan

    @abc.abstractmethod
    def _select_events(self, src_ra, src_dec, scramble=True, inject=None):
        """Select events for log-likelihood evaluation.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        scramble : Optional[bool]
            Scramble events prior to selection.
        inject : Optional[ndarray]
            Structured array containing additional events to append to
            selection

        Returns
        -------
        int:
            Number of selected events

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
    def size(self):
        """int: Number of events the likelihood model is evaluated for.
        """
        pass

    def fit_source(self, src_ra, src_dec, scramble=True, inject=None,
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
            Parameters passed to the L-BFGS-B minimizer

        Returns
        -------
        fmin : float
            Minimal negative log-likelihood converted into the test
            statistic ``-sign(ns)*llh``
        pbest : Dict[str, float]
            Parameters minimizing the negative log-likelihood function

        Warnings
        --------
        Only set `scramble` to `False` if you want to unblind the data.

        """
        # Set all weights once for this source location, if not already cached.
        self._nselected = self._select_events(
            src_ra, src_dec, scramble, inject)

        if self.size < 1:
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

        # Minimize negative log-likelihood; repeat minimization with different
        # seeds for nsources until the fit looks nice. Do not more than 100
        # iterations.
        kwargs.setdefault("pgtol", self._pgtol)

        xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
            llh, params, bounds=self.par_bounds, **kwargs)

        niterations = 1

        while success["warnflag"] == 2 and "FACTR" in success["task"]:
            if niterations > 100:
                raise RuntimeError("Did not manage good fit.")

            params[0] = self.random.uniform(0., 2.*params[0])

            xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
                llh, params, bounds=self.par_bounds, **kwargs)

            niterations += 1

        # If the null-hypothesis is part of minimization, the fit should be
        # negative; log only if the distance is too large.
        if fmin > 0. and (
                self.par_bounds[0][0] <= 0. and self.par_bounds[0][1] >= 0.):
            if abs(fmin) > kwargs["pgtol"]:
                print(
                    "Fitter returned positive value, force to be zero at "
                    "null-hypothesis. Minimum found {0} with fmin "
                    "{1}".format(xmin, fmin))

            fmin = 0.
            xmin[0] = 0.

        if self.size > 0 and abs(xmin[0]) > self._rho_max * self._nselected:
            print(
                "nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                "nsources = {2:8.1f}".format(
                    self._rho_max, self._nselected, xmin[0]))

        pbest = dict(zip(self.params, xmin))

        # Separate over and under fluctuations.
        fmin *= -np.sign(pbest["nsources"])

        return fmin, pbest

    def fit_source_loc(self, src_ra, src_dec, size, seed, **kwargs):
        """Minimize the negative log-likelihood function around
        interesting position.

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
                self._nselected = self._select_events(x[0], x[1])
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

        if self.size > 0 and abs(xmin[0]) > self._rho_max * self._nselected:
            print(
                "nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                "nsources = {2:8.1f}".format(
                    self._rho_max, self._nselected, xmin[0]))

        pbest = dict(ra=xmin[0], dec=xmin[1])
        pbest.update(dict(zip(self.params, xmin[2:])))

        # Separate over and under fluctuations.
        fmin *= -np.sign(pbest["nsources"])

        # Clear cache.
        self._src_ra = np.nan
        self._src_dec = np.nan

        return fmin, pbest

    def do_trials(self, src_ra, src_dec, n_iter=int(1e5), mu_gen=None, ncpu=1,
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
        mu_gen : Optional[Injector]
            Inject additional events into the scrambled map.
        ncpu : int
            Number of processing units; a number larger than 1 enables
            multi-processing support.
        \*\*kwargs
            Parameters passed to `fit_source`

        Returns
        -------
        ndarray:
            Structured array containing number of injected events
            ``n_inj``, test statistic ``TS`` and best-fit values for
            `params` per trial

        """
        if mu_gen is None:
            mu_gen = itertools.repeat((0, None))

        inject = [mu_gen.next() for i in range(n_iter)]

        # Minimize negative log-likelihood function for every trial. In case of
        # multi-processing, each process needs its own sampling seed.
        if ncpu > 1 and n_iter > ncpu and ncpu <= multiprocessing.cpu_count():
            args = [(
                self, src_ra, src_dec, True, inject[i][1], kwargs,
                self.random.randint(2**32)) for i in range(n_iter)
                ]

            pool = multiprocessing.Pool(ncpu)
            results = pool.map(fs, args)

            pool.close()
            pool.join()
            del pool
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
        ts : float
            Test statistic that corresponds to the type I error with
            respect to the background test statics distribution
        beta : float
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
        mu : int
            Number of injected signal events to fulfill sensitivity
            criterion
        flux : float
            Sensitivity flux corresponding to `mu`
        weigths : ndarray
            Weight trials to a Poisson distribution with a mean of
            `flux`.
        trials : ndarray
            Structured array containing previous and all newly generated
            trials

        """
        print("Estimate sensitivity for declination {0:5.1f} deg.\n"
              "\tTS    = {1:6.2f}\n"
              "\tbeta  = {2:7.2%}\n".format(src_dec, ts, beta))

        if trials is None:
            dtype = [("n_inj", np.int), ("TS", np.float)]
            dtype.extend((p, np.float) for p in self.params)
            trials = np.empty((0, ), dtype=dtype)

        start = time.time()

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
        while not stop:
            bounds = np.percentile(
                trials["n_inj"][trials["n_inj"] > 0],
                q=[self._ub_perc, 100. - self._ub_perc])

            if bounds[0] == 1:
                bounds[0] = np.count_nonzero(trials["n_inj"] == 1) /\
                    np.sum(trials["n_inj"] < 2)

            print("\t\tEstimate sensitivity in region {0:5.1f} to "
                  "{1:5.1f}.".format(*bounds))

            def residual(n):
                return np.log10(utils.poisson_percentile(
                    n, trials["n_inj"], trials["TS"], ts)[0] - beta)**2

            seed = np.argmin(residual(n) for n in range(bounds[-1]))

            xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
                residual, [seed], bounds=[bounds], approx_grad=True)

            mu = np.asscalar(xmin)

            # Get the statistical uncertainty of the quantile.
            b, b_err = utils.poisson_percentile(
                mu, trials["n_inj"], trials["TS"], ts)

            print("\t\tBest estimate: {0:6.2f}, "
                  "({1:7.2%} +/- {2:8.3%})".format(mu, b, b_err))

            # If precision is high enough and fit did converge, the wanted
            # value is reached and we can stop the trial computation after
            # this iteration.
            stop = (
                b_err < eps and
                mu > bounds[0] and mu < bounds[-1] and
                np.fabs(b - beta) < eps
                )

            # To avoid a spiral with too few events, we want only half of
            # all events to be background scrambles after iterations.
            p_bckg = np.sum(trials["n_inj"] == 0) / len(trials)
            mu_min = np.log(1. / (1. - p_bckg))
            mu = np.amax([mu, mu_min])

            print("\t\tDo {0:6d} trials with mu = {1:6.2f} events.".format(
                  n_iter, mu))

            # Do trials with best estimate.
            trials = np.append(
                trials, self.do_trials(
                    src_ra, src_dec, n_iter,
                    mu_gen=inj.sample(src_ra, mu), **kwargs))

            sys.stdout.flush()

        flux = inj.mu2flux(mu)

        stop = time.time()
        mins, secs = divmod(stop - start, 60)
        hours, mins = divmod(mins, 60)

        print("Finished after {0:3d}h {1:2d}' {2:4.2f}''.\n"
              "\tInjected: {3:6.2f}\n"
              "\tFlux    : {4:.2e}\n"
              "\tTrials  : {5:6d}\n"
              "\tTime    : {6:6.2f} trial(s) / sec\n".format(
                  int(hours), int(mins), secs, mu, flux, len(trials),
                  len(trials) / (stop - start)))

        sys.stdout.flush()

        # weights = utils.poisson_weight(trials["n_inj"], flux)

        # return mu, flux, weights, trials
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
                np.count_zonzero(residuals > 0.)/len(residuals) > beta or
                np.all(residuals > 0.)
                )

        mu = len(trials) * beta
        print("\tActive region: {0:5.1f}\n".format(mu))

        # Do trials around active region.
        trials = np.append(
            trials, self.do_trials(
                src_ra, src_dec, n_iter,
                mu_gen=inj.sample(src_ra, mu), **kwargs))

        return trials


class GrbLLH(BaseLLH):
    """Log-likelihood function for gamma-ray burst analyses

    Parameters
    ----------
    data : Tuple[array_like]
        Experimental data in on and off-source time range
    livetime : Tuple[float]
        On and off-source time range
    \*\*kwargs
        Parameters passed to base class

    Attributes
    ----------
    data : Dict[str, ndarray]
        Experimental data in ``on`` and ``off``-source time range
    livetime : Dict[str, float]
        On and off-source time range in days
    llh_model : NullModel
        Likelihood model, derived from `ps_model.NullModel`
    nbackground : float
        Number of expected background events in on-source time range

    """
    def __init__(self, data, livetime, llh_model, **kwargs):
        self.data = {}
        self.livetime = {}

        for key, events, live in zip(("on", "off"), data, livetime):
            self.data[key] = numpy.lib.recfunctions.append_fields(
                events, names="B", data=llh_model.background(events),
                usemask=False)

            self.livetime[key] = live

        self.llh_model = llh_model

        self.nbackground = (
            self.livetime["on"] / self.livetime["off"] * self.data["off"].size
            )

        super(GrbLLH, self).__init__(**kwargs)

    def _select_events(self, src_ra=None, src_dec=None, scramble=True,
                       inject=None):
        """Select events for log-likelihood evaluation.

        Parameters
        ----------
        scramble : Optional[bool]
            If `scramble` is `True`, `nbackground` (plus Poisson
            fluctuations) events are selected from the off-source time
            range. Otherwise, the on-source events ``data["on"]`` are
            selected.
        inject : Optional[ndarray]
            Structured array containing additional events to append to
            selection

        Returns
        -------
        int:
            Number of selected events

        Note
        ----
        In the current implementation, the selection depends only on the
        on-source time range. Hence, `src_ra` and `src_dec` are ignored.

        Warnings
        --------
        Only set `scramble` to `False` if you want to unblind the data.

        """
        # We will chose new events, so it is time to clean the likelihood
        # model's cache.
        self.llh_model.reset()

        if scramble:
            N = self.random.poisson(self.nbackground)

            if N > 0:
                self._events = self.random.choice(self.data["off"], N)
                self._events["ra"] = self.random.uniform(0., 2.*np.pi, N)
            else:
                self._events = np.empty(0, dtype=self.data["off"].dtype)
        else:
            self._events = self.data["on"]

        if inject is not None:
            remove = np.logical_or(
                inject["sinDec"] < self.llh_model.sinDec_range[0],
                inject["sinDec"] > self.llh_model.sinDec_range[-1])

            if np.any(remove):
                inject = inject[np.logical_not(remove)]

            inject = numpy.lib.recfunctions.append_fields(
                inject, names="B", data=self.llh_model.backgkround(inject),
                usemask=False)

            self._events = np.append(self._events, inject)

        self._signal = self.llh_model.signal(src_ra, src_dec, self._events)

        return self._events.size

    def llh(self, nsources, **others):
        SoB = self._signal / self._events["B"]

        weights, wgrad = self.llh_model.weight(self._events, **others)
        x = SoB * weights

        ts = 2. * (-nsources + np.log1p(nsources / self.nbackground * x).sum())
        nsgrad = -1. + (x / (self.nbackground + x * nsources)).sum()

        if wgrad is not None:
            pgrad = np.sum(
                nsources / (self.nbackground + x*nsources) * SoB * wgrad,
                axis=-1)
        else:
            pgrad = np.zeros((0, ))

        grad = 2. * np.append(nsgrad, pgrad)

        return ts, grad

    llh.__doc__ = BaseLLH.llh.__doc__

    @property
    def params(self):
        """List[str]: Log-likelihood parameter names
        """
        return super(GrbLLH, self).params + self.llh_model.params.keys()

    @property
    def par_seeds(self):
        """ndarray: Log-likelihood parameter seeds
        """
        seeds = [self.llh_model.params[p][0] for p in self.params[1:]]
        return np.hstack((super(GrbLLH, self).par_seeds, seeds))

    @property
    def par_bounds(self):
        """ndarray: Lower and upper log-likelihood parameter bounds
        """
        bounds = [self.llh_model.params[p][1] for p in self.params[1:]]
        return np.vstack((super(GrbLLH, self).par_bounds, bounds))

    @property
    def size(self):
        """int: Total number of events contained in `data`
        """
        return self.data["on"].size + self.data["off"].size
