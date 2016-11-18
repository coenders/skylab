from __future__ import division

import abc
import itertools
import multiprocessing
import sys
import time

import numpy as np
import scipy.optimize

from . import utils


def fs(args):
    llh, src_ra, src_dec, inject, scramble, kwargs, seed = args

    if scramble:
        llh.random.RandomState(seed)

    return llh.fit_source(src_ra, src_dec, scramble, inject, **kwargs)


class BaseLLH(object):
    __metaclass__ = abc.ABCMeta

    _pgtol = 1e-3
    _rho_max = 0.95
    _ub_perc = 1.

    def __init__(self, nsource=15., nsource_rho=0.9, nsource_bounds=(0., 1e3)):
        self.nsource = nsource
        self.nsource_rho = nsource_rho
        self.nsource_bounds = nsource_bounds
        self.random = np.random.RandomState()

        self._n = 0
        self._N = 0

        self._src_ra = np.nan
        self._src_dec = np.nan

    @abc.abstractmethod
    def _select_events(self, src_ra, src_dec, scramble=False, inject=None):
        """Select events for log-likelihood evaluation.

        The method is intended to update the number of select events
        `_n` and the total number of events `_N` and cache the source
        position as `_src_ra` and `_src_dec`.

        """
        pass

    @abc.abstractmethod
    def llh(self, **fit_pars):
        """Evaluate log-likelihood function given the parameter values
        specified in `fit_pars`.

        Returns
        -------
        float:
            Log-likelihood for the given parameter values.

        """
        pass

    @property
    def params(self):
        """List[str]: Log-likelihood parameter names
        """
        return ["nsources"]

    @property
    def par_seeds(self):
        """ndarray: Log-likelihood parameter seeds.
        """
        if self._n > 0.:
            ns = min(self.nsource, self.nsource_rho * self._n)
        else:
            ns = self.nsource

        return np.atleast_1d(ns)

    @property
    def par_bounds(self):
        """ndarray: Lower and upper log-likelihood parameter bounds.
        """
        return np.atleast_1d(self.nsource_bounds)

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
            Scramble events prior to selection.
        inject : Optional[ndarray]
            Structured array containing additional events to append to
            selection.
        \*\*kwargs
            Parameters passed to the L-BFGS-B minimizer

        Returns
        -------
        fmin : float
            Minimal negative log-likelihood converted into the test
            statistic ``-sign(ns)*llh``.
        pbest : Dict[str, float]
            Parameters minimizing the negative log-likelihood function.

        """
        # Set all weights once for this source location, if not already cached.
        self._select_events(src_ra, src_dec, scramble, inject)

        if self._N < 1:
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

        if self._N > 0 and abs(xmin[0]) > self._rho_max * self._n:
            print(
                "nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                "nsources = {2:8.1f}".format(self._rho_max, self._n, xmin[0]))

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
            statistic ``-sign(ns)*llh``.
        pbest : Dict[str, float]
            Parameters minimizing the negative log-likelihood function.

        """
        def llh(x, *args):
            """Wrap log-likelihood to work with arrays and return the
            negative log-likelihood, which will be minimized.

            """
            # If the minimizer is testing a new position, different events have
            # to be selected.
            if not (x[0] == self._src_ra and x[1] == self._src_dec):
                self._select_events(x[0], x[1])

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

        if self._N > 0 and abs(xmin[0]) > self._rho_max * self._n:
            print(
                "nsources > {0:7.2%} * {1:6d} selected events, fit-value "
                "nsources = {2:8.1f}".format(self._rho_max, self._n, xmin[0]))

        pbest = dict(ra=xmin[0], dec=xmin[1])
        pbest.update(dict(zip(self.params, xmin[2:])))

        # Separate over and under fluctuations.
        fmin *= -np.sign(pbest["nsources"])

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
            Other keyword arguments are passed to `fit_source`.

        Returns
        -------
        ndarray:
            Structured array containing of number of injected events
            ``"n_inj"``, test statistic ``"TS"`` and best-fit parameters
            per trial.

        """
        if mu_gen is None:
            mu_gen = itertools.repeat(0, None)

        inject = [mu_gen.next() for i in range(n_iter)]

        # Minimize negative log-likelihood function for every trial. In case of
        # multi-processing, each process needs its own sampling seed.
        if ncpu > 1 and n_iter > ncpu and ncpu <= multiprocessing.cpu_count():
            args = [(
                self, src_ra, src_dec, True, inject[i][1], kwargs.items(),
                self.random.randint(2**32)) for i in range(n_iter)
                ]

            pool = multiprocessing.Pool(self.ncpu)
            results = pool.map(fs, args)

            pool.close()
            pool.join()
            del pool
        else:
            results = [
                self.fit_source(src_ra, src_dec, True, inject[i][1], **kwargs)
                for i in range(n_iter)
                ]

        dtype = [("n_inj", np.int), ("TS", np.float)] +\
            [(p, np.float) for p in self.params]

        trials = np.empty((n_iter, ), dtype=dtype)

        for i in range(n_iter):
            trials["n_inj"][i] = inject[i][0]
            trials["TS"][i] = results[i][0]

            for key in results[i][1]:
                trials[key][i] = results[i][1][key]

        return trials

    def weighted_sensitivity(self, src_ra, src_dec, alpha, beta, inj, mc,
                             n_bckg=int(1e5), n_iter=1000, eps=5e-3, fit=None,
                             **kwargs):
        """Calculate sensitivity for a given source hypothesis.

        All trials calculated are used at each step and weighted using
        the Poisson probability.

        Parameters
        ----------
        src_ra : float
            Right ascension of source position
        src_dec : float
            Declination of source position
        alpha : array_like
            Error of first kind
        beta : array_like
            Error of second kind
        inj : Injector
            Inject additional events into the scrambled map.
        mc : array_like
            Monte Carlo events to use for injection; needs all fields
            that are stored in experimental data, plus true information
            that the injector uses:
            ``"trueRa"``, ``"trueDec"``, ``"trueE"``, ``"ow"``.
        n_bckg : Optional[int]
            Number of background trials to create if `fun` is `None`
        n_iter : Optional[int]
            Number of trials to create per iteration
        eps : Optional[float]
            Precision for breaking point
        fit : Optional[object]
            Parametrization of background test statistic distribution;
            object should have an inverse survival function `isf`.
        TSval : Optional[array_like]
            TS value to use for calculation; if not given
            `fun.isf(alpha)` is used; if `fun` is `None`, background
            trials are created and fitted with a chi-square
            distribution; makes alpha obsolete.
        \*\*kwargs
            Other keyword arguments are passed to `do_trials`.

        Returns
        -------
        Dict[str, array_like]:
            Dictionary containing the flux ``"flux"`` needed to reach
            sensitivity given by `alpha` and `beta`, the number ``"mu"``
            of injected events corresponding to the flux; the background
            test statistic ``"TS"`` corresponding to `alpha`, and a
            structured array ``"trials"`` containing all generated
            trials.

        """
        if fit is not None and not hasattr(fit, "isf"):
            raise AttributeError("fit must have attribute 'isf(alpha)'!")

        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        tsval = np.atleast_1d(kwargs.pop("TSval", [None for i in alpha]))

        if not (len(alpha) == len(beta) == len(tsval)):
            raise ValueError(
                "alpha, beta, and (if given) TSval must have same length!")

        # Setup source injector.
        inj.fill(src_dec, mc, self.livetime)

        print("Estimate sensitivity for declination {0:5.1f} deg.".format(
              np.degrees(src_dec)))

        def do_estimation(ts, beta, trials):
            print("\tTS    = {0:6.2f}".format(ts))
            print("\tbeta  = {0:7.2%}\n".format(beta))

            # If no events have been injected, do a quick an estimation of
            # active region by doing a few trials.
            if (len(trials) < 1 or
                    not np.any(trials["n_inj"] > 0) or
                    not np.any(trials["TS"][trials["n_inj"] > 0] > 2.*ts)):

                if len(trials) > 0:
                    n_inj = int(np.mean(trials["n_inj"]))
                else:
                    n_inj = 0

                print("Quick estimate of active region, inject increasing "
                      "number of events, starting with {0:d} "
                      "events...".format(n_inj + 1))

                more_trials = True

                while more_trials:
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

                    more_trials = (
                        np.count_zonzero(residuals > 0.)/len(residuals) <= beta
                        or np.any(residuals <= 0.)
                        )

                mu_eff = len(trials) * beta
                print("\tActive region: {0:5.1f}\n".format(mu_eff))

                # Do trials around active region.
                trials = np.append(
                    trials, self.do_trials(
                        src_ra, src_dec, n_iter,
                        mu_gen=inj.sample(src_ra, mu_eff), **kwargs))

            # Fit closest point to beta value; use existing scrambles to
            # determine a seed for the minimization; restrict fit to region
            # where sampled before.
            stop = False
            while not stop:
                bounds = np.percentile(
                    trials["n_inj"][trials["n_inj"] > 0],
                    q=[self._ub_perc, 100. - self._ub_perc])

                if bounds[0] == 1:
                    bounds[0] = np.count_nonzero(trials["n_inj"] == 1) /\
                        np.sum(trials["n_inj"] < 2)

                print("\tEstimate sensitivity in region {0:5.1f} to "
                      "{1:5.1f}".format(*bounds))

                def residual(n):
                    return np.log10(utils.poisson_percentile(
                        n, trials["n_inj"], trials["TS"], ts)[0] - beta)**2

                seed = np.argmin(residual(n) for n in range(bounds[-1]))

                xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
                    residual, [seed], bounds=[bounds], approx_grad=True)

                mu_eff = np.asscalar(xmin)

                # Get the statistical uncertainty of the quantile.
                b, b_err = utils.poisson_percentile(
                    mu_eff, trials["n_inj"], trials["TS"], ts)

                print("\t\tBest estimate: {0:6.2f}, "
                      "({1:7.2%} +/- {2:8.3%})".format(mu_eff, b, b_err))

                # If precision is high enough and fit did converge, the wanted
                # value is reached and we can stop the trial computation after
                # this iteration.
                stop = (
                    b_err < eps and
                    mu_eff > bounds[0] and mu_eff < bounds[-1] and
                    np.fabs(b - beta) < eps)

                # To avoid a spiral with too few events, we want only half of
                # all events to be background scrambles after iterations.
                p_bckg = np.sum(trials["n_inj"] == 0) / len(trials)
                mu_eff_min = np.log(1. / (1. - p_bckg))
                mu_eff = np.amax([mu_eff, mu_eff_min])

                print("\tDo {0:6d} trials with mu = {1:6.2f} events".format(
                      n_iter, mu_eff))

                # Do trials with best estimate.
                trials = np.append(
                    trials, self.do_trials(
                        src_ra, src_dec, n_iter,
                        mu_gen=inj.sample(src_ra, mu_eff), **kwargs))

                sys.stdout.flush()

            return mu_eff, trials

        TS = []
        mu_flux = []
        flux = []

        dtype = [("n_inj", np.int), ("TS", np.float)] +\
            [(p, np.float) for p in self.params]

        trials = np.empty((0, ), dtype=dtype)

        # Calculate number of injected events needed so that beta percent of
        # the trials have a test statistic above the alpha percentile  of the
        # background test statistic distribution (test statistic larger than
        # tsval). Calculate tsval if not given, using the distribution fun or
        # based on background scrambles.
        start = time.time()

        for ts, a, b in zip(tsval, alpha, beta):
            if ts is None:
                if fit is None:
                    print("\tDo background scrambles for estimation of TS "
                          "value for alpha = {0:7.2%}".format(a))

                    trials = np.append(
                        trials, self.do_trials(
                            src_ra, src_dec, n_iter=n_bckg, **kwargs))

                    stop = time.time()
                    mins, secs = divmod(stop - start, 60)
                    hours, mins = divmod(mins, 60)

                    print("\t{0:6d} background scrambles finished after "
                          "after {1:3d}h {2:2d}' {3:4.2f}''.".format(
                            len(trials), int(hours), int(mins), secs))

                    print("Fit background function to scrambles.")

                    if self.par_bounds[0][0] < 0:
                        print("Fit two sided chi2 to background scrambles.")
                        fitfunc = utils.twoside_chi2
                    else:
                        print("Fit delta chi2 to background scrambles.")
                        fitfunc = utils.delta_chi2

                    fit = fitfunc(
                        trials["TS"][trials["n_inj"] == 0],
                        df=2., floc=0., fscale=1.)

                    print(fit)

                ts = np.asscalar(fit.isf(a))

            mu, trials = do_estimation(ts, b, trials)

            TS.append(ts)
            mu_flux.append(mu)
            flux.append(inj.mu2flux(mu))

            stop = time.time()
            mins, secs = divmod(stop - start, 60)
            hours, mins = divmod(mins, 60)

            print("\tFinished after {0:3d}h {1:2d}' {2:4.2f}''".format(
                  int(hours), int(mins), secs))

            print("\t\tInjected: {0:6.2f}".format(mu))
            print("\t\tFlux    : {0:.2e}".format(flux[-1]))
            print("\t\tTrials  : {0:6d}".format(len(trials)))
            print("\t\tTime    : {0:6.2f} trial(s) / sec\n".format(
                  len(trials) / (stop - start)))

            sys.stdout.flush()

        weights = np.vstack([
            utils.poisson_weight(trials["n_inj"], f) for f in mu_flux
            ])

        result = dict(
            flux=flux, mu=mu_flux, TSval=TS, alpha=alpha, beta=beta, fit=fit,
            trials=trials, weights=weights)

        return result
