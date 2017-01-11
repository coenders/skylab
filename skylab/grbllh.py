from __future__ import division

import numpy as np
import numpy.lib.recfunctions

from . import basellh


class GRBLlh(basellh.BaseLLH):
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

        super(GRBLlh, self).__init__(**kwargs)

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
        int
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
                inject, names="B", data=self.llh_model.background(inject),
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

    llh.__doc__ = basellh.BaseLLH.llh.__doc__

    @property
    def params(self):
        """List[str]: Log-likelihood parameter names
        """
        return super(GRBLlh, self).params + self.llh_model.params.keys()

    @property
    def par_seeds(self):
        """ndarray: Log-likelihood parameter seeds
        """
        seeds = [self.llh_model.params[p][0] for p in self.params[1:]]
        return np.hstack((super(GRBLlh, self).par_seeds, seeds))

    @property
    def par_bounds(self):
        """ndarray: Lower and upper log-likelihood parameter bounds
        """
        bounds = [self.llh_model.params[p][1] for p in self.params[1:]]
        return np.vstack((super(GRBLlh, self).par_bounds, bounds))

    @property
    def size(self):
        """int: Total number of events contained in `data`
        """
        return self.data["on"].size + self.data["off"].size

    @property
    def sinDec_range(self):
        """Tuple[float]: Lower and upper allowed sine declination
        """
        return self.llh_model.sinDec_range
