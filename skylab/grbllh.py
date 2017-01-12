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
from __future__ import division

import numpy as np
import numpy.lib.recfunctions

from . import basellh


class GRBLlh(basellh.BaseLLH):
    r"""Log-likelihood function for gamma-ray burst analyses

    Parameters
    ----------
    data : tuple(ndarray)
        Experimental data in on and off-source time range
    livetime : tuple(float)
        On and off-source time range
    \*\*kwargs
        Parameters passed to base class

    Attributes
    ----------
    data : dict(str, ndarray)
        Experimental data in ``on`` and ``off``-source time range
    livetime : dict(str, float)
        On and off-source time range in days
    llh_model : NullModel
        Likelihood model, derived from `ps_model.NullModel`
    nbackground : float
        Number of expected background events in on-source time range

    See Also
    --------
    basellh.BaseLLH

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

        self._events = None
        self._signal = None

        super(GRBLlh, self).__init__(**kwargs)

    def _select_events(self, src_ra, src_dec, scramble=False, inject=None):
        r"""Select events for log-likelihood evaluation.

        If `scramble` is `True`, `nbackground` (plus Poisson
        fluctuations) events are selected from the off-source time
        range. Otherwise, the on-source events ``data["on"]`` are
        selected.

        Note
        ----
        In the current implementation, the selection depends only on the
        on-source time range. Hence, `src_ra` and `src_dec` are ignored.

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

        # Method has to set number of events and number of selected
        # events. Here, both numbers are equal.
        self._nevents = self._events.size
        self._nselected = self._nevents

    def llh(self, nsources, **others):
        SoB = self._signal / self._events["B"]

        weights, wgrad = self.llh_model.weight(self._events, **others)
        x = SoB * weights

        # Multiply by two for chi-square distributed test-statistic.
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
        r"""list(str): Log-likelihood parameter names
        """
        return super(GRBLlh, self).params + self.llh_model.params.keys()

    @property
    def par_seeds(self):
        r"""ndarray: Log-likelihood parameter seeds
        """
        seeds = [self.llh_model.params[p][0] for p in self.params[1:]]
        return np.hstack((super(GRBLlh, self).par_seeds, seeds))

    @property
    def par_bounds(self):
        r"""ndarray: Lower and upper log-likelihood parameter bounds
        """
        bounds = [self.llh_model.params[p][1] for p in self.params[1:]]
        return np.vstack((super(GRBLlh, self).par_bounds, bounds))

    @property
    def sinDec_range(self):
        r"""ndarray: Lower and upper allowed sine declination
        """
        return np.array(self.llh_model.sinDec_range)
