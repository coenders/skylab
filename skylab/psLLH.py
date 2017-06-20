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

import logging
import warnings

import numpy as np
import numpy.lib.recfunctions

from . import basellh


class PointSourceLLH(basellh.BaseLLH):
    r"""Log-likelihood function for single point source analyses

    Handles experimental data for one event sample, calculating the
    unbinned point source Likelihood

    .. math::

        \mathcal{L} = \prod_\limits_{i = 1}^{N}
            \left(
                \frac{n_s}{N}\mathcal{S} +
                \left(1 - \frac{n_s}{N}\right)\mathcal{B}
            \right)

    Parameters
    ----------
    mc : ndarray
        Structured array describing the simulated data; same fields as
        `exp` plus true right ascension ``trueRa``, true declination
        ``trueDec``, true energy ``trueE``, and weight ``ow``. It is
        passed to the likelihood model `llh_model``.
    scramble : bool, optional
        Scramble experimental data in right ascension.
    \*\*kwargs
        Parameters passed to base class

    Attributes
    ----------
    exp : ndarray
        Structured array describing the experimental data; for a
        likelihood model without energy weights, the essential fields
        are right ascension ``ra``, declination ``dec`` or sine
        declination ``sinDec``, and angular uncertainty ``sigma``.
    livetime : float
        Livetime of experimental data in days
    llh_model : NullModel
        Likelihood model derived from `ps_model.NullModel`
    mode : {"all", "band", "box"}
        Select either all events, events in the declination band of the
        source, or events in a box around around the source.
    delta_ang : float
        Boundaries of declination selection band or box
    thresh_S : float
        Select only events with a signal probability above `thresh_S`.

    See Also
    --------
    basellh.BaseLLH

    """
    # The log-likelihood function will be taylor-expanded around this treshold
    # value; see llh method.
    _aval = 1e-3

    def __init__(self, exp, mc, livetime, llh_model, scramble=True, mode="box",
                 delta_ang=np.deg2rad(10.), thresh_S=0., **kwargs):
        super(PointSourceLLH, self).__init__(**kwargs)

        # Add sine declination to experimental data if not available.
        if "sinDec" not in exp.dtype.fields:
            exp = numpy.lib.recfunctions.append_fields(
                exp, names="sinDec", data=np.sin(exp["dec"]),
                dtypes=np.float, usemask=False)

        # Scramble experimental data in right ascension if data was not
        # unblinded yet.
        if scramble:
            exp["ra"] = self.random.uniform(0., 2.*np.pi, exp.size)
        else:
            logger = logging.getLogger(self._logname)
            logger.warn("Working on >> UNBLINDED << data.")

        self.exp = exp
        self.livetime = livetime
        self.set_llh_model(llh_model, mc=mc)

        # Calculate background probability here because it does not change.
        self.exp = numpy.lib.recfunctions.append_fields(
            self.exp, names="B", data=llh_model.background(exp), usemask=False)

        self.mode = mode
        self.delta_ang = delta_ang
        self.thresh_S = thresh_S

        self._events = None
        self._signal = None

    @classmethod
    def upscale(cls, exp, livetime, llh_model, **kwargs):
        r"""Up-scale experimental data to new livetime.

        Parameters
        ----------
        livetime : tuple(float)
            Old and new livetime of experimental data

        Returns
        -------
        PointSourceLLH
            Instance of basic point source likelihood

        """
        logname = "{0:s}.{1:s}".format(cls.__module__, cls.__name__)
        logger = logging.getLogger(logname)

        logger.info("Up-scale experimental data distribution.")

        # Draw number of events for up-scaling from Poisson distribution.
        scale = livetime[1] / livetime[0]
        random = np.random.RandomState(kwargs.get("seed", "None"))
        nevents = random.poisson(scale * exp.size)

        logger.info(
            "Sample {0:d} events from experimental data from {1:.1f} to "
            "{2:.1f} days (x {3:.1f}).".format(
                nevents, livetime[1], livetime[0], scale))

        # Over-sample experimental data.
        exp = random.choice(exp, size=nevents)

        # Up-scaling only makes sense using blinded experimental data.
        kwargs["scramble"] = True
        likelihood = cls(exp, livetime[1], llh_model, **kwargs)
        likelihood.random = random

        return likelihood

    def __str__(self):
        lines = [repr(self)]
        lines.append(67 * "-")
        lines.append("Number of Data Events: {0:7d}".format(self.exp.size))

        if self.exp.size > 0:
            srange = (
                np.rad2deg(np.arcsin(np.amin(self.exp["sinDec"]))),
                np.rad2deg(np.arcsin(np.amax(self.exp["sinDec"])))
                )

            if "logE" in self.exp.dtype.fields:
                erange = (
                    np.amin(self.exp["logE"]),
                    np.amax(self.exp["logE"])
                    )
        else:
            srange = (np.nan, np.nan)
            erange = (np.nan, np.nan)

        lines.append("\tDeclination Range  : {0:6.1f} - {1:6.1f} deg".format(
            *srange))
        lines.append("\tlog10 Energy Range : {0:6.1f} - {1:6.1f}".format(
            *erange))
        lines.append("\tLivetime of sample : {0:7.2f} days".format(
            self.livetime))

        lines.append(67 * "-")

        if self.mode == "all":
            lines.append("Using all events")
        else:
            lines.append("Selected Events - {0:8s}: {1:7d}".format(
                self.mode, self._nselected))

        lines.append(67 * "-")
        lines.append("Likelihood model:")

        lines.append("\n\t".join(
            l if len(set(l)) > 2 else l[:-len("\t".expandtabs())]
            for l in str(self.llh_model).splitlines()))

        lines.append("Fit Parameter\tSeed\tBounds")

        lines.extend(
            "{0:15s}\t{1:.2f}\t{2:.2f} to {3:.2f}".format(p, s, *b)
            for p, s, b in zip(self.params, self.par_seeds, self.par_bounds)
            )

        lines.append(67 * "-")

        return "\n".join(lines)

    def _select_events(self, src_ra, src_dec, scramble=False, inject=None):
        r"""Select events for log-likelihood evaluation.

        Depending on `mode`, select either all events, events in the
        declination band of the source, or events in a box around around
        the source.

        """
        # We will chose new events, so it is time to clean the likelihood
        # model's cache.
        self.llh_model.reset()

        # Select either all events or the ones within the declination band.
        self._nevents = self.exp.size

        if self.mode == "all":
            mask = np.ones_like(self.exp["sinDec"], dtype=np.bool)
        elif self.mode in ["band", "box"]:
            dmin = max(-np.pi/2., src_dec - self.delta_ang)
            dmax = min(src_dec + self.delta_ang, np.pi/2.)

            mask = np.logical_and(
                self.exp["sinDec"] > np.sin(dmin),
                self.exp["sinDec"] < np.sin(dmax))
        else:
            raise ValueError("Not supported mode: {0:s}".format(self.mode))

        events = self.exp[mask]

        if scramble:
            events["ra"] = self.random.uniform(0., 2.*np.pi, size=events.size)

        if self.mode == "box":
            # Select events inside right ascension box: the solid angle is a
            # function of declination, i.e., for a fixed solid angle, the right
            # ascension value has to change with declination.
            dra = np.fabs(
                np.mod(events["ra"] - src_ra + np.pi, 2.*np.pi) - np.pi)

            cosmin = np.amin(np.cos([dmin, dmax]))
            dphi = np.amin([2.*np.pi, 2.*self.delta_ang/cosmin])
            events = events[dra < dphi/2.]

        if inject is not None:
            inject = numpy.lib.recfunctions.append_fields(
                inject, names="B", data=self.llh_model.background(inject),
                usemask=False)

            events = np.append(events, inject)
            self._nevents += inject.size

        signal = self.llh_model.signal(src_ra, src_dec, events)

        # Ignore events with a signal probability below threshold.
        mask = signal > self.thresh_S
        self._events = events[mask]
        self._signal = signal[mask]

        self._nselected = self._events.size

        if (self._nselected < 1 and
                np.sin(self._src_dec) < self.sinDec_range[0] and
                np.sin(self._src_dec) > self.sinDec_range[-1]):
            warnings.warn(
                "No event was selected, fit will go to -inf.",
                RuntimeWarning)

    def llh(self, nsources, **others):
        SoB = self._signal / self._events["B"]

        weights, wgrad = self.llh_model.weight(self._events, **others)
        x = (SoB*weights - 1.) / self._nevents

        alpha = nsources * x
        ts = np.empty_like(alpha, dtype=np.float)

        # Taylor-expand likelihood function and gradients around threshold
        # alpha value in order to avoid divergences.
        aval = -1. + self._aval
        mask = alpha > aval

        ts[mask] = np.log1p(alpha[mask])

        arel = (alpha[~mask] - aval) / self._aval
        ts[~mask] = np.log1p(aval) + arel - arel**2 / 2.

        ts = ts.sum()

        nsgrad = np.empty_like(alpha, dtype=np.float)
        nsgrad[mask] = x[mask] / (1. + alpha[mask])
        nsgrad[~mask] = x[~mask] * (1. - arel) / self._aval
        nsgrad = nsgrad.sum()

        if self._nevents > self._nselected:
            ndiff = self._nevents - self._nselected
            ts += ndiff * np.log1p(-nsources / self._nevents)
            nsgrad -= ndiff / (self._nevents - nsources)

        if wgrad is not None:
            pgrad = SoB * wgrad / self._nevents
            pgrad[:, mask] *= nsources / (1. + alpha[mask])
            pgrad[:, ~mask] *= nsources * (1. - arel) / self._aval
            pgrad = pgrad.sum(axis=-1)

        else:
            pgrad = np.zeros((0,))

        # Multiply by two for chi-square distributed test-statistic.
        ts *= 2.
        grad = 2. * np.append(nsgrad, pgrad)

        return ts, grad

    llh.__doc__ = basellh.BaseLLH.llh.__doc__

    def set_llh_model(self, model, mc=None):
        r"""Set ``llh_model`` to new likelihood model.

        Parameters
        ----------
        model : NullModel
            Likelihood model derived from `ps_model.NullModel`
        mc : ndarray, optional
            Structured array describing the simulated data; same fields
            as `exp` plus true right ascension ``trueRa``, true
            declination ``trueDec``, true energy ``trueE``, and weight
            ``ow``. It is passed to the likelihood model `llh_model``.

        """
        if mc is not None:
            model(self.exp, mc, livetime=self.livetime)

        self.llh_model = model

    @property
    def params(self):
        r"""list(str): Log-likelihood parameter names
        """
        params = self.llh_model.params.keys()
        return super(PointSourceLLH, self).params + params

    @property
    def par_seeds(self):
        r"""ndarray: Log-likelihood parameter seeds
        """
        seeds = [self.llh_model.params[p][0] for p in self.params[1:]]
        return np.hstack((super(PointSourceLLH, self).par_seeds, seeds))

    @property
    def par_bounds(self):
        r"""ndarray: Lower and upper log-likelihood parameter bounds
        """
        bounds = list(super(PointSourceLLH, self).par_bounds)
        bounds.extend(self.llh_model.params[p][1] for p in self.params[1:])
        return np.vstack(bounds)

    @property
    def sinDec_range(self):
        r"""tuple(float): Lower and upper allowed sine declination
        """
        return self.llh_model.sinDec_range


class MultiPointSourceLLH(basellh.BaseLLH):
    r"""Log-likelihood function for multi-year point source analyses

    Handles multiple event samples that are distinct of each other.
    Different samples have different effective areas that have to be
    taken into account for parting the number of expected neutrinos in
    between the different samples. Each sample is represented as an
    instance of `PointSourceLLH`.

    Arguments and keyword arguments are passed to `basellh.BaseLLH`.

    See Also
    --------
    basellh.BaseLLH

    """
    def __init__(self, *args, **kwargs):
        super(MultiPointSourceLLH, self).__init__(*args, **kwargs)
        self._enums = {}
        self._samples = {}

    def __str__(self):
        lines = [repr(self)]
        lines.append(67 * "=")
        lines.append("Number of samples: {0:2d}".format(len(self)))

        if len(self) > 0:
            lines.append("\t{0:>2s} {1:>10s} {2:>8s} {3:>6s}".format(
                "#", "Name", "Livetime", "Events"))

        nevents = 0
        livetime = 0

        for enum in sorted(self._enums.keys()):
            n = self._samples[enum].exp.size
            tlive = self._samples[enum].livetime

            lines.append("\t{0:2d} {1:>10s} {2:8.2f} {3:6d}".format(
                enum, self._enums[enum], tlive, n))

            nevents += n
            livetime += tlive

        lines.append("Number of events: {0:6d}".format(nevents))
        lines.append("Total livetime  : {0:9.2f}".format(livetime))
        lines.append(67 * "-")

        for enum in sorted(self._enums.keys()):
            lines.append("Dataset {0:2d}\n".format(enum))
            lines.append(67 * "-")

            lines.append("\n\t".join(
                l if len(set(l)) > 2 else l[:-len("\t".expandtabs())]
                for l in str(self._samples[enum]).splitlines()))

            lines.append(67 * "-")

        lines.append("Fit Parameter\tSeed\tBounds")

        lines.extend(
            "{0:15s}\t{1:.2f}\t{2:.2f} to {3:.2f}".format(p, s, *b)
            for p, s, b in zip(self.params, self.par_seeds, self.par_bounds)
            )

        lines.append(67 * "-")

        return "\n".join(lines)

    def __len__(self):
        r"""Return number of event samples

        """
        return len(self._enums)

    def add_sample(self, name, llh):
        r"""Add log-likelihood function object.

        Parameters
        -----------
        name : str
            Name of event sample
        llh : PointSourceLLH
            Log-likelihood function using single event sample

        """
        if not isinstance(llh, PointSourceLLH):
            raise ValueError("'{0}' is not LLH-style".format(llh))

        names = self._enums.values()

        if name in names:
            enum = self._enums.keys()[names.index(name)]
            logger = logging.getLogger(self._logname)
            logger.warn("Overwrite {0:d} - {1}".format(enum, name))
        else:
            if len(names) > 0:
                enum = max(self._enums) + 1
            else:
                enum = 0

        self._enums[enum] = name
        self._samples[enum] = llh

    def _select_events(self, src_ra, src_dec, scramble=False, inject=None):
        self._nevents = 0
        self._nselected = 0

        for enum in self._samples:
            if isinstance(inject, dict):
                events = inject.pop(enum, None)
            else:
                events = inject

            self._samples[enum]._select_events(
                src_ra, src_dec, scramble=scramble, inject=events)

            self._nevents += self._samples[enum]._nevents
            self._nselected += self._samples[enum]._nselected

        # Cache source position because it is used when evaluating the
        # log-likelihood functions.
        self._src_ra = src_ra
        self._src_dec = src_dec

    _select_events.__doc__ = basellh.BaseLLH._select_events.__doc__

    def llh(self, nsources, **others):
        r"""Evaluate log-likelihood function given the source strength
        `nsources` and the parameter values specified in `others`.

        The final log-likelihood is the sum over all log-likelihood
        functions. The source strength is distributed between the
        event samples according to their effective area at the
        evaluated declination.

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
        nsamples = len(self._enums)
        weights = np.empty(nsamples, dtype=np.float)
        wgrad = np.zeros((nsamples, len(self.params) - 1), dtype=np.float)

        for i, enum in enumerate(self._samples):
            weights[i], grad = self._samples[enum].llh_model.effA(
                self._src_dec, **others)

            if grad is None:
                continue

            for j, param in enumerate(self.params[1:]):
                if param not in grad:
                    continue

                wgrad[i, j] = grad[param]

        # Normalize weights to one.
        wgrad /= weights.sum()
        weights /= weights.sum()

        # Normalized sum is bound to one; gradients need to account for this
        # boundary by cross-talk.
        wgrad -= weights[..., np.newaxis] * np.sum(wgrad, axis=0)[np.newaxis]

        ts = 0.
        grad = np.zeros_like(self.params, dtype=np.float)

        for i, enum in enumerate(self._samples):
            fval, fgrad = self._samples[enum].llh(
                nsources * weights[i], **others)

            ts += fval
            grad[0] += fgrad[0] * weights[i]

            for j, param in enumerate(self.params[1:]):
                grad[j + 1] += fgrad[0] * nsources * wgrad[j]

                if param not in self._samples[enum].params:
                    continue

                grad[j + 1] += fgrad[self._samples[enum].params.index(param)]

        return ts, grad

    @property
    def livetime(self):
        r"""dict(int, float): Livetime of experimental data
        """
        return {enum: self._samples[enum].livetime for enum in self._samples}

    @property
    def params(self):
        r"""list(str): Log-likelihood parameter names
        """
        # We need to minimize over all parameters given by any likelihood;
        # gamma will be always minimized over, because it is used in the
        # weighting.
        params = set(
            p for e in self._samples
            for p in self._samples[e].llh_model.params)

        return super(MultiPointSourceLLH, self).params + list(params)

    @property
    def par_seeds(self):
        r"""ndarray: Log-likelihood parameter seeds
        """
        # Use seeds' median for all parameters.
        seeds = [
            np.median([
                self._samples[e].llh_model.params[p][0]
                for e in self._samples
                if p in self._samples[e].llh_model.params
                ])
            for p in self.params[1:]
            ]

        return np.hstack((super(MultiPointSourceLLH, self).par_seeds, seeds))

    @property
    def par_bounds(self):
        r"""ndarray: Lower and upper log-likelihood parameter bounds
        """
        bounds = list(super(MultiPointSourceLLH, self).par_bounds)

        # Use tightest parameter bounds.
        pbounds = [
            np.vstack([
                self._samples[e].llh_model.params[p][1]
                for e in self._samples
                if p in self._samples[e].llh_model.params
                ])
            for p in self.params[1:]
            ]

        bounds.extend((np.amax(b[:, 0]), np.amin(b[:, 1])) for b in pbounds)
        return np.vstack(bounds)

    @property
    def sinDec_range(self):
        r"""ndarray: Lower and upper allowed sine declination
        """
        srange = [s.llh_model.sinDec_range for s in self._samples.itervalues()]
        srange.append(super(MultiPointSourceLLH, self).sinDec_range)
        srange = np.vstack(srange)
        return np.array([np.amin(srange[:, 0]), np.amax(srange[:, 1])])
