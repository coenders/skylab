# -*-coding:utf8-*-

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

import abc
import logging

import numpy as np
import numpy.lib.recfunctions
import scipy.interpolate

from . import utils


def rotate_struct(ev, ra, dec):
    r"""Wrapper around `utils.rotated` for structured arrays

    Parameters
    ----------
    ev : ndarray
        Structured array describing events that will be rotated
    ra : float
        Right ascension of direction events will be rotated on
    dec : float
        Declination of direction events will be rotated on

    Returns
    --------
    ndarray:
        Structured array describing rotated events; true information are
        deleted.

    """
    rot = np.copy(ev)

    rot["ra"], rot_dec = utils.rotate(
        ev["trueRa"], ev["trueDec"], ra * np.ones(len(ev)),
        dec * np.ones(len(ev)), ev["ra"], np.arcsin(ev["sinDec"]))

    if "dec" in ev.dtype.names:
        rot["dec"] = rot_dec

    rot["sinDec"] = np.sin(rot_dec)

    # Delete Monte Carlo information from sampled events.
    mc = ["trueRa", "trueDec", "trueE", "ow"]

    return numpy.lib.recfunctions.drop_fields(rot, mc)


class Injector(object):
    r"""Base class for signal injectors

    Derived classes must implement the methods:

        * `fill`
        * `flux2mu`,
        * `mu2flux`, and
        * `sample`.

    The base constructor declares the private attribute `_logging` that
    can used to log messages.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # Use module name and class name of derived class for logging.
        self._logging = logging.getLogger("{0:s}.{1:s}".format(
            self.__class__.__module__, self.__class__.__name__))

    @abc.abstractmethod
    def fill(self, *args, **kwargs):
        r"""Fill injector with sample(s) to draw events from.

        """
        pass

    @abc.abstractmethod
    def flux2mu(self, flux):
        r"""Convert flux to mean number of expected events.

        Parameters
        ----------
        flux : float
            Neutrino flux

        Returns
        -------
        float:
            Mean number of expected neutrino events given `flux`

        """
        pass

    @abc.abstractmethod
    def mu2flux(self, mu):
        r"""Calculate source flux given `mu`.

        Parameters
        ----------
        mu : float
            Mean number of source neutrino events

        Returns
        -------
        float:
            Source flux given `mu`

        """
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        r"""Generator method that returns sampled events.

        """
        pass


class PointSourceInjector(Injector):
    r"""Single point source injector

    The source's energy spectrum follows a power law.

    .. math::

        \frac{\mathrm{d}\Phi}{\mathrm{d}E} =
            \Phi_{0} E_{0}^{2 - \gamma}
            \left(\frac{E}{E_{0}}\right)^{-\gamma}

    By this definiton, the flux is equivalent to a power law with a
    spectral index of two at the normalization energy ``E0*GeV``GeV.
    The flux is given in units GeV^{\gamma - 1} s^{-1} cm^{-2}.

    Parameters
    ----------
    seed : int, optional
        Random seed initializing the pseudo-random number generator.

    Attributes
    -----------
    gamma : float
        Spectral index; use positive values for falling spectrum.
    sinDec_range : tuple(float)
        Shrink allowed declination range.
    sinDec_bandwith : float
        Select events inside declination band around source position.
    src_dec : float
        Declination of source position
    e_range : tuple(float)
        Select events only in a certain energy range.
    random : RandomState
        Pseudo-random number generator

    """
    def __init__(self, gamma, sinDec_range=(-1., 1.), sinDec_bandwidth=0.1,
                 e_range=(0., np.inf), E0=1., GeV=1., seed=None):
        super(PointSourceInjector, self).__init__()

        self.gamma = gamma

        self.sinDec_range = sinDec_range
        self.sinDec_bandwidth = sinDec_bandwidth
        self.src_dec = np.nan

        self.e_range = e_range

        self.E0 = E0
        self.GeV = GeV

        self.random = np.random.RandomState(seed)

    def __str__(self):
        lines = [repr(self)]
        lines.append(67 * "-")

        lines.append(
            "\tSpectral index     : {0:6.2f}\n"
            "\tSource declination : {1:5.1f} deg\n"
            "\tlog10 Energy range : {2:5.1f} to {3:5.1f}".format(
                self.gamma, np.degrees(self.src_dec), *self.e_range))

        lines.append(67 * "-")

        return "\n".join(lines)

    def _setup(self):
        r"""Reset solid angle and declination band.

        """
        A, B = self.sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # Solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

    def _weights(self):
        r"""Setup weights for assuming a power-law flux.

        """
        # Weights given in days; weighted to the point source flux
        self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega
        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        # Normalize weights.
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # Double-check if no weight is dominating the sample.
        if self._norm_w.max() > 0.1:
            self._logging.warn("Maximal weight exceeds 10%: {0:.2%}".format(
                self._norm_w.max()))

    def fill(self, src_dec, mc, livetime):
        r"""Fill injector with Monte Carlo events, selecting events
        around the source position.

        Parameters
        -----------
        src_dec : float
            Declination of source position
        mc : ndarray, dict(enum, ndarray)
            Either structured array describing Monte Carlo events or a
            mapping of `enum` to such arrays
        livetime : float, dict(enum, float)
            Livetime per sample

        Raises
        ------
        TypeError
            If `mc` and `livetime` are not of the same type.

        See Also
        --------
        psLLH.MultiPointSourceLLH

        """
        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise TypeError("mc and livetime are not compatible.")

        # Reset solid angle and declination band.
        self.src_dec = src_dec
        self._setup()

        dtype = [
            ("idx", np.int), ("enum", np.int),
            ("trueE", np.float), ("ow", np.float)
            ]

        self.mc_arr = np.empty(0, dtype=dtype)

        self.mc = dict()

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # Get MC events in the selected energy and sine declination range.
            band_mask = np.logical_and(
                np.sin(mc_i["trueDec"]) > np.sin(self._min_dec),
                np.sin(mc_i["trueDec"]) < np.sin(self._max_dec))

            band_mask &= np.logical_and(
                mc_i["trueE"] / self.GeV > self.e_range[0],
                mc_i["trueE"] / self.GeV < self.e_range[1])

            if not np.any(band_mask):
                self._logging.warn(
                    "Sample {0:d}: no events were selected.".format(key))

                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            self._logging.info(
                "Sample {0}: selected {1:d} events at {2:.2f} deg.".format(
                    key, N, np.degrees(src_dec)))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        self._logging.info("Selected {0:d} events in total.".format(
            len(self.mc_arr)))

        self._weights()

    def flux2mu(self, flux):
        gev_flux = flux * (self.E0 * self.GeV)**(self.gamma - 1.) *\
            self.E0**(self.gamma - 2.)

        return self._raw_flux * gev_flux

    flux2mu.__doc__ = Injector.flux2mu.__doc__

    def mu2flux(self, mu):
        gev_flux = mu / self._raw_flux

        flux = gev_flux * self.GeV**(1. - self.gamma) *\
            self.E0**(2. - self.gamma)

        return flux

    mu2flux.__doc__ = Injector.mu2flux.__doc__

    def sample(self, src_ra, mean_mu, poisson=True):
        r"""Sample events for given source location.

        Parameters
        -----------
        src_ra : float
            Right ascension of source position
        mean_mu : float
            Mean number of events to sample
        poisson : bool, optional
            Use Poisson fluctuations, otherwise sample `mean_mu`.

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            Sampled events for each loop iteration; either as simple
            array or as dictionary for each sample

        """
        while True:
            # Generate event numbers using Poisson events.
            if poisson:
                num = self.random.poisson(mean_mu)
            else:
                num = int(np.around(mean_mu))

            self._logging.info("Mean number of events {0:.1f}".format(mean_mu))
            self._logging.info("Generated number of events {0:d}".format(num))

            if num < 1:
                # No events will be sampled.
                yield num, None
                continue

            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)

            # Get the events that were sampled.
            enums = np.unique(sam_idx["enum"])

            if len(enums) == 1 and enums[0] < 0:
                # Only one event will be sampled.
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])
                yield num, rotate_struct(sam_ev, src_ra, self.src_dec)
                continue

            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                sam_ev[enum] = rotate_struct(sam_ev_i, src_ra, self.src_dec)

            yield num, sam_ev


class ModelInjector(PointSourceInjector):
    r"""Model-dependent point source injector

    Inject events according to a specific neutrino flux model. Fluxes
    are measured in percent of the input flux.

    Parameters
    -----------
    logE : array_like
        Bins in base 10 logarithm of energy; energy should be given in
        units of `GeV` GeV.
    logFlux : array_like
        Base 10 logarithm of flux; flux should be given in units `GeV`
        GeV cm^{-2} s^{-1}.
    deg : int, optional
        Degree of smoothing spline; must be ``1 <= deg <= 5``.
    \*\*kwargs
        Parameters passed to base class

    """
    def __init__(self, logE, logFlux, gamma, deg=4, **kwargs):
        # Make sure that energy bins are of increasing order.
        sorter = np.argsort(logE)
        energy = logE[sorter]
        flux = logFlux[sorter]

        # Make sure that energy bins contain only unique values.
        unique = np.argwhere(np.diff(energy) > 0.)
        energy = energy[unique]
        flux = flux[unique]

        self._spline = scipy.interpolate.InterpolatedUnivariateSpline(
            energy, flux, k=deg)

        # Use default energy range of flux parametrization.
        kwargs.setdefault("e_range", [10.**np.amin(logE), 10.**np.amax(logE)])

        super(ModelInjector, self).__init__(gamma, **kwargs)

    def _weights(self):
        mcenergy = np.log10(self.mc_arr["trueE"]) - np.log10(self.GeV)
        flux = self._spline(mcenergy)
        flux = np.power(10., flux - 2. * mcenergy) / self.GeV

        mask = (flux > 0.) & np.isfinite(flux)
        self.mc_arr = self.mc_arr[mask]
        self.mc_arr["ow"] *= flux[mask] / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

    def flux2mu(self, flux):
        return self._raw_flux * flux

    flux2mu.__doc__ = PointSourceInjector.flux2mu.__doc__

    def mu2flux(self, mu):
        return mu / self._raw_flux

    mu2flux.__doc__ = PointSourceInjector.mu2flux.__doc__
