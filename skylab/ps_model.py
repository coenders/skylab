# -*-coding:utf8-*-

from __future__ import print_function

"""
This file is part of SkyLab

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

# python packages
from copy import deepcopy
import logging

# scipy-project imports
import healpy as hp
import numpy as np
import scipy.interpolate
from scipy.stats import norm
from scipy.signal import convolve

# local package imports
from . import set_pars

# get module logger
def trace(self, message, *args, **kwargs):
    """ Add trace to logger with output level beyond debug
    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# default values for parameters

# model-parameters
_gamma_params = dict(gamma=[2., (1., 4.)])

# histogramming
_sinDec_bins = 25
_sinDec_range = None
_2dim_bins = 25

# spline
_ratio_perc = 99.
_1dim_order = 2
_2dim_order = 2
_gamma_precision = 0.1
_par_val = np.nan
_parab_cache = np.zeros((0, ), dtype=[("S1", np.float), ("a", np.float),
                                      ("b", np.float)])


class NullModel(object):
    r"""Base class of models for likelihood fitting, this defines every core
    class of the likelihood fitting that is needed in the point source
    calculation without implementing any functionality. Use this class as
    starting point for a unbinned point source likelihood model

    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
                "NullModel only to be used as abstract superclass".format(
                    self.__repr__()))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

        return

    @params.deleter
    def params(self):
        self._params = dict()

        return

    def __raise__(self, *args, **kwargs):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                self.__repr__()))

    def __call__(self, *args, **kwargs):
        r"""Calling the class-object will set it up for use of the other
        functions, i.e., creating splines out of data, etc.

        """
        self.__raise__()

    def background(self, *args, **kwargs):
        r"""Calculation of the background probability *B* in the point source
        likelihood, mainly a spatial dependent term.

        """
        self.__raise__()

    def signal(self, *args, **kwargs):
        r"""Calculation of the signal probability *S* in the point source
        likelihood, mainly a spatial dependent term.

        """
        self.__raise__()

    def reset(self, *args, **kwargs):
        r"""Resetting the llh-model to delete possible cached values

        """

        self.__raise__()

    def weight(self, *args, **kwargs):
        r"""Additional weights calculated for each event, commonly used to
        implement energy weights in the point source likelihood.

        It differs from signal and background distributions that is (to
        first approximation) does not depend on the source position.

        """
        self.__raise__()


class ClassicLLH(NullModel):
    r"""Classic likelihood model for point source searches, only using spatial
    information of each event

    """

    sinDec_bins = _sinDec_bins
    sinDec_range = _sinDec_range

    order = _1dim_order

    _bckg_spline = np.nan

    def __init__(self, *args, **kwargs):
        r"""Constructor of ClassicLLH. Set all configurations here.

        """

        self.params = kwargs.pop("params", dict())

        # Set all attributes passed to class
        set_pars(self, **kwargs)

        return

    def __call__(self, exp, mc):
        r"""Use experimental data to create one dimensional spline of
        declination information for background information.

        Parameters
        -----------
        exp : structured array
            Experimental data with all neccessary fields, i.e., sinDec for
            ClassicLLH
        mc : structured array
            Same as exp for Monte Carlo plus true information.

        """

        hist, bins = np.histogram(exp["sinDec"], density=True,
                                  bins=self.sinDec_bins,
                                  range=self.sinDec_range)

        # overwrite range and bins to actual bin edges
        self.sinDec_bins = bins
        self.sinDec_range = (bins[0], bins[-1])

        if np.any(hist <= 0.):
            bmids = (self.sinDec_bins[1:] + self.sinDec_bins[:-1]) / 2.
            estr = ("Declination hist bins empty, this must not happen! "
                    +"Empty bins: {0}".format(bmids[hist <= 0.]))
            raise ValueError(estr)
        elif np.any((exp["sinDec"] < bins[0])|(exp["sinDec"] > bins[-1])):
            raise ValueError("Data outside of declination bins!")

        self._bckg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
                                (bins[1:] + bins[:-1]) / 2.,
                                np.log(hist), k=self.order)

        return

    def __str__(self):
        r"""String representation of ClassicLLH.

        """
        out_str = "{0:s}\n".format(self.__repr__())
        out_str += 67*"~"+"\n"
        out_str += "Spatial background hist:\n"
        out_str += "\tSinDec bins  : {0:3d}\n".format(len(self.sinDec_bins)-1)
        out_str += "\tSinDec range : {0:-4.2f} to {1:-4.2f}\n".format(
                        *self.sinDec_range)
        out_str += 67*"~"+"\n"

        return out_str

    @property
    def bckg_spline(self):
        return self._bckg_spline

    @bckg_spline.setter
    def bckg_spline(self, val):
        if not hasattr(val, "__call__"):
            print(">>> WARNING: {0} is not callable! Not spline-ish".format(val))
            return

        self._bckg_spline = val

        return

    def background(self, ev):
        r"""Spatial background distribution.

        For IceCube is only declination dependent, in a more general scenario,
        it is dependent on zenith and
        azimuth, e.g. in ANTARES, KM3NET, or using time dependent information.

        Parameters
        -----------
        ev : structured array
            Event array, importand information *sinDec* for this calculation

        Returns
        --------
        P : array-like
            spatial background probability for each event to be found
            at *sinDec*

        """
        return 1. / 2. / np.pi * np.exp(self.bckg_spline(ev["sinDec"]))

    def reset(self):
        r"""Classic likelihood does only depend on spatial part, needs no
        caching

        """
        return

    def signal(self, src_ra, src_dec, ev):
        r"""Spatial distance between source position and events

        Signal is assumed to cluster around source position.
        The distribution is assumed to be well approximated by a gaussian
        locally.

        Parameters
        -----------
        ev : structured array
            Event array, import information: sinDec, ra, sigma

        Returns
        --------
        P : array-like
            Spatial signal probability for each event

        """
        cos_ev = np.sqrt(1. - ev["sinDec"]**2)
        dist = np.arccos(np.cos(src_ra - ev["ra"])
                            * np.cos(src_dec) * cos_ev
                         + np.sin(src_dec) * ev["sinDec"])

        return (1./2./np.pi/ev["sigma"]**2
                * np.exp(-dist**2 / 2. / ev["sigma"]**2))

    def weight(self, ev, **params):
        r"""For classicLLH, no weighting of events

        """
        return np.ones(len(ev)), None


class WeightLLH(ClassicLLH):
    r"""Likelihood class supporting weights for the calculation.

    The weights are calculated using N observables for exp. data and Monte
    Carlo.

    Abstract class, not incorporating a weighting scheme for Monte Carlo.

    """

    _mc = None

    _precision = _gamma_precision

    _g1 = _par_val
    _w_cache = _parab_cache

    def __init__(self, params, pars, bins, *args, **kwargs):
        r"""Constructor

        Parameters
        -----------
        params : dict
            List of fit parameters. Each entry is a tuple out of
            (seed, [lower bound, upper bound])
        pars : list
            Parameter names to use for histogram, without sinDec, which is
            added as last normalisation parameter
        bins : int, ndarray
            Binning for each parameter

        Other Parameters
        -----------------
        range : ndarray
            Bin ranges for each parameter
        kernel : ndarray, int, float
            Smoothing filter defining the kernel for smoothing. Smoothing done
            solely for dimensions that are not normalised. A ndarray specifies
            the filter directly, an int is used for a flat kernel with size
            of *filter* in direction of both sides, a float uses a normal
            distributed kernel with approximately one standard deviation per
            bin.

        """

        params = params

        self.hist_pars = pars

        self._ndim_bins = bins
        self._ndim_range = kwargs.pop("range", None)
        self._ndim_norm = kwargs.pop("normed", 0)

        # define kernel
        kernel = kwargs.pop("kernel", 0)
        if np.all(np.asarray(kernel) == 0):
            # No smoothing
            self._kernel_func = lambda X: X
        elif isinstance(kernel, (list, np.ndarray)):
            kernel_arr = np.asarray(kernel)
            assert(np.all(kernel_arr >= 0))
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            XX = np.product(XX, axis=0).T
            self._kernel_func = (lambda X: convolve(X, XX, mode="same")
                                            / convolve(np.ones_like(X), XX, mode="same"))
        elif isinstance(kernel, int):
            assert(kernel > 0)
            kernel_arr = np.ones(2 * kernel + 1, dtype=np.float)
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            XX = np.product(XX, axis=0).T
            self._kernel_func = (lambda X: convolve(X, XX, mode="same")
                                            / convolve(np.ones_like(X), XX, mode="same"))
        elif isinstance(kernel, float):
            assert(kernel >= 1)
            val = 1.6635
            r = np.linspace(-val, val, 2 * int(kernel) + 1)
            kernel_arr = norm.pdf(r)
            XX = np.meshgrid(*([kernel_arr for i in range(len(self.hist_pars)
                                                          - self._ndim_norm)]
                              + [[1] for i in range(self._ndim_norm)]))
            XX = np.product(XX, axis=0).T
            self._kernel_func = (lambda X: convolve(X, XX, mode="same")
                                            / convolve(np.ones_like(X), XX, mode="same"))
        else:
            raise ValueError("Kernel has to be positive int / float or array")

        super(WeightLLH, self).__init__(*args, params=params, **kwargs)

        self._w_spline_dict = dict()

        return

    def __call__(self, exp, mc):
        r"""In addition to *classicLLH.__call__(),
        splines for energy-declination are created as well.

        """

        # create spatial splines of classic LLH class
        super(WeightLLH, self).__call__(exp, mc)

        # store monte carlo events for calculation of background weights
        self._mc = mc

        self._setup(exp)

        return

    def __str__(self):
        r"""String representation

        """
        out_str = super(WeightLLH, self).__str__()
        out_str += "Weighting hist:\n"
        for p, b, r in zip(self.hist_pars, self._ndim_bins, self._ndim_range):
            out_str += "\t{0:11s} : {1:3d}\n".format(p + " bins", len(b)-1)
            out_str += "\t{0:11s} : {1:-4.2f} to {2:-4.2f}\n".format(
                                        p + " range", *r)
        out_str += "\tPrecision : {0:4.2f}\n".format(self._precision)
        out_str += 67*"~"+"\n"

        return out_str

    @property
    def hist_pars(self):
        return self._hist_pars

    @hist_pars.setter
    def hist_pars(self, val):
        self._hist_pars = list(val)

        return

    def _around(self, value):
        r"""Round a value to a precision defined in the class.

        Parameters
        -----------
        value : array-like
            Values to round to precision.

        Returns
        --------
        round : array-like
            Rounded values.

        """
        return np.around(float(value) / self._precision) * self._precision

    def _get_weights(self, **params):
        r"""Calculate weights using the given parameters.

        Parameters
        -----------
        params : dict
            Dictionary containing the parameter values for the weighting.

        Returns
        --------
        weights : array-like
            Weights for each event

        """

        raise NotImplementedError("Weigthing not specified, using subclass")

    def _hist(self, arr, weights=None):
        r"""Create histogram of data so that it is correctly normalized.

        The edges of the histogram are copied so that the spline is defined for
        the entire data range.

        """

        h, binedges = np.histogramdd(arr, bins=self._ndim_bins,
                                     range=self._ndim_range,
                                     weights=weights, normed=True)

        if self._ndim_norm > 0:
            norms = np.sum(h, axis=tuple(range(h.ndim - self._ndim_norm)))
            norms[norms==0] = 1.

            h /= norms

        return h, binedges

    def _ratio_spline(self, **params):
        r"""Create the ratio of signal over background probabilities. With same
        binning, the bin hypervolume cancels out, ensuring correct
        normalisation of the histograms.

        Parameters
        -----------
        params : dict
            (Physics) parameters used for signal pdf calculation.

        Returns
        --------
        spline : scipy.interpolate.RectBivariateSpline
            Spline for parameter values *params*

        """

        # use previous calculations if existing
        if self._w_spline_dict.has_key(tuple(params.items())):
            return self._w_spline_dict[tuple(params.items())]

        mcvars = [self._mc[p] if not p == "sinDec"
                              else np.sin(self._mc["trueDec"])
                  for p in self.hist_pars]

        # create MC histogram
        wSh, wSb = self._hist(mcvars, weights=self._get_weights(**params))
        wSh = self._kernel_func(wSh)
        wSd = wSh > 0.

        # calculate ratio
        ratio = np.ones_like(self._wB_hist, dtype=np.float)

        ratio[wSd & self._wB_domain] = (wSh[wSd & self._wB_domain]
                                        / self._wB_hist[wSd & self._wB_domain])

        # values outside of the exp domain, but inside the MC one are mapped to
        # the most signal-like value
        min_ratio = np.percentile(ratio[ratio>1.], _ratio_perc)
        np.copyto(ratio, min_ratio, where=wSd & ~self._wB_domain)

        binmids = [(wSb_i[1:] + wSb_i[:-1]) / 2. for wSb_i in wSb]
        binmids[-1][[0, -1]] = wSb_i[0], wSb_i[-1]
        binmids = tuple(binmids)

        spline = scipy.interpolate.RegularGridInterpolator(
                                                binmids, np.log(ratio),
                                                method="linear",
                                                bounds_error=False,
                                                fill_value=0.)

        self._w_spline_dict[tuple(params.items())] = spline

        return spline

    def _setup(self, exp):
        r"""Set up everything for weight calculation.

        """
        # set up weights for background distribution, reset all cached values
        self._w_spline_dict = dict()

        expvars = [exp[p] for p in self.hist_pars]

        self._wB_hist, self._wB_bins = self._hist(expvars)
        self._wB_hist = self._kernel_func(self._wB_hist)
        self._wB_domain = self._wB_hist > 0

        # overwrite bins
        self._ndim_bins = self._wB_bins
        self._ndim_range = tuple([(wB_i[0], wB_i[-1])
                                  for wB_i in self._wB_bins])

        return

    def _spline_eval(self, spline, ev):
        r"""Evaluate spline on coordinates using the important parameters.

        """
        return spline(np.vstack([ev[p] for p in self.hist_pars]).T)

    def reset(self):
        r"""Energy weights are cached, reset all cached values.

        """
        self._w_cache = _parab_cache

        return

    def weight(self, ev, **params):
        r"""Evaluate spline for given parameters.

        Parameters
        -----------
        ev : structured array
            Events to be evaluated

        params : dict
            Parameters for evaluation

        Returns
        --------
        val : array-like (N), N events
            Function value.

        grad : array-like (N, M), N events in M parameter dimensions
            Gradients at function value.

        """
        # get params
        gamma = params["gamma"]

        # evaluate on finite gridpoints in spectral index gamma
        g1 = self._around(gamma)
        dg = self._precision

        # check whether the grid point of evaluation has changed
        if g1 == self._g1 and len(ev) == len(self._w_cache):
            S1 = self._w_cache["S1"]
            a = self._w_cache["a"]
            b = self._w_cache["b"]
        else:
            # evaluate neighbouring gridpoints and parametrize a parabola
            g0 = g1 - dg
            g2 = g1 + dg

            S0 = self._spline_eval(self._ratio_spline(gamma=g0), ev)
            S1 = self._spline_eval(self._ratio_spline(gamma=g1), ev)
            S2 = self._spline_eval(self._ratio_spline(gamma=g2), ev)

            a = (S0 - 2.*S1 + S2) / (2. * dg**2)
            b = (S2 - S0) / (2. * dg)

            # cache values
            self._g1 = g1

            self._w_cache = np.zeros((len(ev),),
                                     dtype=[("S1", np.float), ("a", np.float),
                                            ("b", np.float)])
            self._w_cache["S1"] = S1
            self._w_cache["a"] = a
            self._w_cache["b"] = b

        # calculate value at the parabola
        val = np.exp(a * (gamma - g1)**2 + b * (gamma - g1) + S1)
        grad = val * (2. * a * (gamma - g1) + b)

        return val, np.atleast_2d(grad)


class PowerLawLLH(WeightLLH):
    r"""Weighted LLH class assuming unbroken power-law spectra for weighting.

    Optional Parameters
    --------------------
    seed : float
        Seed for gamma parameter
    bonds : ndarray (len 2)
        Bounds for minimisation

    """
    def __init__(self, *args, **kwargs):

        params = dict(gamma=(kwargs.pop("seed", _gamma_params["gamma"][0]),
                             deepcopy(kwargs.pop("bounds", deepcopy(_gamma_params["gamma"][1])))))

        super(PowerLawLLH, self).__init__(params, *args, **kwargs)

        return

    def _get_weights(self, **params):
        r"""Calculate weights using the given parameters.

        Parameters
        -----------
        params : dict
            Dictionary containing the parameter values for the weighting.

        Returns
        --------
        weights : array-like
            Weights for each event

        """

        return self._mc["ow"] * self._mc["trueE"]**(-params["gamma"])



class EnergyLLH(PowerLawLLH):
    r"""Likelihood using Energy Proxy and declination, where declination is
    used for normalisation to account for changing energy distributions.

    """
    def __init__(self, twodim_bins=_2dim_bins, twodim_range=None,
                 **kwargs):
        r"""Constructor

        """
        super(EnergyLLH, self).__init__(["logE", "sinDec"],
                                        twodim_bins, range=twodim_range,
                                        normed=1,
                                        **kwargs)

        return


class EnergyDistLLH(PowerLawLLH):
    r"""Likelihood using Energy Proxy and starting distance for evaluation.
    Declination is not used for normalisation assuming that the energy does not
    change rapidly with declination.

    """
    def __init__(self, twodim_bins=_2dim_bins, twodim_range=None,
                 **kwargs):
        r"""Constructor

        """
        super(EnergyDistLLH, self).__init__(["logE", "dist"],
                                            twodim_bins, range=twodim_range,
                                            **kwargs)

        return


