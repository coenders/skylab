# -*-coding:utf8-*-

r"""
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


utils
======

Helper methods for the other classes

"""

import numpy as np
from scipy.signal import convolve
from scipy.stats import chi2, kstest, poisson


def kernel_func(X, Y):
    r"""Smooth histogram Y with kernel Y

    """
    if Y is None:
        return X

    return (convolve(X, Y, mode="same")
            / convolve(np.ones_like(X), Y, mode="same"))


def poisson_percentile(mu, x, y, yval):
    r"""Calculate upper percentile using a Poissonian distribution.

    Parameters
    ----------
    mu : float
        Mean value of Poissonian distribution

    x : array-like, dtype int
        Trials of variable that is expected to be poissonian distributed

    y : array-like
        Observed variable connected to poissonian variable *x*

    yval : float
        Value to calculate the percentile at

    Returns
    -------
    score : float
        Value at percentile *alpha*

    """

    x = np.asarray(x, dtype=np.int)
    y = np.asarray(y, dtype=np.float)

    w = poisson_weight(x, mu)

    # get percentile at yval
    m = y > yval
    u = np.sum(w[m], dtype=np.float)

    err = np.sqrt(np.sum(w[m]**2)) / np.sum(w)

    if u == 0.:
        return 1., 1.

    return u / np.sum(w, dtype=np.float), err


def poisson_weight(vals, mean, weights=None):
    r"""Calculate weights for a sample that it resembles a poissonian.

    Parameters
    -----------
    vals : array-like
        Random numbers to be weighted, should be integer

    mean : float
        Poisson mean number

    Returns
    --------
    weights : array-like
        Weights for each event

    Optional Parameters
    --------------------
    weights : array-like
        Weights for each event before poissonian

    """

    mean = float(mean)
    vals = np.asarray(vals, dtype=np.int)

    if weights is None:
        weights = np.ones_like(vals, dtype=np.float)

    # get occurences of integers
    bincount = np.bincount(vals, weights=weights)

    n_max = len(bincount)

    # get poisson probability
    if mean > 0:
        p = poisson(mean).pmf(range(n_max))
    else:
        p = np.zeros(n_max, dtype=np.float)
        p[0] = 1.

    # weights for each integer
    w = np.zeros_like(bincount, dtype=np.float)
    m = bincount > 0
    w[m] = p[m] / bincount[m]

    w = w[np.searchsorted(np.arange(n_max), vals)]

    return w * weights


class delta_chi2(object):
    """ A probability density function similar to scipy's rvs functions.
    It consisist of a chi2 distribution plus a delta-distribution at zero.

    """

    def __init__(self, data, **kwargs):
        """ Constructor, evaluates the percentage of events equal to zero and
        fits a chi2 to the rest of the data.

        Parameters
        -----------
        data : array
            Data values to be fit

        """
        data = np.asarray(data)

        if len(data) == 2:
            self.eta = data[0]
            self.par = [data[1], 0., 1.]

            self.eta_err = np.nan
            self.ks = np.nan

            self.f = chi2(*self.par)

            return

        self.par = chi2.fit(data[data > 0], **kwargs)

        self.f = chi2(*self.par)

        self.eta = float(np.count_nonzero(data > 0)) / len(data)
        self.eta_err = np.sqrt(self.eta * (1. - self.eta) / len(data))

        self.ks = kstest(data[data > 0], "chi2", args=self.par)[0]

        return

    def __getstate__(self):
        return dict(par=self.par, eta=self.eta, eta_err=self.eta_err,
                    ks=self.ks)

    def __setstate__(self, state):
        for key, val in state.iteritems():
            setattr(self, key, val)

        self.f = chi2(*self.par)

        return

    def __str__(self):
        return ("Delta Distribution plus chi-square {0:s}\n".format(
                    self.__repr__())
               +"\tSeparation factor = {0:8.3%} +/- {1:8.3%}\n".format(
                    self.eta, self.eta_err)
               +"\t\tNDoF  = {0:6.2f}\n".format(self.par[0])
               +"\t\tMean  = {0:6.2f}\n".format(self.par[1])
               +"\t\tScale = {0:6.2f}\n".format(self.par[2])
               +"\t\tKS    = {0:7.2%}".format(self.ks)
               )

    def pdf(self, x):
        """ Probability density function.
        """
        return np.where(x > 0, self.eta * self.f.pdf(x), 1. - self.eta)

    def logpdf(self, x):
        """ Logarithmic pdf.
        """
        return np.where(x > 0, np.log(self.eta) + self.f.logpdf(x),
                               np.log(1. - self.eta))

    def cdf(self, x):
        """ Probability mass function.
        """
        return (1. - self.eta) + self.eta * self.f1.cdf(x)

    def logcdf(self, x):
        """ Logarithmic cdf.
        """
        return np.log(1. - self.eta) + np.log(self.eta) + self.f1.logcdf(x)

    def sf(self, x):
        """ Survival probability function.
        """
        return np.where(x > 0, self.eta * self.f.sf(x), 1.)

    def logsf(self, x):
        """ Logarithmic sf.
        """
        return np.where(x > 0, np.log(self.eta) + self.f.logsf(x), 0.)

    def isf(self, x):
        """ Inverse survival function.
        """
        return np.where(x < self.eta, self.f.isf(x/self.eta), 0.)


class delta_exp(object):
    r"""Class that approximates the test statistic using a gaussian tail.

    """
    def __init__(self, data, **kwargs):
        r"""Constructor. The data above zero is fitted with a gaussian tail.

        """
        data = np.asarray(data)

        self.deg = kwargs.pop("deg")

        # amount of overfluctuations
        self.eta = float(np.count_nonzero(data > 0)) / len(data)
        self_eta_err = np.sqrt(self.eta * (1. - self.eta) / len(data))

        # sort data and construct cumulative distribution
        x = np.sort(data[data > 0])
        y = np.linspace(1., 0., len(x) + 1)[:-1]

        self.p = np.polyfit(x, np.log(y), self.deg)

        return

    def __getstate__(self):
        return dict(eta=self.eta, eta_err=self.eta_err, p=self.p, deg=self.deg)

    def __setstate__(self, state):
        self.eta = state.pop("eta")
        self.eta_err = state.pop("eta_err")
        self.p = state.pop("p")
        self.deg = state.pop("deg")

        return

    def pdf(self, val):
        r"""Probability densitiy function

        """
        return np.where(val > 0, np.polyval(np.polyder(self.p), val)
                                    * np.exp(np.polyval(self.p, val)),
                        self.eta)

    def sf(self, val):
        r"""Survival function

        """
        return np.where(val > 0, np.exp(np.polyval(self.p, val)), self.eta)

    def isf(self, val):
        r"""Inverse survival function

        """
        if val > self.eta:
            return 0.

        p = np.copy(self.p)
        p[-1] -= np.log(val)

        r = np.roots(p)
        r = np.real(r[np.isreal(r)])
        r = r[r > 0]

        return np.amax(r)


class twoside_chi2(object):
    r"""A probability density function similar to scipy's rvs functions.
    It consists of two chi2 distributions, which are normalized to conserve
    the total normalization, where one of the chi2 functions is defined for
    positive values, the other one for negative values.

    """

    def __init__(self, data, **kwargs):
        r"""Constructor. This will fit both chi2 function in the different
        regimes.
            *data*      -   Data sample to use for fitting

        Keyword Argument:
            *chi1/2*    -   Keyword arguments like floc, fshape, etc. that are
                            passed to the constructor of the corresponding
                            chi2 scipy object.

        """
        data = np.asarray(data)

        c1 = kwargs.pop("chi1", dict())
        c2 = kwargs.pop("chi2", dict())

        self.par1 = chi2.fit(data[data > 0.], **c1)
        self.par2 = chi2.fit(-data[data < 0.], **c2)

        self.f1 = chi2(*self.par1)
        self.f2 = chi2(*self.par2)

        self.eta = float(np.count_nonzero(data > 0.)) / len(data)
        self.eta_err = np.sqrt(self.eta * (1. - self.eta) / len(data))

        # get fit-quality
        self.ks1 = kstest(data[data > 0.], "chi2", args=self.par1)[1]
        self.ks2 = kstest(-data[data < 0.], "chi2", args=self.par2)[1]

        return

    def __getstate__(self):
        return dict(par1=self.par1, par2=self.par2, eta=self.eta,
                    eta_err=self.eta_err, ks1=self.ks1, ks2=self.ks2)

    def __setstate__(self, state):
        for key, val in state.iteritems():
            setattr(self, key, val)

        self.f1 = chi2(*self.par1)
        self.f2 = chi2(*self.par2)

        return

    def __str__(self):
        return ("Two-sided chi-square {0:s}\n".format(self.__repr__())
               +"\tSeparation factor = {0:8.3%} +/- {1:8.3%}\n".format(
                    self.eta, self.eta_err)
               +"\tRight side:\n"
               +"\t\tNDoF  = {0:6.2f}\n".format(self.par1[0])
               +"\t\tMean  = {0:6.2f}\n".format(self.par1[1])
               +"\t\tScale = {0:6.2f}\n".format(self.par1[2])
               +"\t\tKS    = {0:7.2%}\n".format(self.ks1)
               +"\tLeft side:\n"
               +"\t\tNDoF  = {0:6.2f}\n".format(self.par2[0])
               +"\t\tMean  = {0:6.2f}\n".format(self.par2[1])
               +"\t\tScale = {0:6.2f}\n".format(self.par2[2])
               +"\t\tKS    = {0:7.2%}\n".format(self.ks2)
               )

    def pdf(self, x):
        r"""Probability density function.

        """
        return self.eta * self.f1.pdf(x) + (1. - self.eta) * self.f2.pdf(-x)

    def logpdf(self, x):
        r"""Logarithmic pdf.

        """
        return np.where(x > 0, np.log(self.eta) + self.f1.logpdf(x),
                               np.log(1. - self.eta) + self.f2.logpdf(-x))

    def cdf(self, x):
        r"""Probability mass function.

        """
        return self.eta * self.f1.cdf(x) + (1. - self.eta) * self.f2.sf(-x)

    def logcdf(self, x):
        r"""Logarithmic cdf.

        """
        return np.where(x > 0, np.log(self.eta) + self.f1.logcdf(x),
                               np.log(1. - self.eta) + self.f2.logsf(-x))

    def sf(self, x):
        r"""Survival probability function.

        """
        return self.eta * self.f1.sf(x) + (1. - self.eta) * self.f2.cdf(-x)

    def logsf(self, x):
        r"""Logarithmic sf.

        """
        return np.where(x > 0, np.log(self.eta) + self.f1.logsf(x),
                               np.log(1. - self.eta) + self.f2.logcdf(-x))

    def isf(self, x):
        r"""Inverse survival function.

        """
        return np.where(x < self.eta, self.f1.isf(x/self.eta),
                                     -self.f2.ppf(1. - (1.-x)/(1.-self.eta)))


def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2). The
    rotation is performed on (ra3, dec3).

    """
    def cross_matrix(x):
        r"""Calculate cross product matrix

        A[ij] = x_i * y_j - y_i * x_j

        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(len(ra1) == len(dec1)
                == len(ra2) == len(dec2)
                == len(ra3) == len(dec3))

    alpha = np.arccos(np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2)
                      + np.sin(dec1) * np.sin(dec2))
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1.-np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec

