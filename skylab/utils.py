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
import numpy as np
import scipy.signal
import scipy.stats


def kernel_func(X, Y):
    r"""Smooth histogram Y with kernel Y

    """
    if Y is None:
        return X

    return (
        scipy.signal.convolve(X, Y, mode="same") /
        scipy.signal.convolve(np.ones_like(X), Y, mode="same")
        )


def poisson_percentile(mu, x, y, yval):
    r"""Calculate upper percentile using a Poisson distribution.

    Parameters
    ----------
    mu : float
        Mean value of Poisson distribution
    x : array_like,
        Trials of variable that is expected to be Poisson distributed
    y : array_like
        Observed variable connected to `x`
    yval : float
        Value to calculate the percentile at

    Returns
    -------
    score : float
        Value at percentile *alpha*
    err : float
        Uncertainty on `score`

    """
    x = np.asarray(x, dtype=np.int)
    y = np.asarray(y, dtype=np.float)

    w = poisson_weight(x, mu)

    # Get percentile at yval.
    m = y > yval
    u = np.sum(w[m], dtype=np.float)

    if u == 0.:
        return 1., 1.

    err = np.sqrt(np.sum(w[m]**2)) / np.sum(w)

    return u / np.sum(w, dtype=np.float), err


def poisson_weight(vals, mean, weights=None):
    r"""Calculate weights for a sample that it resembles a Poisson.

    Parameters
    ----------
    vals : array_like
        Random integers to be weighted
    mean : float
        Poisson mean
    weights : array_like, optional
        Weights for each event

    Returns
    -------
    ndarray
        Weights for each event

    """
    mean = float(mean)
    vals = np.asarray(vals, dtype=np.int)

    if weights is None:
        weights = np.ones_like(vals, dtype=np.float)

    # Get occurences of integers.
    bincount = np.bincount(vals, weights=weights)

    n_max = len(bincount)

    # Get poisson probability.
    if mean > 0:
        p = scipy.stats.poisson(mean).pmf(range(n_max))
    else:
        p = np.zeros(n_max, dtype=np.float)
        p[0] = 1.

    # Weights for each integer
    w = np.zeros_like(bincount, dtype=np.float)
    m = bincount > 0
    w[m] = p[m] / bincount[m]

    w = w[np.searchsorted(np.arange(n_max), vals)]

    return w * weights


class delta_chi2(object):
    """Modified chi-square distribution

    Combine chi-square distribution and delta distribution at zero.

    Parameters
    ----------
    df : float
        Number of degree of freedom
    loc : float, optional
        Shift probability density.
    scale : float, optional
        Scale probability density.

    Attributes
    ----------
    params : tuple(float)
        Shape, location, and scale parameters of chi-square distribution
    eta : float
        Fraction of over-fluctuations
    eta_err : float
        Uncertainty on `eta`
    ks : float
        KS test stastistic

    """
    def __init__(self, eta, df, loc=0., scale=1.):
        self.eta = eta
        self.params = (df, loc, scale)
        self._chi2 = scipy.stats.chi2(df, loc, scale)
        self.eta_err = np.nan
        self.ks = np.nan

    def __getstate__(self):
        return dict(
            params=self.params, eta=self.eta, eta_err=self.eta_err, ks=self.ks)

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

        self._chi2 = scipy.stats.chi2(*self.params)

    def __str__(self):
        return (
            "Delta Distribution plus chi-square {0:s}\n"
            "\tSeparation factor = {1:8.3%} +/- {2:8.3%}\n"
            "\t\tNDoF  = {3:6.2f}\n"
            "\t\tMean  = {4:6.2f}\n"
            "\t\tScale = {5:6.2f}\n"
            "\t\tKS    = {6:7.2%}").format(
                repr(self), self.eta, self.eta_err,
                self.params[0], self.params[1], self.params[2], self.ks)

    def pdf(self, x):
        r"""Probability density function

        """
        x = np.asarray(x)
        density = np.where(x > 0., self.eta * self._chi2.pdf(x), 1. - self.eta)

        if density.ndim == 0:
            density = np.asscalar(density)

        return density

    def logpdf(self, x):
        r"""Logarithmic probability density function

        """
        x = np.asarray(x)

        density = np.where(
            x > 0., np.log(self.eta) + self._chi2.logpdf(x),
            np.log(1. - self.eta))

        if density.ndim == 0:
            density = np.asscalar(density)

        return density

    def cdf(self, x):
        r"""Probability mass function

        """
        return (1. - self.eta) + self.eta * self._chi2.cdf(x)

    def logcdf(self, x):
        r"""Logarithmic probability mass function

        """
        return np.log(1. - self.eta) + np.log(self.eta) + self._chi2.logcdf(x)

    def sf(self, x):
        r"""Survival function

        """
        x = np.asarray(x)
        probability = np.where(x > 0., self.eta * self._chi2.sf(x), 1.)

        if probability.ndim == 0:
            probability = np.asscalar(probability)

        return probability

    def logsf(self, x):
        r"""Logarithmic survival function

        """
        x = np.asarray(x)

        probability = np.where(
            x > 0., np.log(self.eta) + self._chi2.logsf(x), 0.)

        if probability.ndim == 0:
            probability = np.asscalar(probability)

        return probability

    def isf(self, x):
        r"""Inverse survival function

        """
        x = np.asarray(x)
        ts = np.where(x < self.eta, self._chi2.isf(x / self.eta), 0.)

        if ts.ndim == 0:
            ts = np.asscalar(ts)

        return ts


class FitDeltaChi2(object):
    """Fit `delta_chi2` to test statistic.

    Parameters
    ----------
    df : float, optional
        Seed for number of degree of freedom
    \*\*others
        Optional keyword arguments passed to chi-square function

    See Also
    --------
    scipy.stats.chi2.fit

    """
    def __init__(self, df=np.nan, **others):
        self.df = df
        self.others = others

    def fit(self, data):
        r"""Computes the fraction of over-fluctuations, fits a
        chi-square distribution to the values larger than zero, and
        performs a KS test.

        Parameters
        ----------
        data : array_like
            Test statistic values

        Returns
        -------
        delta_chi2
            Probability density function

        """
        data = np.asarray(data)
        seeds = []

        if np.isfinite(self.df):
            seeds.append(self.df)

        params = scipy.stats.chi2.fit(data[data > 0.], *seeds, **self.others)
        eta = float(np.count_nonzero(data > 0.)) / len(data)

        pdf = delta_chi2(eta, *params)
        pdf.eta_err = np.sqrt(pdf.eta * (1. - pdf.eta) / len(data))
        pdf.ks = scipy.stats.kstest(data[data > 0.], "chi2", args=params)[0]

        return pdf


class delta_exp(object):
    r"""Gaussian tail

    Approximate test statistic using a polynomial fit to the cumulative
    test statistic distribution.

    Attributes
    ----------
    coeff : ndarray
        Polynomial coefficients, highest power first
    eta : float
        Fraction of over-fluctuations
    eta_err : float
        Uncertainty on `eta`

    """
    def __init__(self, coeff, eta, eta_err=None):
        self.coeff = coeff
        self.eta = eta
        self.eta_err = eta_err

    def __getstate__(self):
        return dict(coeff=self.coeff, eta=self.eta, eta_err=self.eta_err)

    def __setstate__(self, state):
        self.coeff = state.pop("coeff")
        self.eta = state.pop("eta")
        self.eta_err = state.pop("eta_err")

    def pdf(self, x):
        r"""Probability densitiy function

        """
        x = np.asarray(x)

        density = np.polyval(np.polyder(self.coeff), x) *\
            np.exp(np.polyval(self.coeff, x))

        density = np.where(x > 0., density, self.eta)

        if density.ndim == 0:
            density = np.asscalar(density)

        return density

    def sf(self, x):
        r"""Survival function

        """
        x = np.asarray(x)

        probability = np.where(
            x > 0., np.exp(np.polyval(self.coeff, x)), self.eta)

        if probability.ndim == 0:
            probability = np.asscalar(probability)

        return probability

    def isf(self, x):
        r"""Inverse survival function

        """
        @np.vectorize
        def get_root(x):
            if x > self.eta:
                return 0.

            coeff = np.copy(self.coeff)
            coeff[-1] -= np.log(x)

            roots = np.roots(coeff)
            roots = np.real(roots[np.isreal(roots)])

            return np.amax(roots[roots > 0])

        ts = get_root(x)

        if ts.ndim == 0:
            ts = np.asscalar(ts)

        return ts


class FitDeltaExp(object):
    r"""Fit polynomial to cumulative test statistic distrubtion.

    Attributes
    ----------
    deg : int
        Degree of the fitting polynomial

    """
    def __init__(self, deg):
        self.deg = deg

    def fit(self, data):
        r"""Perform fit given `data`.

        Parameters
        ----------
        data : array_like
            Test statistic values

        Returns
        -------
        delta_exp_frozen
            Probability density function

        """
        data = np.asarray(data)

        # Get amount of over-fluctuations.
        eta = float(np.count_nonzero(data > 0.)) / len(data)
        eta_err = np.sqrt(eta * (1. - eta) / len(data))

        # Sort data and construct cumulative distribution.
        x = np.sort(data[data > 0.])
        y = np.linspace(1., 0., len(x) + 1)[:-1]

        coeff = np.polyfit(x, np.log(y), self.deg)

        return delta_exp(coeff, eta, eta_err)


class twoside_chi2(object):
    r"""Modified chi-square distribution

    Combine two chi-square distributions, which are normalized to
    conserve the total normalization, where one of the functions is
    defined for positive values and  the other one for negative values.

    Parameters
    ----------
    df : tuple(float)
        Numbers of degree of freedom
    loc : tuple(float), optional
        Shift probability densities.
    scale : tuple(float), optional
        Scale probability densities.

    Attributes
    ----------
    params : ndarray
        Shape, location, and scale parameters of left and right
        chi-square distributions
    eta : float
        Fraction of over-fluctuations
    eta_err : float
        Uncertainty on `eta`
    ks : tuple(float)
        KS test stastistics

    """
    def __init__(self, eta, df, loc=0., scale=1.):
        params = np.empty(shape=(2, 3))
        params[:, 0] = df
        params[:, 1] = loc
        params[:, 2] = scale

        self.eta = eta
        self.params = params

        self._chi2 = tuple(scipy.stats.chi2(*p) for p in params)

        self.eta_err = np.nan
        self.ks = (np.nan, np.nan)

    def __getstate__(self):
        return dict(
            params=self.params, eta=self.eta, eta_err=self.eta_err, ks=self.ks)

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

        self._chi2 = tuple(scipy.stats.chi2(*p) for p in self.params)

    def __str__(self):
        return (
            "Two-sided chi-square {0:s}\n"
            "\tSeparation factor = {1:8.3%} +/- {2:8.3%}\n"
            "\tRight side:\n"
            "\t\tNDoF  = {3[0]:6.2f}\n"
            "\t\tMean  = {3[1]:6.2f}\n"
            "\t\tScale = {3[2]:6.2f}\n"
            "\t\tKS    = {5[1]:7.2%}\n"
            "\tLeft side:\n"
            "\t\tNDoF  = {4[0]:6.2f}\n"
            "\t\tMean  = {4[1]:6.2f}\n"
            "\t\tScale = {4[2]:6.2f}\n"
            "\t\tKS    = {5[1]:7.2%}\n"
            ).format(
                repr(self), self.eta, self.eta_err,
                self.params[0], self.params[1], self.ks)

    def pdf(self, x):
        r"""Probability density function

        """
        x = np.asarray(x)

        density = self.eta * self._chi2[0].pdf(x) +\
            (1. - self.eta) * self._chi2[1].pdf(-x)

        return density

    def logpdf(self, x):
        r"""Logarithmic probability density function

        """
        x = np.asarray(x)

        density = np.where(
            x > 0., np.log(self.eta) + self._chi2[0].logpdf(x),
            np.log(1. - self.eta) + self._chi2[1].logpdf(-x))

        if density.ndim == 0:
            density = np.asscalar(density)

        return density

    def cdf(self, x):
        r"""Probability mass function

        """
        x = np.asarray(x)

        probability = self.eta * self._chi2[0].cdf(x) +\
            (1. - self.eta) * self._chi2[1].sf(-x)

        return probability

    def logcdf(self, x):
        r"""Logarithmic probability mass function

        """
        x = np.asarray(x)

        probability = np.where(
            x > 0., np.log(self.eta) + self._chi2[0].logcdf(x),
            np.log(1. - self.eta) + self._chi2[1].logsf(-x))

        if probability.ndim == 0:
            probability = np.asscalar(probability)

        return probability

    def sf(self, x):
        r"""Survival function

        """
        x = np.asarray(x)

        probability = self.eta * self._chi2[0].sf(x) +\
            (1. - self.eta) * self._chi2[1].cdf(-x)

        return probability

    def logsf(self, x):
        r"""Logarithmic survival function

        """
        x = np.asarray(x)

        probability = np.where(
            x > 0., np.log(self.eta) + self._chi2[0].logsf(x),
            np.log(1. - self.eta) + self._chi2[1].logcdf(-x))

        if probability.ndim == 0:
            probability = np.asscalar(probability)

        return probability

    def isf(self, x):
        r"""Inverse survival function

        """
        x = np.asarray(x)

        ts = np.where(
            x < self.eta, self._chi2[0].isf(x / self.eta),
            -self._chi2[1].ppf(1. - (1. - x) / (1. - self.eta)))

        if ts.ndim == 0:
            ts = np.asscalar(ts)

        return ts


class FitTwoSideChi2(object):
    """Fit `twoside_chi2` to test statistic.

    Parameters
    ----------
    df : tuple(float), optional
        Seeds for number of degree of freedom
    left : dict(str, float), optional
        Optional keyword arguments passed to left chi-square function
    right : dict(str, float), optional
        Optional keyword arguments passed to right chi-square function

    See Also
    --------
    scipy.stats.chi2.fit

    """
    def __init__(self, df=np.nan, left={}, right={}):
        self.df = np.empty(2)
        self.df[:] = df
        self.others = (left, right)

    def fit(self, data):
        r"""Computes the fraction of over-fluctuations, fits chi-square
        distributions, and performs KS tests.

        Parameters
        ----------
        data : array_like
            Test statistic values

        Returns
        -------
        twoside_chi2
            Probability density function

        """
        data = np.asarray(data)
        seeds = [[n] if np.isfinite(n) else [] for n in self.df]

        params = np.vstack((
            scipy.stats.chi2.fit(data[data > 0.], *seeds[0], **self.others[0]),
            scipy.stats.chi2.fit(-data[data < 0.], *seeds[1], **self.others[1])
            ))

        eta = float(np.count_nonzero(data > 0.)) / len(data)

        pdf = twoside_chi2(eta, *params.T)
        pdf.eta_err = np.sqrt(pdf.eta * (1. - pdf.eta) / len(data))

        pdf.ks = (
            scipy.stats.kstest(data[data > 0.], "chi2", args=params[0])[1],
            scipy.stats.kstest(-data[data < 0.], "chi2", args=params[1])[1]
            )

        return pdf


def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).

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

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
        )

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
