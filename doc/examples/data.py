# -*-coding:utf8-*-

r"""Example script to create data in the right format and load it correctly
to the LLH classes.

"""

from __future__ import print_function

# Python

# SciPy
import numpy as np

# skylab
from skylab.psLLH import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_model import UniformLLH, EnergyLLH

log_mean = np.log(np.radians(2.5))
log_sig = 0.5
logE_res = 0.3

np.random.seed(1)

def exp(N=100):
    r"""Create uniformly distributed data on sphere. """
    g = 3.7

    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float)])

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["sinDec"] = np.random.uniform(-1., 1., N)
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    x = np.random.uniform(0., 1., size=N)
    arr["logE"] = np.log10(1. - x) / (1. - g) + logE_res * np.random.normal(size=N)

    return arr

def MC(N=1000):
    r"""Create uniformly distributed MC data on sphere. """
    g = 2.

    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float),
                                 ("trueRa", np.float), ("trueDec", np.float),
                                 ("trueE", np.float), ("ow", np.float)])

    # true information

    arr["trueRa"] = np.random.uniform(0., 2.*np.pi, N)
    arr["trueDec"] = np.arcsin(np.random.uniform(-1., 1., N))
    x = np.random.uniform(0., 1., N)
    arr["trueE"] = (1. - x)**(1. / (1. - g))
    arr["ow"] = arr["trueE"]
    arr["ow"] /= arr["ow"].sum()

    eta = np.random.uniform(0., 2.*np.pi, len(arr))
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    arr["ra"] = arr["trueRa"] + np.cos(eta) * arr["sigma"] / np.cos(arr["trueDec"])
    arr["sinDec"] = np.sin(arr["trueDec"] + np.sin(eta) * arr["sigma"])
    arr["logE"] = np.log10(arr["trueE"]) + logE_res * np.random.normal(size=len(arr))

    return arr

def init(Nexp, NMC, energy=False, **kwargs):
    arr_exp = exp(Nexp)
    arr_mc = MC(NMC)

    if energy:
        llh_model = EnergyLLH(sinDec_bins=max(3, Nexp // 200),
                              sinDec_range=[-1., 1.],
                              twodim_bins=max(3, Nexp // 200),
                              twodim_range=[[0.9 * min(arr_exp["logE"].min(),
                                                       arr_mc["logE"].min()),
                                             1.1 * max(arr_exp["logE"].max(),
                                                       arr_exp["logE"].max())],
                                             [-1., 1.]])
    else:
        llh_model = UniformLLH(sinDec_bins=max(3, Nexp // 200),
                               sinDec_range=[-1., 1.])

    llh = PointSourceLLH(arr_exp, arr_mc, 365., llh_model=llh_model,
                         mode="all", hemispheres=dict(Full=[-np.inf, np.inf]),
                         nsource=Nexp / 100.,
                         nsource_bounds=(-Nexp / 2., Nexp / 2.)
                                        if not energy else (0., Nexp / 2.),
                         seed=np.random.randint(2**32),
                         **kwargs)

    return llh

def multi_init(n, Nexp, NMC, **kwargs):
    energy = kwargs.pop("energy", False)

    llh = MultiPointSourceLLH(hemispheres=dict(Full=[-np.inf, np.inf]),
                              nsource=Nexp / 100.,
                              nsource_bounds=(-Nexp / 2., Nexp / 2.)
                                             if not energy else (0., Nexp / 2.),
                              seed=np.random.randint(2**32),
                              **kwargs)

    for i in xrange(n):
        llh.add_sample(str(i), init(Nexp, NMC, energy=energy))

    return llh


