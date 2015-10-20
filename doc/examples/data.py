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
from skylab.ps_model import ClassicLLH, EnergyLLH

log_mean = np.log(np.radians(2.5))
log_sig = 0.5
logE_mean = np.log(1.)
logE_sig = 1.

np.random.seed(1)

def exp(N=100):
    r"""Create uniformly distributed data on sphere. """
    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float)])

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["sinDec"] = np.random.uniform(-1., 1., N)
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    arr["logE"] = logE_mean + logE_sig * np.random.normal(size=N)

    return arr

def MC(N=1000):
    r"""Create uniformly distributed MC data on sphere. """
    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float),
                                 ("trueRa", np.float), ("trueDec", np.float),
                                 ("trueE", np.float), ("ow", np.float)])

    # true information

    arr["trueRa"] = np.random.uniform(0., 2.*np.pi, N)
    arr["trueDec"] = np.arcsin(np.random.uniform(-1., 1., N))
    arr["trueE"] = np.random.lognormal(mean=logE_mean, sigma=logE_sig, size=N)
    arr["ow"] = arr["trueE"]
    arr["ow"] /= arr["ow"].sum()

    eta = np.random.uniform(0., 2.*np.pi, len(arr))
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    arr["ra"] = arr["trueRa"] + np.cos(eta) * arr["sigma"] / np.cos(arr["trueDec"])
    arr["sinDec"] = np.sin(arr["trueDec"] + np.sin(eta) * arr["sigma"])
    arr["logE"] = np.log10(arr["trueE"]) + np.random.normal(size=len(arr)) #/ 4

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
        llh_model = ClassicLLH(sinDec_bins=max(3, Nexp // 200),
                               sinDec_range=[-1., 1.])

    llh = PointSourceLLH(arr_exp, arr_mc, 365., llh_model=llh_model,
                         mode="all", hemispheres=dict(Full=[-np.inf, np.inf]),
                         rho_nsource_bounds=(-0.8, 0.8) if not energy else (0., 0.8),
                         seed=np.random.randint(2**32),
                         **kwargs)

    return llh

def multi_init(n, Nexp, NMC, **kwargs):
    energy = kwargs.pop("energy", False)

    llh = MultiPointSourceLLH(hemispheres=dict(Full=[-np.inf, np.inf]),
                              rho_nsource_bounds=(-0.8, 0.8) if not energy else (0., 0.8),
                              seed=np.random.randint(2**32),
                              **kwargs)

    for i in xrange(n):
        llh.add_sample(str(i), init(Nexp, NMC, energy=energy))

    return llh


