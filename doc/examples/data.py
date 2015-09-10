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

log_mean = np.log(np.radians(3.))
log_sig = np.log(1.5)
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

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["sinDec"] = np.random.uniform(-1., 1., N)
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    arr["logE"] = logE_mean + logE_sig * np.random.normal(size=N)

    eta = np.random.uniform(0., 2.*np.pi, N)
    arr["trueRa"] = arr["ra"] + np.cos(eta) * arr["sigma"] / np.sqrt(1. - arr["sinDec"]**2)
    arr["trueDec"] = np.arcsin(arr["sinDec"]) + np.sin(eta) * arr["sigma"]
    arr["trueRa"] = np.arccos(np.cos(arr["trueRa"]))
    arr["trueDec"] = np.arcsin(np.sin(arr["trueDec"]))
    arr["trueE"] = np.random.lognormal(mean=logE_mean, sigma=logE_sig, size=N)
    arr["ow"] = 1./N

    return arr

def init(Nexp, NMC, energy=False, **kwargs):
    arr_exp = exp(Nexp)
    arr_mc = MC(NMC)

    if energy:
        llh_model = EnergyLLH(sinDec_bins=max(2, Nexp // 25),
                              sinDec_range=[-1., 1.],
                              twodim_bins=max(3, Nexp // 100),
                              twodim_range=[[0.9 * min(arr_exp["logE"].min(),
                                                       arr_mc["logE"].min()),
                                             1.1 * max(arr_exp["logE"].max(),
                                                       arr_mc["logE"].max())],
                                            [-1., 1.]])
    else:
        llh_model = ClassicLLH(sinDec_bins=max(2, Nexp // 25),
                               sinDec_range=[-1., 1.])

    llh = PointSourceLLH(arr_exp, arr_mc, 365., llh_model=llh_model,
                         seed=kwargs.pop("seed", 1234),
                         mode="all", hemispheres=dict(FullSky=(-np.inf, np.inf)),
                         nsource_bounds=((-0.5 if not energy else 0.) * Nexp,
                                         0.5 * Nexp),
                         **kwargs)

    return llh

def multi_init(n, Nexp, NMC, **kwargs):
    energy = kwargs.pop("energy", False)
    seed = kwargs.pop("seed", 0)
    llh = MultiPointSourceLLH(seed=seed, **kwargs)

    seed += 1

    for i in range(n):
        llh.add_sample("S{0:02d}".format(i), init(np.random.poisson(Nexp // n),
                                                  NMC, energy=energy,
                                                  seed=seed + i))

    return llh

