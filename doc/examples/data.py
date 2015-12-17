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
from skylab.ps_model import UniformLLH, EnergyLLH, PowerLawLLH

log_mean = np.log(np.radians(0.5))
log_sig = 0.2
logE_res = 0.1

np.random.seed(1)

def exp(N=100):
    r"""Create uniformly distributed data on sphere. """
    g = 3.7

    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float)])

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["sinDec"] = np.random.uniform(-1., 1., N)

    E = np.log10(np.random.pareto(g, size=N) + 1)
    arr["sigma"] = np.random.lognormal(mean=log_mean, sigma=log_sig, size=N)
    arr["logE"] = E + logE_res * np.random.normal(size=N)

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
    arr["trueE"] = np.random.pareto(g, size=N) + 1
    arr["ow"] = arr["trueE"]**(g)
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
        '''
        llh_model = EnergyLLH(sinDec_bins=min(50, Nexp // 50),
                              sinDec_range=[-1., 1.],
                              twodim_bins=min(50, Nexp // 50),
                              twodim_range=[[0.9 * min(arr_exp["logE"].min(),
                                                       arr_mc["logE"].min()),
                                             1.1 * max(arr_exp["logE"].max(),
                                                       arr_exp["logE"].max())],
                                             [-1., 1.]])
        '''
        llh_model = PowerLawLLH(["logE"], min(50, Nexp // 50),
                                twodim_range=[0.9 * arr_mc["logE"].min(),
                                              1.1 * arr_mc["logE"].max()],
                                sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.])
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


