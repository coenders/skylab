# -*-coding:utf8-*-

import data

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from skylab.ps_injector import PointSourceInjector
from skylab.psLLH import MultiPointSourceLLH

if __name__=="__main__":

    # init likelihood class
    llh = data.multi_init(4, 1000, 100000, ncpu=1, energy=True)

    N_MC = 1000
    if isinstance(llh, MultiPointSourceLLH):
        mc = dict([(key, data.MC(N_MC)) for key in llh._enum.iterkeys()])
    else:
        mc = data.MC(N_MC)

    print(llh)

    # init a injector class sampling events at a point source
    inj = PointSourceInjector(2.)

    # start calculation for dec = 0
    result = llh.weighted_sensitivity(0., [0.5, 2.87e-7], [0.9, 0.5],
                                      inj, mc,
                                      n_bckg=1000,
                                      n_iter=1000,
                                      eps=5.e-2)

