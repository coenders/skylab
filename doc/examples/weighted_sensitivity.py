# -*-coding:utf8-*-

import data

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from skylab.ps_injector import PointSourceInjector
from skylab.psLLH import MultiPointSourceLLH
from skylab.utils import poisson_weight

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))
if __name__=="__main__":
    import matplotlib.pyplot as plt

    # init likelihood class
    llh = data.multi_init(4, 1000, 250000, ncpu=4, energy=False)

    N_MC = 1000
    if isinstance(llh, MultiPointSourceLLH):
        mc = dict([(key, data.MC(N_MC)) for key in llh._enum.iterkeys()])
    else:
        mc = data.MC(N_MC)

    print(llh)

    # init a injector class sampling events at a point source
    inj = PointSourceInjector(2., sinDec_bandwidth=0.1)

    # start calculation for dec = 0
    result = llh.weighted_sensitivity(0., [0.5, 2.87e-7], [0.9, 0.5],
                                      inj, mc,
                                      n_bckg=10000,
                                      n_iter=2000,
                                      eps=2.e-2)

    trials = result["trials"]
    mu = np.arange(0, 11, 2)

    plt.hist([pVal_func(trials["TS"], 0.) for i in mu],
             weights=[poisson_weight(trials["n_inj"], i) for i in mu],
             bins=200,
             label=[r"$\mu={0:.0f}$".format(i) for i in mu],
             histtype="stepfilled", alpha=0.5, log=True)
    plt.legend()
    plt.show()

