# -*-coding:utf8-*-

import data

from scipy.stats import chi2
import healpy as hp
import numpy as np
from scipy.signal import convolve2d
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
    #llh = data.multi_init(2, 1000, 250000, ncpu=4, energy=True)
    llh = data.init(2000, 250000, ncpu=4, energy=True)

    N_MC = 1000
    if isinstance(llh, MultiPointSourceLLH):
        mc = dict([(key, data.MC(N_MC)) for key in llh._enum.iterkeys()])
    else:
        mc = data.MC(N_MC)

    print(llh)

    gamma = 2.
    # init a injector class sampling events at a point source
    inj = PointSourceInjector(gamma, sinDec_bandwidth=1.)

    # start calculation for dec = 0
    result = llh.weighted_sensitivity(0., [0.5, 2.87e-7], [0.9, 0.5],
                                      inj, mc,
                                      n_bckg=10000,
                                      n_iter=10000,
                                      eps=1.e-2)

    trials = result["trials"]

    fig, (ax, ) = plt.subplots(nrows=1, ncols=len(llh.params) + 1, squeeze=False)

    mu = np.arange(0, 11, 2)
    ax[0].hist([pVal_func(trials["TS"], 0.) for i in mu],
               weights=[poisson_weight(trials["n_inj"], i) for i in mu],
               bins=200, label=[r"$\mu={0:.0f}$".format(i) for i in mu],
               histtype="stepfilled", alpha=0.5, log=True)
    ax[0].legend()

    m = trials["n_inj"] > 0
    for i, par in enumerate(llh.params):
        xbins = np.arange(trials["n_inj"].max() + 2) - 0.5
        ybins = np.linspace(*(xbins[[0, -1]] + 0.5 if i < 1 else llh.par_bounds[i]),
                            num=1000)
        h, xbins, ybins = np.histogram2d(trials["n_inj"], trials[par],
                                         bins=[xbins, ybins])

        kernel = np.ones((3, 3), dtype=np.float)
        h = convolve2d(h, kernel, mode="same") / convolve2d(np.ones_like(h), kernel, mode="same")

        norm = np.sum(h, axis=1, dtype=np.float)
        h = np.cumsum(h, axis=1, dtype=np.float)
        norm[norm < 1] = 1.
        np.set_printoptions(precision=1, linewidth=100)
        h /= norm[np.newaxis].T
        h = np.where(h < 0.5, h, -h + 1) / 0.5

        p = ax[i + 1].contourf((xbins[1:] + xbins[:-1]) / 2.,
                               (ybins[1:] + ybins[:-1]) / 2., h.T,
                               levels=np.linspace(0., 1., 11)[2:],
                               cmap=plt.cm.Spectral_r)
        plt.colorbar(mappable=p, ax=ax[i + 1])

        if i < 1:
            ax[i + 1].set_aspect("equal", adjustable="box")
        elif par == "gamma":
            ax[i + 1].axhline(gamma, color="black")
        ax[i + 1].set_xlim(0., xbins[1:][norm > 20][-1])


    plt.show()

