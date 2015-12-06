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

if __name__=="__main__":
    import matplotlib.pyplot as plt

    # init likelihood class
    #llh = data.multi_init(2, 1000, 250000, ncpu=4, energy=True)
    llh = data.init(10000, 250000, ncpu=4, energy=True)

    N_MC = 100000
    if isinstance(llh, MultiPointSourceLLH):
        mc = dict([(key, data.MC(N_MC)) for key in llh._enum.iterkeys()])
    else:
        mc = data.MC(N_MC)

    print(llh)

    # data

    gamma = np.linspace(1., 2.7, 4)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.hist([llh.exp["logE"]] + [mc["logE"] for i in gamma],
             weights=[np.ones(len(llh.exp))]
                      + [mc["ow"] * mc["trueE"]**(-g) for g in gamma],
             histtype="step", bins=100, log=True, normed=True)
    dec = np.arcsin(mc["sinDec"])
    angdist = np.degrees(np.arccos(np.cos(mc["trueRa"] - mc["ra"])
                                    * np.cos(mc["trueDec"]) * np.cos(dec)
                                   + np.sin(mc["trueDec"]) * mc["sinDec"]))
    ax2.hist([np.log10(np.degrees(llh.exp["sigma"]))] + [np.log10(np.degrees(mc["sigma"])) for i in gamma],
             weights=[np.ones(len(llh.exp))] + [mc["ow"] * mc["trueE"]**(-g) for g in gamma],
             histtype="step", bins=100, normed=True)

    plt.show()

    ### LLH SCAN ###

    Gamma = 2.
    # init a injector class sampling events at a point source
    inj = PointSourceInjector(Gamma, sinDec_bandwidth=1., seed=0)
    inj.fill(0., mc, 333.)

    mu = 50.
    n = 50
    gamma = np.linspace(*llh.par_bounds[1], num=n)

    ntrials = 25
    T = list()
    for i in range(ntrials):
        inject = inj.sample(mu, poisson=True).next()[1]

        TS, xmin = llh.fit_source(np.pi, 0., inject=inject)

        nsources = np.linspace(0.5 * xmin["nsources"],
                               2. * xmin["nsources"], 2 * n)

        X, Y = np.meshgrid(nsources[:-1], gamma[:-1])

        X = X.ravel()
        Y = Y.ravel()

        Z = np.empty_like(X)

        for i, (x, y) in enumerate(zip(X, Y)):
            Z[i] = llh.llh(nsources=x, gamma=y)[0]

        Z = -2. * (Z - Z.max())

        h, xb, yb, = np.histogram2d(X, Y, weights=Z, bins=[nsources, gamma])

        T.append(h)

        plt.contour(nsources[:-1], gamma[:-1], h.T, levels=[2.3], colors="grey", alpha=0.25)
        plt.scatter(xmin["nsources"], xmin["gamma"], marker="x")

    T = sum(T) / ntrials
    T -= T.min()
    plt.contour(nsources[:-1], gamma[:-1], T.T, levels=[2.3], colors="red")
    plt.scatter(mu, Gamma, marker="o", color="red")
    plt.xlim(0., np.amax(nsources))
    plt.show()

