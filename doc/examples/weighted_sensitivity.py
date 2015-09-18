# -*-coding:utf8-*-

import data

import numpy as np

from skylab.ps_injector import PointSourceInjector
from skylab.utils import poisson_weight

import matplotlib
matplotlib.use("QT4Agg")
import matplotlib.pyplot as plt

if __name__=="__main__":

    # init likelihood class
    llh = data.init(1000, 1000, ncpu=4)
    mc = data.MC(100000)

    print(llh)

    # init a injector class sampling events at a point source
    inj = PointSourceInjector(2., seed=0)

    # start calculation for dec = 0
    x = list()
    y = list()
    y2 = list()
    ndec = 1
    nmu = 7
    for dec in np.linspace(-np.pi/2., np.pi/2., ndec + 2)[1:-1]:
        inj.fill(dec, mc)
        #llh.do_trials(dec, n_iter=10)
        #continue
        result = llh.weighted_sensitivity(dec, [0.5, 2.87e-7],
                                               [0.9, 0.5],
                                          inj,
                                          #fit="exp",
                                          n_bckg=1000,
                                          n_iter=250,
                                          eps=5.e-2)

        mu = np.unique(np.array(np.linspace(0., max(result["mu"]), nmu),
                                dtype=np.int))

        t = result["trials"]

        bins = np.linspace(*np.percentile(t["TS"], [5., 100.]), num=500)

        w = [poisson_weight(t["n_inj"], i) for i in mu]

        plt.hist(t["TS"], weights=w[0], bins=bins, histtype="step",
                 label=r"$\mu=0$", cumulative=-1, normed=True)
        plt.hist([t["TS"] for i in mu[1:]][::-1], weights=w[1:][::-1],
                 bins=bins, histtype="step",
                 label=[r"$\mu={0:d}$".format(i) for i in mu[1:]][::-1],
                 cumulative=1,
                 normed=True)
        plt.legend(loc="best")
        plt.show()

