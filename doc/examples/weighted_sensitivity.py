# -*-coding:utf8-*-

from __future__ import print_function

# SciPy
from scipy.stats import chi2
import healpy as hp
import numpy as np
from scipy.signal import convolve2d

# skylab
from skylab.ps_injector import PointSourceInjector
from skylab.psLLH import MultiPointSourceLLH
from skylab.ps_model import UniformLLH
from skylab.utils import poisson_weight

# local
import utils

if __name__=="__main__":
    plt = utils.plotting(backend="pdf")

    llh, mc = utils.startup()
    print(llh)

    N = 10
    Gamma = np.linspace(1., 4., N)

    fig, ax = plt.subplots()

    for energy in [True, False]:

        l = "w/"
        if not energy:
            l += "o"

            llh.llh_model = (UniformLLH(sinDec_bins=max(3, len(llh.exp) // 200),
                                        sinDec_range=[-1., 1.]),
                             mc)

        props = dict()
        sens = list()
        disc = list()
        for j, gamma in enumerate(Gamma):
            # init a injector class sampling events at a point source
            inj = PointSourceInjector(gamma, sinDec_bandwidth=1.)

            # start calculation for dec = 0
            result = llh.weighted_sensitivity(0., [0.5, 2.87e-7], [0.9, 0.5],
                                              inj, mc,
                                              n_bckg=10000,
                                              n_iter=1000,
                                              eps=1.e-2,
                                              **props)

            props["TSval"] = result["TSval"]

            sens.append(result["mu"][0])
            disc.append(result["mu"][1])

        p = ax.plot(Gamma, disc, label="{0:s} energy".format(l))
        ax.plot(Gamma, sens, color=p[0].get_color(), linestyle="dashed")

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mu_\mathrm{inj}$")

    ax.set_ylim(ymin=0.)

    ax.legend(loc="best")

    fig.savefig("figures/nevents.pdf")

    plt.show()

