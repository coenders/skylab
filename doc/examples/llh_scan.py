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
from skylab.utils import poisson_weight

# local
import utils


if __name__=="__main__":

    plt = utils.plotting(backend="pdf")

    # init likelihood class
    llh, mc = utils.startup()

    print(llh)

    # data plot

    gamma = np.array([2., 2.3, 2.7])

    # energy
    fig_E, ax_E = plt.subplots()
    h, b, p = ax_E.hist([llh.exp["logE"]] + [mc["logE"] for i in gamma],
                        weights=[np.ones(len(llh.exp))]
                                 + [mc["ow"] * mc["trueE"]**(-g) for g in gamma],
                        label=["Data"] + [r"$\gamma={0:.1f}$".format(g) for g in gamma],
                        color=["black"]
                        + [ax_E._get_lines.color_cycle.next() for i in range(len(gamma))],
                        histtype="step", bins=25, log=True, normed=True)
    ax_E.legend(loc="best")

    d = np.power(10., np.floor(np.log10(np.percentile(h[0][h[0]>0], 5.))))
    ax_E.set_ylim(ymin=d)
    ax_E.set_xlim(xmax=np.ceil(np.percentile(mc["logE"], 99.9)))

    ax_E.set_xlabel("Energy Proxy")
    ax_E.set_ylabel("Probability Density")

    # mrs
    fig_p, ax_p = plt.subplots()
    h, b, p = ax_p.hist([np.degrees(llh.exp["sigma"])]
                        + [np.degrees(mc["sigma"]) for i in gamma],
                        weights=[np.ones(len(llh.exp))]
                                 + [mc["ow"] * mc["trueE"]**(-g) for g in gamma],
                        label=["Data"] + [r"$\gamma={0:.1f}$".format(g) for g in gamma],
                        color=["black"]
                        + [ax_p._get_lines.color_cycle.next() for i in range(len(gamma))],
                        histtype="step", bins=25, normed=True)
    ax_p.legend(loc="best")

    ax_p.set_xlabel("Sigma / $1^\circ$")
    ax_p.set_ylabel("Probability Density")

    fig_E.savefig("figures/energy.pdf")
    fig_p.savefig("figures/mrs.pdf")
    plt.show()
    plt.close("all")

    ### LLH SCAN ###

    fig, ax = plt.subplots()

    for mu, Gamma in zip([20, 40, 30], gamma):
        print("\tgamma =", Gamma)
        # init a injector class sampling events at a point source
        inj = PointSourceInjector(Gamma, sinDec_bandwidth=1., seed=0)
        inj.fill(0., mc, 333.)

        n = 25
        gamma = np.linspace(*llh.par_bounds[1], num=n)

        ntrials = 100
        T = list()

        nsources = np.linspace(0.5 * mu, 2. * mu, 2 * n)

        for i in range(ntrials):
            # Inject always the same number of events for this plot
            n_inj, inject = inj.sample(mu, poisson=False).next()

            TS, xmin = llh.fit_source(np.pi, 0., inject=inject)

            X, Y = np.meshgrid(nsources[:-1], gamma[:-1])

            X = X.ravel()
            Y = Y.ravel()

            Z = np.empty_like(X)

            for i, (x, y) in enumerate(zip(X, Y)):
                Z[i] = llh.llh(nsources=x, gamma=y)[0]

            Z = -2. * (Z - Z.max())

            h, xb, yb, = np.histogram2d(X, Y, weights=Z, bins=[nsources, gamma])

            T.append(h)

        T = np.sum(T, axis=0) / ntrials
        T -= T.min()

        xmin, ymin = np.unravel_index(np.argmin(T), T.shape)

        col = ax._get_lines.color_cycle.next()
        ax.contour(nsources[:-1], gamma[:-1], T.T, levels=[2.3, 4.6],
                   colors=col, linestyles=["dashed", "solid"],
                   antialiased=True)
        ax.scatter(nsources[xmin], gamma[ymin], marker="x", color=col)
        ax.scatter(mu, Gamma, marker="o", color=col, label=r"$\gamma={0:.1f}$".format(Gamma))

    ax.set_xlim(0, 70)

    ax.set_xlabel(r"$n_\nu$")
    ax.set_ylabel(r"$\gamma$")

    leg = ax.legend(ncol=len(gamma), scatterpoints=1,
                    bbox_to_anchor=(0., 1.02, 1., .102),
                    loc=3, mode="expand", borderaxespad=0, columnspacing=1.,
                    handlelength=0.5, frameon=False)

    fig.savefig("figures/contour.pdf", bbox_extra_artists=(leg, ), bbox_inches="tight")

    plt.show()

    plt.close("all")

