# -*-coding:utf8-*-

import healpy as hp
import numpy as np

import data

tw = 5.31

def startup(NN=1, multi=False, **kwargs):
    n = 4
    Nexp = 10000 // NN
    NMC = 500000 // NN
    if multi:
        llh = data.multi_init(n, Nexp, NMC, ncpu=4, **kwargs)
        mc = dict([(i, data.MC(NMC)) for i in range(n)])
    else:
        llh = data.init(Nexp, NMC, ncpu=4, **kwargs)
        mc = data.MC(NMC)

    return llh, mc

def plotting(backend="QT4Agg"):
    import matplotlib as mpl
    if backend is not None:
        mpl.use(backend)

    rcParams = dict()
    rcParams["text.size"] = 10
    rcParams["font.size"] = 10
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern"]
    rcParams["mathtext.fontset"] = "cm"
    rcParams["text.usetex"] = True
    rcParams["lines.linewidth"] = 1.1
    rcParams["figure.dpi"] = 72.27
    rcParams["figure.figsize"] = (tw, tw / 1.6)
    rcParams["figure.autolayout"] = True
    rcParams["axes.color_cycle"] =["#d7191c", "#2b83ba", "#756bb1",
                                   "#fdae61", "#abdda4"]
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10

    mpl.rcParams.update(rcParams)

    import matplotlib.pyplot as plt

    return plt

def skymap(plt, vals, **kwargs):
    fig, ax = plt.subplots(subplot_kw=dict(projection="aitoff"))

    gridsize = 1000

    x = np.linspace(np.pi, -np.pi, 2 * gridsize)
    y = np.linspace(np.pi, 0., gridsize)

    X, Y = np.meshgrid(x, y)

    r = hp.rotator.Rotator(rot=(-180., 0., 0.))

    YY, XX = r(Y.ravel(), X.ravel())

    pix = hp.ang2pix(hp.npix2nside(len(vals)), YY, XX)

    Z = np.reshape(vals[pix], X.shape)

    lon = x[::-1]
    lat = np.pi /2.  - y

    cb = kwargs.pop("colorbar", dict())
    cb.setdefault("orientation", "horizontal")
    cb.setdefault("fraction", 0.075)

    title = cb.pop("title", None)

    p = ax.pcolormesh(lon, lat, Z, **kwargs)

    cbar = fig.colorbar(p, **cb)

    cbar.solids.set_edgecolor("face")
    cbar.update_ticks()
    if title is not None:
        cbar.set_label(title)

    ax.xaxis.set_ticks([])

    return fig, ax

