# -*-coding:utf8-*-

from data import init

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(1).sf(TS) + chi2(1).cdf(-TS)))

if __name__=="__main__":

    # init the llh class
    llh = init(1000, 10000)

    print(llh)

    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(nside=16,
                                                         pVal=pVal_func)):
        if i > 0:
            # break after first follow up
            break

    # plot results
    hp.mollview(scan["pVal"], min=0., cmap=plt.cm.afmhot)
    hp.projscatter(np.degrees(llh.exp["ra"]),
                   np.degrees(np.arcsin(llh.exp["sinDec"])),
                   lonlat=True, marker="x", color="red")
    plt.show()

