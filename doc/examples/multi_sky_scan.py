# -*-coding:utf8-*-

from data import multi_init

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(1).sf(TS) + chi2(1).cdf(-TS)))

if __name__=="__main__":

    # init the llh class
    llh = multi_init(3, 1000, 10000,
                     energy=False, ncpu=4)

    print(llh)

    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(nside=32,
                                                         pVal=pVal_func)):
        if i > 0:
            # break after first follow up
            break

    # plot results
    hp.mollview(scan["pVal"], min=0., cmap=plt.cm.afmhot)
    col = plt.gca()._get_lines.color_cycle
    for enum, llh_i in llh._sams.iteritems():
        hp.projscatter(np.degrees(llh_i.exp["ra"]),
                       np.degrees(np.arcsin(llh_i.exp["sinDec"])),
                       lonlat=True, marker="x", color=col.next())
    plt.show()

