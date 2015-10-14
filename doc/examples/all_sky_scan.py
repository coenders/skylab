# -*-coding:utf8-*-

import data
from skylab.psLLH import MultiPointSourceLLH

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(1).sf(TS) + chi2(1).cdf(-TS)))

if __name__=="__main__":

    # init the llh class
    llh = data.multi_init(4, 1000, 10000, ncpu=4)

    print(llh)

    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(nside=32,
                                                         pVal=pVal_func,
                                                         decRange=np.radians([-90., 90.]))):
        if i > 0:
            # break after first follow up
            break

    # plot results
    hp.mollview(scan["pVal"], min=0., cmap=plt.cm.afmhot)
    if isinstance(llh, MultiPointSourceLLH):
        for llh in llh._sams.itervalues():
            hp.projscatter(np.degrees(llh.exp["ra"]),
                           np.degrees(np.arcsin(llh.exp["sinDec"])),
                           lonlat=True, marker="x", color=plt.gca()._get_lines.color_cycle.next())
    else:
        hp.projscatter(np.degrees(llh.exp["ra"]),
                       np.degrees(np.arcsin(llh.exp["sinDec"])),
                       lonlat=True, marker="x", color="red")
    plt.show()

