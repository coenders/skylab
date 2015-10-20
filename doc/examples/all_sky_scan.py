# -*-coding:utf8-*-

import data
from skylab.psLLH import MultiPointSourceLLH

from scipy.stats import chi2
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

if __name__=="__main__":

    # init the llh class
    llh = data.multi_init(4, 1000, 100000, ncpu=1, energy=True)

    print(llh)

    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(nside=16,
                                                         pVal=pVal_func,
                                                         decRange=np.radians([-90., 90.]))):
        if i > 0:
            # break after first follow up
            break

    # plot results
    hp.mollview(scan["pVal"], min=0., cmap=plt.cm.afmhot, rot=[-180., 0., 0.])
    if isinstance(llh, MultiPointSourceLLH):
        for llh in llh._sams.itervalues():
            hp.projscatter(np.degrees(llh.exp["ra"]),
                           np.degrees(np.arcsin(llh.exp["sinDec"])),
                           rot=[-180., 0., 0.],
                           lonlat=True, marker="x", color=plt.gca()._get_lines.color_cycle.next())
    else:
        hp.projscatter(np.degrees(llh.exp["ra"]),
                       np.degrees(np.arcsin(llh.exp["sinDec"])),
                       rot=[-180., 0., 0.],
                       lonlat=True, marker="x", color="red")

    plt.show()

    for key in llh.params:
        eps = 0.1
        q = scan[key]
        if not key == "nsources":
            q = np.ma.masked_array(q)
            q.mask = ~(scan["nsources"]
                       > np.percentile(scan["nsources"][scan["nsources"] > 0],
                                       eps))
        hp.mollview(q, unit=key, cmap=plt.cm.cubehelix,
                    rot=[-180., 0., 0.],
                    min=min(0., np.percentile(scan[key], eps)),
                    max=np.percentile(scan[key], 100. - eps))
        plt.show()

