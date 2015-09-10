# -*-coding:utf8-*-

import numpy as np
import time

from data import init

if __name__=="__main__":
    import matplotlib.pyplot as plt

    ndec = 5
    ntrials = 100

    X = np.arange(3.5, 5.5, 0.5)
    T = list()
    Terr = list()

    print("Spatial")
    for i in X:
        print(i)
        llh = init(10**i, 1)

        t = list()
        for dec in np.random.uniform(-np.pi/2., np.pi/2., ndec):
            start = time.clock()
            llh.do_trials(dec, n_iter=ntrials)
            t.append(time.clock() - start)
        T.append(np.mean(t))
        Terr.append(np.std(t))

    plt.errorbar(10.**X, T, yerr=Terr, label="Spatial LLH")
    plt.xlabel("Number of exp. events")
    plt.ylabel("Execution time for {0:d} trials".format(ntrials))
    plt.show()
