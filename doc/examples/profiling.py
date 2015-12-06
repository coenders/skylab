#! /usr/bin/env python
# -*-coding:utf8-*-

import time

import data

import numpy as np

#from skylab.ps_injector import PointSourceInjector
from skylab.psLLH import MultiPointSourceLLH

if __name__=="__main__":
    import matplotlib.pyplot as plt

    for ncpu in range(1, 3):
        N = list()
        T = list()
        for ndec in range(3, 6):
            start = time.time()

            # init likelihood class
            llh = data.init(10**ndec, 10**ndec, ncpu=ncpu, energy=True)
            print(llh)

            trials = llh.do_trials(0., n_iter=100 * max(1, 100 // 10**ndec))

            stop = time.time()

            print("Finished {0:d} trials in {1:.2f} sec".format(len(trials), stop - start))

            N.append(10**ndec)
            T.append((stop - start) / len(trials))

        plt.plot(N, T, label="{0:d} events, {1:d} cores".format(10**ndec, ncpu))

    plt.semilogx(nonposx="clip")
    plt.legend(loc="best")
    plt.show()

