# coding: utf-8

# modules
import numpy as np
import scipy.optimize as op
from mps import BayesEstimator
from mps.funcs import gaussian2d, gauss1d

# functions
def lnlike(params, d, derr, fmodel, *x):
    model = fmodel(*x, *params)

    # Likelihood function (in log)
    exp = -0.5*np.sum((d-model)**2/(derr*derr) + np.log(2.*np.pi*derr*derr))
    if np.isnan(exp):
        return -np.inf
    else:
        return exp

# main
def main():
    # ------------- input ----------------
    # simulated data
    x = np.linspace(-1,1,32)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    dd = gaussian2d(xx, yy, 1., 0., 0., 0.1, 0.1) # answer
    sig_d = 0.2
    d_sim = np.random.normal(dd, sig_d, dd.shape) # simulated data

    # initial parameter/parameter range
    pini = [1., 0.1, 0.1, 0.1, 0.1]
    pranges = [
        [0.001, -1., -1., 0.001, 0.001],
        [10., 1., 1., 1., 1.]
    ]
    # ------------------------------------

    be = BayesEstimator([xx, yy], d_sim, sig_d, gaussian2d)
    be.run_mcmc(pini, pranges, outname='test_wrap_mcmc_2d',
        nrun=1000, nburn=200,)# nwalkers=10)
    #print(be.criterion)

if __name__ == '__main__':
    main()