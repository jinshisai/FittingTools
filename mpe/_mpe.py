# coding: utf-8
'''
Model-Parameter Estimator (MPE) is a python package containing various methods
to estimate the best parameters for a given model and a data set.

Developed by Jinshi Sai (Insa Choi)
Dec 23 2022
'''


### Modules
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner
from typing import Callable
from multiprocessing.dummy import Pool
#from mpipool import MPIPool
import gc
from scipy.stats import gaussian_kde

from .funcs import gauss1d


### Functions for MCMC
# Logarithm of prior distribution
def uniform_lnprior(params, pranges):
    # If all parameters are within the given parameter ranges
    if np.all((pranges[0] < params) * (params < pranges[1])):
        return 0.0
    else:
        return -np.inf

# Logarithm of likelihood function assuming Gaussian distribution
def gauss_lnlike(params, d, derr, fmodel, *x):
    model = fmodel(*x, *params)

    # Likelihood function (in log)
    exp = -0.5*np.nansum((d-model)**2/(derr*derr) + np.log(2.*np.pi*derr*derr))
    if np.isnan(exp):
        return -np.inf
    else:
        return exp

# Logarithm of posterior distribution
#def lnprob(params, pranges, d, derr, fmodel, *x):
#    # According to Bayes' theorem
#    return  lnprior(params, pranges) + lnlike(params, d, derr, fmodel, *x)


### Python class for model fitting
class BayesEstimator():
    '''
    Estimate the best parameters for a given model and a data set using Bayesian methods.
    '''

    # initialize
    def __init__(self, axes: list, data: np.ndarray,
        sig_d: float or np.ndarray, model: Callable,
        lnlike: Callable = gauss_lnlike, lnprior: Callable = uniform_lnprior):
        '''

        Parameters
        ----------
        axes (list): Coordinates of the data set. Must be given as a list containing
            all dimentional coordinates (e.g., [x, y, z] for three dimention case,
            where x, y and z each is a coordinate array for the axis).
        data (ndarray): Observed/simulated data set.
        sig_d (ndarray): Uncertainty of the data
        model (function): The model to be tested for the given data set.
        '''
        # Given data/model
        self.axes = axes
        self.data = data
        self.sig_d = sig_d
        self.model = model

        # initialize
        self.pini    = None
        self.pranges = None

        # prior/likelihood functions
        self.lnlike = lnlike
        self.lnprior = lnprior


    # Run mcmc
    def run_mcmc(self, pini, pranges, outname=None,
        nwalkers=None, nrun=5000, nburn=500, labels=[], show_progress=True,
        f_rand_init=0.1, credible_interval=0.68, show_results=True,
        optimize_ini=True, moves=emcee.moves.StretchMove(), symmetric_error=False,
        npool=1, errtype='gauss', savefig = True, savesampler = True, restart = False):
        '''
        A wrapper to run MCMC with emcee.

        Parameters
        ----------
        pini (list or array): Parameters to determine initial positions for mcmc run
        pranges (list or array): Parameter ranges to search for.
            Must be given as pranges = np.array([min_values,...,],[max_values,...,]])
        '''
        ### setting of parameters for mcmc
        # ndim: number of free parameters
        # nwalkers: number of mcmc run path
        # maxiter: ?
        # nrun: step number
        # nburn: cutoff of mcmc run result. You'll use only results after nburn mcmc run.
        # Nthin: ?
        # N_posterior:

        # parameters
        ndim = len(pini)
        if nwalkers is None:
            nwalkers = 2*ndim

        if outname is None:
            outname = 'modelfit'

        # Output name
        out_chain    = outname + '_chain'
        out_triangle = outname + '_triangle.pdf'
        out_text     = outname + '_results.txt'

        # Param labels
        if len(labels) == 0:
            labels = np.array([r'$p_{%i}$'%i for i in range(ndim)])
        elif len(labels) == ndim:
            pass
            #print ('Use given labels.')
        else:
            print ('WARNING: Number of given labels does not match nparams. Ignore input labels.')
            labels = np.array([r'$p_{%i}$'%i for i in range(ndim)])

        # Initial set of positions for walkers
        # Maximum likelihood
        if optimize_ini:
            nll = lambda params, args: -self.lnlike(params, *args) # negative ln likelihood
            res = op.minimize(nll, pini, [self.data, self.sig_d, self.model, *self.axes],
                bounds=[ [pranges[0][i], pranges[1][i]] for i in range(ndim) ],
                method='Nelder-Mead', tol=1e-3)
            pini = res.x
            print('Optimized initial: ', pini)
        else:
            if type(pini) == list:
                pini = np.array(pini)

        # initial values for each walk
        p_pertb = pini * f_rand_init * 0.5
        random = np.array([
            [np.random.uniform(low=pranges[0][j] - pini[j], high=pranges[1][j] - pini[j])
            for i in range(nwalkers)]
            for j in range(ndim)]) * f_rand_init
        p0 = [pini + random[:,i] for i in range(nwalkers)]


        # save samples
        sample_file = outname + '_sample.h5'
        if savesampler:
            backend = emcee.backends.HDFBackend(sample_file)

            # continue from saved samples
            if restart:
                if backend.iteration == 0:
                    raise ValueError(f'Cannot restart: {sample_file} has no saved samples.')

                nwalkers_backend, ndim_backend = backend.shape
                if nwalkers_backend != nwalkers or ndim_backend != ndim:
                    raise ValueError('Saved sampler shape does not match this run.')

                p0_run = None
                print(f'Restarting MCMC from {sample_file} at step {backend.iteration}')
            else:
                backend.reset(nwalkers, ndim)
                p0_run = p0
        else:
            backend = None
            p0_run = p0


        # Begin MCMC run
        print ('Run MCMC!')

        # save parameters
        self.pini = pini
        self.pranges = pranges
        self.p0 = p0
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nburn = nburn
        self.nrun = nrun


        # Logarithm of posterior distribution
        def lnprob(params, pranges, d, derr, fmodel, *x):
            # According to Bayes' theorem
            return  self.lnprior(params, pranges) + self.lnlike(params, d, derr, fmodel, *x)


        # Multi processing
        if npool > 1:
            with Pool(npool) as pool:
                # Choose sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                        args=[self.pranges, self.data, self.sig_d, self.model, *self.axes],
                        pool=pool, moves=moves, backend = backend)
                # Run nrun steps showing progress
                results = sampler.run_mcmc(p0_run, nrun, progress=show_progress)
        else:
            # Choose sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                    args=[self.pranges, self.data, self.sig_d, self.model, *self.axes],
                    moves=moves, backend = backend)
            # Run nrun steps showing progress
            results = sampler.run_mcmc(p0_run, nrun, progress=show_progress)
        self.sampler = sampler
        self.results = results


        # Burn first nburn steps
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))


        # Output fitting results
        dt = datetime.now()
        dtstr = dt.strftime('%Y-%m-%d %H:%M:%S')
        if symmetric_error:
            p_fit = np.empty((2, ndim))
        else:
            p_fit = np.empty((3, ndim))
        print('Best paramters')
        with open(out_text, '+w') as f:
            f.write('# Fitting results\n')
            f.write('# Date: '+ dtstr + '\n')
            # results + error
            if symmetric_error:
                print('mean 1sigma')
                f.write('# param mean sigma\n')
                for i in range(ndim):
                    hist, bin_e = np.histogram(samples[:, i], bins=int(np.sqrt(len(samples[:, i]))))
                    bin_c = 0.5*(bin_e[:-1] + bin_e[1:])
                    if errtype == 'gauss':
                        p_mcmc, _ = op.curve_fit(gauss1d, bin_c, hist,
                            p0=[np.nanmax(hist), np.mean(samples[:, i]), np.std(samples[:, i])])
                        mn = p_mcmc[1]
                        err = p_mcmc[2]
                    else:
                        mn = np.mean(samples[:, i])
                        err = np.std(samples[:, i])
                    outtxt = '%s %13.6e %13.6e\n'%(labels[i], mn, err)
                    print(outtxt)
                    f.write(outtxt)
                    p_fit[:,i] = [mn, err]
            else:
                print('Credible interval: %i percent'%(credible_interval*100.))
                print('mode lower upper')
                f.write('# param 50th %ith %ith\n'%(50*(1. - credible_interval), 50*(1. + credible_interval)))
                for i in range(ndim):
                    p_mcmc = np.percentile(samples[:, i],
                        [50*(1. - credible_interval), 50, 50*(1. + credible_interval)])
                    q = np.diff(p_mcmc)
                    outtxt = '%s %13.6e %13.6e %13.6e\n'%(labels[i], p_mcmc[1], q[0], q[1])
                    print(outtxt)
                    f.write(outtxt)
                    p_fit[:,i] = [p_mcmc[1], q[0], q[1]]

        # Result figures
        if any([show_results, savefig]):
            # Chain plot
            for n0, fout in zip(
                [0, nburn],
                [out_chain + '_full.pdf', out_chain + '_conv.pdf']):
                fig, axes = plt.subplots(ndim, 1, figsize=(11.69, 8.27), sharex=True)
                xplot = np.arange(n0, nrun, 1)
                for i, ax in enumerate(axes):
                    chain_plot = np.array(
                    [ax.plot(xplot, sampler.chain[iwalk,n0:,i].T, 'k')
                     for iwalk in range(nwalkers)])
                    ax.set_ylabel(labels[i])
                    ax.tick_params(which='both', direction='in',
                        bottom=True, top=True, left=True, right=True, labelbottom=False)
                # x-label
                axes[0].set_xlim(n0,nrun)
                axes[-1].tick_params(labelbottom=True)
                axes[-1].set_xlabel('Step number')
                if savefig: fig.savefig(fout, transparent=True)
                if show_results: plt.show()

            # Corner plot
            fig2 = corner.corner(samples, labels=labels)#, quantiles=[0.16, 0.5, 0.84])
            if savefig: fig2.savefig(out_triangle, transparent=True)
            if show_results: plt.show()

        self.pfit = p_fit
        self.get_criterion()
        print('Criterion: ', self.criterion)


    def get_criterion(self):
        # best parameter
        popt = self.pfit[0]
        # Akaike's Information Criterion (AIC)
        aic = - 2. * self.lnlike(popt, self.data, self.sig_d, self.model, *self.axes)\
         + 2.* self.ndim
        # Bayesian Information Criterion (BIC)
        bic = - 2. * self.lnlike(popt, self.data, self.sig_d, self.model, *self.axes)\
         + self.ndim * np.log(self.data.size)
        # output
        self.criterion = {'AIC': aic, 'BIC': bic}



def get_sample_mode(samples, ngrid = 1000):
    ns, ndim = samples.shape

    modes = []
    for i in range(ndim):
        _samples = samples[:,i]
        kde = gaussian_kde(_samples)
        x = np.linspace(min(_samples), max(_samples), ngrid)
        mode = x[np.argmax(kde(x))]
        modes.append(mode)

        # for check
        #print('mode %.2f'%mode)
        #print('median %.2f'%np.median(_samples))

    return modes


def HPDI(samples, ngrid = 1000, credible_interval = 0.683):
    ns, ndim = samples.shape

    modes = []
    plow = []
    pup = []
    for i in range(ndim):
        _samples = samples[:,i]
        kde = gaussian_kde(_samples)
        x = np.linspace(min(_samples), max(_samples), ngrid)
        pdf = kde(x) # probability density function

        # get mode
        mode = x[np.argmax(pdf)]
        modes.append(mode)


        # get credible interval
        # Sort by density
        sorted_indices = np.argsort(-pdf)
        x_sorted = x[sorted_indices]
        pdf_sorted = pdf[sorted_indices]

        # Cumulative density
        cumulative_density = np.cumsum(pdf_sorted)
        cumulative_density /= cumulative_density[-1]

        # HPDI computation
        hpdi_indices = np.where(cumulative_density <= credible_interval)[0]
        hpdi_range = (np.nanmin(x_sorted[hpdi_indices]),
            np.nanmax(x_sorted[hpdi_indices]))
        plow.append(mode - hpdi_range[0])
        pup.append(hpdi_range[1] - mode)

    return modes, plow, pup



def ETI(samples, credible_interval = 0.683):
    ns, ndim = samples.shape
    pmid = []
    plow = []
    pup = []
    for i in range(ndim):
        eti_range = np.percentile(samples[:, i],
            [50*(1. - credible_interval), 50, 50*(1. + credible_interval)])
        q = np.diff(eti_range)

        pmid.append(eti_range[1])
        plow.append(q[0])
        pup.append(q[1])
    return pmid, plow, pup



class PosteriorSamples():

    def __init__(self, samples, labels = None):

        if type(samples) == str:
            reader = emcee.backends.HDFBackend(samples)
            self.samples = np.transpose(reader.get_chain(discard=0, flat=False), (1,0,2))
        else:
            self.samples = samples

        self.nwalkers, self.nsteps_per_walker, self.ndim = self.samples.shape


        # Parameter labels
        if labels is None:
            labels = np.array([r'$p_{%i}$'%i for i in range(self.ndim)])
        elif len(labels) == self.ndim:
            pass
        else:
            print ('WARNING: Number of given labels does not match nparams. Ignore input labels.')
            labels = np.array([r'$p_{%i}$'%i for i in range(self.ndim)])
        self.labels = labels


    def get_credible_range(self, nburnin=500,
        labels=[], credible_interval=0.683,
        symmetric_error = False, credible_type = 'HDPI',
        symmetric_error_type = 'gauss'
        ):
        '''
        Calculate credible range.

        Parameters
        ----------
         nburnin (int): burnin
        '''
        # flatten sample array
        samples = self.samples[:, nburnin:, :].reshape((-1, self.ndim))

        #print(samples.shape)
        # calculate credible range
        if symmetric_error:
            p_fit = np.empty((2, self.ndim))
            for i in range(self.ndim):
                hist, bin_e = np.histogram(samples[:, i], 
                    bins=int(np.sqrt(len(samples[:, i]))))
                bin_c = 0.5*(bin_e[:-1] + bin_e[1:])
                if errtype == 'gauss':
                    p_mcmc, _ = op.curve_fit(gauss1d, bin_c, hist, 
                        p0=[np.nanmax(hist), np.mean(samples[:, i]), np.std(samples[:, i])])
                    mn = p_mcmc[1]
                    err = p_mcmc[2]
                else:
                    mn = np.mean(samples[:, i])
                    err = np.std(samples[:, i])
                p_fit[:,i] = [mn, err]
        else:
            if credible_type == 'HPDI':
                popt, plow, pup = HPDI(samples, credible_interval = credible_interval,
                    ngrid = 1000)
                p_fit = np.array([popt, plow, pup])
            elif credible_type == 'ETI':
                popt, plow, pup = ETI(samples, credible_interval = credible_interval,)
                p_fit = np.array([popt, plow, pup])
            else:
                print('ERROR\tcredible_range: input credible_type is wrong.')
                print('ERROR\tcredible_range: Currently only credible_type = HPDI and ETI are supported.')
                return 0

        self.credible_range = p_fit
        return p_fit


    def plot_chains(self, nburnin = 0, fig = None, outname = None,
        showfig = True, savefig = False, return_fig = False, figsize=(11.69, 8.27)):

        # flatten sample array
        samples = self.samples[:, nburnin:, :].reshape((-1, self.ndim))

        if fig is None:
            fig, axes = plt.subplots(self.ndim,
                1, figsize=figsize, sharex=True)

        xplot = np.arange(nburnin, self.nsteps_per_walker, 1)

        for i, ax in enumerate(axes):
            for iwalk in range(self.nwalkers):
                ax.plot(xplot, self.samples[iwalk, nburnin:, i].T, 'k')
            ax.set_ylabel(self.labels[i])
            ax.tick_params(which='both', direction='in',
                bottom=True, top=True, left=True, right=True, labelbottom=False)
        # x-label
        axes[0].set_xlim(nburnin, self.nsteps_per_walker)
        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel('Step number')
        if savefig: fig.savefig(outname, transparent=True)
        if showfig: plt.show()
        if return_fig:
            return figs
        else:
            plt.close()


    def plot_corner(self, nburnin = 0,
        fig = None, outname = None,
        showfig = True, savefig = False, return_fig = False,):

        # flatten sample array
        samples = self.samples[:, nburnin:, :].reshape((-1, self.ndim))
        nsteps, _ = samples.shape

        fig = corner.corner(samples, labels=self.labels)
        if savefig: fig.savefig(outname, transparent=True)
        if showfig: plt.show()
        if return_fig:
            return figs
        else:
            plt.close()