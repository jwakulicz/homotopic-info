import numpy as np
import itertools
from functools import partial

def logdet(cov):
    _, logdet = np.linalg.slogdet(cov)
    return logdet

def gaussian_kl(g1, g2):
    """ Computes the KL divergence between two gaussian distributions

    Inputs
    --------
        g1, g2: tuple
    tuples of (mean, covariance,) of the gaussian distribution.

    Returns
    --------
        float
    """
    
    mu1 = g1[0]
    cov1 = g1[1]
    mu2 = g2[0]
    cov2 = g2[1]

    n = np.shape(cov1)[0]
    inv_cov2 = np.linalg.inv(cov2)
    kl = 0.5 * (
                logdet(cov2) - logdet(cov1) + \
                (mu2 - mu1) @ inv_cov2 @ (mu2 - mu1).T + \
                np.trace(inv_cov2 @ cov1) - n
               )
    return kl

def gaussian_bhattacharya(g1, g2):
    """ Computes the Bhattacharya distance between two gaussian distributions

    Inputs
    --------
        g1, g2: tuple
    tuples of (mean, covariance,) of the gaussian distribution.

    Returns
    --------
        float
    """
    mu1 = g1[0]
    cov1 = g1[1]
    mu2 = g2[0]
    cov2 = g2[1]

    cov_mean = 0.5 * (cov1 + cov2)
    bhattacharya = 1/8 * (mu1 - mu2) @ np.linalg.inv(cov_mean) @ (mu1 - mu2).T + \
                   0.5 * (logdet(cov_mean) - 0.5 * logdet(cov1 @ cov2))

    return bhattacharya

def weighted_exp(g1, g2, dist_func=None):
    ''' Returns negative disribution distance between two GMM components,
    exponentiated and weighted by component weight.
    
    Inputs
    --------
        g1, g2: tuple
    tuples of (mean, covariance, weight) of the GMM components.

    Returns
    --------
        float
    '''
    return g2[2] * np.exp(-dist_func(g1,g2))

def weighted_exp_pairwise(X, Y, dist_func=None):
    ''' Evaluates weighted_exp method between lists of GMM components X and Y
    in a pairwise fashion.

    Inputs
    --------
        X, Y : list of tuples
    lists of tuples describing the GMM components in (mean, cov, weight) format.
        dist_func : function
    the distance function being used to evaluate distances between components,
    either gaussian_kl or bhattacharya_kl

    Returns
    ---------
        np.array
    dim(X)*dim(Y) x 1 numpy array of evaluated results.
    '''
    kwargs = {'dist_func': dist_func}
    return np.array(list(
                itertools.starmap(
                                  partial(weighted_exp, **kwargs),
                                  itertools.product(X, Y)
                                 )
                )).reshape(len(X) * len(Y), 1)

def bound(g1,Y, dist_func):
    ''' Returns '''
    return g1[2] * np.log(np.sum(weighted_exp_pairwise([g1], Y, dist_func)))


def mi_bound(msmt_means, msmt_covs, msmt_weights, lower=False):
    '''Calculates a tight upper bound on mutual information of measurement
    and underlying random variables, where both have gaussian mixture model distributions.
    From Kolchinsky 2017, Estimating Mixture Entropy with Pairwise Distances.
    
    Inputs
    --------
        msmt_means : list
    list of means of the measurement GMM  at time t (should be equal to underlying, 
    as long as zero mean measurement moise)
        msmt_covs : list
    list of covarances of the measurement GMM components at time t
        msmt_weights : list
    list of weights of the measurement GMM components
        lower : boolean
    set to True if evaluating lower bound, otherwise defaults to upper bound

    Returns
    -------
        float 
    Upper/lower bound on mutual information MI(M ; X)
    '''
    if lower == True:
        dist_func = gaussian_bhattacharya
    else:
        dist_func = gaussian_kl
        
    gmms = list(zip(msmt_means, msmt_covs, msmt_weights))
    kwargs = {'dist_func': dist_func}

    vals = np.array(list(
                itertools.starmap(
                                  partial(bound, **kwargs),
                                  itertools.product(gmms, [gmms])
                                 )
                )).reshape(len(gmms), 1)
    return -np.nansum(vals)
