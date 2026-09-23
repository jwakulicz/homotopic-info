import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
# from hausdorff import hausdorff_distance #pip install hausdorff

def broadcast_dist(x, means):
    assert(x.shape == (1,2))
    dists = np.sqrt(((x-means)**2).sum(axis=2))
    return dists

def get_msmt_gmm(covs, msmt_noise_cov=None):
    """Returns the measurement gmm based on gmm provided.
    Assumes zero-mean Gaussian additive noise on measurements.

    Inputs
    ---------
        covs : list
    list of covariances of the base gmm
        msmt_noise_cov : list
    list of the measurement noise covariances to be added

    Returns
    ---------
        tuple
    (means, covs, weights) of the resulting measurement GMM
    """
    for i, cov in enumerate(msmt_noise_cov):
        if cov is None:
            msmt_noise_cov[i] = np.eye(2) * np.nan
    msmt_covs = np.add(covs, np.asarray(msmt_noise_cov))

    return msmt_covs

def calc_msmt_cov(x, mean, r=5):
    '''Naive model of the measurement covariance based on a limited
    FOV model, where noise increases linearly with distance between
    sensor and a component's mean
    
    Distance taken is Euclidean'''
    alpha = 0.1
    shape = np.shape(mean)
    dists = euclidean_distances(x, mean).flatten()
    if len(dists) == 1:
        if dists < r:
            msmt_covs = np.eye(shape[-1]) * dists[0] * alpha
        else:
            msmt_covs = None
    else:
        msmt_covs = [np.eye(shape[-1]) * d * alpha if d < r else None for d in dists]
    return msmt_covs

def broadcast_msmt_cov(x, mean, r=5):
    alpha = 1
    shape = np.shape(mean)
    dists = broadcast_dist(x, mean)
    eyes = np.array([np.eye(2)] * shape[0]).reshape(shape[0],2,2,2)
    msmt_covs = np.array(
        [i * d * alpha if d < r else i * np.nan for eye,dee in zip(eyes,dists) for i,d in zip(eye,dee)]
        ).reshape(shape[0],2,2,2)
    return msmt_covs