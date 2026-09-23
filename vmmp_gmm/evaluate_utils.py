import numpy as np
from vmmp_gmm.gmm import generate_samples_flattened, flatten_trajectories, get_marginal_covariances
from vmmp_gmm.env_utils import check_obs_intersect_traj
from ipp.cost import logdet
from tqdm import tqdm

def compute_collision_ratio(obstacle_set, weights, means, covariances, n_samples = 100):
    samples = generate_samples_flattened(n_samples, weights, means, covariances)
    collisions = np.array( [
        check_obs_intersect_traj(sample, obstacle_set) for sample in samples        
    ], dtype = np.float32)
    return np.mean(collisions)

def compute_displacement_error(test_path, weights, means, covariances):

    idx_max = np.argmax(weights)

    return compute_displacement_error_gaussian(
        test_path,
        means[idx_max, ...]
    )

def compute_expected_displacement_error(test_path, weights, means, covariances):
    return np.sum(
        weights * compute_expected_displacement_error_gaussian(
            test_path,
            means,
            covariances
        )
    )

def compute_weighted_mahalanobis_error(test_path, weights, means, covariances):
    return np.sum(
        weights * compute_mahalanobis_error(test_path, means, covariances)
    )

def compute_displacement_error_gaussian(test_path, means):

    displacement = test_path - means
    error = np.sqrt(np.sum(displacement * displacement, axis=-1)) # elementwise squaring, sum over last dimension
    total_error = np.sum(error, axis=-1) # sum over time axis    

    return total_error

def compute_expected_displacement_error_gaussian(test_path, means, covariances):
    return compute_displacement_error_gaussian(test_path, means) + np.sum(np.trace(covariances, axis1=-2, axis2=-1), axis=-1)

def compute_mahalanobis_error(test_path, means, covariances):
    displacement = test_path - means
    weighted_displacement = np.linalg.solve(covariances, displacement)
    error = np.sqrt(np.sum(displacement * weighted_displacement, axis=-1))
    total_error = np.sum(error, axis=-1)
    return total_error

def safe_log(num):
    return np.log(num + 1E-20)

def compute_cross_entropy(p, q, axis=0):
    return np.sum(-p * safe_log(q), axis=axis)

def compute_kl_divergence(p, q, axis=0):
    return np.sum(p * (safe_log(p) - safe_log(q)), axis=axis)

def compute_kl_divergence_dict(p_dict, q_dict):
    return sum(
        [ p * (safe_log(p) - safe_log(q_dict.get(item, 1E-20))) for item, p in p_dict.items()]
    )

def compute_cross_entropy_dict(p_dict, q_dict):
    return sum(
        [ p * (- safe_log(q_dict.get(item, 1E-20))) for item, p in p_dict.items()]
    )

def gaussian_kl(mu1, cov1, mu2, cov2, logdetcov1=None):
    """ Computes the KL divergence between two normal distributions

    Inputs
    --------
        g1, g2: tuple
    tuples of (mean, covariance,) of the gaussian distribution.

    Returns
    --------
        float
    """
    # if logdetcov1 is None:
    #     logdetcov1 = logdet(cov1)
    
    # mu1 = np.array(mu1)
    # mu2 = np.array(mu2)
    cov1 = np.squeeze(np.array(cov1), axis=1)
    cov2 = np.squeeze(np.array(cov2), axis=1)
    # mu1 = flatten_trajectories(mu1)
    # mu2 = flatten_trajectories(mu2)

    n = np.shape(cov1)[-1]
    inv_cov2 = np.linalg.inv(cov2)
    kl = 0.5 * (
                (logdet(cov2) - logdet(cov1)).reshape((-1,1)) + \
                np.squeeze((mu2 - mu1) @ inv_cov2 @ (mu2 - mu1).transpose(0,2,1), axis=-1) + \
                np.trace(inv_cov2 @ cov1, axis1=1, axis2=2).reshape((-1,1)) - n
               )
    return kl

def compute_metric_info_gain(full_posterior, partial_posterior):
    '''The variational bound approximation for KL divergence between two GMMs, as given in
    https://www.researchgate.net/publication/4249249_Approximating_the_Kullback_Leibler_Divergence_Between_Gaussian_Mixture_Models
    
    The approximation is not perfect, i.e. it does not satisfy positivity property of a metric,
    is not as accurate as other approximations e.g. MC approximations.
    
    But for the purpose of just comparing KL divergences against each other it should be suitable enough.'''

    self_components = full_posterior[0].shape
    diff_components = partial_posterior[0].shape

    pairwise_self = np.zeros(self_components) #shape of weights
    pairwise_diff = np.zeros(self_components)
    for i, component in tqdm(enumerate(zip(*full_posterior))):
        self_means = [component[1]] * self_components[0]
        self_covs = [component[2]] * self_components[0]
        diff_means = [component[1]] * diff_components[0]
        diff_covs = [component[2]] * diff_components[0]
        # print(diff_means.shape, partial_posterior[1].shape, diff_covs.shape, partial_posterior[2].shape)
        pairwise_self[i] = np.average(
                                      np.exp(-gaussian_kl(self_means,
                                                          self_covs,
                                                          full_posterior[1],
                                                          full_posterior[2]
                                                        #   logdetcov1=logdet(np.array(self_covs))
                                                          )
                                            ),
                                       weights=full_posterior[0],
                                       axis=0
                                       )[0]
        pairwise_diff[i] = np.average(
                                      np.exp(-gaussian_kl(diff_means,
                                                          diff_covs,
                                                          partial_posterior[1],
                                                          partial_posterior[2]
                                                        #   logdetcov1=logdet(np.array(diff_covs))
                                                          )
                                            ),
                                      weights=partial_posterior[0],
                                      axis=0
                                      )[0]
    
    return np.sum(full_posterior[0] * (safe_log(pairwise_self) - safe_log(pairwise_diff)), axis=0)

def compute_decorrelated_metric_info_gain(full_posterior, partial_posterior):
    decorrelated_mi = np.zeros(100)
    timesteps = np.arange(0,100,1)
    full_means_flattened = flatten_trajectories(full_posterior[1])
    partial_means_flattened = flatten_trajectories(partial_posterior[1])
    for t in timesteps:
        flattened_time_idx = np.array([2 * t, 2 * t + 1])
        full_marginal_means = full_means_flattened[..., flattened_time_idx]
        full_marginal_covariances = full_posterior[2][...,t,:,:]

        partial_marginal_means = partial_means_flattened[..., flattened_time_idx]
        partial_marginal_covariances = partial_posterior[2][...,t,:,:]
        # print(full_marginal_covariances.shape)

        decorrelated_mi[t] = compute_metric_info_gain(full_posterior = (full_posterior[0],
                                                                         full_marginal_means,
                                                                         full_marginal_covariances),
                                                         partial_posterior = (partial_posterior[0],
                                                                              partial_marginal_means,
                                                                              partial_marginal_covariances)
                                                                              )
    return np.sum(decorrelated_mi)
