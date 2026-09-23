import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
from vmmp_gmm.gmm import diagonalise_covariances, flatten_trajectories, extract_block_diagonals, get_valid_components
from vmmp_gmm.h_signature import HSignature

def p_detect_analytic(x_robot, gmm_weights, gmm_means, gmm_covariances, t, r,PD=1):
    # might be better as a weighted sum of probs, same like prob_ray_crossing
    means_at_t = gmm_means[...,t,:]
    covs_at_t = gmm_covariances[...,t,:,:]
    weighted_mean, weighted_cov = weighted_sum_2d_normals(gmm_weights, means_at_t, covs_at_t) 
    new_cov = weighted_cov + r ** 2 * np.eye(2)

    n_factor = PD / np.sqrt(np.linalg.det(weighted_cov @ (np.eye(2) * r) + np.eye(2)))

    bernoulli_prob = n_factor * gaussian_analytic(x=x_robot,
                                        mean=weighted_mean,
                                        inv_cov=np.linalg.inv(new_cov)
                                        )
    if bernoulli_prob > 1:
        print('prob over 1')
    
    return bernoulli_prob

def p_detect_components(x_robot, gmm_means, gmm_covariances, t, r, PD=1):
    means_at_t = gmm_means[...,t,:]
    covs_at_t = gmm_covariances[...,t,:,:]

    n_factor = PD / np.sqrt(np.linalg.det(covs_at_t @ (np.eye(2) * r) + np.eye(2)))

    bernoulli_probs = n_factor * gaussian_analytic_vectorised(x=x_robot,
                                        means=means_at_t,
                                        covs=covs_at_t + (r ** 2) * np.eye(2)
                                        )
    return bernoulli_probs

def p_detect(x_robot, gmm_weights, gmm_means, gmm_covariances, t, r):
    means_at_t = gmm_means[...,t,:]
    covs_at_t = gmm_covariances[...,t,:,:]
    gmm_pdf = weighted_sum_2d_normals(gmm_weights, means_at_t, covs_at_t)
    return integrate_2d_normal_analytic(r, x_robot, gmm_pdf[0], gmm_pdf[1])

def get_msmt(x, test_trajectory, t, msmt_traj, r):
    ground_truth = test_trajectory[t,...].reshape(1,-1)
    cov = calc_msmt_cov(x, ground_truth, r=r)
    if cov is not None:
        z = ground_truth + np.random.multivariate_normal(np.zeros((2,)), cov)
        if msmt_traj is not None:
            msmt_traj = np.vstack((msmt_traj, z))
        else:
            msmt_traj = z.reshape(1,2)
    return msmt_traj, cov

def get_msmt_vectorised(x, test_trajectory, t, r, msmt_traj=None, msmt_covs=None, msmt_times=None):
    ground_truth = test_trajectory[t,...].reshape((1,1,2))
    cov = calc_msmt_cov_vectorised(x, ground_truth, r=r)
    if np.any(cov==-1):
        z = np.array([[np.nan, np.nan]])
    else:
        print('target detected')
        z = ground_truth[0] + np.random.multivariate_normal(np.zeros((2,)), cov[0,0,...])
    if msmt_traj is not None:
        msmt_traj = np.vstack((msmt_traj, z))
        msmt_covs = np.hstack((msmt_covs,cov))
        msmt_times = np.hstack((msmt_times, t))
    else:
        msmt_traj = z.reshape(1,2)
        msmt_covs = cov
        msmt_times = np.array([t])
    return msmt_traj, msmt_covs, msmt_times

def calc_msmt_cov(x, mean, r):
    '''Naive model of the measurement covariance based on a limited
    FOV model, where noise increases linearly with distance between
    sensor and a component's mean
    
    Distance taken is Euclidean'''
    alpha = 0.01
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

def prob_ray_crossing(x, means, covariances, rep_pts):
    '''Probabilistic model for ray crossing
    Inputs
    -------- 
    x : np.array
        (1,2) robot sensing position
    means : np.array
        (2,2) array of gmm means at t and t+1
    covariances : np.array
        (4,4) array of covariance and cross covariances from the gmm,
        corresponding to mean at t and t+1
    rep_pts : list
        list of representative points from the dataset
    
    Returns
    ---------
    prob_ray_crossing : list
        list of probabilities indexed by obstacles, i.e. 0-th index is
        probability of a observing a ray crossing of obstacle 1 at time t'''
    msmt_covs = calc_msmt_cov(x, means)
    if any(cov is None for cov in msmt_covs):
        probs = [None] * len(rep_pts)
    else:
        flattened_means = flatten_trajectories(means)
        predicted_covariances = covariances + diagonalise_covariances(msmt_covs)
        mvg = multivariate_normal(flattened_means, predicted_covariances)
        probs = []
        for pt in rep_pts:
            if flattened_means[0] > flattened_means[2]:
                lower_bounds = [pt[0], pt[1], -np.inf, pt[1]]
                upper_bounds = [np.inf, np.inf, pt[0], np.inf]
            else:
                lower_bounds = [-np.inf, pt[1], pt[0], pt[1]]
                upper_bounds = [pt[0], np.inf, np.inf, np.inf]
            probs.append(mvg.cdf(upper_bounds, lower_limit=lower_bounds))
    return probs

def calc_msmt_cov_vectorised(x, means, r, alpha=0.01):
    shape = means.shape
    dists = euclidean_distances(x, means.reshape((shape[0]* shape[1], shape[2])))
    msmt_covs = np.stack([
                np.eye(shape[-1]) * d * alpha
                if d < r
                else
                np.eye(shape[-1]) * -1 #not too sure how this will work out...
                for d in dists[0]
                ],
                axis=0
                ).reshape(shape[0],shape[1],2,2)
     
    return msmt_covs


def prob_ray_crossing_vectorised(x, means, covariances, rep_pts, obs_x_bounds, p_sig, msmts, r):
    '''Vectorised probability of ray crossing to hopefully improve computation
    Inputs
    --------
    x : np.array
        (1,2) shape robot sensing location
    means : np.array
        (num_components, 2, 2) shaped array of the means of each component at t and t+1
    covariances : np.array
        (num_components, 4, 4) shape array of covariance and cross covariance from the
        predictive gmm, corresponding to state at t and t+1
    p_sig : HSignature object
        the current partial signature according to the current measurement set
    msmts : np.array
        the current measurement set I guess
    rep_pts : list
        list of the locatoins of representative points from the environment
    
    Returns
    ---------
    prob_ray_crossing : np.array
        (num_components, 1, num_rep_pts) shaped array of each of the probabilities of
        each component crossing a particular representative point ray
    '''
    probs = np.zeros((means.shape[0], len(rep_pts)*2))
    msmt_covs = calc_msmt_cov_vectorised(x, means, r)
    valid_components = get_valid_components(msmt_covs)

    if len(valid_components) == 0:
        return probs
    
    valid_msmt_covs = msmt_covs[valid_components,:,:,:]
    flattened_means = flatten_trajectories(means[valid_components,:,:])
    predicted_msmt_covariances = extract_block_diagonals(diagonalise_covariances(valid_msmt_covs),size=4,a=0)
    predicted_covariances = covariances[valid_components,:,:] + predicted_msmt_covariances
    
    for c, predictive_mean, predictive_cov in zip(valid_components,flattened_means,predicted_covariances):
        pred_msmt = np.vstack((msmts,predictive_mean[2:]))
        pred_sig = HSignature.get_hsig_from_path(pred_msmt, rep_pts, obs_x_bounds)
        if pred_sig.sig == p_sig.sig or len(p_sig.sig) > len(pred_sig.sig):
            continue
        mvg = multivariate_normal(predictive_mean,predictive_cov)
        for i, pt in enumerate(rep_pts[::-1]):
            if predictive_mean[0] > predictive_mean[2]: #travelling right to left
                if predictive_mean[0] > pt[0]: #Assume you cannot go backwards
                    # mvg = multivariate_normal(predictive_mean,predictive_cov)
                    lower_bounds = [pt[0], -np.inf, -np.inf, -np.inf]
                    upper_bounds = [np.inf, np.inf, pt[0], np.inf]
                    probs[c][i] = mvg.cdf(upper_bounds, lower_limit=lower_bounds)

            else:
                if predictive_mean[0] < pt[0]:
                    # mvg = multivariate_normal(predictive_mean,predictive_cov)
                    lower_bounds = [-np.inf, -np.inf, pt[0], -np.inf]
                    upper_bounds = [pt[0], np.inf, np.inf, np.inf]
                    probs[c][-(i+1)] = mvg.cdf(upper_bounds, lower_limit=lower_bounds)
    return probs

def gaussian_analytic_vectorised(x, means, covs):
    means = means[...,np.newaxis,:]
    exponents = (x - means) @ np.linalg.solve(covs, (x-means).transpose(0,2,1))
    return np.exp(-0.5 * exponents).flatten()


def gaussian_analytic(x, mean, inv_cov):
    exponent = -0.5 * ((x - mean) @ inv_cov) @ (x - mean).T
    return np.exp(exponent)


def pdf_2d_normal_analytic(x, mean, inv_cov, det_cov):
    """
    Returns the probability density of a two-dimensional normal distribution
    at a query point x, using the analytic expression.
    
    Args:
    x (np.ndarray): 1x2 array representing the query point.
    mean (np.ndarray): 1x2 array representing the mean of the distribution.
    cov (np.ndarray): 2x2 array representing the covariance matrix of the distribution.
    
    Returns:
    float: The probability density at point x.
    """
    
    # Calculate the exponent in the probability density expression
    exponent = -0.5 * ((x - mean) @ inv_cov) @ (x - mean).T
    
    # Calculate the normalization factor in the probability density expression
    norm_factor = 1 / (2 * np.pi * np.sqrt(det_cov))
    
    # Calculate the probability density at the query point x
    pdf = norm_factor * np.exp(exponent)
    
    return pdf

def weighted_sum_2d_normals(weights, means, covs):
    """
    Computes the weighted sum of N two-dimensional normal distributions,
    and returns the resulting normal distribution.
    
    Args:
    weights (list): A list of N weights for each distribution.
    means (list): A list of N 1x2 numpy arrays representing the means of the distributions.
    covs (list): A list of N 2x2 numpy arrays representing the covariance matrices of the distributions.
    
    Returns:
    tuple: A tuple containing the mean and covariance matrix of the resulting normal distribution.
    """
    # Calculate the weighted sum of the means and covariances
    weighted_mean_sum = np.average(means, axis=0, weights=weights)
    weighted_cov_sum = np.average(covs, axis=0, weights=np.square(weights))
    
    # Return the mean and covariance matrix of the resulting normal distribution
    return weighted_mean_sum, weighted_cov_sum

def integrate_2d_normal_analytic(R, x_robot, mean, cov):
    """
    Integrates the probability density of a two-dimensional normal distribution
    over a circular disc of radius R centered at point x_robot, using the analytic expression.
    
    Args:
    R (float): The radius of the circular disc.
    x_robot (np.ndarray): 1x2 array representing the center of the disc.
    mean (np.ndarray): 1x2 array representing the mean of the distribution.
    cov (np.ndarray): 2x2 array representing the covariance matrix of the distribution.
    
    Returns:
    float: The integrated probability density over the circular disc.
    """
    # Define the integrand as a function of r and theta
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    def integrand(r, theta):
        # Convert polar coordinates to Cartesian coordinates
        x = x_robot[0][0] + r * np.cos(theta)
        y = x_robot[0][1] + r * np.sin(theta)
        point = np.array([x, y])
        
        # Evaluate the probability density at the current point
        pdf = pdf_2d_normal_analytic(point, mean, inv_cov, det_cov)
        
        # Return the product of the probability density and the Jacobian of the transformation
        return pdf * r
    
    # Integrate the integrand over the circular disc using polar coordinates
    integral, _ = nquad(integrand, ranges=[(0, R), (0, 2*np.pi)])
    
    return integral