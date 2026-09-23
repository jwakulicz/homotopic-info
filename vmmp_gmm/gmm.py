from audioop import cross
import numpy as np
import sklearn.mixture
import scipy
from itertools import product
from functools import reduce
from vmmp_gmm.plotting import plot_partial_trajectory, plot_gmm
from vmmp_gmm.env_utils import check_obs_intersect_traj
from vmmp_gmm.h_signature import HSignature
#from evaluate_utils import compute_collision_ratio


def learn_gmm(
    trajectories,
    **kwargs # this includes n_components for gmm
):
    reshaped_trajectories = flatten_trajectories(trajectories)

    gmm = sklearn.mixture.GaussianMixture(
        **kwargs
    )

    gmm.fit(reshaped_trajectories)
    return gmm

def predict_vmmp_gmm_given_partial(
    gmm_dict,
    pdf,
    trajectories
):
    prior_weights_given_hsig = np.concatenate([
        pdf['probs'].get(sig, 0.0) * gmm.weights_
        for sig, gmm in gmm_dict.items()
    ])

    prior_means_given_hsig = np.concatenate([
        gmm.means_
        for sig_str, gmm in gmm_dict.items()
    ])

    prior_covariances_given_hsig = np.concatenate([
        gmm.covariances_
        for sig_str, gmm in gmm_dict.items()
    ])

    hsig_map = sum([[sig_str] * gmm.weights_.shape[0] for sig_str, gmm in gmm_dict.items()], [])

    # tuple of weights, means, and covariances
    posterior_gmm_given_hsig = predict_gmm_given_partial_expanded(
        prior_weights_given_hsig,
        prior_means_given_hsig,
        prior_covariances_given_hsig,
        trajectories
    )

    return posterior_gmm_given_hsig, hsig_map


def predict_gmm_given_partial(
    gmm,
    trajectories
):
    return predict_gmm_given_partial_expanded(gmm.weights_, gmm.means_, gmm.covariances_, trajectories)

def predict_gmm_given_msmt(
    gmm,
    msmts,
    msmt_covariances,
    msmt_times
):
    return predict_gmm_given_msmt_expanded(
        gmm.weights_, 
        gmm.means_, 
        gmm.covariances_, 
        msmt_covariances, 
        msmts, 
        msmt_times
        )

def predict_vmmp_gmm_given_msmt(
    gmm_dict,
    pdf,
    msmt_covariances,
    msmts,
    msmt_times
):
    '''
    Inputs
    --------
    gmm_dict : 
        gmm dictionary output from training
    pdf : 
        conditional distribution for partial h_signature of observed trajectory given by VMMP
        now assuming that pdf is a pandas dataframe with indexes 'hsig_string' and column 'probs'
    msmt_covariances : np.array
        array of shape (1,n_msmts,2,2) with msmt covariances corresponding to measurements in 
        msmts array
    msmts : np.array
        array of size (n_msmts, 1, 2) with all measurements
    msmt_times : np.array
        array of size (1, n_msmts) with measurement times
    '''
    prior_weights_given_hsig = np.concatenate([
        pdf['probs'].get(sig, default=0.0) * gmm.weights_
        for sig, gmm in gmm_dict.items()
    ])

    prior_means_given_hsig = np.concatenate([
        gmm.means_
        for sig_str, gmm in gmm_dict.items()
    ])

    prior_covariances_given_hsig = np.concatenate([
        gmm.covariances_
        for sig_str, gmm in gmm_dict.items()
    ])

    hsig_map = sum([[sig_str] * gmm.weights_.shape[0] for sig_str, gmm in gmm_dict.items()], [])

    posterior_gmm_given_hsig = predict_gmm_given_msmt_expanded(
        prior_weights_given_hsig,
        prior_means_given_hsig,
        prior_covariances_given_hsig,
        msmt_covariances,
        msmts,
        msmt_times
    )
    return posterior_gmm_given_hsig, hsig_map

def predict_gmm_given_msmt_expanded(
    weights,
    means,
    covariances,
    msmt_noise,
    msmts,
    msmt_times
):
    valid_msmt_idx = get_valid_components(msmt_noise.transpose(1,0,2,3))
    valid_msmt_covs = msmt_noise[...,valid_msmt_idx,:,:]
    valid_msmts = msmts[valid_msmt_idx]
    valid_msmt_times = msmt_times[valid_msmt_idx]

    reshaped_msmts = flatten_trajectories(valid_msmts)
    flattened_time_idxes = np.array([[2 * t, 2 * t + 1] for t in valid_msmt_times]).flatten()

    marginal_means = means[..., flattened_time_idxes]
    marginal_covariances = get_marginal_covariances(covariances, flattened_time_idxes)

    log_prob = gaussian_log_likelihood(reshaped_msmts, marginal_means, marginal_covariances)
    log_posterior = log_prob + np.log(weights)
    log_posterior = log_posterior - scipy.special.logsumexp(log_posterior, axis=0)
    weights = np.exp(log_posterior)

    predictive_prior_means = means
    predictive_prior_covariances = covariances
    cross_covariances = get_cross_covariances(covariances, flattened_time_idxes)

    msmt_covariances = marginal_covariances + diagonalise_covariances(valid_msmt_covs)

    inv_residuals = np.linalg.solve(msmt_covariances, reshaped_msmts - marginal_means)

    innovations = (cross_covariances @ (inv_residuals[..., np.newaxis]))[..., 0]

    predictive_posterior_means = predictive_prior_means + innovations

    predictive_posterior_covariances = predictive_prior_covariances - cross_covariances @ np.linalg.solve(msmt_covariances, transpose(cross_covariances))

    return weights, unflatten_trajectories(predictive_posterior_means), extract_block_diagonals(predictive_posterior_covariances), predictive_posterior_covariances

def predict_gmm_with_p_detect(
    weights,
    means,
    covariances,
    pdf_dict,
    vmmp,
    hsig_map,
    msmt_noise,
    msmt,
    msmt_time,
    p_detect,
    rep_pts,
    obs_x_bounds,
    gmm_hsigs
):
    '''
    Inputs
    --------
    pdf : 
        conditional distribution for partial h_signature of observed trajectory given by VMMP
        now assuming that pdf is a pandas dataframe with indexes 'hsig_string' and column 'probs'
    msmt_covariances : np.array
        array of shape (1,1,2,2) with msmt covariances corresponding to measurements in 
        msmts array
    msmts : np.array
        array of size (1, 1, 2) with all measurements
    msmt_times : np.array
        array of size (1, 1) with measurement times
    p_detects : np.array
        array of shape (num_components,1) with the probability of detection of each measurement
        given the component.
    '''
    if np.any(msmt_noise==-1):
        #update the weights according to probability of midetection
        # log_p = np.log(1 - p_detect)
        # print(log_p)
        # log_posterior = log_p + np.log(weights)
        # log_posterior = np.log(weights) - scipy.special.logsumexp(np.log(weights), axis=0)
        weights = weights * (1-p_detect)
        # print('no detect')
        #leave GMM components untouched
        predictive_posterior_means = flatten_trajectories(means)
        predictive_posterior_covariances = covariances
    else:
        # update the weights and components by marginalising over probability of detection
        flattened_time_idxes = np.array([2 * msmt_time, 2 * msmt_time + 1])
        means = flatten_trajectories(means)
        
        marginal_means = means[..., flattened_time_idxes]
        marginal_covariances = get_marginal_covariances(covariances, flattened_time_idxes)

        likelihood = gaussian_log_likelihood(msmt, marginal_means, marginal_covariances)
        # log_prob = likelihood + np.log(p_detect)
        log_posterior = likelihood + np.log(weights)
        log_posterior = log_posterior - scipy.special.logsumexp(log_posterior, axis=0)
        weights = np.exp(log_posterior) * p_detect

        predictive_prior_means = means
        predictive_prior_covariances = covariances
        cross_covariances = get_cross_covariances(covariances, flattened_time_idxes)

        msmt_covariances = marginal_covariances + diagonalise_covariances(msmt_noise)

        inv_residuals = np.linalg.solve(msmt_covariances, msmt - marginal_means)

        innovations = (cross_covariances @ (inv_residuals[..., np.newaxis]))[..., 0]

        predictive_posterior_means = predictive_prior_means + innovations

        predictive_posterior_covariances = predictive_prior_covariances - cross_covariances @ np.linalg.solve(msmt_covariances, transpose(cross_covariances))
        # print('weights before homotopic belief:', weights)
        # unnormalised_weights = weights * [pdf['probs'].get(sig, default=0.0) for sig in hsig_map]
        # print('unnormalised:', unnormalised_weights)
        # print('pdf keys:', pdf.head(n=10))
    mus = unflatten_trajectories(predictive_posterior_means[..., :2*msmt_time+4])
    psig_list = [HSignature.get_hsig_from_path(mu, rep_pts, obs_x_bounds) for mu in mus]
    pdf_list = []
    for sig in psig_list:
        sig_str = sig.as_string()
        if sig_str not in pdf_dict.keys():
            pdf = vmmp.get_homotopic_belief(sig, 
                                            gmm_hsigs)
            pdf_dict[sig_str] = pdf
        else:
            pdf = pdf_dict[sig_str]
        pdf_list.append(pdf)
        
    unnormalised_weights = weights * [np.sum(
                                            [pdf['probs'].get(sig, default=0.0)
                                            for pdf in pdf_list]
                                            )
                                            for sig in hsig_map]
    weights = unnormalised_weights / sum(unnormalised_weights)

    return weights, unflatten_trajectories(predictive_posterior_means), extract_block_diagonals(predictive_posterior_covariances), predictive_posterior_covariances

def predict_gmm_given_partial_expanded(
    weights,
    means,
    covariances,
    trajectories
):
    """
    Inputs:

    gmm: an sklearn.mixture.GaussianMixture object
    trajectories: np.ndarray of size [n_timesteps, 2] #TODO: enable batching
    msmt_times: np.ndarray of size (1, n_msmts) detailing the time of each measurement given in trajectories

    Outputs:
    weights: np.ndarray of size [n_components] representing weight of each component
    means: np.ndarray of size [n_components, n_timesteps, 2] for each component's mean
    covariances: generator of np.ndarrays [n_components, 2, 2] over n_timesteps
    """

    reshaped_trajectories = flatten_trajectories(trajectories)
    # get marginal mean/covariance for the times desired
    
    marginal_means = means[..., :reshaped_trajectories.shape[-1]]
    marginal_covariances = covariances[..., :reshaped_trajectories.shape[-1], :reshaped_trajectories.shape[-1]]

    # evaluate log probability for each marginal component
    log_prob = gaussian_log_likelihood(reshaped_trajectories, marginal_means, marginal_covariances)
    log_posterior = log_prob + np.log(weights)
    log_posterior = log_posterior - scipy.special.logsumexp(log_posterior, axis=0)
    weights = np.exp(log_posterior)
    predictive_prior_means = means
    predictive_prior_covariances = covariances
    cross_covariances  = covariances[..., :, :reshaped_trajectories.shape[-1]]
    # print(np.shape(cross_covariances), cross_covariances)


    measurement_covariances = marginal_covariances + 0.1 * np.eye(marginal_covariances.shape[-1])


    # todo: fix this for batching
    inv_residuals = np.linalg.solve(measurement_covariances, reshaped_trajectories - marginal_means)

    innovations = (cross_covariances @ (inv_residuals[..., np.newaxis]))[..., 0]

    predictive_posterior_means = predictive_prior_means + innovations

    predictive_posterior_covariances = predictive_prior_covariances - cross_covariances @ np.linalg.solve(measurement_covariances, transpose(cross_covariances))

    return weights, unflatten_trajectories(predictive_posterior_means), extract_block_diagonals(predictive_posterior_covariances), predictive_posterior_covariances

def unwrap_vmmp_gmm(gmm_dict):
    prior_weights_given_hsig = np.concatenate([
        gmm.weights_
        for sig, gmm in gmm_dict.items()
    ])

    prior_means_given_hsig = np.concatenate([
        gmm.means_
        for sig_str, gmm in gmm_dict.items()
    ])

    prior_covariances_given_hsig = np.concatenate([
        gmm.covariances_
        for sig_str, gmm in gmm_dict.items()
    ])
    
    prior_block_covariances_given_hsig = np.concatenate([
        extract_block_diagonals(gmm.covariances_)
        for sig_str, gmm in gmm_dict.items()
    ])

    prior_gmm = (prior_weights_given_hsig, unflatten_trajectories(prior_means_given_hsig), prior_block_covariances_given_hsig, prior_covariances_given_hsig)

    hsig_map = sum([[sig_str] * gmm.weights_.shape[0] for sig_str, gmm in gmm_dict.items()], [])

    return prior_gmm, hsig_map

def gaussian_log_likelihood(
    data,
    mean,
    covariance
):
    return -0.5 * ( 
        + logdet(2 * np.pi * covariance) 
        + mahalanobis_distance(data, mean, covariance)
    )

def mahalanobis_distance(
    data,
    mean,
    covariance
):
    residuals = data - mean
    inv_residuals = np.linalg.solve(covariance, residuals)
    return np.sum(residuals * inv_residuals, axis=-1) # basically a dot product

def logdet(
    covariance
):
    (_, val) = np.linalg.slogdet(covariance)
    return val

def transpose(
    array
):
    return np.swapaxes(array, -1, -2)

def flatten_trajectories(
    trajectories
):
    return np.reshape(
        trajectories,
        (*trajectories.shape[:-2], -1)
    )

def unflatten_trajectories(
    reshaped_trajectories
):
    return np.reshape(
        reshaped_trajectories,
        (*reshaped_trajectories.shape[:-1], -1, 2)
    )

def extract_block_diagonals(
    covariances,
    size=2,
    a=1
):

    return np.stack(
        [
            covariances[..., idx_time:idx_time+size, idx_time:idx_time+size]
            for idx_time in range(0, covariances.shape[-1], size)
        ],
        axis=a
    )


def get_valid_components(cov_matrices):
    m = np.where(~np.any(cov_matrices == -1, axis=(1,2,3)))
    return m[0]

def get_marginal_covariances(
    covariances,
    flattened_msmt_times
):
    ''''
    Outputs
    ---------
    covs : np.array
        shape (num_components, num_msmts, num_msmts) covariance matrix'''
    idxs = [i for i in product(flattened_msmt_times, flattened_msmt_times)]
    cov_eles = np.asarray(
        [[reduce(lambda m, idx: m[...,idx], idx, cov) for idx in idxs] for cov in covariances]
    )
    n = len(flattened_msmt_times)
    m = covariances.shape[0]
    return cov_eles.reshape(m,n,n)

def get_cross_covariances(
    covariances,
    flattened_msmt_times
):
    return covariances[...,:,flattened_msmt_times]

def diagonalise_covariances(
    covs
):
    # if sparse_flag is True:
    flattened_covs = flatten_msmt_covariances(covs)#, sparse_flag)
    diagonalised = np.diagflat(flattened_covs)
    return diagonalised
    # else:
        # flattened_covs = flatten_msmt_covariances(covs)
        # return np.diag(flattened_covs)

def get_valid_components(cov_matrices):
    m = np.where(~np.any(cov_matrices == -1, axis=(1,2,3)))
    return m[0]

def flatten_msmt_covariances(
    msmt_covariances
):  
    # if sparse_flag is True:
    len_shape = len(msmt_covariances.shape)
    return np.diagonal(msmt_covariances, axis1=len_shape-2, axis2=len_shape-1)
    # else:
        # return np.array(
        #     [np.diag(cov) for cov in msmt_covariances]
        # ).reshape(1,-1)[0]

def construct_block_diagonals(
    covariances,
    axis=-3
):
    pass

def generate_samples_gaussian_flattened(
    n_samples, means, covariances
):
    samples = np.stack(
        [np.random.multivariate_normal(mean, covariance, n_samples)
        for mean, covariance in zip(means, covariances)], # iterate over time axis
        axis = 1
    )
    return samples

def generate_samples_flattened(
    n_samples,
    weights,
    means,
    covariances
):
    n_samples_to_compute = np.random.multinomial(n_samples, weights)
    samples = np.concatenate(
        [generate_samples_gaussian_flattened(int(n), mean, covariance)
        for n, mean, covariance in zip(n_samples_to_compute, means, covariances)], # iterate over components
        axis = 0
    )
    return samples 

def interpolate_trajectory(
    trajectory,
    num_interpolation_points = 100
):
    # assume the -2nd dimension is time
    xp_initial = np.linspace(0., 1., trajectory.shape[-2])

    xp_query = np.linspace(0., 1., num_interpolation_points)
    
    trajectory_func = scipy.interpolate.interp1d(
        xp_initial,
        trajectory,
        axis=-2
    )
    return trajectory_func(
        xp_query
    )

def interpolate_all_trajectories(
    trajectories_list,
    **kwargs
):
    return np.stack(
        [interpolate_trajectory(
            trajectory,
            **kwargs
        ) for trajectory in trajectories_list if trajectory.shape[0] > 2]
    )

def safe_log(num):
    return np.log(num + 1E-20)

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    from env_graph import draw_obstacles

#%% Load data and prepare
    with open('./data/rdnm_starts', 'rb') as f:
        data_dict = pickle.load(f)

    trajectories = map(np.stack, data_dict["training_paths"])
    obstacle_set = np.array([[[20, 20], [30, 30]], [[5,5],[10,10]]])

    num_interpolation_points = 40

    interpolated_trajectories = interpolate_all_trajectories(trajectories, num_interpolation_points=num_interpolation_points)

#%% Learn GMM
    num_components = 4
    gmm = learn_gmm(interpolated_trajectories, n_components=num_components)

#%% Plotting
    cmap = plt.cm.get_cmap('hsv', num_components)
    obstacles_spec = {
        'color': 'black',
        'linewidth': 1
    }

    fig_learning_result, ax_learning_result = plt.subplots()

    ax_learning_result.set_xlim(
        interpolated_trajectories[:, :, 0].min(),
        interpolated_trajectories[:, :, 0].max(),
    )
    ax_learning_result.set_ylim(
        interpolated_trajectories[:, :, 1].min(),
        interpolated_trajectories[:, :, 1].max(),
    )

    ax_learning_result.set_aspect('equal')

    for obs in obstacle_set:
        draw_obstacles(ax_learning_result, obs, **obstacles_spec)

    fig_prediction_result, ax_prediction_result = plt.subplots()

    ax_prediction_result.set_xlim(
        interpolated_trajectories[:, :, 0].min(),
        interpolated_trajectories[:, :, 0].max(),
    )
    ax_prediction_result.set_ylim(
        interpolated_trajectories[:, :, 1].min(),
        interpolated_trajectories[:, :, 1].max(),
    )

    ax_prediction_result.set_aspect('equal')

    for obs in obstacle_set:
        draw_obstacles(ax_prediction_result, obs, **obstacles_spec)

    reshaped_prior_mean = unflatten_trajectories(gmm.means_)



    plt.show(block=False)
    plt.pause(0.1)


    trajectory_under_test = interpolated_trajectories[400, :, :]


    for idx_time in range(2, num_interpolation_points):

        weights, reshaped_means, covariances = predict_gmm_given_partial(
            gmm,
            trajectory_under_test[:idx_time, :]
        )

        print(weights)

        for line in ax_prediction_result.lines:
            ax_prediction_result.lines.remove(line)

        for patch in ax_prediction_result.patches:
            ax_prediction_result.patches.remove(patch)


        plot_partial_trajectory(
            ax_prediction_result,
            trajectory_under_test,
            idx_time
        )

        # plot_gmm(
        #     ax_prediction_result,
        #     weights,
        #     reshaped_means,
        #     covariances,
        #     cmap = cmap
        # )
        samples = generate_samples_flattened(10, weights, reshaped_means, covariances)

        for sample in samples:
            ax_prediction_result.plot(
                sample[:, 0].T,
                sample[:, 1].T,
                'r' if check_obs_intersect_traj(sample, obstacle_set) else 'g',
                alpha = 0.1
            )

        print(compute_collision_ratio(obstacle_set, weights, reshaped_means, covariances))

        plt.show(block=False)
        plt.pause(0.1)
        # fig_prediction_result.savefig(f'figs/gmm_{idx_time}.png')
