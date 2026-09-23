import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import seaborn as sns
import pandas as pd
import vmmp_gmm.pickle_helper as pickle_helper
from vmmp_gmm.gmm import interpolate_all_trajectories, predict_vmmp_gmm_given_partial
from vmmp_gmm.evaluate_utils import compute_kl_divergence, compute_decorrelated_metric_info_gain
from vmmp_gmm.h_signature import HSignature
from vmmp_gmm.vmmp_utils import get_all_prefixes

eval_data_folder = sys.argv[1]
eval_data_file_name = sys.argv[2]
test_data_file = sys.argv[3]
train_data_file = sys.argv[4]
min_idx = int(sys.argv[5])
max_idx = int(sys.argv[6])
save_pfx = sys.argv[7]


def get_full_posterior(test_path, rep_pts, vmmp, gmm_dict, obs_x_bounds):
    full_sig = HSignature.get_hsig_from_path(
        test_path,
        rep_pts,
        obs_x_bounds
    )

    hsig_strs = np.unique(list(gmm_dict.keys()))
    hsigs = [HSignature.get_hsig_from_string(sig) for sig in hsig_strs]
    gmm_hsigs = list(set(prefix for sig in hsigs for prefix in get_all_prefixes(sig) + [sig]))

    pdf_all = vmmp.get_homotopic_belief(full_sig, gmm_hsigs)

    posterior_gmm_given_all, _ = predict_vmmp_gmm_given_partial(
        gmm_dict,
        pdf_all,
        test_path
    )
    return posterior_gmm_given_all

def compute_displacement_error_gaussian(test_path, weights, means):

    max_comp = np.argmax(weights)
    displacement = test_path - means[max_comp,...]
    error = np.sqrt(np.sum(displacement * displacement, axis=-1)) # elementwise squaring, sum over last dimension    

    return error

test_paths = pickle_helper.pickle_load_data(folder_path='./data', file_name=test_data_file)
train_data = pickle_helper.pickle_load_data(folder_path='./data', file_name=train_data_file)

obs_x_bounds = [np.array([-30.90,-29.65]),np.array([-22.30,-20.23]),
                    np.array([-14.12,-12.88]), np.array([-7.29,-5.27]),
                    np.array([2.13,3.22])]
rep_pts = train_data['rep_pts']
vmmp = train_data['vmmp']
gmm_dict = train_data['gmm_dict']

# min_idx = 0
# max_idx = 100
num_data = max_idx-min_idx

fig, ax = plt.subplots(1,1)
rwds = ['kl', 'ent']
data_dict = {}
for r in rwds:
    error = np.empty((num_data, 100))
    n_failure = 0
    fail_idxs = []
    msmts = []
    n_msmts = []
    time_diff = []
    kl_divs = np.empty((num_data, 100))
    kl_divs.fill(np.nan)
    mis = np.empty((num_data,100))
    robot_states = []
    mis.fill(np.nan)
    for n, idx in enumerate(range(min_idx,max_idx)):
        if os.path.isfile(f'./{eval_data_folder}/' + r + '_' + eval_data_file_name + f'_{idx}'):
            data = pickle_helper.pickle_load_data(folder_path=f'./{eval_data_folder}', file_name= r + '_' + eval_data_file_name + f'_{idx}')
            posterior_gmm = data['posterior_gmms'][-1]
            test_path = test_paths[idx]
            full_gmm = get_full_posterior(test_path, rep_pts, vmmp, gmm_dict, obs_x_bounds)
            full_keep_idxs = np.argwhere((full_gmm[0] / np.nanmax(full_gmm[0])) > 0.00)
            full_keep_gmm = (full_gmm[0][full_keep_idxs].flatten(),
                             full_gmm[1][full_keep_idxs,...],
                             full_gmm[2][full_keep_idxs,...],
                             full_gmm[3][full_keep_idxs,...])
            num_msmts = np.count_nonzero(np.sum(~np.isnan(data['msmts']),axis=1))
            robot_states.append(data['robot_states'])
            if num_msmts > 1:
                error[n] = compute_displacement_error_gaussian(test_path, posterior_gmm[0], posterior_gmm[1])
                msmts.append(np.nanmean(np.sqrt(np.sum(np.diff(data['msmts'], axis=0)**2, axis=1))))
                n_msmts.append(num_msmts)
                time_diff.append(np.nanmean(np.diff(data['msmt_times'])))
                msmt_idx = np.where((~np.any(np.isnan(data['msmts']),axis=1)))
                for i in msmt_idx[0]:
                    prior = data['posterior_gmms'][i]
                    keep_idxs = np.argwhere((prior[0] / np.nanmax(prior[0])) > 0.00)
                    keep_gmm = (prior[0][keep_idxs].flatten(),
                                prior[1][keep_idxs,...],
                                prior[2][keep_idxs,...],
                                prior[3][keep_idxs,...])
                    kl_divs[n][i] = compute_kl_divergence(full_gmm[0], prior[0])
                    mis[n][i] = compute_decorrelated_metric_info_gain((full_keep_gmm[0], full_keep_gmm[1], full_keep_gmm[2]),
                                                         (keep_gmm[0], keep_gmm[1],keep_gmm[2])) / 100

            else:
                error[n] = np.nan
                n_failure += 1
                fail_idxs.append(idx)

    data_dict[r] = {'av_distance_between' : msmts,
                    'n_msmts' : n_msmts,
                    'ADE' : error,
                    'num_failures' : n_failure,
                    'fail_cases' : fail_idxs,
                    'time_diff' : time_diff,
                    'KL' : kl_divs,
                    'MI' : mis,
                    'msmt_locations' : robot_states
                    }
c_dict = {'kl' : 'tab:blue',
          'ent' : 'tab:orange'}

pickle_helper.pickle_save_data(file_name=save_pfx, folder_path=f'./{eval_data_folder}', data=data_dict)

