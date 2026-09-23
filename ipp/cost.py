import numpy as np
from math import log10
from vmmp_gmm.h_signature import HSignature
from vmmp_gmm.vmmp import VMMP
from ipp.msmt import prob_ray_crossing, p_detect, calc_msmt_cov, p_detect_analytic, prob_ray_crossing_vectorised, p_detect_components
from vmmp_gmm.env_utils import reshape_grid_to_vector
from vmmp_gmm.gmm import predict_vmmp_gmm_given_msmt, get_marginal_covariances, predict_gmm_with_p_detect
from ipp.mixture_mi import mi_bound
import ipp.msmt_gmm as msmt_gmm

def kl_div(p, q):
    '''p and q should be the conditional probability distributions p(h|p_curr) and q(h|p_obs)
    the kl divergence between the two distributions is returned'''
    q = q.reindex(p.index, fill_value=0.0, copy=False)
    # print(q)
    kl = np.sum(p['probs'] * (safe_log(p['probs']) - safe_log(q['probs'])))
    return kl

def kl_div_np(p, q):
    if (p == 0).all().all():
        return 0
    if (q == 0).all().all():
        return 0
    kl = np.sum(p['probs'] * (safe_log(p['probs']) - safe_log(q['probs'])))
    return kl

def kl_detection_vectorised(p_curr,
                            pdf_dict,
                            x,
                            gmm_weights,
                            gmm_means,
                            gmm_block_covariances,
                            gmm_full_covariances,
                            rep_pts,
                            obs_x_bounds,
                            vmmp,
                            gmm_hsigs,
                            t,
                            msmts,
                            r):
    time_idxs = [t, t+1]
    means = gmm_means[...,time_idxs, :]
    covariances = get_marginal_covariances(gmm_full_covariances, [2 * t, 2 * t +1, 2*t +2, 2*t+3])
    probs = prob_ray_crossing_vectorised(x=x,
                                         means=means,
                                         covariances=covariances,
                                         rep_pts=rep_pts,
                                         obs_x_bounds=obs_x_bounds,
                                         p_sig=p_curr,
                                         msmts=msmts,
                                         r=r
                                         )
    if not np.any(probs):
        return 0, pdf_dict
    
    if p_curr.as_string() in pdf_dict.keys():
        pdf_curr = pdf_dict[p_curr.as_string()]
    else:
        pdf_curr = vmmp.get_homotopic_belief(p_curr,gmm_hsigs)
        pdf_dict[p_curr.as_string()] = pdf_curr

    next_letters = [i for i in range(-len(rep_pts), len(rep_pts)+1) if i != 0]

    possible_next_hsigs = [HSignature.concat(p_curr, i)
                           if i != p_curr.get_sfx()
                           else p_curr
                           for i in next_letters
                           ]
    possible_next_pdfs = [vmmp.get_homotopic_belief(next_sig, gmm_hsigs)
                          if next_sig.as_string() not in pdf_dict.keys()
                          else pdf_dict[next_sig.as_string()]
                          for next_sig in possible_next_hsigs]
    pdf_dict.update({next_sig.as_string() : pdf for next_sig, pdf in zip(possible_next_hsigs, possible_next_pdfs)})

    kl_divs = np.zeros((1,len(rep_pts)*2))
    for i, pdf_next in enumerate(possible_next_pdfs):
        kl_divs[0][i] = kl_div_np(pdf_curr, pdf_next)
    expected_kl_divs = probs @ kl_divs.T
    p_detect_component = p_detect_components(x, gmm_means, gmm_block_covariances,t,r=r).reshape(-1,1)
    detect_expected_kl_divs = p_detect_component * expected_kl_divs
    weighted_exp_kl_divs = np.average(detect_expected_kl_divs, axis=0, weights=gmm_weights)
    return weighted_exp_kl_divs, pdf_dict


def safe_log(num):
    return np.log(num + 1E-20)

def generate_heatmaps(
        prior_gmm,
        hsig_map,
        gmm_hsigs,
        vmmp,
        rep_pts,
        obs_x_bounds,
        env_grid,
        msmt_set,
        msmt_covs,
        msmt_times,
        r,
        pdf_dict = dict(),
        t_start = 1,
        t_stop = 99,
        rwd_fn = 'kl',
        p_detect=None,
        atc_flag=False
        ):

    sensing_locs = reshape_grid_to_vector(env_grid[0], env_grid[1])
    rewards = np.zeros((t_stop-t_start, len(sensing_locs),1))

    partial_sig = HSignature.get_hsig_from_path(
        msmt_set,
        rep_pts,
        obs_x_bounds
    )
    partial_sig.reduce()

    print('partial sig is:', partial_sig.as_string())

    if partial_sig.as_string() in pdf_dict.keys():
        pdf = pdf_dict[partial_sig.as_string()]
        print('got cached pdf')
    else:
        pdf_dict[partial_sig.as_string()] = vmmp.get_homotopic_belief(partial_sig,gmm_hsigs)
        pdf = pdf_dict[partial_sig.as_string()]
        print('calculated new pdf')
    posterior_gmm = predict_gmm_with_p_detect(weights=prior_gmm[0],
                                              means=prior_gmm[1],
                                              covariances=prior_gmm[3],
                                              pdf_dict=pdf_dict,
                                              vmmp=vmmp,
                                              hsig_map=hsig_map,
                                              msmt_noise=msmt_covs[-1][-1],
                                              msmt=msmt_set[-1],
                                              msmt_time=msmt_times[-1],
                                              p_detect=p_detect,
                                              rep_pts=rep_pts,
                                              obs_x_bounds=obs_x_bounds,
                                              gmm_hsigs=gmm_hsigs
                                             )
    
    for k, idx_time in enumerate(range(t_start,t_stop,1)):
        reward = np.zeros((len(sensing_locs),1))
        for n, x in enumerate(sensing_locs):
            if atc_flag and (x[1] > 15 or x[1] < -13 or x[1] - (17/20) * x[0] - 17 * 2 > 0 or x[1] + (17/20) * x[0] - (17**2/20) > 0):
                rwd = 0.0
            else:
            ################### VOMP KL + PROB OF DETECTION ######################
                if rwd_fn == 'kl':
                    rwd, pdf_dict = kl_detection_vectorised(partial_sig,
                                                            pdf_dict,
                                                            x.reshape(1,-1),
                                                            *posterior_gmm,
                                                            rep_pts,
                                                            obs_x_bounds,
                                                            vmmp,
                                                            gmm_hsigs,
                                                            idx_time,
                                                            msmts=msmt_set,
                                                            r=r)

            #################### WEIGHTED SUM OF ENTROPY #############################
                elif rwd_fn == 'ent':
                    msmt_covs = calc_msmt_cov(x.reshape(1,-1), posterior_gmm[1][:,int(idx_time),...],r=r)
                    msmt_gmm_covs = msmt_gmm.get_msmt_gmm(posterior_gmm[2][:,int(idx_time),...], msmt_covs)
                    rwd = np.nansum([w * 0.5 * logdet(2 * np.pi * np.exp(1) * c)
                                for 
                                w,c
                                in 
                                zip(posterior_gmm[0], msmt_gmm_covs)
                                ])
                    rwd *= p_detect_analytic(x.reshape(1,-1), posterior_gmm[0], posterior_gmm[1], posterior_gmm[2], idx_time,r=r)
            if rwd == 0:
                reward[n] = np.nan
            else:
                reward[n] = rwd        
        rewards[k] = reward

    return rewards, posterior_gmm, pdf_dict


def logdet(M):
    _, logdet = np.linalg.slogdet(M)
    return logdet