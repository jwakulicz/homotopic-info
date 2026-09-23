import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os.path
from matplotlib import cm
np.random.seed(0)

from ipp.planning import MCTS, do_mcts
from ipp.robot import SparseRobot
import vmmp_gmm.pickle_helper as pickle_helper
from vmmp_gmm.gmm import interpolate_all_trajectories, plot_gmm, plot_partial_trajectory, unwrap_vmmp_gmm
from ipp.msmt import get_msmt_vectorised, p_detect_components, p_detect_analytic
from ipp.cost import generate_heatmaps
from vmmp_gmm.h_signature import HSignature
from ipp.utils import init_map_data, get_blob_dictionary
from vmmp_gmm.vmmp_utils import get_all_prefixes
np.random.seed(1)

def run_single_target_robot_experiment(test_path_idx, train_data, test_paths, map_info, rwd_fn, sfx, save_folder, threshold=0.8, r=5, atc_flag=True):
    test_path = test_paths[test_path_idx]
    (map_X, map_Y, map_data, mall_cmap, sense_X, sense_Y, sensing_locs) = map_info
    vmmp = train_data['vmmp']
    gmm_dict = train_data['gmm_dict']
    rep_pts = train_data['rep_pts']
    new_ys = [0.25, -0.43, 8.75, -0.81, -0.64]
    rep_pts = [np.array([rep_pts[i][0], new_ys[i]]) for i in range(len(new_ys))]
    obs_x_bounds = [np.array([-30.90,-29.65]),np.array([-22.30,-20.23]),
                    np.array([-14.12,-12.88]), np.array([-7.29,-5.27]),
                    np.array([2.13,3.22])]

    colours = cm.get_cmap("tab10").colors
    hsig_cmap_dict = {
    hsig: colours[idx % len(colours)]
    for idx, hsig in enumerate(gmm_dict.keys())
    }

    fig, ax = plt.subplots()

    '''Generate initial heatmaps'''
    msmt_traj, msmt_covs, msmt_times = get_msmt_vectorised(x = test_path[0].reshape(1,-1) + 0.01,
                             test_trajectory=test_path,
                             t=0,
                             msmt_traj=None,
                             r=r
                             )
    prior_gmm, hsig_map = unwrap_vmmp_gmm(gmm_dict)
    hsigs = [HSignature.get_hsig_from_string(sig) for sig in np.unique(hsig_map)]
    gmm_hsigs = list(set(prefix for sig in hsigs for prefix in get_all_prefixes(sig) + [sig]))

    p_detect = p_detect_components(x_robot=test_path[0].reshape(1,-1) + 0.01,
                                   gmm_means=prior_gmm[1],
                                   gmm_covariances=prior_gmm[2],
                                   t=0,
                                   r=r) 
    
    kls, posterior_gmm, pdf_dict = generate_heatmaps(prior_gmm,
                                                     hsig_map,
                                                     gmm_hsigs,
                                                     vmmp,
                                                     rep_pts,
                                                     obs_x_bounds,
                                                     [sense_X,sense_Y],
                                                     msmt_traj,
                                                     msmt_covs,
                                                     msmt_times,
                                                     r=r,
                                                     rwd_fn=rwd_fn,
                                                     p_detect=p_detect,
                                                     atc_flag=atc_flag
                                                     )
    
    reward_dictionary = get_blob_dictionary(kls, sensing_locs, threshold, 1)
    def reward_fn(s):
        return reward_dictionary[s]
    
    robot = SparseRobot(state=(test_path[0][0], test_path[0][1], 0),
                        possible_next_states=list(reward_dictionary.keys()),
                        velocity=1, 
                        reward_fn=reward_fn 
                        )
    
    eval_data_to_save = {'posterior_gmms': [posterior_gmm],
                         'hsig_maps': [hsig_map],
                         'robot_states': [robot.state]}
    
    t_index=0
    while t_index < 100:
        ax.cla()

        robot, _ = do_mcts(robot, horizon=100-t_index, c=np.sqrt(2), n_iter=5000)
        if robot is None:
            print('No blobs reachable')
            break
        eval_data_to_save['robot_states'].append(robot.state)

        # get measurement
        msmt_traj, msmt_covs, msmt_times = get_msmt_vectorised(x=np.array(robot.state[:-1]).reshape(1,-1),
                                                               test_trajectory=test_path,
                                                               t=robot.t,
                                                               msmt_traj=msmt_traj,
                                                               msmt_covs=msmt_covs,
                                                               msmt_times=msmt_times,
                                                               r=r
                                                               )

        p_detect = p_detect_components(x_robot=np.array(robot.state[:-1]).reshape(1,-1),
                                       gmm_means=posterior_gmm[1],
                                       gmm_covariances=posterior_gmm[2],
                                       t = robot.t,
                                       r=r)
        
        kls, posterior_gmm, pdf_dict = generate_heatmaps(prior_gmm=posterior_gmm,
                                                                    hsig_map=hsig_map,
                                                                    gmm_hsigs=gmm_hsigs,
                                                                    vmmp=vmmp,
                                                                    rep_pts=rep_pts,
                                                                    obs_x_bounds=obs_x_bounds,
                                                                    env_grid=[sense_X,sense_Y],
                                                                    msmt_set=msmt_traj,
                                                                    msmt_covs=msmt_covs,
                                                                    msmt_times=msmt_times,
                                                                    pdf_dict=pdf_dict,
                                                                    t_start=robot.t+1,
                                                                    # t_stop=t_stop,
                                                                    rwd_fn=rwd_fn,
                                                                    r=r,
                                                                    p_detect= p_detect,
                                                                    atc_flag = atc_flag
                                                                    )
        
        max_weight_idx = (-posterior_gmm[0]).argsort()[:5].tolist()
        temp_gmm = (posterior_gmm[0][max_weight_idx], posterior_gmm[1][max_weight_idx,:,:], posterior_gmm[2][max_weight_idx,:,:])
        max_hsigs = np.asarray(hsig_map)[max_weight_idx]
        ax.pcolormesh(map_X, map_Y, map_data, cmap=mall_cmap)
        plot_gmm(
            ax,
            *temp_gmm,
            max_hsigs,
            cmap = lambda idx: hsig_cmap_dict[max_hsigs[idx]],
            alpha_func= lambda weight: 0.5* weight
        )
        plot_partial_trajectory(ax, test_path, robot.t)
        ax.scatter(msmt_traj[:,0], msmt_traj[:,1], c='r', marker='x', s=6)
        ax.scatter(robot.state[0], robot.state[1], c='k', marker='x', s=6)
        ax.set_xlim((-43,17))
        ax.set_ylim((-16,17))

        fig.savefig('./'+save_folder+'_figs/' + rwd_fn + '_' + sfx+f'_{test_path_idx}_{t_index}_{robot.t}.png', dpi=300)
        
        eval_data_to_save['posterior_gmms'].append(posterior_gmm)
        eval_data_to_save['hsig_maps'].append(hsig_map)
        
        reward_dictionary = get_blob_dictionary(kls, sensing_locs, threshold, robot.t+1)
        if not reward_dictionary:
            print('no future reward blobs')
            break
        def reward_fn(s):
            return reward_dictionary[s]
        robot.reward_fn = reward_fn
        robot.possible_next_states=list(reward_dictionary.keys())
        t_index = robot.t
    
    eval_data_to_save['msmts'] = msmt_traj
    eval_data_to_save['msmt_covs'] = msmt_covs
    eval_data_to_save['msmt_times'] = msmt_times

    return eval_data_to_save
    

if __name__ == '__main__':
    '''Load train and test data'''
    train_data = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_trained_model_tro')
    test_paths = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_test_traces_tro')
    test_path_idx = 13

    '''Getting the map and sensing environment ready'''
    map_data = init_map_data()

    '''Running experiment and returning results'''
    sfx = 'debug_misdetection_70pct'
    rwd_fn = 'kl'
    save_folder = 'eval_data'
    eval_data_dict = run_single_target_robot_experiment(test_path_idx,
                                                        train_data,
                                                        test_paths,
                                                        map_data,
                                                        rwd_fn=rwd_fn,
                                                        sfx=sfx,
                                                        save_folder=save_folder,
                                                        threshold=0.8,
                                                        r=4)