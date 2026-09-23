from run_orienteering_mcts_experiment import run_single_target_robot_experiment
from ipp.utils import init_map_data
from vmmp_gmm.gmm import interpolate_all_trajectories
import vmmp_gmm.pickle_helper as pickle_helper

import sys
import time

start_test_path_idx = int(sys.argv[1])
rwd_fn = sys.argv[2]
sfx = sys.argv[3]
save_folder = sys.argv[4]

'''Load train and test data'''
# train_data = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_vmmp_train_three_component')#file_name='atc_vmmp_train_three_component')
# test_paths = interpolate_all_trajectories(
# pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_clean_test_traces'),
#     num_interpolation_points = 100
# )
train_data = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_trained_model_tro')
test_paths = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_test_traces_tro')

'''Load sensing and map environment'''
map_data = init_map_data()

for path_idx in range(start_test_path_idx, start_test_path_idx+100):
    start_time = time.perf_counter()
    print(f'Running experiment for test trajectory {path_idx} of {start_test_path_idx+99}')
    eval_data = run_single_target_robot_experiment(path_idx, train_data, test_paths, map_data, rwd_fn=rwd_fn, sfx=sfx, save_folder=save_folder, r=4, threshold=0.8)
    end_time = time.perf_counter()
    compute_time = end_time - start_time
    eval_data['compute_time'] = compute_time
    # eval_data = run_single_target_rhc_greedy_experiment(path_idx, train_data, test_paths, map_data, rwd_fn=rwd_fn, sfx=sfx)
    print(f'Saving test trajectory {path_idx} data')
    pickle_helper.pickle_save_data(data=eval_data, folder_path='./'+save_folder, file_name= rwd_fn + '_' + sfx +f'_{path_idx}')