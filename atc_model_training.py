#%%
from inspect import trace
import numpy as np
from vmmp_gmm.env_graph import Graph
import vmmp_gmm.env_utils as env_utils
from vmmp_gmm.gmm import interpolate_all_trajectories, learn_gmm
import matplotlib
import matplotlib.pyplot as plt
from vmmp_gmm.h_signature import HSignature
from vmmp_gmm.vmmp import VMMP
import vmmp_gmm.pickle_helper as pickle_helper
from svgpathtools import svg2paths

#%%
'''Initialising parameters of environment and plotting environments'''
rep_pts = np.array([[-30.40,0.78], [-21.36, 0.57], [-13.68,9.50], [-6.61, 0.14], [2.50,-0.02]])
path, attributes = svg2paths('./map_files/boundary.svg')
increments = 1000
bndry = np.zeros((increments, 2))
for n, i in enumerate(np.linspace(0,1,increments)):
    pos = path[0].point(i)
    bndry[n][:] = np.array([pos.real *0.05 - 60, (-pos.imag * 0.05) + 20])

# %%
'''Loading training data'''
train_traces = pickle_helper.pickle_load_data(folder_path='./data', file_name='atc_train_traces_tro')

hsig_freq = {k : len(v) for k , v in train_traces.items()}
hsig_data = []
for sig_str, freq in hsig_freq.items():
    hsig_data = hsig_data + [HSignature.get_hsig_from_string(sig_str)] * freq

train_paths = [p[:] for v in train_traces.values() for p in v]

print(len(train_paths))
#%% train VMMP
'''Training VMMP'''
atc_vmmp = VMMP.get_vmmp_from_data(hsig_data, [(1,),(2,),(3,),(4,),(5,),(-1,),(-2,),(-3,),(-4,),(-5,)], 0.001, 2)

'''Training GMM'''
train_traces_interpolated = {
    sig_str: np.array(trajectories)#interpolate_all_trajectories(trajectories, num_interpolation_points=100) #TODO: make this use timesteps!
    for sig_str, trajectories in train_traces.items() 
}

num_components_per_hsig = 3
gmm_dict = {
    sig_str: learn_gmm(interpolated_trajectories, n_components=num_components_per_hsig)
    for sig_str, interpolated_trajectories in train_traces_interpolated.items()
    if interpolated_trajectories.shape[0] > num_components_per_hsig
    # and sig_str != ''
}

# num_components_for_empty_hsig = 6
# empty_sig_trajectories = train_traces_interpolated['']
# gmm_dict[''] = learn_gmm(empty_sig_trajectories, n_components=num_components_for_empty_hsig)


'''Training naive GMM'''

gmm_training_paths = np.concatenate([
    paths for paths in train_traces_interpolated.values() # initially a dict from hsig to paths
], axis = 0)

naive_gmm = learn_gmm(gmm_training_paths,
                      n_components = (len(train_traces_interpolated) * num_components_per_hsig) #+ (num_components_for_empty_hsig - num_components_per_hsig))
                      )

train_data = {'hsig_data': hsig_data, 
              'bndry': bndry,
              'rep_pts': rep_pts,
              'vmmp' : atc_vmmp,
              'gmm_dict': gmm_dict,
              'naive_gmm': naive_gmm
              }


pickle_helper.pickle_save_data(data = train_data, folder_path='./data', file_name='atc_trained_model_tro')

# test_probs = test_vmmp.get_conditional_prob_dist(HSignature(()))
# print([(k.sig,v) for k,v in test_probs.items()])
# %% generate prediction figures for paper

# test = test_paths[1]
# idxs=[]

# for i in range(len(test)):
#     for pt in rep_pts:
#         if test[i][0] > pt[0] and test[i-1][0] < pt[0]:
#             idxs.append(i) 

# idxs = [int(idxs[0] / 2)] + idxs
# idxs = list(set(idxs))

# colours = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
#             'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# for n, idx in enumerate(idxs):
#     n = n+1
#     path = test[:idx]
#     path_sig = HSignature.get_hsig_from_path(path, rep_pts)
#     pdf = test_vmmp.get_conditional_prob_dist(path_sig)
#     ax.flat[n] = testing_vmmp.plot_line_seg(
#                                             path, 
#                                             ax.flat[n], 
#                                             kwargs = {'color':'black', 'alpha': 1, 'linewidth' : 1}
#                                             )
#     trajs = testing_vmmp.all_viable_paths(
#                                           path, 
#                                           [x.sig for x in list(pdf.keys())], 
#                                           obstacle_set, 
#                                           np.array([grid_sz, grid_sz]), 
#                                           rep_pts
#                                           )
#     for m, hsig in enumerate(trajs.keys()):
#         key = [k for k in pdf.keys() if k.sig == hsig]
#         alpha = pdf[key[0]]
#         c = colours[m]
#         for traj in trajs[hsig]:
#             ax.flat[n] = testing_vmmp.plot_line_seg(
#                                                 traj, 
#                                                 ax.flat[n], 
#                                                 kwargs = {'color':c, 'alpha': alpha, 'linewidth' : 1}
#                                                 )
# plt.savefig('testing_traj.png', dpi=1200,bbox_inches='tight')




# %%
