#%%
# from distutils.command.clean import clean
import numpy as np
import matplotlib.pyplot as plt
import vmmp_gmm.pickle_helper as pickle_helper
# from pgm_reader import Reader
# import yaml
from vmmp_gmm.h_signature import HSignature
from vmmp_gmm.gmm import interpolate_trajectory
from ipp.utils import init_map_data
# from cycler import cycler
from svgpathtools import svg2paths
from scipy.spatial.distance import cdist
from tqdm import tqdm
from random import sample, shuffle
from collections import defaultdict
#%%
'''Importing data and setting up variables'''
traces = pickle_helper.pickle_load_data(folder_path='./atc_mall', file_name='processed_data_20121024')
more_traces = pickle_helper.pickle_load_data(folder_path='./atc_mall', file_name='processed_data_20121028')
pids = list(traces.keys())
more_pids = list(more_traces.keys())
traj_list = [np.stack(traces[pid][int(pid)].to_numpy()) / 1000.0 for pid in pids]
trajectories = traj_list + [np.stack(more_traces[pid][int(pid)].to_numpy()) / 1000.0 for pid in more_pids]
path, attributes = svg2paths('./map_files/boundary.svg')
# print(bndry, attributes)

# (map_X, map_Y, map_data, mall_cmap, sense_X, sense_Y, sensing_locs) = init_map_data()
# manually taken from open_map.py output...
rep_pts = np.array([[-30.40,0.78], [-21.36, 0.57], [-13.68,9.50], [-6.61, 0.14], [2.50,-0.02]])
hsig_data = {}

#%%
increments = 1000
bndry = np.zeros((increments, 2))
for n, i in enumerate(np.linspace(0,1,increments)):
    pos = path[0].point(i)
    bndry[n][:] = np.array([pos.real *0.05 - 60, (-pos.imag * 0.05) + 20])

#%% right 15, top 15, left -38
clean_trajectories = []
bad_trajectories = {'boundary condition fail':[], 'displacement fail':[], 'distance travelled fail':[], 'squiggle fail':[]}
epsilon = 2.0

# (map_X, map_Y, map_data, mall_cmap, sense_X, sense_Y, sensing_locs) = init_map_data()

for trajectory in tqdm(trajectories):
    #remove any points that land exactly on a ray
    trajectory = trajectory[
        [float('{:.2f}'.format(trajectory[i,0])) not in rep_pts[...,0] for i in range(len(trajectory))]
        ]
    #crop trajectories
    r_idx = np.where(trajectory[...,0] > 13)
    if r_idx[0].shape[0] != 0:
        r_idx_min = np.min(r_idx)
        r_idx_max = np.max(r_idx)
        if r_idx_min != 0 and trajectory[r_idx_min - 3][0] < trajectory[r_idx_min][0]:
            trajectory = trajectory[:r_idx_min]
        else:
            trajectory = trajectory[r_idx_max:]
    trajectory = trajectory[trajectory[...,0] > -38]
    trajectory = trajectory[trajectory[...,1] < 15]
    trajectory = trajectory[trajectory[...,1] > -11]
    if trajectory.shape[0] < 10:
        continue
    #only boundary trajectories
    interpolated_trajectory = interpolate_trajectory(trajectory=trajectory)
    start_d = np.min(cdist(interpolated_trajectory[0].reshape((1,2)), bndry))
    end_d = np.min(cdist(interpolated_trajectory[-1].reshape((1,2)), bndry))
    displacement = np.linalg.norm(interpolated_trajectory[0]-interpolated_trajectory[-1]) #added abs to remove odd trajectories that stand still
    bad_flag = False
    if start_d < epsilon and end_d < epsilon:
        if displacement > 10:
            if np.sum(np.sqrt(np.sum(np.diff(interpolated_trajectory, axis=0)**2, axis=1))) > 10: #added to get long trajectories
                direction_vector = interpolated_trajectory[-1] - interpolated_trajectory[0]
                for t, pos in enumerate(interpolated_trajectory[:-1]):
                    delta_direction = interpolated_trajectory[t+1] - pos
                    delta_projection = np.dot(direction_vector, delta_direction)
                    if delta_projection <= 0:
                        print(pos,delta_projection)
                        bad_trajectories['squiggle fail'].append(interpolated_trajectory)
                        bad_flag = True
                        break
                if not bad_flag:
                    clean_trajectories.append(interpolated_trajectory)
            else:
                bad_trajectories['distance travelled fail'].append(interpolated_trajectory)
        else:
            bad_trajectories['displacement fail'].append(interpolated_trajectory)
    else:
        bad_trajectories['boundary condition fail'].append(interpolated_trajectory)

print('total traj left', len(clean_trajectories))


#%%
obs_x_bounds = [np.array([-30.90,-29.65]),np.array([-22.30,-20.23]),
                np.array([-14.12,-12.88]), np.array([-7.29,-5.27]),
                np.array([2.13,3.22])]
hsig_data = defaultdict(list)
for i, trajectory in tqdm(enumerate(clean_trajectories)):
    hsig = HSignature.get_hsig_from_path(trajectory, rep_pts, obs_x_bounds)
    hsig.reduce()
    if hsig.as_string() != '':
        hsig_data[hsig.as_string()].append(trajectory)
        # filter_trajectories.append(trajectory)

num_trajectories = sum([len(v) for v in hsig_data.values()])
# num_train = int(num_trajectories * 0.8)
print('after empty sig removal:', num_trajectories)

#%%
# clean_trajectories = filter_trajectories
train_hsig_data = defaultdict(list)
test_hsig_data = defaultdict(list)
for k,v in hsig_data.items():
    num_train = int(len(v) * 0.8)
    train_trajectories_idx = set(sample([i for i in range(len(v))], num_train))
    test_trajectories_idx = list(set([i for i in range(len(v))]) - train_trajectories_idx)
    if len(train_trajectories_idx) >= 100:
        train_hsig_data[k] = list(np.array(v)[sample(list(train_trajectories_idx), 100)])
        test_hsig_data[k] = list(np.array(v)[sample(test_trajectories_idx,20)])
    else:
        train_hsig_data[k] = list(np.array(v)[list(train_trajectories_idx)])
        test_hsig_data[k] = list(np.array(v)[test_trajectories_idx])

    print(k, num_train, len(train_hsig_data[k]), len(test_hsig_data[k]))



#%%
too_few_training_sigs = []
for k in train_hsig_data.keys():
    if len(train_hsig_data[k]) <100:
        too_few_training_sigs.append(k)

for k in too_few_training_sigs:
    train_hsig_data.pop(k)
    test_hsig_data.pop(k)

print(too_few_training_sigs)
print('remaining train trajectories:', sum([len(v) for v in train_hsig_data.values()]))

ticks = np.arange(0, len(train_hsig_data.keys()))
counts = [len(v) for v in train_hsig_data.values()]
fig,ax = plt.subplots()
ax.bar(ticks, counts)
ax.set_xticks(ticks)
ax.set_xticklabels(list(train_hsig_data.keys()), rotation=65)
plt.savefig('train_hsig_distro.png')

ticks = np.arange(0, len(test_hsig_data.keys()))
counts = [len(v) for v in test_hsig_data.values()]
fig,ax = plt.subplots()
ax.bar(ticks, counts)
ax.set_xticks(ticks)
ax.set_xticklabels(list(test_hsig_data.keys()), rotation=65)
plt.savefig('test_hsig_distro.png')

keep_test_data = []
for k,v in test_hsig_data.items():
    keep_test_data = keep_test_data + v

shuffle(keep_test_data)
print('remaining test trajectories:', len(keep_test_data))

#%%

pickle_helper.pickle_save_data(folder_path='./data', file_name='atc_train_traces_tro', data=train_hsig_data)
pickle_helper.pickle_save_data(folder_path='./data', file_name='atc_test_traces_tro', data=keep_test_data) 

#%% Uncomment this section to enable plotting hsigs
# reader = Reader()
# data = np.flipud(reader.read_pgm('./map_files/localisation_grid.pgm'))
# with open('./map_files/localization_grid.yaml', 'r') as f:
#     metadata = yaml.safe_load(f)

# X, Y = np.meshgrid(
#     np.arange(metadata['origin'][0], metadata['origin'][0]+reader.width * metadata['resolution'], metadata['resolution']),
#     np.arange(metadata['origin'][1], metadata['origin'][1]+reader.height * metadata['resolution'], metadata['resolution']))

# fig, ax = plt.subplots()
# ax.pcolormesh(X, Y, data)
# ax.plot(bndry[...,0], bndry[...,1])
# # ax.tick_params(axis='both', which='both', bottom=False, top=False,labelbottom=False, left=False, right=False, labelleft=False)
# cycle1 = cycler('color', plt.get_cmap('tab20').colors)
# cycle2 = cycler('color', plt.get_cmap('Accent').colors)
# cycle = cycle1.concat(cycle2)
# cycle = cycle.concat(cycle)
# #print(hsig_data.keys(), cycle)

# for i, hsig in enumerate(hsig_data.keys()):
#     fig, ax = plt.subplots()
#     ax.pcolormesh(X,Y,data)
#     ax.plot(bndry[...,0], bndry[...,1])
#     for traj in hsig_data[hsig]:
#         ax.plot(traj[...,0], traj[...,1], color=cycle.by_key()['color'][i])
#     plt.savefig(f'../vmmp/hsig_visualisation/fix_sig{hsig}.png')
#     plt.close(fig)


# plt.show()

