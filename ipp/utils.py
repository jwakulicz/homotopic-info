import numpy as np
import matplotlib as mpl
import yaml
from pgm_reader import Reader
from vmmp_gmm.env_utils import reshape_grid_to_vector

def get_travel_time(a, b, velocity=1):
    dist = np.linalg.norm(a-b)
    return dist / velocity

def init_map_data_thor():
    map_data = np.loadtxt('./THOR_map_data.txt', delimiter=',')
    x_min, x_max = -7, 8
    y_min, y_max = -2.5, 2.5

    xx = np.linspace(x_min, x_max, 8)
    yy = np.linspace(y_min, y_max, 8)

    [XX,YY] = np.meshgrid(xx,yy)
    sensing_locs = reshape_grid_to_vector(XX,YY)
    return XX, YY, sensing_locs, map_data

def init_map_data_sims():
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10

    xx = np.linspace(x_min, x_max, 10)
    yy = np.linspace(y_min, y_max, 10)

    [XX,YY] = np.meshgrid(xx,yy)
    sensing_locs = reshape_grid_to_vector(XX,YY)
    return XX, YY, sensing_locs

def init_map_data():
    reader = Reader()
    map_data = np.flipud(reader.read_pgm('./map_files/localisation_grid.pgm'))
    with open('./map_files/localization_grid.yaml', 'r') as f:
        metadata = yaml.safe_load(f)

    X, Y = np.meshgrid(
    np.arange(metadata['origin'][0], metadata['origin'][0]+reader.width * metadata['resolution'], metadata['resolution']),
    np.arange(metadata['origin'][1], metadata['origin'][1]+reader.height * metadata['resolution'], metadata['resolution']))

    x_min, x_max = -40, 17
    y_min, y_max = -15, 17

    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

    subset_X = X[mask]
    subset_Y = Y[mask]
    subset_map_data = map_data[mask]

    xx = np.linspace(x_min,x_max,12)
    yy = np.linspace(y_min,y_max,12)
    [XX,YY] = np.meshgrid(xx,yy)
    sensing_locs = reshape_grid_to_vector(XX, YY)

    n_rows = int(round((y_max - y_min) / metadata['resolution']))
    n_cols = int(round((x_max - x_min) / metadata['resolution']))

    subset_map_data = subset_map_data.reshape((n_rows, n_cols))
    subset_X = subset_X.reshape((n_rows, n_cols))
    subset_Y = subset_Y.reshape((n_rows, n_cols))
    subset_map_data = np.ma.masked_array(subset_map_data, subset_map_data > 0)
    mall_cmap = mpl.cm.get_cmap('binary').reversed()

    return subset_X, subset_Y, subset_map_data, mall_cmap, XX, YY, sensing_locs

def get_blob_dictionary(rewards, sensing_locs, threshold, start_time = 0, stop_time=99):
    blob_dictionary = dict()
    T = np.arange(start_time+1,stop_time,1)
    for j, t in enumerate(T):
        for k, xy in enumerate(sensing_locs):
            x = xy[0]
            y = xy[1]
            v = rewards[j,k,0]/np.nanmax(rewards)

            if v >= threshold:
                blob_dictionary[(x,y,int(t))] = v
    return blob_dictionary

def plot_environment(ax, rep_pts, obs_x_bounds):
    for centre, brdr in zip(rep_pts, obs_x_bounds):
        width = brdr[1] - brdr[0]
        corners_x = [centre[0] - width / 2, centre[0] + width / 2, centre[0] + width / 2, centre[0] - width / 2, centre[0] - width / 2]
        corners_y = [centre[1] - width / 2, centre[1] - width / 2, centre[1] + width / 2, centre[1] + width / 2, centre[1] - width / 2]
        ax.plot(corners_x, corners_y, c='k',linewidth=1)
    return ax

def plot_environment_rect(ax, rep_pts, obs_x_bounds, obs_y_bounds):
    for centre, x_brdr, y_brdr in zip(rep_pts, obs_x_bounds, obs_y_bounds):
        width = x_brdr[1] - x_brdr[0]
        height = y_brdr[1] - y_brdr[0]
        corners_x = [centre[0] - width / 2, centre[0] + width / 2, centre[0] + width / 2, centre[0] - width / 2, centre[0] - width / 2]
        corners_y = [centre[1] - height / 2, centre[1] - height / 2, centre[1] + height / 2, centre[1] + height / 2, centre[1] - height / 2]
        ax.plot(corners_x, corners_y, c='k',linewidth=1)
    return ax