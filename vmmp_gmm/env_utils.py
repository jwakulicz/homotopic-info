import numpy as np
from random import choice
from numpy.linalg import LinAlgError

def euclid_dist(node1, node2):
    '''Calculates Euclidean distance between two points
    
    Input
    --------
    node1/2 : array-like
        array or list containing the x, y coordinate of points'''
    return np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)

def sample_environment(env, obstacle_shapes, num_samples, bndry_pts, dim=2):
    """sampling n points from the environment while rejecting points that fall
    within defined obstacles.

    Input
    -----------
    env : array-like
        2 x 2 array or list of x and y bounds of the environment within which to sample points.
    obstacle_shapes : array-like
        list of bottom left and top right coordinates of each rectangular, 
        axis-aligned object.
    num_samples : int
        number of desired samples.
    dim : int
        dimensions of environment, hard-coded as 2 for now.
    
    Returns
    ----------
    samples : np.array
        num_samples x 2 dimensional array with x, y coordinates of sample points.
    """
    samples = []

    #Assuming that no obstacles intersect with boundary of the domain

    for i in range(bndry_pts):
        coord_y = np.random.rand(1, 1) * (env[1][1] - env[1][0]) + env[1][0]
        coord_x = np.random.rand(1, 1) * (env[0][1] - env[0][0]) + env[0][0]
        samples.append(np.array([choice([(choice([env[0][0], env[0][1]]), coord_y[0][0]),
                                    (coord_x[0][0], choice([env[1][0], env[1][1]]))])][0]))

    while len(samples) < num_samples:
        success = []
        test_pt_y = np.random.rand(1, 1) * (env[1][1] - env[1][0]) + env[1][0]
        test_pt_x = np.random.rand(1, 1) * (env[0][1] - env[0][0]) + env[0][0]
        for obs in obstacle_shapes:
            # print(obs[0][0], test_pt[0][0], obs[1][0])
            if obs[0][0] < test_pt_x[0][0] < obs[1][0] and \
                    obs[0][1] < test_pt_y[0][0] < obs[1][1]:
                success.append(0)
            else:
                success.append(1)
        # print(success)
        if all(success):
            pt = np.array([test_pt_x[0][0], test_pt_y[0][0]])
            samples.append(pt)
    
    samples = np.asarray(samples)
    return samples.reshape((num_samples, 2))

def get_closest_brder(loc, env):
    BR = np.array([env[0][1], env[1][0]])
    BL = np.array([env[0][0], env[1][0]])
    TL = np.array([env[0][0], env[1][1]])
    TR = np.array([env[0][1], env[1][1]])

    #cant be bothered
    borders = [[BL, TL], [BL, BR], [TL, TR], [BR, TR]]
    brdr_dist = [np.abs(np.cross(loc-b[0],b[1]-b[0]))/np.linalg.norm(b[1]-b[0]) for b in borders]
    min_idx = np.argmin(brdr_dist)
    # print(loc, borders[min_idx])
    return borders[min_idx]

def sample_closest_bndry(loc, env):
    bndry_pts = []
    brdr = get_closest_brder(loc, env)
    for i in range(50):
        u = np.random.rand(1, 1)
        # print((1-u) * brdr[1] + u * brdr[0])
        bndry_pts.append(((1-u) * brdr[1] + u * brdr[0])[0])
    dists = [euclid_dist(x, loc) for x in bndry_pts]
    bndry_pt_indx = np.argmin(dists)
    
    return bndry_pts[bndry_pt_indx]

def check_ray_intersect(line_segment, ray_origin):
    '''Function to check if a path intersects with ray drawn from an obstacle. Taken 
    directly from Seth's code
    Inputs:
    ---------
    line_segment: list of 1x2 np.arrays with [x, y] of start and end of line segment
    ray_origin: 1x2 np.array with [x, y] coordinates of representative point for an obstacle

    Returns:
    ---------
    boolean. true if intersection occurs, false otherwise.
    '''
    ray_direction = np.array([0, 1])

    p1 = line_segment[0]
    p2 = line_segment[1]


    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_direction[1], ray_direction[0]])

    if np.dot(v2, v3) == 0:
        return False

    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)

    return t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0

def check_obs_intersect_traj(traj, obstacles):
    return any(
        check_obs_intersect(
            start,
            end,
            obstacles
        )
        for start, end in zip(traj[:-1, ...], traj[1:, ...])
    )

def check_obs_intersect(l1_start, l1_end, obstacles):
    '''Checks whether the straight line joining two points intersects any provided
    obstacle regions.'''
    # this function is so scuffed, can be improved
    l1_param = [l1_end[0] - l1_start[0], l1_end[1] - l1_start[1]]
    for obs in obstacles:
        diags_start = [obs[0], [obs[0][0], obs[1][1]]]
        diags_end = [obs[1], [obs[1][0], obs[0][1]]]
        for l2_start, l2_end in zip(diags_start, diags_end):
            l2_param = [l2_end[0] - l2_start[0], l2_end[1] - l2_start[1]]
            A = np.array([[l2_param[0], -l1_param[0]],
                         [l2_param[1], -l1_param[1]]])
            b = np.array([[l1_start[0] - l2_start[0]],
                         [l1_start[1] - l2_start[1]]])
            try:
                line_params = np.linalg.solve(A, b)
            except LinAlgError:
                continue

            if np.any(line_params > 1) or np.any(line_params < 0):
                continue
            else:
                return 1
    return 0

def get_rep_pts(obstacles):
    '''Returns a representative point for given obstacles. A representative point 
    can be any point inside the obstacle region. Here, the centroid of the obstacle
    is chosen.
    
    Inputs
    -----------
    obstacles : array-like
        list of bottom left and top right coordinates of each rectangular, 
        axis-aligned object.
    
    Returns
    -----------
    rep_pts : np.array
        1xm array of representative points, where m is the number of obstacles given.
    '''
    rep_pts = np.zeros((len(obstacles), 2))
    for i, obs in enumerate(obstacles):
        rep_x, rep_y = obs[0][0], (obs[1][1] + obs[0][1]) / 2
        rep_pts[i][0], rep_pts[i][1] = rep_x, rep_y
    return rep_pts

def reshape_grid_to_vector(x_g, y_g):
    return np.concatenate(
        [x_g.reshape( (-1, 1) ), y_g.reshape( (-1, 1) )],
        axis=-1
    )