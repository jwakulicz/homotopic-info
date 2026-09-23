import sys
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import vmmp_gmm.env_utils as env_utils
from vmmp_gmm.h_signature import HSignature

class graphNode:
    """
    Node object

    Inputs 
    ----------
    node : 
        label of the node.
    pos : 1x2 np.array container
        x-y coordinates of the node
    """
    def __init__(self, id, pos):
        self.id = id
        self.position = np.array([pos[0], pos[1]])
        self.adjacent = {}

    def add_neighbour(self, neighbour, weight=0):
        self.adjacent[neighbour] = weight

    def get_neighbours(self):
        return self.adjacent.keys()

    def get_weight(self, neighbour):
        return self.adjacent[neighbour]


class Graph:
    '''Graph object
    '''
    def __init__(self):
        self.node_dict = {}
        self.edge_dict = {}
        self.paths = {}
        self.best_cost = {}

    def add_node(self, id, loc):
        """ Add new node(s) to the graph.

        Inputs
        --------
        node : str
            New node label, str(position)
        loc : array-like
            x, y coordinates of the new node.
        """
        new_node = graphNode(id, loc)
        self.node_dict[id] = new_node
        return new_node

    def get_node(self, node):
        '''Returns a node object specified by user.'''
        # probably need a better way to access nodes than labels? position? idk
        try:
            return self.node_dict[node]
        except KeyError:
            print("Node doesn't exist")

    def add_edge(self, from_node, to_node, weight=0):
        '''Adds and edge between two nodes in the graph, with default 0 weight. If
        specified nodes do not exist in the graph, do nothing.
        
        Inputs
        ---------
        from_node : str
            Label of the 'from' node in the graph. node.id
        to_node : str
            Label of the 'to' node in the graph. node.id
        weight : array-like
            Weight(s) of the edge.'''
        if from_node not in self.node_dict:
            return 
        if to_node not in self.node_dict:
            return 

        self.node_dict[from_node].add_neighbour(to_node, weight)
        self.node_dict[to_node].add_neighbour(from_node, weight)

    def get_edge_weight(self, from_node, to_node):
        '''Returns weight of an edge between two nodes.
        from_node and to_node are both node.ids
        '''
        return self.node_dict[from_node].adjacent[to_node]

    def get_nodes(self):
        '''Returns all node objects in the graph instance.'''
        return self.node_dict.values()
    
    def get_bndry_nodes(self, env):
        '''
        Inputs
        --------
        env : 2x2 array-like container
            Array or list of lists with the x and y boundaries of the environment
        
        Returns
        ---------
        bndry_nodes : list
            list of all nodes that lie on the boundary of the domain specified by env'''
        
        bndry_nodes = []
        for node in list(self.get_nodes()):
            if node.position[0] in env[0] or node.position[1] in env[1]: 
                bndry_nodes.append(node)
        return bndry_nodes
                

    def get_path_cost(self, path):
        '''Returns the cost of a given path through the graph.

        Inputs
        --------
        path : array-like container
            Contains consecutive node labels in the path.

        Returns
        --------
        path_cost : float
            Cost of the path
        '''
        # hsig_shape = self.get_edge_weight(path[0], path[1]).shape
        path_cost = 0
        for current, step in zip(path, np.hstack(path[1:])):
            path_hsig += self.get_edge_weight(current, step)
        return path_cost
    
    def dijkstra(self, start_node_id):
        #list of Node objects...
        unvisited_nodes = list(self.get_nodes())

        shortest_path = {}

        previous_nodes = {}

        max_value = sys.maxsize
        for node in unvisited_nodes:
            shortest_path[node.id] = max_value
        
        shortest_path[start_node_id] = 0

        while unvisited_nodes:
            current_min_node = None
            # print('1')
            for node in unvisited_nodes:
                # print('2:', node.id)
                if current_min_node == None:
                    current_min_node = node
                elif shortest_path[node.id] < shortest_path[current_min_node.id]:
                    current_min_node = node
                    # print('3', current_min_node.id)
        
            #list of keys of adjacency dictionary, node.id's, not node objects
            nb_nodes = list(current_min_node.get_neighbours())
            for nb in nb_nodes:
                candidate_cost = shortest_path[current_min_node.id] + self.get_edge_weight(current_min_node.id, nb)
                if candidate_cost < shortest_path[nb]:
                    shortest_path[nb] = candidate_cost
                    previous_nodes[nb] = current_min_node

            unvisited_nodes.remove(current_min_node)

        self.paths = previous_nodes
        self.best_cost = shortest_path

    def get_shortest_path(self, start_node_id, goal_node_id, pos_not_id=False):
        path = []
        node = goal_node_id

        while node != start_node_id:
            path.append(node)
            node = self.paths[node].id
        
        path.append(start_node_id)

        path_ids, path_costs = list(reversed(path)), self.best_cost[goal_node_id]

        if pos_not_id == False:
            return path_ids, path_costs
        else: 
            return [self.get_node(x).position for x in path_ids], path_costs


    def get_all_hsigs(self, start_node_id, env, rep_pts):
        ''' doesn't do what it's supposed to, returns only the hsigs of shortest paths.
        So, no looping or doubling back allowed with this function.'''
        bndry_ids = [x.id for x in self.get_bndry_nodes(env)]
        hsigs = []
        for bndry_node in bndry_ids:
            path, _ = self.get_shortest_path(start_node_id, bndry_node, pos_not_id=True)
            hsig = HSignature.get_hsig_from_path(path, rep_pts)
            hsigs.append(hsig.sig)
        
        return list(set(hsigs))

    @classmethod
    def construct_delaunay(cls, obstacle_set, samples, cost_fn, kwargs={}):
        '''Constructs a graph with connections according to delaunay triangulation of
        a user-input set of points. Edges that pass through any obstacles in the 
        environment are found and removed, so the resulting triangulation may not be
        strictly Delaunay.
        
        Inputs
        ----------
        obstacle_set : (num_obstacles)x2x2 np.array
            A list/array of obstacle descriptions. Individual obstacles are represented
            by the coordinates of the bottom left corner, and top-right corner.
            Obstacles must be axis-aligned rectangles.
        samples : np.array
            An array of points sampled from free-space in the environment.
        cost_fn : python function
            Function that calculates the cost function/weight between two nodes in the graph
        kwargs : dictionary
            keyword arguments to be input to cost_fn
        
        Returns
        --------
        Graph object.
        '''
        tri = Delaunay(samples)
        delaunay_graph = cls()

        for i, pos in enumerate(samples):
            delaunay_graph.add_node(i, pos)

        for simplex in tri.simplices:
            for current, step in zip(simplex, np.hstack((simplex[1:], simplex[:1]))):
                x1 = samples[current][0]
                y1 = samples[current][1]
                x2 = samples[step][0]
                y2 = samples[step][1]
                # obstacle check, lots of room for improvement here
                for obs in obstacle_set:
                    # print(obs)
                    # if (x1 < obs[0][0] and x2 < obs[0][0]) or \
                    # (y1 < obs[0][1] and y2 < obs[0][1]) or \
                    # (x1 > obs[1][0] and x2 > obs[1][0]) or \
                    # (y1 > obs[1][1] and y2 > obs[1][1]):
                    #     weight = cost_fn([x1,y1], [x2,y2], **kwargs)
                    #     delaunay_graph.add_edge(current, step, weight)
                    # else:
                    #     #print(current, step)
                    if not env_utils.check_obs_intersect([x1, y1], [x2, y2], obstacle_set):
                        weight = cost_fn([x1,y1], [x2,y2], **kwargs)
                        delaunay_graph.add_edge(current, step, weight)
        return delaunay_graph

    @classmethod
    def construct_uniform(cls, grid_sz, num_samples, obstacle_set, cost_fn, kwargs={}):
        '''Constructs a uniformly discretized, 8-connected graph from specified environment
        bounds and number of samples per row/column.
        
        Inputs
        --------
        grid_sz : array-ike
            Bounds for the environment.
        num_samples : int
            Number of nodes desired in a given column/row of the grid.
        obstacle_set : array-like
            A list/array of obstacle descriptions. Individual obstacles are represented
            by the coordinates of the bottom left corner, and top-right corner.
            Obstacles must be axis-aligned rectangles.
        cost_fn : python function
            Function that calculates the cost function/weight between two nodes in the graph
        kwargs : dictionary
            keyword arguments to be input to cost_fn
        
        Returns
        ----------
        Graph object
        '''
        xs = np.linspace(grid_sz[0], grid_sz[1], num_samples)
        ys = np.linspace(grid_sz[0], grid_sz[1], num_samples)

        XX, YY = np.meshgrid(xs, ys)
        #for some reason z array (obstacle info) works as z[y, x]
        z = np.ones((num_samples, num_samples))

        uni_graph = cls()

        for obs in obstacle_set:
            z[obs[0][1]:obs[1][1]+1, obs[0][0]:obs[1][0]+1] = 0

        idx_cnt = 0
        for X, Y in zip(XX, YY):
            for i, (x, y) in enumerate(zip(X, Y)):
                if z[int(y), int(x)]:
                    uni_graph.add_node(i + idx_cnt, np.array([x,y]))

            idx_cnt += num_samples

        for node in uni_graph.get_nodes():
            idx = node.id
            for n in [-1, 1,
                    -(num_samples+1), (num_samples+1),
                    -(num_samples), (num_samples),
                    -(num_samples-1), (num_samples-1)]:
                nb_idx = idx + n

                #this part might be very bad coding, idk
                nb_node = uni_graph.get_node(nb_idx)
                if bool(nb_node):
                    dist = env_utils.euclid_dist(node.position, nb_node.position)
                    if dist <= np.sqrt(2):
                        weight = cost_fn(node.position, nb_node.position, **kwargs)
                        uni_graph.add_edge(node.id, nb_node.id, weight=weight)
        return uni_graph

    def draw(self, obstacle_set, ax, alpha, add_label=True, save_graph=False, label_pos=False):
        '''Draws a given graph, with nodes and all edges. Also draws borders of obstacles if
        given.
        
        Inputs
        ---------
        obstacle_set : an array-like container
            A list/array of obstacle descriptions. Individual obstacles are represented
            by the coordinates of the bottom left corner, and top-right corner.
            Obstacles must be axis-aligned rectangles.
        add_label : boolean
            Toggles addition of node labels on plot on/off.
        save_graph : boolean
            Toggles saving of a high-res .png of output on/off.
        '''
        nodes = self.get_nodes()
        num_nodes = len(nodes)
        # edges
        positions = np.zeros((num_nodes, 2))
        labels = []
        for i, node in enumerate(nodes):
            positions[i] = node.position
            labels.append('[{:0.2f},{:0.2f}]'.format(node.position[0], node.position[1]))

        edges = {}
        for node in nodes:
            for neighbour in node.get_neighbours():
                nb_node = self.node_dict[neighbour]
                # neighbour_pos = nb_node.position
                new_edge = str(node.id) + '_to_' + str(nb_node.id)
                new_edge_flip = str(nb_node.id) + '_to_' + str(node.id)
                line_seg = [node.position, nb_node.position]

                if (new_edge and new_edge_flip) not in edges:
                    # print("yes")
                    edges[new_edge] = line_seg
                else:
                    continue

        seg = LineCollection(edges.values(), colors='g', alpha=alpha, linewidth=1)

        # fig, ax = plt.subplots(figsize=(10,10))

        for obs in obstacle_set:
            draw_obstacles(ax, obs, colors='black', linewidth=1)

        ax.add_collection(seg)
        ax.scatter(positions[:, 0], positions[:, 1], s=50, marker='o',alpha=alpha)
        
        if add_label:
            for i, label in enumerate(labels):
                if label_pos:
                    ax.text(positions[i, 0], positions[i, 1], label, fontsize=5)
                else:
                    ax.text(positions[i, 0], positions[i, 1], i, fontsize=8)
        
        if save_graph:
            plt.savefig('fig_test.png', format="png", dpi=1200)
        
        return ax

def draw_obstacles(ax, obs, **kwargs):
    obs_line = [[(obs[0][0],obs[0][1]), (obs[0][0],obs[1][1])],
                [(obs[0][0],obs[1][1]), (obs[1][0],obs[1][1])],
                [(obs[1][0],obs[1][1]), (obs[1][0],obs[0][1])],
                [(obs[1][0],obs[0][1]), (obs[0][0],obs[0][1])]]
    lines = LineCollection(obs_line, **kwargs)
    ax.add_collection(lines)
    return lines


