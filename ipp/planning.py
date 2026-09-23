import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


def do_mcts(robot_node, c=1, horizon=100, n_iter=1000):
    mcts = MCTS(root_node=robot_node, c=c, horizon=horizon)
    for _ in range(n_iter):
        mcts.run_mcts(robot_node)
    soln = mcts.solve(robot_node)
    return soln, mcts#mcts.solve(robot_node), mcts

class DecMCTS:
    def __init__(self, c=2, gamma=0.9, horizon=100):
        self.R = defaultdict(float)
        self.N = defaultdict(float)
        self.last_visit = defaultdict(int)
        self.children = dict()
        self.c = c
        self.gamma = gamma
        self.rollout_horizon = horizon
        self.action_prob_dist = None

    def solve(self, node):
        if node.is_terminal():
            return None
        
        if node not in self.children:
            return node.get_random_child()
        
        def score(child):
            if self.N[child] == 0:
                return -np.inf
            return self.R[child]  + self.c * np.sqrt(
                np.log(self.N[node]) / self.N[child]
            )
        
        new_node = max(self.children[node], key = score)
        
        return new_node

    def _update_action_prob_dist(self, node):
        def score(child):
            if self.N[child] == 0:
                return 1E-20
            return self.R[child]  + self.c * np.sqrt(
                np.log(self.N[node]) / self.N[child]
            )
        
        children = self.children[node]
        p_unnormalised = {child : score(child) for child in children}
        normalisation_factor = sum(list(p_unnormalised.values()))
        p_normalised = {k : v / normalisation_factor
                        for 
                        k,v 
                        in
                        p_unnormalised.items()
                        }
        return p_normalised

    def run_dec_mcts(self, node, other_robot_nodes, reward_fn):
        path = self._selection(node)
        leaf = path[-1]
        self._expansion(leaf)
        reward = self._rollout(leaf, reward_fn, other_robot_nodes)
        self._backpropagation(path, reward)
        self._update_action_prob_dist(node)

    def _selection(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # this node hasn't been explored, explore it
                return path
            # get all children that haven't been explored yet
            # this could be made faster with set compliment...
            unexplored = [child for child in self.children[node] if child not in self.children]
            # unexplored = self.children[node] - self.children.keys()
            if unexplored:
                # take one unexplored child at random
                next_node = unexplored.pop()
                path.append(next_node)
                return path
            # otherwise, continue down the tree to a leaf as according to UCB
            node = self._get_max_d_ucb_child(node)

    def _expansion(self, node):
        "Expands a node"
        if node in self.children:
            return
        self.children[node] = node.get_children()

    def _rollout(self, node, reward_fn, other_robot_nodes):
        n = node.t
        reward = 0
        while True:
            if node.is_terminal() or node.t == n + self.rollout_horizon:
                return reward
            else:
                node = node.get_random_child() #root node should not be included in reward calculation
                reward += reward_fn(other_robot_nodes.append(node)) - reward_fn(other_robot_nodes)
                
    
    def _backpropagation(self, path, reward, rollout_iter):
        for node in reversed(path):
            discount_factor = self.gamma^(rollout_iter - self.last_visit[node])
            N_new = discount_factor * self.N[node] + 1
            self.R[node] = (discount_factor * self.R[node] * self.N[node] + reward) / N_new
            self.N[node] = N_new
            self.last_visit[node] = rollout_iter
            
    def _get_max_d_ucb_child(self, node):
        # make sure all children are explored
        assert all(child in self.children for child in self.children[node])

        log_N_parent = np.log(self.N[node])

        def ucb(child): #discounted ucb now
            return self.R[child] + self.c * np.sqrt(
                log_N_parent / self.N[child]
            )

        return max(self.children[node], key=ucb)



# Adapted from https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
class MCTS:
    def __init__(self, root_node, c = 1, horizon = 100):
        self.R = defaultdict(float)
        self.N = defaultdict(float) #will return 0.0 if key doesn't exist
        self.children = {root_node : []} #keys are states, values are sets of states.
        self.c = c
        self.rollout_horizon = horizon
        self.root_node = root_node
        self.root_node.parent = None
        self.root_node = root_node
        self.root_node.parent = None

    def solve(self, node):
        # print([(k.state, k.parent.state, v/self.N[k]) for k,v in self.R.items() if k.parent is not None])
        if node.is_terminal():
            return None
        
        if node not in self.children:
            return node.get_random_child()
        
        def score(child):
            if self.N[child] == 0:
                return -np.inf
            return self.R[child] / self.N[child] 
        
        new_node = max(self.children[node], key = score)
        
        return new_node

    def run_mcts(self, node):
        path = self._selection(node)
        # print('selection', [(n.state, n.parent) for n in path])
        path = self._expansion(path)
        # leaf = path[-1]
        reward = self._rollout(path)
        self._backpropagation(path, reward)
        # print('after rollout:', [self.R[n] for n in path])

    def _selection(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # this node hasn't been explored, explore it
                return path
            unexplored = [child for child in self.children[node] if child not in self.children]
            if unexplored:
                # take one unexplored child at random
                next_node = unexplored.pop()
                path.append(next_node)
                return path
            # otherwise, continue down the tree to a leaf as according to UCB
            node = self._get_max_ucb_child(node)

    def _expansion(self, path):
        "Expands a node"
        leaf = path[-1]
        self.children[leaf] = leaf.get_children()
        try:
            leaf.get_random_child()
        except IndexError:
            return path
        else:
            path.append(leaf.get_random_child())
        return path
        

    def _rollout(self, path):
        node = path[-1]
        n = node.t
        # get cost to come
        reward = sum([self.R[ancestor]/self.N[ancestor] if self.N[ancestor] != 0 else ancestor.reward() for ancestor in path])
        while True:
            if node.is_terminal() or node.t == n + self.rollout_horizon:
                return reward
            else:
                # add estimated cost to go
                node = node.get_random_child()
                reward += node.reward()
                
    
    def _backpropagation(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.R[node] += reward
            
    def _get_max_ucb_child(self, node):
        # make sure all children are explored
        assert all(child in self.children for child in self.children[node])

        log_N_parent = np.log(self.N[node])

        def ucb(child):
            return self.R[child] / self.N[child] + self.c * np.sqrt(
                log_N_parent / self.N[child]
            )

        return max(self.children[node], key=ucb)

class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __hash__(self):
        return hash((self.state, self.parent))
    
    def __eq__(self, other):
        return self.state == other.state and self.parent == other.parent

    @abstractmethod
    def get_children(self):
        return NotImplemented
    
    @abstractmethod
    def get_random_child(self):
        #returns a new Node instance
        return NotImplemented
    
    @abstractmethod
    def is_terminal(self):
        #true if this node is terminal, ends the rollout
        return NotImplemented
    
    @abstractmethod
    def reward(self):
        #should only apply to terminal nodes
        return NotImplemented
