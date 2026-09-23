from abc import ABC, abstractmethod
import numpy as np
import itertools
# from skimage.draw import line

class Dynamics:
    def __init__(self, env):
        self.env = env

    def get_next_states(self, init_state):
        possible_next_states = [
            self.get_next_state(
                init_state,
                ctrl
                )
            for ctrl in self.ctrl_space
        ]      

        valid_next_states = [
            state
            for state in possible_next_states
            if self.is_valid(state, init_state)
        ]
        return valid_next_states
    
    def is_valid(self, state, init_state):
        if state is None:
            print('rejected due to OOB')
            return False
        else:
            init_env_state = self.state_to_env(init_state)
            next_env_state = self.state_to_env(state)
            
            if self.env[init_env_state] == 1 or self.env[next_env_state] == 1:
                print('rejected because in obstacle')
                return False
            rr, cc = line(init_env_state[0], init_env_state[1], next_env_state[0], next_env_state[1])
            if np.any(self.env[rr, cc] == 1):
                print('rejected because path goes through obstacle')
                return False
            else:
                return True
    
    @abstractmethod
    def get_next_state(self, init_state, ctrl):
        return NotImplemented
    
class GridDynamics(Dynamics):
    def __init__(self, env, n_neighbours, x_grid, y_grid):
        Dynamics.__init__(self, env)
        self.init_ctrl_space(n_neighbours)
        # self.init_ctrl_space()
        self.x = x_grid
        self.y = y_grid
        self.n_neighbours = n_neighbours

    def init_ctrl_space(self, n_neighbours):
        #8-connected grid
        pos_ctrl = np.linspace(0, n_neighbours, n_neighbours+1)
        neg_ctrl = pos_ctrl * -1
        ctrls = np.unique(np.concatenate((pos_ctrl, neg_ctrl)))
        ctrl_space = itertools.product(ctrls,ctrls)
        self.ctrl_space = np.array(list(ctrl_space))

    def state_to_env(self, state):
        j = np.argwhere(self.x == state[0][0])[0][0]
        i = self.y.shape[0] - 1 - np.argwhere(self.y == state[0][1])[0][0]
        return i, j

    def env_to_state(self, coords):
        if any(i < 0 for i in coords):
            pass
        elif coords[0] > self.env.shape[0]-1 or coords[1] > self.env.shape[1]-1:
            pass
        else:
            return np.array([[self.x[int(coords[1])], 
                              self.y[self.y.shape[0]-1-int(coords[0])]]])
    
    def get_next_state(self, init_state, ctrl):
        init_env_coords = self.state_to_env(init_state)
        next_env_coords = init_env_coords + ctrl
        return self.env_to_state(next_env_coords)
    
class SparseDynamics:
    def __init__(self, possible_next_states, velocity=1):
        self.possible_next_states = possible_next_states
        self.velocity = velocity

    def get_next_states(self, init_state):
        valid_next_states = [
            s for s in self.possible_next_states
            if 
            self.is_valid(s, init_state)
        ]
        return valid_next_states

    def is_valid(self, state, init_state):
        #robot must be able to travel to the next state in time, and next
        #state time must be greater than initial state time.
        x1 = np.array(state[:-1])
        x2 = np.array(init_state[:-1])
        return (np.linalg.norm(x1-x2) / self.velocity) < (state[-1] - init_state[-1]) and ((state[-1] - init_state[-1]) > 0)