from ipp.planning import MCTS, Node
from ipp.dynamics import GridDynamics, SparseDynamics
from ipp.cost import kl_detection_vectorised
from random import choice
import numpy as np
from math import ceil

class Robot(Node, GridDynamics):
    def __init__(self,
                 init_state,
                 init_time,
                 env,
                 n_neighbours,
                 x_grid,
                 y_grid,
                 gmm,
                 pdf_dict,
                 p_curr,
                 rep_pts,
                 vmmp,
                 vel=1
                 ):
        GridDynamics.__init__(self, env, n_neighbours, x_grid, y_grid)
        self.state = init_state
        self.t = init_time
        self.gmm = gmm
        self.pdf_dict = pdf_dict
        self.p_curr = p_curr
        self.rep_pts = rep_pts
        self.vmmp = vmmp
        self.vel = vel
 
    def get_children(self):
        states = self.get_next_states(self.state)

        children =[
            Robot(
            s,
            self.get_toa(s) + self.t,
            self.env,
            self.n_neighbours,
            self.x,
            self.y,
            self.gmm,
            self.pdf_dict,
            self.p_curr,
            self.rep_pts,
            self.vmmp
            )
            for s in states
            ]
            
        return children
    
    def get_random_child(self):
        states = self.get_next_states(self.state)
        random_state = choice(states)

        return Robot(
            random_state,
            self.get_toa(random_state) + self.t,
            self.env,
            self.n_neighbours,
            self.x,
            self.y,
            self.gmm,
            self.pdf_dict,
            self.p_curr,
            self.rep_pts,
            self.vmmp
            )
    
    def is_terminal(self):
        is_terminal = False
        if self.t >= 99:
            is_terminal = True
        return is_terminal
    
    def reward(self):
        R, self.pdf_dict = kl_detection_vectorised(
            self.p_curr,
            self.pdf_dict,
            self.state,
            *self.gmm,
            self.rep_pts,
            self.vmmp,
            self.t
        )
        return R
    
    def get_toa(self, next_state):
        d = np.linalg.norm(next_state - self.state)
        return ceil(d / self.vel) #have to round up to get toa
    

class SparseRobot(Node, SparseDynamics):
    def __init__(self,
                 state,
                 possible_next_states,
                 velocity = 1,
                 reward_fn=None,
                 parent=None
                 ):
        '''
        Inputs
        -------------
        state : tuple
            tuple of state info (x,y,t). Time now included in the state
        reward_dict : dict
            dictionary with keys states (x,y,t) and values rewards
        '''
        SparseDynamics.__init__(self,
                                possible_next_states=possible_next_states,
                                velocity=velocity
                               )
        self.state = state
        self.t = state[-1]
        self.reward_fn = reward_fn
        self.parent = parent

    def get_children(self):
        states = self.get_next_states(self.state)

        children = [
                    SparseRobot(state = s,
                                possible_next_states=self.possible_next_states,
                                velocity = self.velocity,
                                reward_fn=self.reward_fn,
                                parent=self)
                    for s in states
                    ]
        return children

    def get_random_child(self):
        states = self.get_next_states(self.state)
        random_state = choice(states)

        return SparseRobot(
                           state = random_state,
                           possible_next_states=self.possible_next_states,
                           velocity = self.velocity,
                           reward_fn=self.reward_fn,
                           parent=self
                           )
    
    def is_terminal(self):
        is_terminal = False
        if self.state[-1] >= 99:
            is_terminal = True
        if len(self.get_next_states(self.state)) == 0:
            is_terminal = True
        return is_terminal
    
    def reward(self):
        if self.parent is None:
            return 0
        else:
            return self.reward_fn(self.state)
