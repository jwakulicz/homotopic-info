import numpy as np
import vmmp_gmm.env_utils as env_utils
from copy import copy

def tuple_to_string(hsig_tuple):
    '''Returns the h signature as a string to be used with VMMP structure.'''
    hsig_as_string = ''
    for i in range(len(hsig_tuple)):
        hsig_as_string += str(hsig_tuple[i])
    return hsig_as_string


class HSignature:
    def __init__(self, sig):
        self.sig = sig

    def reduce(self):
        '''Reduces a h signature by removing cases where letters are followed by their "inverse".'''
        if len(self.sig) < 2:
            return

        # sig_copy = [0, *self.sig, 0] # HACK: adding zeros as a padding, which is never there. obstacles are counted from 1

        # self.sig = tuple([
        #     sig_copy[idx]
        #     for idx in range(1, len(sig_copy)-1)
        #     if sig_copy[idx] != -sig_copy[idx+1] and sig_copy[idx-1] != -sig_copy[idx]
        # ])
        for h1_idx in range(len(self.sig)-1):
            if self.sig[h1_idx] == -self.sig[h1_idx+1]:
                self.sig = self.sig[:h1_idx] + self.sig[h1_idx+2:]
                break
        # after iterating though all elements, return
        else:
            return

        self.reduce()
    
    def get_sfx(self):
        if self.sig != ():
            return self.sig[-1]
        else:
            return self.sig
    
    def as_string(self):
        '''Returns the h signature as a string to be used with VMMP structure.'''
        return tuple_to_string(self.sig)
    
    @classmethod
    def concat(cls, sig, ele):
        h_sig = sig.sig + (ele,)
        return cls(h_sig)

    @classmethod
    def get_hsig_from_path(cls, path, rep_pts, obs_x_bounds):
        ''' get hsig of a path, where path is a bunch of line segments.
        Inputs 
        ---------
        path: list of 1x2 np.arrays
            [x,y] points long the path
        rep_pts: list of 1x2 np.arrays
            one [x,y] point inside each obstacle
        obs_x_bounds : list of 1x2 np.arrays
            list of [x_L, x_R] left and right boundaries of a bounding box around
            each obstacle
        
        Returns
        ---------
        HSignature object with signature of the path (not reduced)'''
        h_sig = ()
        
        path = path[~np.isnan(path).any(axis=1)]
        sorted_rep_pts = list(enumerate(sorted(rep_pts, key = lambda pt: pt[0])))
        sorted_obs_bounds = sorted(obs_x_bounds, key = lambda pt: pt[0])

        for p1, p2 in zip(path[:-1], path[1:]):
            if p1[0] <= p2[0]:  #if the segment goes from left to right it is a positive crossing
                segment = [p1, p2]
                for pt_idx, rep_pt in sorted_rep_pts:
                    if env_utils.check_ray_intersect(segment,
                                                     np.array([sorted_obs_bounds[pt_idx][0],rep_pt[1]])):
                        # print(segment)
                        h_sig += (pt_idx+1,)
            
            else:
                segment = [p2, p1]
                for pt_idx, rep_pt in sorted_rep_pts[::-1]:
                    if p1[0] > p2[0]: # if the segment goes from right to left, it is a negative crossing
                        if env_utils.check_ray_intersect(segment,
                                                         np.array([sorted_obs_bounds[pt_idx][1], rep_pt[1]])):
                            # print(segment)
                            h_sig += (-(pt_idx+1),)

        return cls(h_sig)

    @classmethod
    def get_hsig_from_string(cls, string):
        h_sig = ()
        for i, letter in enumerate(string):
            if letter == '-':
                letter += string[i+1]
            
            if string[i-1] == '-':
                continue
            
            h_sig += (int(letter),)
        return cls(h_sig)
    
    def __str__(self):
        return self.as_string()

    # def __eq__(self, other):
    #     return str(self) == str(other)

    def __eq__(self, other):
        if isinstance(other, HSignature):
            return self.sig == other.sig
        elif isinstance(other, str):
            return self.sig == other
        else:
            return False

    def __hash__(self) -> int:
        return hash(str(self))

if __name__ == '__main__':
    import numpy as np
    #path = [Location(xlon=0., ylat=0.), Location(xlon=1., ylat=0.), Location(xlon=2., ylat=1.), Location(xlon=0.5, ylat=1.)]
    path = [np.array([0,0]), np.array([1,0]), np.array([2,1]), np.array([1,1])]
    ref_pts = [np.array([0.25,-1]), np.array([0.75,-1]), np.array([1.5,-1])]

    foo = HSignature.get_hsig_from_path(path, ref_pts)
    print(foo.sig)
    print(foo.sig[3])
    # foo.reduce()
    # print(foo.sig)
    print(foo.as_string())
    import vmmp_utils
    print(vmmp_utils.get_suffix(foo))