import numpy as np
class ObservationSpace(object):
    def __init__(self,shape,sample = None):
        self.shape = shape
        if sample is None:
            self.sample = lambda: np.random.randn(shape)
        else:
            self.sample = sample
class ContGrid():
    max_s = 1
    observation_space = ObservationSpace(2,lambda: np.random.rand(2)*ContGrid.max_s)
    def __init__(self):
        self.goal = np.array([.1,.5])*self.max_s
        print(self.goal)
        self.goal_size = .15*self.max_s
        self.reset()
        self.a_scale = 1
    def reset(self):
        #self.s = np.asarray([.1,.1])
        self.s = np.random.rand(2)*self.max_s
        return self.s
    def get_transition(self,s,a):
        term = False
        r = -1.0
        sPrime =  s + self.a_scale*a #+ np.random.randn(2)*self.a_scale/1000
        if np.any(sPrime > self.max_s) or np.any(sPrime < 0):
            sPrime[sPrime >= self.max_s] = self.max_s-.01*self.max_s
            sPrime[sPrime < 0] = 0+.01*self.max_s
            r = -1.0
            #term = True
        elif np.all(sPrime > self.goal) and np.all(sPrime < (self.goal+self.goal_size)):
            r = 0
            term = True
        return sPrime.copy(),r,term,False
    def step(self,a):
        self.s,r,term,_ = self.get_transition(self.s,a)
        return sPrime.copy(),r,term,False
