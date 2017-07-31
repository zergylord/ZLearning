from utils import *
import matplotlib.pyplot as plt
import numpy as np
from environment import ContGrid
def z_iteration(Theta,R,Z):
    return np.matmul(np.matmul(np.diag(np.exp(R)),Theta),Z)
env = ContGrid()
s_dim = 2
gamma = .95
b = .01
n_mem = 1000
S = np.random.randn(n_mem,s_dim)
R = np.random.randn(n_mem)
SPrime = np.random.randn(n_mem,s_dim)
#uniform sampling
for i in range(n_mem):
    S[i,:] = env.observation_space.sample()
    SPrime[i,:],R[i],term,_ = env.get_transition(S[i,:],np.random.randn(2)*.01)
Theta = rbf(SPrime,S,b)
Z = np.ones([n_mem,1])
old_Z = Z.copy()
for i in range(100):
    Z = z_iteration(Theta,R,Z**gamma)
    print(np.sum(np.abs(Z-old_Z)))
    old_Z = Z.copy()
plt.scatter(SPrime[:,0],SPrime[:,1],c=np.log(Z[:,0]))
plt.show()
