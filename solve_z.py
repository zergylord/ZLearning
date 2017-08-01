from utils import *
import matplotlib.pyplot as plt
import numpy as np
from environment import ContGrid
def z_iteration(Theta,R,Z):
    return np.matmul(np.matmul(np.diag(np.exp(R)),Theta),Z)
env = ContGrid()
s_dim = 2
b = .01
n_mem = 1000
S = np.zeros([n_mem,s_dim])
R = np.zeros([n_mem])
term = np.zeros([n_mem],dtype=np.int32)
SPrime = np.random.randn(n_mem,s_dim)
#uniform sampling
for i in range(n_mem):
    S[i,:] = env.observation_space.sample()
    SPrime[i,:],R[i],term[i],_ = env.get_transition(S[i,:],np.random.randn(2)*.01)
n_term = term.sum()
Theta = np.zeros([n_mem,n_mem+n_term])
Theta[:,:-n_term] = rbf(SPrime,S,b)
term_R = np.zeros([n_term,1])
for i,ind in enumerate(np.nonzero(term)[0]):
    Theta[ind,:] = 0
    Theta[ind,-n_term+i] = 1.0
    term_R[i] = R[ind]
NN = Theta[:,:-n_term]
NT = Theta[:,-n_term:]
print(NN.shape,NT.shape)
M = np.diag(np.exp(R))
A = np.eye(n_mem)-np.matmul(M,NN)
b = np.matmul(np.matmul(M,NT),np.exp(term_R))
print(A.shape,b.shape)
Z = np.linalg.solve(A,b)
plt.scatter(SPrime[:,0],SPrime[:,1],c=np.log(Z[:,0]))
plt.colorbar()
plt.show()
