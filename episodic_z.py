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
R = np.zeros([n_mem+1])
term = np.zeros([n_mem+1],dtype=np.int32)
SPrime = np.random.randn(n_mem,s_dim)
#uniform sampling
for i in range(n_mem):
    S[i,:] = env.observation_space.sample()
    SPrime[i,:],R[i],term[i],_ = env.get_transition(S[i,:],np.random.randn(2)*.01)
Theta = np.zeros([n_mem+1,n_mem+1])
Theta[:-1,:-1] = rbf(SPrime,S,b)
Theta[-1,-1] = 1.0
for i in np.nonzero(term)[0]:
    Theta[i,:] = 0
    Theta[i,-1] = 1.0
NN = Theta[:-1,:-1]
NT = Theta[:-1,-1]
print(NN.shape,NT.shape)
M = np.diag(np.exp(R[:-1]))
A = np.eye(n_mem)-np.matmul(M,NN)
b = np.matmul(M,np.expand_dims(NT,-1))*np.exp(R[-1])
print(A.shape,b.shape)
foo = np.linalg.solve(A,b)
Z = np.ones([n_mem+1,1])
old_Z = Z.copy()
for i in range(1000):
    Z = z_iteration(Theta,R,Z)
    diff = np.sum(np.abs(Z-old_Z))
    old_Z = Z.copy()
    print(i,diff)
    if diff == 0.0:
        break
#val,vec = np.linalg.eig(np.matmul(np.diag(np.exp(R)),Theta))
#print(val)
#print(Z)
#plt.ion()
print(np.sum(np.abs(foo[:,0]-Z[:-1,0])))
plt.subplot(1,3,1)
plt.scatter(SPrime[:,0],SPrime[:,1],c=np.log(foo[:,0]))
plt.colorbar()
plt.subplot(1,3,2)
plt.scatter(SPrime[:,0],SPrime[:,1],c=np.log(Z[:-1,0]))
plt.colorbar()
plt.subplot(1,3,3)
plt.scatter(SPrime[:,0],SPrime[:,1],c=(foo[:,0]-Z[:-1,0]))
plt.colorbar()
#plt.pause(.01)
plt.show()
