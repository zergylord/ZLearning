from utils import *
import matplotlib.pyplot as plt
import numpy as np
from environment import ContGrid
env = ContGrid()
s_dim = 100
if s_dim == 2:
    M = np.eye(2)
else:
    M = np.random.randn(s_dim,2)
a_dim = 2
b = 1e-4
n_mem = 1000
S = np.zeros([2,n_mem,s_dim])
A = np.zeros([2,n_mem,a_dim])
R = np.zeros([2,n_mem])
SPrime = np.random.randn(2,n_mem,s_dim)
#uniform sampling
for c in range(2):
    for i in range(n_mem):
        s = env.observation_space.sample()
        A[c,i,:] = np.random.randn(2)*.01
        sPrime,R[c,i],term,_ = env.get_transition(s,A[c,i,:])
        S[c,i,:] = np.squeeze(np.matmul(M,np.expand_dims(s,-1)))
        SPrime[c,i,:] = np.squeeze(np.matmul(M,np.expand_dims(sPrime,-1)))
        
Theta = rbf(SPrime[1]-S[1],SPrime[0]-S[0],b)
print(Theta.shape,A[0].shape)
pred_A = np.matmul(Theta,A[0])
diff = np.sum(np.square(pred_A-A[1]),-1)
rand_diff = np.sum(np.square(np.random.randn(n_mem,a_dim)*.01-np.random.randn(n_mem,a_dim)*.01),-1)
print(pred_A[0],A[1,0])
print('% error: ',np.mean(diff)/np.mean(rand_diff))
print('% perfect: ',1-np.mean(diff)/np.mean(rand_diff))
'''
plt.hist(diff)
plt.hist(rand_diff,alpha=.5)
plt.figure()
plt.scatter(SPrime[0,:,0],SPrime[0,:,1],c=diff)
#plt.scatter(SPrime[0,:,0]-S[0,:,0],SPrime[0,:,1]-S[0,:,1],c=diff)
plt.colorbar()
plt.show()
'''
