from utils import *
import matplotlib.pyplot as plt
import numpy as np
import gym
env = gym.make("Pendulum-v0")
s_dim = env.observation_space.shape[0]
z_dim = 100
if z_dim == s_dim:
    M = np.eye(s_dim)
else:
    M = np.random.randn(z_dim,s_dim)
a_dim = env.action_space.shape[0]
b = 1e-3
n_mem = 10000
S = np.zeros([2,n_mem,z_dim])
A = np.zeros([2,n_mem,a_dim])
R = np.zeros([2,n_mem])
SPrime = np.random.randn(2,n_mem,z_dim)
#uniform sampling
for c in range(2):
    for i in range(n_mem):
        s = env.reset()
        A[c,i,:] = env.action_space.sample()
        sPrime,R[c,i],term,_ = env.step(A[c,i,:])
        S[c,i,:] = np.squeeze(np.matmul(M,np.expand_dims(s,-1)))
        SPrime[c,i,:] = np.squeeze(np.matmul(M,np.expand_dims(sPrime,-1)))
        
Theta = rbf(SPrime[1]-S[1],SPrime[0]-S[0],b)
print(Theta.shape,A[0].shape)
pred_A = np.matmul(Theta,A[0])
diff = np.sum(np.abs(pred_A-A[1]),-1)
rand_A = np.zeros([2,n_mem,a_dim])
for c in range(2):
    for i in range(n_mem):
        rand_A[c,i,:] = env.action_space.sample()
rand_diff = np.sum(np.abs(rand_A[0]-rand_A[1]),-1)
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
