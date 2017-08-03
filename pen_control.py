from utils import *
import matplotlib.pyplot as plt
import numpy as np
import gym
env = gym.make("Pendulum-v0")
env = gym.make('MountainCarContinuous-v0')
'''
use_rp = True
b = 1e-1
'''
use_rp = True
b = 1e-2
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
RP = np.random.randn(s_dim,100)
n_mem = 1000
S = np.zeros([n_mem,s_dim])
A = np.zeros([n_mem,a_dim])
R = np.zeros([n_mem])
term = np.zeros([n_mem],dtype=np.int32)
SPrime = np.random.randn(n_mem,s_dim)
#uniform sampling
for i in range(n_mem):
    S[i,:] = env.reset()
    A[i,:] = env.action_space.sample()
    SPrime[i,:],R[i],term[i],_ = env.step(A[i,:])
    #term[i] = R[i] > -1.0
n_term = term.sum()
Theta = np.zeros([n_mem,n_mem+n_term])
if use_rp:
    Theta[:,:-n_term] = rbf(np.matmul(SPrime,RP),np.matmul(S,RP),b)
else:
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
LHS = np.eye(n_mem)-np.matmul(M,NN)
RHS = np.matmul(np.matmul(M,NT),np.exp(term_R))
Z = np.linalg.solve(LHS,RHS)
plt.scatter(SPrime[:,0],SPrime[:,1],c=np.log(Z[:,0]))
plt.colorbar()
plt.show()

s = env.reset()
plt.ion()
for i in range(1000):
    #P(s'|s)
    if use_rp:
        theta = rbf(np.matmul(np.expand_dims(s,0),RP),np.matmul(S,RP),b)
    else:
        theta = rbf(np.expand_dims(s,0),S,b)
    part = np.dot(theta[0],Z[:,0])
    control = theta[0]*Z[:,0]/part
    '''
    if i % 10 == 0:
        plt.clf()
        plt.scatter(SPrime[:,0],SPrime[:,1],c=control)
        plt.colorbar()
        #plt.show()
        plt.pause(.001)
    '''
    ind = np.random.choice(n_mem,p=control)
    id_theta = rbf(np.matmul(SPrime,RP)-np.matmul(np.expand_dims(s,0),RP),
            np.matmul(SPrime,RP)-np.matmul(S,RP),b)
    pred_A = np.matmul(id_theta,A)
    a = pred_A[ind]
    sPrime,r,term,_ = env.step(a)
    s = sPrime
    env.render()
    #term = r > -1.0
    print(r,term)
