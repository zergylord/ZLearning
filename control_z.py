from utils import *
import matplotlib.pyplot as plt
import numpy as np
from environment import ContGrid
def z_iteration(Theta,R,Z):
    return np.matmul(np.matmul(np.diag(np.exp(R)),Theta),Z)
def sample_passive():
    return np.random.randn(2)*.01
env = ContGrid()
'''
use_rp = True
b = 1e-1
'''
use_rp = False
b = 1e-2
s_dim = 2
a_dim = 2
RP = np.random.randn(s_dim,100)
oracle_id = False
n_mem = 1000
S = np.zeros([n_mem,s_dim])
A = np.zeros([n_mem,a_dim])
R = np.zeros([n_mem])
term = np.zeros([n_mem],dtype=np.int32)
SPrime = np.random.randn(n_mem,s_dim)
#uniform sampling
for i in range(n_mem):
    S[i,:] = env.observation_space.sample()
    A[i,:] = sample_passive()
    SPrime[i,:],R[i],term[i],_ = env.get_transition(S[i,:],A[i,:])
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

s = np.asarray([.9,.9])#env.observation_space.sample()
plt.ion()
for i in range(1000):
    #P(s'|s)
    if use_rp:
        theta = rbf(np.matmul(np.expand_dims(s,0),RP),np.matmul(S,RP),b)
    else:
        theta = rbf(np.expand_dims(s,0),S,b)
    part = np.dot(theta[0],Z[:,0])
    control = theta[0]*Z[:,0]/part
    if i % 10 == 0:
        plt.clf()
        plt.scatter(SPrime[:,0],SPrime[:,1],c=control)
        plt.colorbar()
        #plt.show()
        plt.pause(.001)
    ind = np.random.choice(n_mem,p=control)
    if oracle_id:
        #ind = np.argmax(control)
        s = SPrime[ind]
    else:
        id_theta = rbf(np.matmul(SPrime,RP)-np.matmul(np.expand_dims(s,0),RP),
                np.matmul(SPrime,RP)-np.matmul(S,RP),b)
        pred_A = np.matmul(id_theta,A)
        a = pred_A[ind]
        sPrime,r,term,_ = env.get_transition(s,a)
        #print(s,a,sPrime)
        s = sPrime.copy()
    if r == 0.0:
        print(i)
        break

#sPrime,r,term,_ = env.get_transition(s,a)
