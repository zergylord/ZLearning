import utils
import tf_utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
import time
env = gym.make("Pendulum-v0")
seed = 1337
env._seed(1337)
np.random.seed(seed)
#env = gym.make('MountainCarContinuous-v0')
'''
use_rp = True
b = 1e-1
'''
use_tf = True
use_rp = False
b = 1e-2
print(env.observation_space.shape,env.action_space.shape)
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
s = env.reset()
for i in range(n_mem):
    S[i,:] = s
    A[i,:] = env.action_space.sample()/1
    '''
    if np.random.rand() < .5:
        A[i,:] = env.action_space.high[0]
    else:
        A[i,:] = env.action_space.low[0]
    '''
    SPrime[i,:],R[i],term[i],_ = env.step(A[i,:])
    term[i] = R[i] > -1.0
    if False or term[i]:
        print('yay!')
        s = env.reset()
    else:
        s = SPrime[i,:]
n_term = term.sum()
Theta = np.zeros([n_mem,n_mem+n_term])
if use_rp:
    Theta[:,:-n_term] = utils.rbf(np.matmul(SPrime,RP),np.matmul(S,RP),b)
else:
    Theta[:,:-n_term] = utils.rbf(SPrime,S,b)
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

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(SPrime[:,0],SPrime[:,1],SPrime[:,2],c=np.log(Z[:,0]))
plt.show()
'''

print('loading id model...')
hid_dim = 100
z_dim = 3
bandwidth = 0.1
n_i = n_mem
def encode(inp,reuse=None):
    hid = tf.layers.dense(inp,hid_dim,activation=tf.nn.relu,name='hid1',reuse=reuse)
    z = tf.layers.dense(hid,z_dim,activation=None,name='hid2',reuse=reuse)
    return z
x_i = tf.placeholder(tf.float32,shape=[None,n_i,s_dim])
a_i = tf.placeholder(tf.float32,shape=[None,n_i,a_dim])
xPrime_i = tf.placeholder(tf.float32,shape=[None,n_i,s_dim])
x = tf.placeholder(tf.float32,shape=[None,s_dim])
a = tf.placeholder(tf.float32,shape=[None,a_dim])
xPrime = tf.placeholder(tf.float32,shape=[None,s_dim])


x_diff = encode(xPrime)-encode(x,True)
x_i_diff = encode(xPrime_i,True)-encode(x_i,True)
print(x_diff.shape,x_i_diff.shape)
#mb x 1 x n_1
print(x_diff.shape,x_i_diff.shape)
prob = tf_utils.rbf(tf.expand_dims(x_diff,1),x_i_diff,b=bandwidth)
a_hat = tf.squeeze(tf.matmul(prob,a_i),1)
loss = tf.losses.mean_squared_error(a,a_hat)

a_rand = tf.placeholder(tf.float32,shape=[None,a_dim])
rand_loss = tf.losses.mean_squared_error(a,a_rand)
perfect = 1 - loss/rand_loss


train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss) 
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'foo')
print('loading complete!')

cur_time = time.clock()
s = env.reset()
cumr = 0
cum_cost = 0
for i in range(1000):
    #P(s'|s)
    if use_rp:
        theta = utils.rbf(np.matmul(np.expand_dims(s,0),RP),np.matmul(S,RP),b)
    else:
        theta = utils.rbf(np.expand_dims(s,0),S,b)
    part = np.dot(theta[0],Z[:,0])
    control = theta[0]*Z[:,0]/part
    ind = np.random.choice(n_mem,p=control)
    if not use_tf:
        id_theta = utils.rbf(np.expand_dims(np.matmul(SPrime[ind],RP)-np.matmul(s,RP),0),
                np.matmul(SPrime,RP)-np.matmul(S,RP),b)
        a = np.matmul(id_theta,A)[0]
    else:
        a = sess.run(a_hat,
                feed_dict={xPrime:np.expand_dims(SPrime[ind],0),x:np.expand_dims(s,0),
                    xPrime_i:np.expand_dims(SPrime,0),
                    x_i:np.expand_dims(S,0),
                    a_i:np.expand_dims(A,0)})[0]
    #a = env.action_space.sample()
    sPrime,r,term,_ = env.step(a)
    term = r > -1.0
    cumr+=term
    cum_cost+=r
    s = sPrime
    #env.render()
print(time.clock()-cur_time,r,cum_cost,cumr)
