import numpy as np
from tf_utils import *
import tensorflow as tf
import gym
import time
cur_time = time.clock()
env = gym.make("Pendulum-v0")
#env = gym.make('MountainCarContinuous-v0')
x_dim = env.observation_space.shape[0]
hid_dim = 100
z_dim = 100
n_i = 1000
a_dim = env.action_space.shape[0]
bandwidth = 0.1
def encode(inp,reuse=None):
    hid = tf.layers.dense(inp,hid_dim,activation=tf.nn.relu,name='hid1',reuse=reuse)
    z = tf.layers.dense(hid,z_dim,activation=None,name='hid2',reuse=reuse)
    return z
x_i = tf.placeholder(tf.float32,shape=[None,n_i,x_dim])
a_i = tf.placeholder(tf.float32,shape=[None,n_i,a_dim])
xPrime_i = tf.placeholder(tf.float32,shape=[None,n_i,x_dim])
x = tf.placeholder(tf.float32,shape=[None,x_dim])
a = tf.placeholder(tf.float32,shape=[None,a_dim])
xPrime = tf.placeholder(tf.float32,shape=[None,x_dim])


x_diff = encode(xPrime)-encode(x,True)
x_i_diff = encode(xPrime_i,True)-encode(x_i,True)
print(x_diff.shape,x_i_diff.shape)
#mb x 1 x n_1
print(x_diff.shape,x_i_diff.shape)
prob = rbf(tf.expand_dims(x_diff,1),x_i_diff,b=bandwidth)
a_hat = tf.squeeze(tf.matmul(prob,a_i),1)
loss = tf.losses.mean_squared_error(a,a_hat)

a_rand = tf.placeholder(tf.float32,shape=[None,a_dim])
rand_loss = tf.losses.mean_squared_error(a,a_rand)
perfect = 1 - loss/rand_loss


train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss) 
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_replay = int(1e5)
X_replay = np.zeros([n_replay,x_dim])
A_replay = np.zeros([n_replay,a_dim])
XPrime_replay = np.zeros([n_replay,x_dim])
def gen_data():
    global X_replay,A_replay,XPrime_replay
    s = env.reset()
    print('generating samples...')
    for i in range(n_replay):
        if i > 0:
            X_replay[i,:] = XPrime_replay[i-1,:]
        else:
            X_replay[i,:] = s
        A_replay[i,:] = env.action_space.sample()
        XPrime_replay[i,:],_,_,_ = env.step(A_replay[i,:])
    print('... done generating')
gen_data()
n_steps = int(1e6)
refresh = int(1e3)
mb_dim = 32
def sample_mb():
    n_sample = mb_dim*(n_i+1)
    inds = np.random.choice(n_replay,size=n_sample,replace=False)
    return X_replay[inds[:(mb_dim*n_i)]].reshape([mb_dim,n_i,x_dim]), \
            A_replay[inds[:(mb_dim*n_i)]].reshape([mb_dim,n_i,a_dim]), \
            XPrime_replay[inds[:(mb_dim*n_i)]].reshape([mb_dim,n_i,x_dim]), \
            X_replay[inds[(mb_dim*n_i):]].reshape([mb_dim,x_dim]), \
            A_replay[inds[(mb_dim*n_i):]].reshape([mb_dim,a_dim]), \
            XPrime_replay[inds[(mb_dim*n_i):]].reshape([mb_dim,x_dim])



cum_loss = 0
for i in range(n_steps):
    X_i,A_i,XPrime_i,X,A,XPrime = sample_mb()
    _,cur_loss,pred = sess.run([train_op,loss,a_hat],
            feed_dict={xPrime:XPrime,x:X,xPrime_i:XPrime_i,x_i:X_i,a:A,a_i:A_i})
    cum_loss += cur_loss
    if (i+1) % refresh == 0:
        gen_data()
        A_rand = np.zeros([mb_dim,a_dim])
        for j in range(mb_dim):
            A_rand[j] = env.action_space.sample()
        percent_perfect = sess.run(perfect,
                feed_dict={a_rand:A_rand,xPrime:XPrime,x:X,xPrime_i:XPrime_i,x_i:X_i,a:A,a_i:A_i})
        #print(np.concatenate([pred[:10],A[:10]],-1))
        print(i+1,time.clock()-cur_time,cum_loss/refresh,percent_perfect)
        cur_time = time.clock()
        cum_loss = 0
        saver.save(sess,'bar')



