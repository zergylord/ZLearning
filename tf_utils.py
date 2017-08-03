import numpy as np
import tensorflow as tf
def softmax(X):
    X = tf.exp(X - tf.reduce_max(X,-1,keep_dims=True))
    return X/tf.reduce_sum(X,-1,keep_dims=True)
#cdist for matrices with arbitary num of batch ranks
def cdist(X,Y):
    rank = len(X.shape)
    XX = tf.reduce_sum(tf.square(X),-1,keep_dims=True)
    YY = tf.expand_dims(tf.reduce_sum(tf.square(Y),-1),-2)
    if rank == 2:
        XY = -2*tf.matmul(X,tf.transpose(Y))
    elif rank == 3:
        XY = -2*tf.matmul(X,tf.transpose(Y,[0,2,1]))
    elif rank == 3:
        XY = -2*tf.matmul(X,tf.transpose(Y,[0,1,3,2]))
    else:
        print("FOOBAR")
    return XX+YY+XY
def rbf(X,Y,b=1.0,normalize=True):
    raw = -cdist(X,Y)/b
    if normalize:
        return softmax(raw)
    else:
        return tf.exp(raw)
        #return raw
