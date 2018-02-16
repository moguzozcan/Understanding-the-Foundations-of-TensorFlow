import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

x = np.arange(100)
y = 10 * x + 2 + np.random.random((100,)) * 100

plt.scatter(x, y, marker='x')
plt.show()

X = np.stack((x, np.ones((100,))), axis=1)
X

Y = y.reshape(-1, 1)

from numpy import dot
from numpy.linalg import inv

W = dot(dot(inv(dot(x.T, x)), x.T), y)

yp = dot(x, W)

plt.scatter(x, y, marker='x')
plt.plot(x, yp, c='r')
plt.show()


g = tf.Graph()

with g.as_default():
    Xt = tf.constant(x.astype(float))
    Yt = tf.constant(y)

sess = tf.InteractiveSession(graph=g)

sess.run(Xt)

Wt = tf.matmul(tf.matmul(
    tf.matrix_inverse(
        tf.matmul(tf.transpose(Xt), Xt)), tf.transpose(Xt)), Yt)

w_ = sess.run(Wt)

w_

from tensorflow.contrib.learn import datasets

boston = datasets.load_boston()

Xb = np.hstack((boston.data, np.ones((506, 1))))
print(Xb.shape)

Yb = boston.target.reshape(-1, 1)

print(Yb.shape)

Wb = dot(dot(inv(dot(Xb.T, Xb)), Xb.T), Yb)

np.mean((dot(Xb, Wb)-Yb)**2)**0.5

#---------

g = tf.Graph()

Wt = tf.matmul(tf.matmul(
    tf.matrix_inverse(
        tf.matmul(tf.transpose(Xt), Xt)), tf.transpose(Xt)), Yt)

sess = tf.InteractiveSession(graph=g)

w_ = sess.run(Wt)

#---------

g = tf.Graph()

Xt = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Yt = tf.placeholder(dtype=tf.float32, shape=[None, 1])

Wt = tf.matmul(tf.matmul(
    tf.matrix_inverse(
        tf.matmul(tf.transpose(Xt), Xt)), tf.transpose(Xt)), Yt)

sess = tf.InteractiveSession(graph=g)

sess.run(Wt, feed_dict={Xt:X, Yt:Y})
