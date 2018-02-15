import tensorflow as tf

# y = Wx + b
W = tf.constant([10, 100], name='const_W')

# Note that these placeholders can hold tensors of any shape
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name="Wx")

y = tf.add(Wx, b, name="y")


with tf.Session() as sess:
    print("Intermediate result: Wx = ", sess.run(Wx, feed_dict={x: [3, 33]}))
    print("Final result: Wx + b = ", sess.run(y, feed_dict={x: [5, 50], b: [7, 9]}))

writer = tf.summary.FileWriter('../m3_example2', sess.graph)
writer.close()
