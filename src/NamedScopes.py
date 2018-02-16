import tensorflow as tf

A = tf.constant([4], tf.int32, name='A')
B = tf.constant([5], tf.int32, name='B')
C = tf.constant([6], tf.int32, name='C')

x = tf.placeholder(tf.int32, name='x')

# y = Ax^2 + Bx+ C
Ax2_1 = tf.multiply(A, tf.pow(x, 2), name="Ax2_1")
Bx = tf.multiply(B, x, name="Bx")
y1 = tf.add_n([Ax2_1, Bx, C], name="y1")

# y = Ax^2 + Bx^2
Ax2_2 = tf.multiply(A, tf.pow(x, 2), name="Ax2_2")
Bx2 = tf.multiply(B, tf.pow(x, 2), name="Bx2")
y2 = tf.add_n([Ax2_2, Bx2], name="y2")

y = y1 + y2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [10]}))

    writer = tf.summary.FileWriter('./tensorboards/m3_example5', sess.graph)
    writer.close()