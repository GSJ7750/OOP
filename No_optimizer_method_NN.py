import tensorflow
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w_1 = tf.Variable(tf.truncated_normal([784, 200]))
b_1 = tf.Variable(tf.truncated_normal([1, 200]))

w_2 = tf.Variable(tf.truncated_normal([200, 100]))
b_2 = tf.Variable(tf.truncated_normal([1, 100]))

w_3 = tf.Variable(tf.truncated_normal([100, 10]))
b_3 = tf.Variable(tf.truncated_normal([1, 10]))

def sigma(x):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

z_1 = tf.add(tf.matmul(X, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)
z_3 = tf.add(tf.matmul(a_2, w_3), b_3)
a_3 = sigma(z_3)

diff = tf.subtract(a_3, y)

def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

d_z_3 = tf.multiply(diff, sigmaprime(z_3))
d_b_3 = d_z_3
d_w_3 = tf.matmul(tf.transpose(a_2), d_z_3)

d_a_2 = tf.matmul(d_z_3, tf.transpose(w_3))
d_z_2 = tf.multiply(d_a_2, sigmaprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(X), d_z_1)

eta = tf.constant(0.5)
step = [
    tf.assign(w_1, tf.subtract(w_1, tf.multiply(eta, d_w_1)))
  , tf.assign(b_1, tf.subtract(b_1, tf.multiply(eta, tf.reduce_mean(d_b_1, axis=[0]))))
  , tf.assign(w_2, tf.subtract(w_2, tf.multiply(eta, d_w_2)))
  , tf.assign(b_2, tf.subtract(b_2, tf.multiply(eta, tf.reduce_mean(d_b_2, axis=[0]))))
  , tf.assign(w_3, tf.subtract(w_3, tf.multiply(eta, d_w_3)))
  , tf.assign(b_3, tf.subtract(b_3, tf.multiply(eta, tf.reduce_mean(d_b_3, axis=[0]))))
]

acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict = {X: batch_xs, y : batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict ={X: mnist.test.images[:1000],y : mnist.test.labels[:1000]})
        print(res)
