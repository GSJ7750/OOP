import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = [[0,0], [0,1], [1,0], [1,1],[0,1],[1,1],[1,0],[0,0]]
y_data = [[1,0],[0,1],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

x = tf.placeholder(tf.float32,[None,2])
y = tf.placeholder(tf.float32,[None,2])

w1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.zeros([2]))
_y1 = tf.sigmoid(tf.matmul(x,w1)+b1)
w2 = tf.Variable(tf.random_normal([2,2]))
b2 = tf.Variable(tf.zeros([2]))
_y2 = tf.matmul(_y1,w2)+b2

h = tf.nn.softmax(_y2)
cost = tf.losses.mean_squared_error(_y2, y)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.argmax(h,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

 for step in range(10001):
        sess.run(train,feed_dict={x: x_data, y: y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}))
            
 h, c, a = sess.run([_y2,predicted, accuracy],feed_dict={x:[[1,1],[0,0],[0,1],[1,0]], y:[[1,0],[1,0],[0,1],[0,1]]})
 print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
