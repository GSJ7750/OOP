from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

url = 'https://www.bbc.com/korean/news-46562902'
r = requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")
mr = soup.find(class_="story-body__introduction")
text = mr.get_text()
split_text = list(text)

spacing = []
for i in range(len(split_text)):
    if split_text[i] == ' ':
        spacing.append(([split_text[i-1], split_text[i+1]]))
        
split1 = split_text
df = pd.get_dummies(split1)
split_text = list(filter((' ').__ne__, split_text))

split2 = split_text
df2 = pd.get_dummies(split2)

xdata = df2.values
y = np.ones([xdata.shape[0], 1])

for k in range(len(spacing)):
    for l in range(len(split2)):
        if split2[l] == spacing[k][0]:
            if split2[l+1] == spacing[k][1]:
                print("spacing : ", l)
                y[l][0] = y[l][0]-1
                
import numpy as np
import tensorflow as tf
import datetime
sess = tf.Session()

x = xdata


y_ = np.zeros([xdata.shape[0], xdata.shape[0]-1])

y = np.concatenate((y, y_),axis = 1)


c_ = tf.zeros([xdata.shape[0],xdata.shape[0]])
h_ = tf.zeros([xdata.shape[0],xdata.shape[0]])


X = tf.placeholder(dtype=tf.float32, shape=[None, xdata.shape[1]])
Y = tf.placeholder(dtype=tf.float32, shape=[None, xdata.shape[0]])
W = tf.Variable(tf.random_normal([xdata.shape[0], 1]))*0.5
b = tf.Variable(tf.random_normal([1]))


#he
#el
#ll
#lo

seq_len = len(x) #len(x)cell 갯수, 인풋이 몇 덩어리인지
num_units = len(sess.run(c_))  #len(sess.run((c_)))# hiddenlayer


class lstm:
    def build(c, h):
        args = tf.concat((X,h), axis=1)
#        print(args)

        out_size = 4 * num_units
        proj_size = args.shape[-1]
#        print(out_size)
#        print(proj_size)

        weights = tf.ones([proj_size, out_size]) * 0.5
#        print(weights)


        out = tf.matmul(args, weights)
#        print(out)

        bias = tf.ones([out_size]) * 0.5
#        print(bias)

        concat = out + bias
#        print(concat)

        i, j, f, o = tf.split(concat, 4, 1)
#        print(i)
#        print(j)
#        print(f)
#        print(o)

        g = tf.tanh(j)
#        print(g)

        def sigmoid_array(x):
            return 1 / (1 + tf.exp(-x))

        forget_bias = 1.0

        sigmoid_f = sigmoid_array(f + forget_bias)
#        print(sigmoid_f)

        sigmoid_array(i) * g

        new_c = c * sigmoid_f + sigmoid_array(i) * g
#        print(new_c)

        new_h = tf.tanh(new_c) * sigmoid_array(o)
#        print(new_h)

#        print('\n new_h:',new_h)
#        print('\n new_c',new_c)

#        print(res[1].h)
#        print(res[1].c)

        return new_c, new_h

bx = x[::-1]

by = y[::-1]



bc_ = tf.zeros([xdata.shape[0],xdata.shape[0]])
bh_ = tf.zeros([xdata.shape[0],xdata.shape[0]])


bX = tf.placeholder(dtype=tf.float32, shape=[None, xdata.shape[1]])
bY = tf.placeholder(dtype=tf.float32, shape=[None, xdata.shape[0]])
bW = tf.Variable(tf.random_normal([xdata.shape[0], 1]))*0.5
bb = tf.Variable(tf.random_normal([1]))

class blstm:
    def build(c, h):
        args = tf.concat((X,h), axis=1)
#        print(args)

        out_size = 4 * num_units
        proj_size = args.shape[-1]
#        print(out_size)
#        print(proj_size)

        weights = tf.ones([proj_size, out_size]) * 0.5
#        print(weights)


        out = tf.matmul(args, weights)
#        print(out)

        bias = tf.ones([out_size]) * 0.5
#        print(bias)

        concat = out + bias
#        print(concat)

        i, j, f, o = tf.split(concat, 4, 1)
#        print(i)
#        print(j)
#        print(f)
#        print(o)

        g = tf.tanh(j)
#        print(g)

        def sigmoid_array(x):
            return 1 / (1 + tf.exp(-x))

        forget_bias = 1.0

        sigmoid_f = sigmoid_array(f + forget_bias)
#        print(sigmoid_f)

        sigmoid_array(i) * g

        new_bc = c * sigmoid_f + sigmoid_array(i) * g
#        print(new_c)

        new_bh = tf.tanh(new_bc) * sigmoid_array(o)
#        print(new_h)

#        print('\n new_h:',new_h)
#        print('\n new_c',new_c)

#        print(res[1].h)
#        print(res[1].c)

        return new_bc, new_bh

##################################################################### Forward lstm

ta_c = tf.TensorArray(size=seq_len, dtype=tf.float32)
ta_h = tf.TensorArray(size=seq_len, dtype=tf.float32)

def body(last_state, last_output, step, ta_c, ta_h):
    
    output = lstm.build(last_state, last_output)[0]
    state = lstm.build(last_state, last_output)[1]
    ta_c = ta_c.write(step, state)
    ta_h = ta_h.write(step, output)
    return state, output, tf.add(step, 1), ta_c, ta_h
    

timesteps = seq_len
steps = lambda a, b, step, c, d: tf.less(step, timesteps)
lstm_output, lstm_state, step, ta_c, ta_h = tf.while_loop(steps, body, (c_, h_, 0, ta_c, ta_h), parallel_iterations=20)

output = lstm_output
logits = tf.matmul(output, W) + b

with tf.name_scope('mean_square_error'):
    mean_square_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.unstack(logits, axis = 1))))
tf.summary.scalar('mean_square_error', mean_square_error)

optimizer = tf.train.AdamOptimizer(0.0003)
minimize = optimizer.minimize(mean_square_error)

with tf.name_scope('error'):
    with tf.name_scope('mistakes'):
        mistakes = tf.not_equal(Y, tf.round(tf.unstack(logits, axis = 1)))
    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
tf.summary.scalar('error', error)

sess = tf.Session()
merged = tf.summary.merge_all()
date = str(datetime.datetime.now())
init_op = tf.global_variables_initializer()
sess.run(init_op)

epoch = 3000
for i in range(epoch):
    if (i + 1) % 100 == 0:
        summary, incorrect, mean_squ_err = sess.run([merged, error, mean_square_error], {X:x, Y:y})
        
        print('Epoch {:4d} | incorrect {: 3.1f}% | mean squ error {: 3.1f}'.format(i + 1, incorrect * 100, mean_squ_err))
    else:
        summary, acc = sess.run([merged, error], {X:x, Y:y})


    sess.run(minimize,{X:x, Y:y})
    
##################################################################### backward lstm
    
bta_c = tf.TensorArray(size=seq_len, dtype=tf.float32)
bta_h = tf.TensorArray(size=seq_len, dtype=tf.float32)

def bbody(last_state, last_output, step, bta_c, bta_h):
    
    boutput = blstm.build(last_state, last_output)[0]
    bstate = blstm.build(last_state, last_output)[1]
    bta_c = bta_c.write(step, bstate)
    bta_h = bta_h.write(step, boutput)
    return bstate, boutput, tf.add(step, 1), bta_c, bta_h
    

timesteps = seq_len


steps = lambda a, b, step, c, d: tf.less(step, timesteps)

blstm_output, blstm_state, step, bta_c, bta_h = tf.while_loop(steps, bbody, (bc_, bh_, 0, bta_c, bta_h), parallel_iterations=20)

boutput = blstm_output
blogits = tf.matmul(boutput, W) + b

with tf.name_scope('mean_square_error'):
    mean_square_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.unstack(logits, axis = 1))))
tf.summary.scalar('mean_square_error', mean_square_error)

optimizer = tf.train.AdamOptimizer(0.0003)
minimize = optimizer.minimize(mean_square_error)

with tf.name_scope('error'):
    with tf.name_scope('mistakes'):
        mistakes = tf.not_equal(Y, tf.round(tf.unstack(logits, axis = 1)))
    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
tf.summary.scalar('error', error)

sess = tf.Session()
merged = tf.summary.merge_all()

date = str(datetime.datetime.now())

init_op = tf.global_variables_initializer()
sess.run(init_op)

epoch = 3000

for i in range(epoch):
    if (i + 1) % 100 == 0:
        summary, incorrect, mean_squ_err = sess.run([merged, error, mean_square_error], {X:x, Y:y})
        
        print('Epoch {:4d} | incorrect {: 3.1f}% | mean squ error {: 3.1f}'.format(i + 1, incorrect * 100, mean_squ_err))
    else:
        summary, acc = sess.run([merged, error], {X:x, Y:y})


    sess.run(minimize,{X:x, Y:y})
    
##################################################################### Application
fw = sess.run(tf.equal(sess.run(Y, feed_dict={X:x, Y:y}),sess.run(tf.round(tf.unstack(logits, axis = 1)),feed_dict={X:x, Y:y})))
fw = sess.run(tf.one_hot(fw, 1, axis=0))

bw = sess.run(tf.equal(sess.run(Y, feed_dict={X:bx, Y:by}),sess.run(tf.round(tf.unstack(blogits, axis = 1)),feed_dict={X:bx, Y:by})))
bw = sess.run(tf.one_hot(bw, 1, axis=0))

fw = fw[0]
fw = fw[:,[0]]

bw = bw[0]
bw = bw[:,[0]]

Bi_Lstm_Output = np.column_stack((fw,bw))
print(Bi_Lstm_Output)

#######################################################################

sentence = split2

s_output = []
for i in range(len(sentence)):
    s_output.append(sentence[i])
    if Bi_Lstm_Output[i][0] == 0:
        s_output.append(' ')
print(s_output)

bs_output = []

for i in range(len(sentence)):
    bs_output.append(sentence[i])
    if Bi_Lstm_Output[-i-1][1] == 0:
        bs_output.append(' ')
print(bs_output)
