#%% ***********************************************************************
# RNN Study materials
#  File Name:RNN_Exercise_Strings
#  Description: Simple RNN Study materials follow up Sunghoon Kim's Lecture
#  Reference: https://www.youtube.com/user/hunkims
#  Authors: Kyuhwan Yeon (KyuhwanYeon@gmail.com)
# *************************************************************************
#%% Step0. Import Library
import tensorflow as tf
import numpy as np
#%%  Data preprocess
tf.set_random_seed(777)  # reproducibility
idx2char = ['A', 'B', 'C', 'D', 'E']
#x_data = [[0, 1, 2, 3, 4, 5]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # A 0
              [0, 1, 0, 0, 0],   # B 1
              [0, 0, 1, 0, 0],   # C 2
              [0, 0, 0, 1, 0],   # D 3
              [0, 0, 0, 0, 1],   # E 4
              ]]  

y_data = [[1, 0, 2, 3, 4]]    # True label: BACDE
# Set Hyper parameters
num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  
batch_size = 1   
sequence_length = 5  # BACDE = 5
learning_rate = 0.1
#%% Build the graph
# Set X and Y
X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label


#RNN Layer
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
with tf.variable_scope("myrnn21") as scope:
    outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)
#%% Run the Computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))