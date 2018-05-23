from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

from plotting import *

data_path = './data/'

mnist = input_data.read_data_sets(data_path, one_hot=True)
train_size = mnist.train.num_examples
test_size = mnist.test.num_examples

epochs = 10
batch_size = 256
classes = 10
features = 784
layer_nodes = [features, 100, 100, 100, classes]
stddev = 0.100
bias_weight_init = 0.100
learning_rate = 1e-4
epoch_errors = []

x = tf.placeholder(dtype=tf.float32, shape=[None, features], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')

W1 = tf.Variable(tf.truncated_normal([layer_nodes[0], layer_nodes[1]], stddev=stddev, name='W1'))
W2 = tf.Variable(tf.truncated_normal([layer_nodes[1], layer_nodes[2]], stddev=stddev, name='W2'))
W3 = tf.Variable(tf.truncated_normal([layer_nodes[2], layer_nodes[3]], stddev=stddev, name='W3'))
W4 = tf.Variable(tf.truncated_normal([layer_nodes[3], layer_nodes[4]], stddev=stddev, name='W4'))

b1 = tf.Variable(tf.truncated_normal([layer_nodes[1]], stddev=stddev, name='b1'))
b2 = tf.Variable(tf.truncated_normal([layer_nodes[2]], stddev=stddev, name='b2'))
b3 = tf.Variable(tf.truncated_normal([layer_nodes[3]], stddev=stddev, name='b3'))
b4 = tf.Variable(tf.truncated_normal([layer_nodes[4]], stddev=stddev, name='b4'))

def nn_model(x):
    input_layer = {'weights': W1, 'biases': b1}
    hidden_layer_1 = {'weights': W2, 'biases': b2}
    hidden_layer_2 = {'weights': W3, 'biases': b3}
    output_layer = {'weights': W4, 'biases': b4}

    input_layer_sum = tf.add(tf.matmul(x, input_layer['weights']), input_layer['biases'])
    input_layer_sum = tf.nn.relu(input_layer_sum)

    hidden_layer_1_sum = tf.add(tf.matmul(input_layer_sum, hidden_layer_1['weights']), hidden_layer_1['biases'])
    hidden_layer_1_sum = tf.nn.relu(hidden_layer_1_sum)
    
    hidden_layer_2_sum = tf.add(tf.matmul(hidden_layer_1_sum, hidden_layer_2['weights']), hidden_layer_2['biases'])
    hidden_layer_2_sum = tf.nn.relu(hidden_layer_2_sum)

    output_layer_sum = tf.add(tf.matmul(hidden_layer_2_sum, output_layer['weights']), output_layer['biases'])

    return output_layer_sum

def nn_train(x):
    pred = nn_model(x)
    pred = tf.identity(pred)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0.0

            for i in range(int(train_size / batch_size) +1):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            epoch_errors.append(epoch_loss)
            print('epoch:', epoch +1, 'of', epochs, ' with loss: ', epoch_loss)
        display_convergence(epoch_errors)

if __name__ == '__main__':
    nn_train(x)