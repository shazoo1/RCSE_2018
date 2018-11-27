import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# To plot figure
accuracy_storage = []

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# https://pypi.org/project/jupyter-tensorboard/
# https://www.tensorflow.org/tutorials/estimators/cnn
# http://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/

# Setting our ops
# Input
with tf.name_scope('input') as scope:
    x = tf.placeholder(tf.float32, [None, 784],name='input')

    # Reshape for convolution recognition
    x_image = tf.reshape(x, [-1,28,28,1])


# Convolutional Layer #1
with tf.name_scope("conv_1") as scope:
    W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev=0.1))
    b_conv_1 = tf.Variable(tf.constant(0.1,shape=[32]))

    conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides=[1,1,1,1], padding='SAME') + b_conv_1
    conv_1 = tf.nn.relu(conv_1)


# Polling Layer #1
with tf.name_scope("pool_1") as scope:
    pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)


# Convolutional Layer #2
with tf.name_scope("conv_2") as scope:
    W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv_2 = tf.Variable(tf.constant(0.1, shape=[64]))


    conv_2 = tf.nn.conv2d(pool_1, W_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2
    conv_2 = tf.nn.relu(conv_2)


# Pooling Layer #2
with tf.name_scope("pool_2") as scope:
    pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)


# Reshaping
    h_pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])


# Fully connected layer
with tf.name_scope("fully_conn") as scope:
    W_1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    fc1 = tf.add(tf.matmul(h_pool_2_flat, W_1), b_1)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.layers.dropout(fc1, rate=0.30)


# Output, class prediction
with tf.name_scope('output') as scope:

    W_2 = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.1, shape=[10]))

    y = tf.add(tf.matmul(fc1, W_2), b_2)
    y = tf.nn.softmax(y)


with tf.name_scope('ground_truth') as scope:
    y_ = tf.placeholder(tf.float32, [None, 10],name='class')



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')
train_step = tf.train.AdamOptimizer(0.001, name='optimizer').minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                           })

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,

                                                         })
            print("Epoch: %s. Accuracy: %s" % (i,acc) )


            accuracy_storage.append(acc * 100)

    file_writer = tf.summary.FileWriter('board/linear_regress', sess.graph)

    prediction = (sess.run(y, feed_dict={x:mnist.test.images[1:2]}))
    for index,r in enumerate(prediction):
        print(index,r)
        print('Label is:', mnist.test.labels[1:2])



data = np.array(accuracy_storage)
print(accuracy_storage)
plt.plot(data)
plt.show()