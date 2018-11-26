import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
    conv_1 = tf.layers.conv2d(x_image, 32, 5, activation=tf.nn.relu)

#Polling Layer #1
with tf.name_scope("pool_1") as scope:
    pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)


# Convolutional Layer #2
with tf.name_scope("conv_2") as scope:
    conv_2 = tf.layers.conv2d(pool_1, 64, 3, activation=tf.nn.relu)

# Pooling Layer #2
with tf.name_scope("pool_2") as scope:
    pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)


h_pool_2_flat = tf.contrib.layers.flatten(pool_2)

# Fully connected layer
with tf.name_scope("fully_conn") as scope:
    fc1 = tf.layers.dense(h_pool_2_flat, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.25, training=True)


# Output, class prediction
with tf.name_scope('output') as scope:
    y = tf.layers.dense(fc1,10)

with tf.name_scope('ground_truth') as scope:
    y_ = tf.placeholder(tf.float32, [None, 10],name='class')




cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')


train_step = tf.train.AdamOptimizer(0.001, name='optimizer').minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                           })

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,

                                                         }))

    file_writer = tf.summary.FileWriter('board/linear_regress', sess.graph)

    prediction = (sess.run(y, feed_dict={x:mnist.test.images[1:2]}))
    for index,r in enumerate(prediction):
        print(index,r)
        print('Label is:', mnist.test.labels[1:2])
