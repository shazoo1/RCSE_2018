import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Setting our ops
# Input
x = tf.placeholder(tf.float32, [None, 784],name='input')

W = tf.Variable(tf.zeros([784, 10]), name='weight')
b = tf.Variable(tf.zeros([10]), name='bias')

relu = tf.nn.relu(tf.matmul(x, W) + b,)

keep_probability = tf.placeholder(tf.float32,name='keep_probability')
dropout = tf.nn.dropout(relu, keep_probability)

y = tf.nn.sigmoid(dropout)

y_ = tf.placeholder(tf.float32, [None, 10],name='class')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')

train_step = tf.train.GradientDescentOptimizer(0.5, name='optimizer').minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                            keep_probability:0.5})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                         y_: mnist.test.labels,
                                                         keep_probability: 0.5
                                                         }))

    file_writer = tf.summary.FileWriter('board/linear_regress', sess.graph)

    prediction = (sess.run(y, feed_dict={x:mnist.test.images[1:2]}))
    for index,r in enumerate(prediction):
        print(index,r)
        print('Label is:', mnist.test.labels[1:2])
