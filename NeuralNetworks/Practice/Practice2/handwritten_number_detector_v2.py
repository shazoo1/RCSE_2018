import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Setting our ops
# Input
x = tf.placeholder(tf.float32, [None, 784],name='input')

keep_probability = tf.placeholder(tf.float32,name='prob')

W_1 = tf.Variable(tf.truncated_normal([784, 784],stddev=0.1), name='weight_1')
b_1 = tf.Variable(tf.truncated_normal([784],stddev=0.1), name='bias_1')

hidden_layer_1 = tf.layers.dropout(tf.nn.relu(tf.matmul(x, W_1) + b_1))


W_2 = tf.Variable(tf.zeros([784, 10]), name='weight_2')
b_2 = tf.Variable(tf.zeros([10]), name='bias_2')

y = tf.nn.softmax(tf.matmul(hidden_layer_1, W_2) + b_2)

y_ = tf.placeholder(tf.float32, [None, 10],name='class')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')

train_step = tf.train.GradientDescentOptimizer(0.5, name='optimizer').minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
             #feed_dict={keep_probability:0.5})

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
