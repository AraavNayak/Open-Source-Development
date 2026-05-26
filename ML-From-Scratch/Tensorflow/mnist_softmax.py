import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'tmp/data'
NUM_STEPS = 1000 # train for a 1000 steps
MINIBATCH_SIZE = 100 # each step uses 100 images

data = input_data.read_data_sets(DATA_DIR, one_hot=True) # one_hot=True means 3 --> [0 0 0 1 0 0 0 0 0 0]

# Input: 28×28 image --> flattened to 784 numbers
x = tf.placeholder(tf.float32, [None, 784]) #special type of tensor. None for the batch size (variable), 784 for the flattened image
W = tf.Variable(tf.zeros([784, 10])) # Weight matrix; maps pixels to digits

y_true = tf.placeholder(tf.float32, [None, 10]) #the correct answers (one hot encoded)
y_pred = tf.matmul(x, W) # model's prediction

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)) # (1) Applies softmax to y_pred   (2) Computes cross-entropy loss   (3) Averages over the batch

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # creates an operation; adds to dataflow graph

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables.initializer()) # Train
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))