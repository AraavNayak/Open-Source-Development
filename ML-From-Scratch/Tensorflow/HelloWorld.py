import tensorflow as tf

print(tf.__version__)
h = tf.constant("Hello")
w = tf.constant("World")
hw = h + w # does not compute sum; just adds summation operation to a graph of computations to be done later


# NOTE: Session has been deprecated since Tensorflow version 2.0
# with tf.Session() as sess: # interface for tensorflow calculation
#     ans = sess.run(hw) # executes the computation
# print(ans)

print(hw.numpy().decode()) # can directly evaluate (eager execution)



