import tensorflow as tf
import numpy as np

def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)
    #return tf.concat(0, [x, c])

def condition(x):
    return tf.reduce_sum(x) < 100

x = tf.Variable(tf.constant(0, shape=[2, 2]))
g = tf.constant(np.array([[0, 1], [0, 0.9999]]), dtype=tf.float32)
g1 = tf.constant(np.array([[0, 1], [0.9999, -0.00001]]), dtype=tf.float32)

with tf.Session():
    tf.initialize_all_variables().run()
    result = []
    result.append(tf.while_loop(condition, body, [x]))
    result = tf.concat(0, result)
    #result = tf.ceil(g1)
    print(result.eval())