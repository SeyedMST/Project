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

elems = tf.convert_to_tensor([[10,30.,423,39]])
elems = tf.reshape(elems, [-1])
#p = tf.convert_to_tensor([0.2000009, 0.2, 0.3, 0.3000000001])

#tf.gather (g)
#tf.gather
#samples = tf.multinomial(tf.log([[0.2, 0.2, 0.3, 0.4]]), 1)

sample_size = 2
p = tf.convert_to_tensor([1.99999999999, 0.2, 0.3, 0.3000000001])
y = tf.convert_to_tensor([[1, 1, 0, 1.0]])
zzz = tf.shape(tf.multiply(p, y))

input_shape = tf.cast(tf.shape(p)[0], tf.int32)
sample_size = tf.minimum(sample_size, input_shape)


indices = tf.py_func(np.random.choice, [input_shape, sample_size, False, p], tf.int64)
y = tf.cast (indices, tf.int32)

y = tf.gather(elems, y)

neg_sample_size = tf.reduce_sum(tf.convert_to_tensor([0.99999]))
neg_sample_size = tf.cast(neg_sample_size, tf.int32)

values, indices = tf.nn.top_k(elems, neg_sample_size, False)


rr = tf.cast(p, tf.int32)
with tf.Session():
    tf.initialize_all_variables().run()
    # result = []
    # result.append(tf.while_loop(condition, body, [x]))
    # result = tf.concat(0, result)
    #result = tf.ceil(g1)
    print(zzz.eval())