from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
batch_size = 64

with tf.Session() as sess:
    # restore model
    saver = tf.train.import_meta_graph("./Model/VAE/test_model-30000.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./Model/VAE"))
    # extract decoder only
    graph = tf.get_default_graph()
    decoder_input = graph.get_tensor_by_name("decoder/decoder_input:0")
    decoder_output = graph.get_tensor_by_name("outputs/output:0")

    # Testing
    # Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder_output, feed_dict={decoder_input: z_mu})
            canvas[(n - i - 1) * 28 : (n - i) * 28, j * 28 : (j + 1) * 28] = x_mean[
                0
            ].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()
