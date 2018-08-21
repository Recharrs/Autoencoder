import sys, os
import numpy as np
import tensorflow as tf

from autoencoder.model import Model


# Utils
# ===========================================================



# Config
# ===========================================================
import tensorflow as tf

# Network Parameters
image_dim = 784  # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# custom loss function (reconstrution loss only)
def custom_loss(ground_true, prediction):
    loss = ground_true * tf.log(1e-10 + prediction) + (1 - ground_true) * tf.log(
        1e-10 + 1 - prediction
    )
    loss = -tf.reduce_sum(loss, 1)
    return loss


# custom random initializer
def custom_random_init(shape, name=None):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), name=name)


# Model Architecture Configure
config = {
    "random_init": custom_random_init,
    "model": [
        {
            "name": "inputs",
            "layers": [{"type": "input", "name": "input", "shape": [None, image_dim]}],
        },
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "FC",
                    "name": "encoder",
                    "input": "input",
                    "output_size": hidden_dim,
                    "activation": tf.nn.tanh,
                },
                {
                    "type": "sampler",
                    "name": "sampler",
                    "input": "encoder",
                    "output_size": latent_dim,
                },
            ],
        },
        {
            "name": "decoder",
            "layers": [
                {"type": "block_input", "name": "decoder_input", "input": "sampler"},
                {
                    "type": "FC",
                    "name": "decoder_1st",
                    "input": "decoder_input",
                    "output_size": hidden_dim,
                    "activation": tf.nn.tanh,
                },
                {
                    "type": "FC",
                    "name": "decoder_2nd",
                    "input": "decoder_1st",
                    "output_size": image_dim,
                    "activation": tf.nn.sigmoid,
                },
            ],
        },
        {
            "name": "outputs",
            "layers": [{"type": "output", "name": "output", "input": "decoder_2nd"}],
        },
    ],
    "loss": [
        {
            "name": "encode_decode_loss",
            "weight": 1,
            "ground_truth": "input",
            "prediction": "output",
            "loss_func": custom_loss,
        }
    ],
}

mode = sys.argv[1]


# Training
# ===========================================================
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder.model import Model
from config_VAE import config

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

display_step = 1000

# Construct model
model = Model(config)
model.train(learning_rate)

# Visualize Graph
writer = tf.summary.FileWriter("Log/VAE")
writer.add_graph(tf.get_default_graph())

# Start training
with tf.Session() as sess:
    # Initialize
    sess.run(model.init)

    # get model input
    graph = tf.get_default_graph()
    model_input = graph.get_tensor_by_name("inputs/input:0")

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run Optimization
        _, l = sess.run([model.optimizer, model.loss], feed_dict={model_input: batch_x})
        # Display loss
        if i % display_step == 0 or i == 1:
            print("Step %i, Loss: %f" % (i, l))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "./Model/VAE/test_model", global_step=num_steps)        


# Testing
# ===========================================================
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
