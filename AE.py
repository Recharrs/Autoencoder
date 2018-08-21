import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from autoencoder.model import Model


# Config
# ===========================================================
# Network Parameters
num_input = 1000 * 6 # Trajectory data
num_hidden_1 = 64  # 1st layer num features
num_hidden_2 = 64  # 2nd layer num features (the latent dim)

# Training Parameters
learning_rate = 10e-2
num_steps = 1000
batch_size = 32

# Location
data_dir = "/tmp/data/"
log_dir = "Log/AE"
model_dir = "Model/AE/test_model"
test_model_dir = "Model/AE/test_model/test_model-30000.meta"

# Mode
mode = "train"

# Model Architecture Configure
config = {
    "model": [
        {   # input lauer
            "name": "inputs",
            "layers": [{"type": "input", "name": "input", "shape": [None, num_input]}],
        }, { 
            "name": "encoder",
            "layers": [
                {
                    "type": "FC",
                    "name": "enc_1",
                    "input": "input",
                    "output_size": num_hidden_1,
                    "activation": tf.nn.sigmoid,
                },
                {
                    "type": "FC",
                    "name": "enc_2",
                    "input": "enc_1",
                    "output_size": num_hidden_2,
                    "activation": tf.nn.sigmoid,
                },
            ],
        }, {
            "name": "decoder",
            "layers": [
                {
                    "type": "block_input", 
                    "name": "decoder_input", 
                    "input": "enc_2"
                }, {
                    "type": "FC",
                    "name": "dec_1",
                    "input": "decoder_input",
                    "output_size": num_hidden_1,
                    "activation": tf.nn.sigmoid,
                }, {
                    "type": "FC",
                    "name": "dec_2",
                    "input": "dec_1",
                    "output_size": num_input,
                    "activation": tf.nn.sigmoid,
                },
            ],
        }, {
            "name": "outputs",
            "layers": [{"type": "output", "name": "output", "input": "dec_2"}],
        },
    ],
    "loss": [
        {
            "name": "enc_dec_loss",
            "weight": 1,
            "ground_truth": "input",
            "prediction": "output",
        }
    ],
}


# Training
# ===========================================================
if mode == "train":
    # Import MNIST data
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    # Construct model
    model = Model(config)
    model.train(learning_rate)

    # Visualize Graph
    writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    # Start training
    with tf.Session() as sess:
        # Initialize
        sess.run(model.init)

        # get model input
        graph = tf.get_default_graph()
        model_input = graph.get_tensor_by_name("inputs/input:0")

        # Traning
        for i in range(1, num_steps + 1):
            # Prepare Data
            batch_x, _ = mnist.train.next_batch(batch_size)

            # Run Optimization
            _, l = sess.run([model.optimizer, model.loss], feed_dict={model_input: batch_x})
            # Display loss
            if i % display_step == 0 or i == 1:
                print("Step %i: Minibatch Loss: %f" % (i, l))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, model_dir, global_step=num_steps)


# Testing
# ===========================================================
if mode == "test":
    pass
