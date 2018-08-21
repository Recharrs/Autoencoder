import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder.model import Model
from config_AE import config

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000

# Construct model
model = Model(config)
model.train(learning_rate)

# Visualize Graph
writer = tf.summary.FileWriter("Log/AE")
writer.add_graph(tf.get_default_graph())

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
    saver.save(sess, "./Model/AE/test_model", global_step=num_steps)

