import numpy as np
import tensorflow as tf
from autoencoder.model import Model
from config_RAE import config

# Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 128
time_step = 8
data_size = 1

display_step = 1000

# Construct model
model = Model(config)
model.train(learning_rate)

# Visualize Graph
writer = tf.summary.FileWriter("Log/RAE")
writer.add_graph(tf.get_default_graph())

# Start training
with tf.Session() as sess:
    # Initialize
    sess.run(model.init)

    # get model input
    graph = tf.get_default_graph()
    model_input = graph.get_tensor_by_name("inputs/input:0")

    # Prepare Data
    d = np.linspace(0, time_step, time_step, endpoint=False).reshape(
        [1, time_step, data_size]
    )
    d = np.tile(d, (batch_size, 1, 1))

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        r = np.random.randint(20, size=batch_size).reshape([batch_size, 1, 1])
        r = np.tile(r, (1, time_step, data_size))
        random_sequences = r + d

        # Run Optimization
        _, l = sess.run(
            [model.optimizer, model.loss], feed_dict={model_input: random_sequences}
        )
        # Display loss
        if i % display_step == 0 or i == 1:
            print("Step %i, Loss: %f" % (i, l))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "./Model/RAE/test_model", global_step=num_steps)


