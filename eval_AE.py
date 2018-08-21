import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

with tf.Session() as sess:
    # restore model
    saver = tf.train.import_meta_graph("./Model/AE/test_model-30000.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./Model/AE"))

    graph = tf.get_default_graph()
    model_input = graph.get_tensor_by_name("inputs/input:0")
    model_output = graph.get_tensor_by_name("outputs/output:0")

    # Testing
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(model_output, feed_dict={model_input: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = batch_x[
                j
            ].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = g[j].reshape(
                [28, 28]
            )

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

