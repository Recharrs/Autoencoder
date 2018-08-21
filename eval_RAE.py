import numpy as np
import tensorflow as tf

# Parameter
batch_size = 128
time_step = 8
data_size = 1

with tf.Session() as sess:
    # restore model
    saver = tf.train.import_meta_graph("./Model/RAE/test_model-10000.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./Model/RAE"))
    
    graph = tf.get_default_graph()
    model_input = graph.get_tensor_by_name("inputs/input:0")
    model_output = graph.get_tensor_by_name("outputs/output:0")
    
    # Prepare Data
    d = np.linspace(0, time_step, time_step, endpoint=False).reshape(
        [1, time_step, data_size]
    )
    d = np.tile(d, (batch_size, 1, 1))
    r = np.random.randint(20, size=batch_size).reshape([batch_size, 1, 1])
    r = np.tile(r, (1, time_step, data_size))
    random_sequences = r + d

    # Testing
    (input_, output_) = sess.run([model_input, model_output], {model_input: random_sequences})
    print('train result:')
    print('input: ', input_[0, :, :].flatten())
    print('output: ', output_[0, :, :].flatten())

