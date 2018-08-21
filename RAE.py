import sys, os
import numpy as np
import tensorflow as tf

from autoencoder.model import Model


# Utils
# ===========================================================



# Config
# ===========================================================
# Network Parameters
batch_size = 128
data_size = 1
time_step = 8
num_hidden = 12

num_steps = 10000
display_step = 1000

# Location
data_dir = "/tmp/data/"
log_dir = "Log/RAE"
model_dir = "Model/RAE/"
test_model_dir = "Model/RAE/test_model-30000.meta"

# Configure
config = {
    "model":[
        {
            "name": "inputs",
            "layers": [
                {
                    "type": "input",
                    "name": "input",  # input name
                    "shape": [batch_size, time_step, data_size],  # input shape
                }
            ]
        },
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "RNN",
                    "name": "encoder",
                    "input": "input",
                    "output_size": num_hidden,  # i.e. hidden state size
                    "activation": None,
                    "sequence_len": time_step,  # recurrent len
                    "init_state": None,  # init state
                    "input_mode": "INPUT_MODE",  # input, zeros, output
                }
            ]
        },
        {
            "name": "decoder",
            "layers": [
                {
                    "type": "block_input",
                    "name": "decoder_input",
                    "input": "encoder/state"
                },
                {
                    "type": "RNN",
                    "name": "decoder",
                    "input": "encoder/input_size",     # specify data size
                    "output_size": num_hidden,
                    "activation": None,
                    "sequence_len": "encoder/sequence_len", # as encoder
                    "init_state": "decoder_input",
                    "input_mode": "OUTPUT_MODE",
                    "fc_activation": None,
                }
            ]
        },
        {
            "name": "outputs",
            "layers": [
                {
                    "type": "output",
                    "name": "output", 
                    "input": "decoder/outputs"
                }
            ]
        }
    ],
    "loss": [
        {
            "name": "enc_dec_loss",
            "weight": 1,
            "ground_truth": "input",
            "prediction": "output" 
        }
    ],
    "lr": 10e-3, 
}

mode = sys.argv[1]


# Training
# ===========================================================
if mode == "train":
    # Construct model
    model = Model(config)

    # make save dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Visualize Graph
    writer = tf.summary.FileWriter(log_dir)
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Initialize
        sess.run(model.init)

        # get model input
        model_input = tf.get_default_graph().get_tensor_by_name("inputs/input:0")

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
            _, loss = sess.run([model.optimizer, model.loss], feed_dict={model_input: random_sequences})
            
            if i % display_step == 0 or i == 1:
                # Display loss
                print("Step %i, Loss: %f" % (i, loss))
                # Save Model
                save_path = os.path.join(model_dir, "model-%02d" % (i // display_step))
                saver.save(sess, save_path, global_step=num_steps)                


# Testing
# ===========================================================
if mode == "test":
    with tf.Session() as sess:
        # restore model
        saver = tf.train.import_meta_graph(model_dir + "/model-10-10000.meta")
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

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
