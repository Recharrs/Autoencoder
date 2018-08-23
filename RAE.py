import sys, os
import csv
import numpy as np
import tensorflow as tf
import pickle

from autoencoder.model import Model

# Config
# ===========================================================
# Network Parameters
batch_size = 32
state_size = 6
max_time_step = 100
num_hidden = 64

num_steps = 50000
display_step = 10

# Location
data_train = "Trajectory/data/data_train.pkl"
data_valid = "Trajectory/data/data_train.pkl"
data_test = "Trajectory/data/data_test.pkl"

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
                    "name": "observation", # input name
                    "shape": [batch_size, max_time_step, state_size], # input shape
                }, 
                {
                    "type": "input",
                    "name": "mask", # input name
                    "shape": [batch_size, max_time_step], # input shape
                }
            ]
        },
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "RNN",
                    "name": "encoder",
                    "input": "observation",
                    "output_size": num_hidden,  # i.e. hidden state size
                    "activation": None,
                    "sequence_len": max_time_step,  # recurrent len
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
            "ground_truth": "observation",
            "prediction": "output",
            "mask": "mask",
        }
    ],
    "lr": 10e-4, 
}

# Model Mode
mode = ""

# Utils
# ===========================================================
def import_data(data_dir):
    data = []
    for env_folder in os.listdir(data_dir):
        if "DS_Store" not in env_folder and "data" not in env_folder and "RollerBall_2" not in env_folder:
            for iter_folder in os.listdir(os.path.join(data_dir, env_folder)):
                if "DS_Store" not in iter_folder:
                    for filename in os.listdir(os.path.join(data_dir, env_folder, iter_folder)):
                        with open(os.path.join(data_dir, env_folder, iter_folder, filename), newline='') as csvfile:
                            single_data = []
                            rows = csv.reader(csvfile, delimiter=" ")
                            for step, row in enumerate(rows):
                                if step != 0:
                                    single_data.append([float(n) for n in row])
                            data.append(np.array(single_data))
    return data

def spilt_data(data):
    n_train = int(len(data) * 0.6)
    n_valid = int(len(data) * 0.2)
    n_test = int(len(data) * 0.2)

    data_train = data[:n_train]
    data_valid = data[n_train:n_train+n_valid]
    data_test = data[n_train+n_valid:]

    with open("./Trajectory/data/data_train.pkl", "wb") as output:
        pickle.dump(data_train, output)
    with open("./Trajectory/data/data_valid.pkl", "wb") as output:
        pickle.dump(data_valid, output)
    with open("./Trajectory/data/data_test.pkl", "wb") as output:
        pickle.dump(data_test, output)

def padding_data(data):
    output, mask = [], []
    for d in data:
        single_output, single_mask = [], []
        for step in range(max_time_step):
            if step > len(d) - 1:
                single_output.append([0.0 for _ in range(state_size)])
                single_mask.append(0)
            else:
                single_output.append(d[step])
                single_mask.append(1)
        output.append(np.array(single_output))
        mask.append(single_mask)
    return output, mask

def get_batch(data, idx):
    num_of_batch = len(data) // batch_size
    batch_id = idx % (num_of_batch)
    
    batch_data = data[batch_id*batch_size:(batch_id+1)*batch_size]
    batch_ob, batch_mask = padding_data(batch_data)
    return batch_ob, batch_mask 

# Training
# ===========================================================
if mode == "train" or True:
    # data = import_data("./Trajectory")
    # spilt_data(data)
    # sys.exit(0)

    # training data
    file = open(data_train, "rb")
    data_train = pickle.load(file)
    
    # Construct model
    model = Model(config)

    # Visualize Graph
    writer = tf.summary.FileWriter(log_dir)
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Initialize
        sess.run(model.init)

        # model input
        graph = tf.get_default_graph()
        observation = graph.get_tensor_by_name("inputs/observation:0")
        mask = graph.get_tensor_by_name("inputs/mask:0")

        # Training
        for i in range(1, num_steps + 1):
            # GET DATA
            data_ob, data_mask = get_batch(data_train, i)

            # Run Optimization
            _, loss = sess.run([model.optimizer, model.loss], feed_dict={
                observation: data_ob,
                mask: data_mask
            })

            # Show Loss
            if i % display_step == 0 or i == 1:
                # Display loss
                print("Step %i, Loss: %f" % (i, loss))
                # Save Model
                save_path = os.path.join(model_dir, "model-%02d" % (i // display_step))
                saver.save(sess, save_path, global_step=num_steps)                

# Validation
# ===========================================================
if mode == "valid" and True:
    num_hidden = [16, 32, 64, 128, 256]
    config["lr"] = [10e-4, 5*10e-4, 10e-3, 5*10e-3, 10e-2]

    with tf.Session() as sess:
        pass

# Testing
# ===========================================================
if mode == "test" and True:
    with tf.Session() as sess:
        # restore model
        saver = tf.train.import_meta_graph(model_dir + "/model-10-10000.meta")
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        graph = tf.get_default_graph()
        model_input = graph.get_tensor_by_name("inputs/input:0")
        model_output = graph.get_tensor_by_name("outputs/output:0")

        # Prepare Data
        d = np.linspace(0, max_time_step, max_time_step, endpoint=False).reshape(
            [1, max_time_step, state_size]
        )
        d = np.tile(d, (batch_size, 1, 1))
        r = np.random.randint(20, size=batch_size).reshape([batch_size, 1, 1])
        r = np.tile(r, (1, max_time_step, state_size))
        random_sequences = r + d

        # Testing
        (input_, output_) = sess.run([model_input, model_output], {model_input: random_sequences})
        print('train result:')
        print('input: ', input_[0, :, :].flatten())
        print('output: ', output_[0, :, :].flatten())
