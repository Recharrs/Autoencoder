import tensorflow as tf

# Network Parameters
batch_size = 128
data_size = 1
time_step = 8
num_hidden = 12

config = {
    "model": [
        {
            "name": "inputs",
            "layers": [
                {
                    "type": "input",
                    "name": "input",  # input name
                    "shape": [batch_size, time_step, data_size],  # input shape
                }
            ],
        },
        {
            "name": "encoder",
            "layers": [
                {
                    "type": "RNN",
                    "name": "encoder",
                    "input": "input",
                    "input_mode": "INPUT_MODE",  # input, zeros, output
                    "init_state": None,  # init state
                    "cell": tf.contrib.rnn.BasicRNNCell,
                    "output_size": num_hidden,  # i.e. hidden state size
                    "activation": None,
                    "sequence_len": time_step,  # recurrent len
                }
            ],
        },
        {
            "name": "decoder",
            "layers": [
                {
                    "type": "block_input",
                    "name": "decoder_input",
                    "input": "encoder/state",
                },
                {
                    "type": "RNN",
                    "name": "decoder",
                    "input": "encoder/input_size",  # specify data size
                    "input_mode": "OUTPUT_MODE",
                    "init_state": "decoder_input",
                    "cell": tf.contrib.rnn.BasicRNNCell,
                    "output_size": num_hidden,
                    "activation": None,
                    "sequence_len": "encoder/sequence_len",  # as encoder
                    "fc_activation": None,
                },
            ],
        },
        {
            "name": "outputs",
            "layers": [
                {"type": "output", "name": "output", "input": "decoder/outputs"}
            ],
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
