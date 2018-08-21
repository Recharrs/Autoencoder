import tensorflow as tf

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)

# Model Architecture Configure
config = {
    "model": [
        {
            "name": "inputs",
            "layers": [{"type": "input", "name": "input", "shape": [None, num_input]}],
        },
        {
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
        },
        {
            "name": "decoder",
            "layers": [
                {"type": "block_input", "name": "decoder_input", "input": "enc_2"},
                {
                    "type": "FC",
                    "name": "dec_1",
                    "input": "decoder_input",
                    "output_size": num_hidden_1,
                    "activation": tf.nn.sigmoid,
                },
                {
                    "type": "FC",
                    "name": "dec_2",
                    "input": "dec_1",
                    "output_size": num_input,
                    "activation": tf.nn.sigmoid,
                },
            ],
        },
        {
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
