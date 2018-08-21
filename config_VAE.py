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
