# Autoencoder - Lite Version
**This is lite version of AutoEncoder**
Try to build many kinds of autoencoder. Currently got vanilla autoencoder and variantional autoencoder only.
RAE, and VRAE will be added in the future.

## Example

Use the example from tensorflow example tutorial.

### AE

The training process will save model in `Model/AE/` folder,
make sure the folder exist otherwise the files will failed to saved.<br>
During the evaluation, pretrained model will be restored for further testing.<br>

```
# train
python train_AE.py
# test
python eval_AE.py
```

Also, the model graph will be save in `Log/AE/` folder before training. Use the following command to visualize the model graph.

```
# launch tensorboard
tensorboard --logdir=Log/AE
```

The model configure defined in `config_AE.py`.

### VAE

Similar to AE, model saved in `Model/VAE/`, graph in `Log/VAE/`, and the model configure defined in `config_VAE.py`.

```
# train
python train_VAE.py
# test
python eval_VAE.py
# launch tensorboard
tensorboard --logdir=Log/VAE
```

Different to the previous example restores whole model for evaluation, this example only restores the decoder for using.<br>

## Usage

### Model Config

Use config to specify model architecture.<br>
In the config, `"model"` and `"loss"` must be defined.<br>
There is also an optional setting `"random_init"` for setting how variables are initialized.<br>

#### model

`"model"` is a list of blocks, and each block has it own `"name"`, which will be the scope name of block.<br>

#### block

Blocks can be nested by define `"blocks"`, which is also a list of bloocks.<br>
Also, each block has `"layers"`, a list of layers, which would be FC, RNN, CNN, anrd etc. (Currently only FC are available.)<br>

#### layer

Each layer must have its own unique `"name"`, and what `"type"` it is.<br>
Different type of layers have different attributes to fill in. Check `model.py` or example config file for more information.

```python
config = {
    "random_init": custom_random_init
    "model": [
        # list of blocks
        {
            "name": "block name",
            "blocks": [
                # list of blocks
            ],
            "layers": [
                # list of layers
                {
                    "type": "FC",
                    "name": "layer_name",
                    "input": "some input layer",
                    "output_size": 100,
                    "activation": tf.nn.tanh
                },
                ...
            ]
        },
        ...
    ],
    "loss": [
        {
            "name": "loss name",
            "weight": 1,
            "ground_truth": "some label layer",
            "prediction": "some output layer",
            "loss_func": custom_loss_func
        }
    ],

}
```

#### loss

`"loss"` define the optimisation objective.<br>
In default, loss function is the MSE between `"ground_truth"` and `"prediction"`. If `"loss_func"` is defined, it will pass `"ground_truth"` and `"prediction"` to the custom loss function.<br>
`"weight"` is the loss weighting to trade with other loss.<br>
For variational autoencoder, loss of sampler, which is KL divergence, will be add to total loss automatically.

### Training and Testing

```python
from autoencoder.model import Model

# build model
model = Model(config)
# specity training parameters, not actually training
model.train(learning_rate)

# init tensorflow variables
sess.run(model.init)

# train
# need to find the tensor with its name in the graph
graph = tf.get_default_graph()
model_input = graph.get_tensor_by_name("input_node_name:0")
sess.run(model.optimizer, feed_dict={model_input: data})

# test full model
sess.run(model_output, feed_dict={model_input: data})
# test encoder
sess.run(model_encoder, feed_dict={model_input: data})
# test decoder
sess.run(model_output, feed_dict={model_decoder_input: data})
```

## Version

`tensorflow 1.7.1`
`tensorboard 1.7.0`

## Reference

[Tensorflow Example](https://github.com/aymericdamien/TensorFlow-Examples)
