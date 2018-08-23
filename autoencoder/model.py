import tensorflow as tf
import numpy as np

# tf utils
# ========================================
def Variable(initial_value, name=None):
    return tf.Variable(
        initial_value=initial_value,
        name=name
    )

def FC(x, weight, bias, activation=None):
    if activation is None:
        return tf.matmul(x, weight) + bias # linear
    else:
        return activation(tf.matmul(x, weight) + bias)

def Cell(cell_type, num_units, activation=None, name=None):
    if cell_type == "RNN":
        return tf.contrib.rnn.BasicRNNCell(num_units, activation=activation, name=name)
    if cell_type == "LSTM":
        return tf.contrib.rnn.LSTMCell(num_units, activation=activation, name=name)
    #if "cell" not in layer_config or layer_config["cell"] is None:
    #    raise NotImplementedError("Configuration")

def GetInput(layer_config):
    input_size = None
    # input size
    if isinstance(layer_config["input"], int):
        input_size = layer_config["input"] # specified fixed length
    elif isinstance(layer_config["input"], str):
        input_size = self.getNode(layer_config["input"]) # other layers parameter
    return input_size

def GetTimeStep(layer_config):
    if "sequence_len" in layer_config and layer_config["sequence_len"] is not None:
        if isinstance(layer_config["sequence_len"], int):                                
            time_step = layer_config["sequence_len"] # specified fixed length
        elif isinstance(layer_config["sequence_len"], str):                                
            time_step = self.getNode(layer_config["sequence_len"]) # other layers paramete
    return time_step

# Build Model
class Model:
    def __init__(self, config):
        # Settings
        # --------------------
        # random initializer
        if "random_init" in config and config["random_init"] is not None:
            # custom random initializer
            self.random_init = config["random_init"]
        else:
            # default: normal distribution
            self.random_init = tf.random_normal

        # Construct model
        # --------------------
        # nodes: every node or layer need register
        self.nodes = {}
        # create blocks
        self.createBlocks(config["model"])

        # Loss Function & Optimizer
        # --------------------
        self.createLoss(config)
        self.createOpt(config)
        
        # Init Model
        # --------------------
        self.init = tf.global_variables_initializer()

    def createLoss(self, config):
        self.loss = 0
        with tf.name_scope("loss"):
            for loss_config in config["loss"]:
                with tf.name_scope(loss_config["name"]):
                    # ground truth & prediction
                    ground_truth = self.getNode(loss_config["ground_truth"])
                    prediction = self.getNode(loss_config["prediction"])
                    mask = self.getNode(loss_config["mask"])

                    # weight
                    loss_weight = loss_config["weight"]
                    if ("loss_func" not in loss_config or loss_config["loss_func"] is None):
                        # default loss: mean square error
                        loss = loss_weight * tf.reduce_mean(tf.multiply(tf.reduce_mean((tf.pow(ground_truth - prediction, 2)), axis=-1), mask))
                    else:
                        # custom loss
                        loss = loss_weight * loss_config["loss_func"](ground_truth, prediction)
                # add loss
                self.loss += loss
            # total loss (mean up batch dim)
            self.loss = tf.reduce_mean(self.loss)

    def createOpt(self, config):
        # optimizer
        self.optimizer = tf.train.RMSPropOptimizer(config["lr"]).minimize(self.loss)

    def createBlocks(self, config):
        for block_config in config:
            # create block scope
            with tf.name_scope(block_config["name"]):
                # layers
                if "layers" in block_config and block_config["layers"] is not None:
                    # create layers
                    self.createLayers(block_config["layers"])
                # blocks
                if "blocks" in block_config and block_config["blocks"] is not None:
                    # create blocks
                    self.createBlocks(block_config["blocks"])

    def createLayers(self, config):
        for layer_config in config:
            # Config Parameters
            # ----------------------------------------
            layer_type = layer_config["type"]
            layer_name = layer_config["name"]

            # Create Layer
            # ----------------------------------------
            with tf.name_scope(layer_name):
                # Fully Connected
                if layer_type == "FC":  
                    input_layer = self.getNode(layer_config["input"])

                    # input: shape should be [batch_size, data_size]
                    (batch_size, input_size) = input_layer.get_shape()

                    # weight & bias
                    weight = Variable(self.random_init([input_size, layer_config["output_size"]]), name="weight")
                    bias = Variable(self.random_init([layer_config["output_size"]]), name="bias")

                    # build layer
                    self.nodes[layer_name] = FC(weight, bias, activation=layer_config["activation"])

                # Vanilla RNN
                elif layer_type == "RNN":
                    # parameters
                    output_size = layer_config["output_size"]

                    # cell type:  "RNN" = layer_config["cell"]
                    cell = Cell("RNN", output_size, activation=layer_config["activation"], name="RNN_Cell")

                    # inputs & initial state
                    if "input_mode" not in layer_config or layer_config["input_mode"] is None:
                        # TODO: zero input w/ shape [batch_size, time_step, data_size]
                        pass

                    elif layer_config["input_mode"] == "INPUT_MODE":
                        # Input: /w shape [batch_size, time_step, input_size]
                        input_layer = self.getNode(layer_config["input"])
                        (batch_size, time_step, input_size) = input_layer.get_shape()
                 
                        # initial state
                        _state = cell.zero_state(batch_size)
                        _outputs = []
                        
                        # time major
                        input_layer = tf.transpose(input_layer, [1, 0, 2])
                        
                        # recurrent
                        for step in range(time_step):
                            _output, _state = cell(input_layer[step], _state)
                            _outputs.append(_output)
                     
                    elif layer_config["input_mode"] == "OUTPUT_MODE":
                        # Preveious Input: /w shape [batch_size, input_size]
                        input_layer = self.getNode(layer_config["init_state"])
                        (batch_size, input_size) = input_layer.get_shape()
                                                
                        # initial state
                        _state = cell.zero_state(batch_size)
                        _output = tf.zeros([batch_size, input_size])
                        _outputs = []

                        # create FC for convert output from [batch_size, output_size] to [batch_size, input_size]
                        fc_weight = Variable(self.random_init([output_size, input_size]), name="weight")
                        fc_bias = Variable(self.random_init([input_size]), name="bias")

                        # recurrent
                        for _ in range(time_step):
                            _output, _state = cell(_output, _state)
                            _output = fc(_output,fc_weight, fc_bias, activation=layer_config["fc_activation"])
                            _outputs.append(_output)

                    # stack outputs: /w shape = [batch_size, time_step, data_size]
                    _outputs = tf.stack(_outputs, axis=1)

                    # register node
                    self.nodes[layer_name] = {
                        "outputs": _outputs,
                        "state": _state,
                        "sequence_len": time_step,
                        "input_size": input_size
                    }

                # LSTM
                elif layer_type == "LSTM":
                    raise NotImplementedError("LSTM: not implemented")

                # Sampler for variational autoencoder
                elif layer_type == "sampler":
                    '''
                        with tf.name_scope(layer_name):
                            # TODO: only standard deviation for each dimension
                            #       might need to add convariance?
                            # TODO: activation function?

                            # input size
                            # input data shape should be [batch, data_size]
                            # TODO: reshape
                            input_layer = self.nodes[layer_config["input"]]
                            input_size = input_layer.get_shape().as_list()[1]
                            # mean
                            with tf.name_scope("mean"):
                                # weight & bias
                                z_mean_w = tf.Variable(
                                    self.random_init([input_size, layer_config["output_size"]]),
                                    name="weight",
                                )
                                z_mean_b = tf.Variable(
                                    self.random_init([layer_config["output_size"]]), name="bias"
                                )
                                # build vector
                                z_mean = tf.matmul(input_layer, z_mean_w) + z_mean_b
                            # standard deviation (actually is 4*log(std)?)
                            with tf.name_scope("standard_deviation"):
                                # weight & bias
                                z_std_w = tf.Variable(
                                    self.random_init([input_size, layer_config["output_size"]]),
                                    name="weight",
                                )
                                z_std_b = tf.Variable(
                                    self.random_init([layer_config["output_size"]]), name="bias"
                                )
                                # build vector
                                z_std = tf.matmul(input_layer, z_std_w) + z_std_b
                            # epsilon
                            epsilon = tf.random_normal(
                                [layer_config["output_size"]],
                                dtype=tf.float32,
                                mean=0.,
                                stddev=1.0,
                                name="epsilon",
                            )
                            # reparameterize trick
                            # z = mean + var*eps
                            z = z_mean + tf.exp(z_std / 2) * epsilon
                            self.nodes[layer_name] = z

                            # loss
                            with tf.name_scope("KL_divergence_loss"):
                                loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
                                # TODO: setting beta (currentlt 0.5)
                                self.loss += -0.5 * tf.reduce_sum(loss, 1)
                    '''
                    raise NotImplementedError("VAE: not implemented")

                # Input
                elif layer_type == "input":
                    # create placeholder
                    self.nodes[layer_name] = tf.placeholder(
                        tf.float32, layer_config["shape"], name=layer_name
                    )

                # Output
                elif layer_type == "output":
                    # input layer
                    input_layer = self.getNode(layer_config["input"])
                    # create identity
                    self.nodes[layer_name] = tf.identity(input_layer, name=layer_name)

                # Block Input
                elif layer_type == "block_input":
                    # input layer
                    input_layer = self.getNode(layer_config["input"])
                    # create placeholder with default input
                    self.nodes[layer_name] = tf.placeholder_with_default(
                        input_layer, input_layer.get_shape(), name=layer_name
                    )

    def getNode(self, path):
        # split path
        path = path.split('/')
        # find the node with the path
        node = self.nodes
        for config in path:
            node = node[config]
        return node