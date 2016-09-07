import lasagne


class LSTMNetwork:

    # Network Parameters
    #########################################################

    # LSTM Layers
    LSTM_LAYERS = 2
    LSTM_LAYER_SIZE = 800
    LSTM_LEARN_INIT = True
    LSTM_CLIPPING = 100

    # Dense Layers
    DENSE_LAYERS = 2
    DENSE_LAYERS_ACTIVATION = lasagne.nonlinearities.sigmoid
    DENSE_LAYERS_W = lasagne.init.GlorotUniform

    # Training options
    #########################################################

    TRAIN_OBJECTIVE = lasagne.objectives.squared_error
    TRAIN_MAX_EPOCHS = 35
    TRAIN_UPDATES = lasagne.updates.nesterov_momentum
    TRAIN_UPDATES_LEARNING_RATE = 0.01
    TRAIN_UPDATES_MOMENTUM = 0.9
    TRAIN_BATCH_SIZE = 100

    # General options
    #########################################################

    ALLOW_INPUT_DOWNCAST = True

    def __init__(self,
                 word_vector_size,
                 lstm_layers=LSTM_LAYERS,
                 lstm_layer_size=LSTM_LAYER_SIZE,
                 lstm_learn_init=LSTM_LEARN_INIT,
                 lstm_clipping=LSTM_CLIPPING,
                 dense_layers=DENSE_LAYERS,
                 dense_layers_activation=DENSE_LAYERS_ACTIVATION,
                 dense_layers_w=DENSE_LAYERS_W):

        ## TODO
        input_var = None
        mask_var = None

        # Building the neural network
        #########################################################

        # Input and mask layer
        l_in = lasagne.layers.InputLayer(shape=(None, None, word_vector_size), input_var=input_var)
        l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

        # LSTM Layer
        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_in, lstm_layer_size, mask_input=l_mask,
                                                    learn_init=lstm_learn_init, grad_clipping=lstm_clipping)
        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(l_in, lstm_layer_size, mask_input=l_mask,
                                                         learn_init=lstm_learn_init, grad_clipping=lstm_clipping,
                                                         backwards=True)

        l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

        n_batch, n_sequence_length, n_features = l_in.input_var.shape
        l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, lstm_layer_size))

        l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=1,
                                            nonlinearity=dense_layers_activation,
                                            W=dense_layers_w())

        l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_sequence_length))

        self.network = l_out

    def train(self,
              objective=TRAIN_OBJECTIVE,
              max_epochs=TRAIN_MAX_EPOCHS,
              batch_size=TRAIN_BATCH_SIZE,
              updates=TRAIN_UPDATES,
              updates_learning_rate=TRAIN_UPDATES_LEARNING_RATE,
              updates_momentum=TRAIN_UPDATES_MOMENTUM
              ):
        # TODO max_epochs should be used as upper bound -- intelligent early termination.
        pass
