import lasagne
import theano.tensor as T
import theano
import numpy as np
import os
import time
from .generic import GenericNetwork


class LSTMNetwork(GenericNetwork):

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
    TRAIN_BATCH_SIZE = 5

    # General options
    #########################################################

    ALLOW_INPUT_DOWNCAST = True
    VERBOSE = True

    def __init__(self,
                 word_vector_size,
                 lstm_layers=LSTM_LAYERS,
                 lstm_layer_size=LSTM_LAYER_SIZE,
                 lstm_learn_init=LSTM_LEARN_INIT,
                 lstm_clipping=LSTM_CLIPPING,
                 dense_layers=DENSE_LAYERS,
                 dense_layers_activation=DENSE_LAYERS_ACTIVATION,
                 dense_layers_w=DENSE_LAYERS_W,
                 *args,
                 **kwargs):

        self.features_no = word_vector_size
        self.input_var = T.tensor3('inputs')
        self.mask_var = T.matrix('mask')
        self.target_var = T.tensor3('targets')

        # Building the neural network
        #########################################################

        # TODO Implement number of LSTM and Dense Layers options -- these are currently ignored.

        # Input and mask layer
        l_in = lasagne.layers.InputLayer(shape=(None, None, self.features_no), input_var=self.input_var)
        l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=self.mask_var)

        # LSTM Layer
        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_in, lstm_layer_size, mask_input=l_mask,
                                                    learn_init=lstm_learn_init, grad_clipping=lstm_clipping)
        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(l_in, lstm_layer_size, mask_input=l_mask,
                                                         learn_init=lstm_learn_init, grad_clipping=lstm_clipping,
                                                         backwards=True)

        l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

        n_batch, n_sequence_length, n_features = l_in.input_var.shape
        l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, lstm_layer_size))

        l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=self.features_no,
                                            nonlinearity=dense_layers_activation,
                                            W=dense_layers_w())

        l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_sequence_length, word_vector_size))
        self.network = l_out

        super(LSTMNetwork, self).__init__(*args, **kwargs)

    def train(self,
              training_data_filename,
              objective=TRAIN_OBJECTIVE,
              max_epochs=TRAIN_MAX_EPOCHS,
              batch_size=TRAIN_BATCH_SIZE,
              updates=TRAIN_UPDATES,
              updates_learning_rate=TRAIN_UPDATES_LEARNING_RATE,
              updates_momentum=TRAIN_UPDATES_MOMENTUM,
              allow_input_downcast=ALLOW_INPUT_DOWNCAST,
              *args, **kwargs
              ):

        self.verbose and print("Compiling functions...")
        prediction = lasagne.layers.get_output(self.network)
        loss = objective(prediction, self.target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params)

        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = objective(test_prediction, self.target_var)
        test_loss = test_loss.mean()

        train_fn = theano.function([self.input_var, self.target_var, self.mask_var],
                                   loss,
                                   updates=updates,
                                   allow_input_downcast=allow_input_downcast)

        val_fn = theano.function([self.input_var, self.target_var, self.mask_var],
                                 test_loss,
                                 allow_input_downcast=allow_input_downcast)

        self.verbose and print("Starting training...")

        def _iterate_batches(tuples, batch_size):
            while True:
                batch = []
                input, target, mask =


        # TODO max_epochs should be used as upper bound -- intelligent early termination.
        for epoch in range(max_epochs):

            train_err, train_batches = 0, 0
            start_time = time.time()

            for batch in _iterate_batches(train, dataset_no, batch_size):
                inputs, targets, mask = batch
                train_err += train_fn(inputs, targets, mask)
                train_batches += 1

            val_err, val_batches = 0, 0
            for batch in _iterate_batches(val, dataset_no, batch_size):
                inputs, targets, mask = batch
                err = val_fn(inputs, targets, mask)
                val_err += err
                val_batches += 1

            print("Epoch %d/%d" % (epoch + 1, max_epochs), end=" ")
            print("took %f seconds." % (time.time() - start_time))
            print("    Training loss....: %f" % (train_err / train_batches))
            print("    Validation loss..: %f" % (val_err / val_batches))

        super(LSTMNetwork, self).train(*args, **kwargs)
