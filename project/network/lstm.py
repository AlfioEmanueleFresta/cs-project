import lasagne
import theano.tensor as T
from .generic import GenericNetwork


class LSTMNetwork(GenericNetwork):

    def defaults(self):
        defaults = super(LSTMNetwork, self).defaults()
        defaults.update({
            # LSTM Layers
            'lstm_layers': 2,
            'lstm_layers_size': 75,
            'lstm_layers_activation': lasagne.nonlinearities.sigmoid,
            'lstm_layers_precompute_input': True,
            'lstm_layers_learn_init': True,
            'lstm_layers_only_return_final': False,

            # Dense Layers
            'dense_layers': 1,
            'dense_layers_activation': lasagne.nonlinearities.sigmoid,
            'dense_layers_w': lasagne.init.GlorotUniform,
            'dense_layers_size': 125,
            'dense_layers_dropout': 0.3,

            'output_layer_activation': lasagne.nonlinearities.softmax,

            # Training options
            'train_objective': lasagne.objectives.categorical_crossentropy,
            'train_max_epochs': 1500,
            'train_updates': lasagne.updates.nesterov_momentum,
            'train_updates_learning_rate': 0.01,
            'train_updates_momentum': 0.9,

            # General configuration
            'allow_input_downcast': True,
            'verbose': True,
            'include_mask': True,
        })
        return defaults

    def __init__(self,
                 max_words_per_sentence,
                 **kwargs):

        kwargs.update({'max_words_per_sentence': max_words_per_sentence})

        self.input_var = T.tensor3('inputs')
        self.mask_var = T.matrix('masks')
        self.target_var = T.matrix('targets')

        super(LSTMNetwork, self).__init__(**kwargs)

    def build_network(self):
        # Building the neural network
        #########################################################

        # TODO Implement number of LSTM and Dense Layers options -- these are currently ignored.

        # Input and mask layer

        # TODO first dimension can be self.train_batch_size, but can't feed fewer data points.
        #      this means that the last mini batch, if not 'full', can't be fed into the network.
        input_shape = (None, self.max_words_per_sentence, self.input_features_no)
        mask_shape = (None, self.max_words_per_sentence)

        l_in = lasagne.layers.InputLayer(shape=input_shape, input_var=self.input_var)
        l_mask = lasagne.layers.InputLayer(shape=mask_shape, input_var=self.mask_var)

        l_forward = l_in
        for i in range(self.lstm_layers):
            l_forward = lasagne.layers.LSTMLayer(
                l_in, self.lstm_layers_size,
                mask_input=l_mask,
                nonlinearity=self.lstm_layers_activation,
                only_return_final=((self.lstm_layers-1) == i) or self.lstm_layers_only_return_final,
                precompute_input=self.lstm_layers_precompute_input,
                learn_init=self.lstm_layers_learn_init)

        l_dense = l_forward
        for _ in range(self.dense_layers):
            l_dense = lasagne.layers.DenseLayer(l_dense, num_units=self.dense_layers_size,
                                                nonlinearity=self.dense_layers_activation,
                                                W=self.dense_layers_w())
            l_dense = lasagne.layers.DropoutLayer(l_dense, p=self.dense_layers_dropout)

        l_out = lasagne.layers.DenseLayer(l_dense, num_units=self.output_categories_no,
                                          nonlinearity=self.output_layer_activation,
                                          W=self.dense_layers_w())

        #l_out = lasagne.layers.ReshapeLayer(l_out, shape=(-1, -1))
        self.network = l_out
