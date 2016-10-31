import lasagne
import theano.tensor as T


from .generic import GenericNetwork


class MLPNetwork(GenericNetwork):

    def defaults(self):
        defaults = super(MLPNetwork, self).defaults()
        defaults.update({
            # Dense Layers
            'dense_layers': 2,
            'dense_layers_activation': lasagne.nonlinearities.sigmoid,
            'dense_layers_w': lasagne.init.GlorotUniform,
            'dense_layers_size': 500,
            'dense_layers_dropout': 0.3,

            'output_layer_activation': lasagne.nonlinearities.softmax,

            # Training options
            'train_objective': lasagne.objectives.categorical_crossentropy,
            'train_max_epochs': 500,
            'train_updates': lasagne.updates.nesterov_momentum,
            'train_updates_learning_rate': 0.01,
            'train_updates_momentum': 0.9,

            # General configuration
            'allow_input_downcast': True,
            'verbose': True,
        })
        return defaults

    def __init__(self,
                 max_words_per_sentence,
                 **kwargs):

        kwargs.update({'max_words_per_sentence': max_words_per_sentence})

        self.input_var = T.tensor3('inputs')
        self.target_var = T.matrix('targets')

        super(MLPNetwork, self).__init__(**kwargs)

    def build_network(self):
        # Building the neural network
        #########################################################

        # TODO Implement number of LSTM and Dense Layers options -- these are currently ignored.

        # Input and mask layer
        input_shape = (None, self.max_words_per_sentence, self.input_features_no)
        l_in = lasagne.layers.InputLayer(shape=input_shape, input_var=self.input_var)
        l_dense = l_in

        for i in range(self.dense_layers):
            l_dense = lasagne.layers.DenseLayer(l_dense, num_units=self.dense_layers_size,
                                                nonlinearity=self.dense_layers_activation,
                                                W=self.dense_layers_w())
            l_dense = lasagne.layers.DropoutLayer(l_dense, p=self.dense_layers_dropout)

        l_out = lasagne.layers.DenseLayer(l_dense, num_units=self.output_categories_no,
                                          nonlinearity=self.output_layer_activation,
                                          W=self.dense_layers_w())

        #l_out = lasagne.layers.ReshapeLayer(l_out, shape=(-1, self.output_categories_no))
        self.network = l_out

