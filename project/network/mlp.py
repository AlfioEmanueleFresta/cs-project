import lasagne
import theano.tensor as T
import theano
import numpy as np
import os
import time
import readline


from .generic import GenericNetwork


class MLPNetwork(GenericNetwork):

    def defaults(self):
        defaults = super(MLPNetwork, self).defaults()
        defaults.update({
            # Dense Layers
            'dense_layers': 4,
            'dense_layers_activation': lasagne.nonlinearities.sigmoid,
            'dense_layers_w': lasagne.init.GlorotUniform,
            'dense_layers_size': 30,
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
            l_dense = lasagne.layers.DropoutLayer(l_in, p=self.dense_layers_dropout)

        l_out = lasagne.layers.DenseLayer(l_dense, num_units=self.output_categories_no,
                                          nonlinearity=self.output_layer_activation,
                                          W=self.dense_layers_w())

        l_out = lasagne.layers.ReshapeLayer(l_out, shape=(-1, self.output_categories_no))
        self.network = l_out

    def train(self,
              *args, **kwargs
              ):

        self.verbose and print("Compiling functions...")
        prediction = lasagne.layers.get_output(self.network)
        loss = self.train_objective(prediction, self.target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params)

        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = self.train_objective(test_prediction, self.target_var)
        test_loss = test_loss.mean()

        train_fn = theano.function([self.input_var, self.target_var],
                                   loss,
                                   updates=updates,
                                   allow_input_downcast=self.allow_input_downcast)

        val_fn = theano.function([self.input_var, self.target_var],
                                 [prediction, test_loss],
                                 allow_input_downcast=self.allow_input_downcast)

        questions_and_answers = self.get_prepared_training_data()
        self.verbose and print("Starting training...")

        # TODO max_epochs should be used as upper bound -- intelligent early termination.
        for epoch in range(self.train_max_epochs):

            train, val, _ = self._get_split_data(questions_and_answers)

            train_err, train_batches = 0, 0
            start_time = time.time()

            for batch in self._iterate_minibatches(train):
                inputs, targets = batch
                err = train_fn(inputs, targets)
                train_err += err
                train_batches += 1

            val_err, val_batches = 0, 0
            #ex_in, ex_ta, ex_pr = [], [], []
            for batch in self._iterate_minibatches(val):
                inputs, targets = batch
                pred, err = val_fn(inputs, targets)
                val_err += err
                val_batches += 1

            print("Epoch %d/%d" % (epoch + 1, self.train_max_epochs), end=" ")
            print("took %f seconds." % (time.time() - start_time))
            print("    Training loss....: %f" % (train_err / train_batches))
            print("    Validation loss..: %f" % (val_err / val_batches))

        #for a, b, c in zip(ex_in, ex_ta, ex_pr):
            #print(" -- Input.....: %s" % a)
            #print("  - Target....: %s" % b)
            #print("  - Prediction: %s" % c)

