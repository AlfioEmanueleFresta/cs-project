import lasagne
import theano.tensor as T
import theano
import time
from project.laplotter import LossAccPlotter
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
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),
                               T.argmax(self.target_var, axis=1)),
                          dtype=theano.config.floatX)

        train_fn = theano.function([self.input_var, self.mask_var, self.target_var],
                                   [loss, test_acc],
                                   updates=updates,
                                   allow_input_downcast=self.allow_input_downcast)

        val_fn = theano.function([self.input_var, self.mask_var, self.target_var],
                                 [prediction, test_loss, test_acc],
                                 allow_input_downcast=self.allow_input_downcast)

        pred_fn = theano.function([self.input_var, self.mask_var],
                                  [prediction],
                                  allow_input_downcast=self.allow_input_downcast)

        self.prediction_function = pred_fn

        questions_and_answers = self.get_prepared_training_data()
        self.verbose and print("Starting training...")

        train, val, _ = self._get_split_data(questions_and_answers)

        print("Epoch      Time        Tr. loss   Val. loss  Val. acc.   B  Best acc. ")
        print("---------  ----------  ---------  ---------  ----------  -  ----------")

        plotter = LossAccPlotter("Training loss and training accuracy",
                                 show_plot_window=True,
                                 show_regressions=False)

        best_loss, best_acc = None, 0

        # TODO max_epochs should be used as upper bound -- intelligent early termination.
        for epoch in range(self.train_max_epochs):

            train_err, train_acc, train_batches = 0, 0, 0
            start_time = time.time()

            for batch in self._iterate_minibatches(train):
                inputs, mask, targets = batch
                err,  acc = train_fn(inputs, mask, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            val_err, val_batches, val_acc = 0, 0, 0
            #ex_in, ex_ta, ex_pr = [], [], []
            for batch in self._iterate_minibatches(val):
                inputs, mask, targets = batch
                pred, err, acc = val_fn(inputs, mask, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            train_loss = train_err / train_batches
            train_acc = train_acc / train_batches
            val_loss = val_err / val_batches
            val_acc = val_acc / val_batches * 100

            is_best_loss = False
            if best_loss is None or val_loss < best_loss:
                is_best_loss = True
                best_loss = val_loss
                best_acc = val_acc
                self.mark_best()

            print("%4d/%4d  %9.6fs  %9.6f  "
                  "%9.6f  %9.5f%%  %s  %9.5f%%" %
                  (epoch + 1, self.train_max_epochs,
                   time.time() - start_time,
                   train_loss, val_loss, val_acc,
                   "*" if is_best_loss else " ", best_acc))

            # Plot!
            plotter.add_values(epoch + 1,
                               loss_train=train_loss,
                               acc_train=train_acc,
                               loss_val=val_loss,
                               acc_val=val_acc)

