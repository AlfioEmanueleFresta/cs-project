import lasagne
import theano.tensor as T
import theano
import numpy as np
import os
import time
import readline


from project.helpers import get_split_data, load_questions_and_answers,\
                            get_all_answers, one_hot, one_hot_decode
from .generic import GenericNetwork


class MLPNetwork(GenericNetwork):

    # Network Parameters
    #########################################################

    OUTPUT_CATEGORIES_NO = 100

    # Dense Layers
    DENSE_LAYERS = 2
    DENSE_LAYERS_ACTIVATION = lasagne.nonlinearities.sigmoid
    DENSE_LAYERS_W = lasagne.init.GlorotUniform
    DENSE_LAYERS_SIZE = 1500
    DENSE_LAYERS_DROPOUT = 0.2

    # Training options
    #########################################################

    TRAIN_OBJECTIVE = lasagne.objectives.categorical_crossentropy
    TRAIN_MAX_EPOCHS = 200
    TRAIN_UPDATES = lasagne.updates.nesterov_momentum
    TRAIN_UPDATES_LEARNING_RATE = 0.01
    TRAIN_UPDATES_MOMENTUM = 0.9
    TRAIN_BATCH_SIZE = 2
    TRAIN_PERCENTAGE = 0.75

    # General options
    #########################################################

    ALLOW_INPUT_DOWNCAST = True
    VERBOSE = True

    def __init__(self,
                 max_words_per_sentence,
                 word_vector_size,
                 output_categories_no=OUTPUT_CATEGORIES_NO,
                 dense_layers=DENSE_LAYERS,
                 dense_layers_activation=DENSE_LAYERS_ACTIVATION,
                 dense_layers_w=DENSE_LAYERS_W,
                 dense_layers_size=DENSE_LAYERS_SIZE,
                 dense_layers_dropout=DENSE_LAYERS_DROPOUT,
                 *args,
                 **kwargs):

        self.max_words_per_sentence = max_words_per_sentence
        self.output_categories_no = output_categories_no
        self.features_no = word_vector_size
        self.input_var = T.tensor3('inputs')
        self.target_var = T.matrix('targets')

        # Building the neural network
        #########################################################

        # TODO Implement number of LSTM and Dense Layers options -- these are currently ignored.

        # Input and mask layer
        l_in = lasagne.layers.InputLayer(shape=(None, self.max_words_per_sentence, self.features_no), input_var=self.input_var)
        l_dense = l_in

        for i in range(dense_layers):
            l_dense = lasagne.layers.DenseLayer(l_dense, num_units=dense_layers_size,
                                                nonlinearity=dense_layers_activation,
                                                W=dense_layers_w())
            l_dense = lasagne.layers.DropoutLayer(l_in, p=dense_layers_dropout)

        l_out = lasagne.layers.DenseLayer(l_dense, num_units=self.output_categories_no,
                                          nonlinearity=dense_layers_activation,
                                          W=dense_layers_w())

        l_out = lasagne.layers.ReshapeLayer(l_out, shape=(-1, self.output_categories_no))
        self.network = l_out

        super(MLPNetwork, self).__init__(*args, **kwargs)

    def _get_answer_by_index(self, i):
        try:
            return self.data['answers'][i]

        except IndexError:
            return 'N/A i=%d, n=%d' % (i, len(self.data['answers']))

    def _get_answer_by_one_hot_vector(self, vector, limit_to=None):
        if limit_to:
            vector = vector[:limit_to]
        return self._get_answer_by_index(one_hot_decode(vector))

    def train(self,
              training_data_filename,
              glove=None,
              objective=TRAIN_OBJECTIVE,
              max_epochs=TRAIN_MAX_EPOCHS,
              batch_size=TRAIN_BATCH_SIZE,
              updates=TRAIN_UPDATES,
              updates_learning_rate=TRAIN_UPDATES_LEARNING_RATE,
              updates_momentum=TRAIN_UPDATES_MOMENTUM,
              allow_input_downcast=ALLOW_INPUT_DOWNCAST,
              train_percentage=TRAIN_PERCENTAGE,
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

        train_fn = theano.function([self.input_var, self.target_var],
                                   loss,
                                   updates=updates,
                                   allow_input_downcast=allow_input_downcast)

        val_fn = theano.function([self.input_var, self.target_var],
                                 [prediction, test_loss],
                                 allow_input_downcast=allow_input_downcast)

        self.verbose and print("Starting training...")

        questions_and_answers = load_questions_and_answers(training_data_filename)

        # Build the answers database
        self.data['answers'] = list(get_all_answers(questions_and_answers))

        if len(self.data['answers']) > self.output_categories_no:
            raise ValueError("There are too many answers in the training file. "
                             "Consider increasing 'output_categories_no'.")

        def _iterate_batches(tuples, batch_size):
            tuples = list(tuples)
            while True:
                this_batch_size = min(batch_size, len(tuples))
                if not this_batch_size:
                    break
                tuples_batch = tuples[0:this_batch_size]
                tuples = tuples[this_batch_size:]
                inputs = list([glove.get_sentence_matrix(x[0], max_words=self.max_words_per_sentence) for x in tuples_batch])
                targets = list([one_hot(n=self.output_categories_no, i=self.data['answers'].index(x[1])) for x in tuples_batch])
                assert len(inputs) == len(targets)
                yield inputs, targets

        train, val, test = get_split_data(questions_and_answers,
                                          train_percentage=train_percentage)

        # TODO max_epochs should be used as upper bound -- intelligent early termination.
        for epoch in range(max_epochs):

            train_err, train_batches = 0, 0
            start_time = time.time()

            for batch in _iterate_batches(train, batch_size):
                inputs, targets = batch
                err = train_fn(inputs, targets)
                train_err += err
                train_batches += 1

            val_err, val_batches = 0, 0
            ex_in, ex_ta, ex_pr = [], [], []
            for batch in _iterate_batches(val, batch_size):
                inputs, targets = batch
                pred, err = val_fn(inputs, targets)
                val_err += err
                val_batches += 1

                #ex_in += [' '.join(glove.matrix_to_words(x)) for x in inputs]
                #ex_ta += [self._get_answer_by_one_hot_vector(x) for x in targets]
                #ex_pr += [self._get_answer_by_one_hot_vector(x) for x in pred]

            print("Epoch %d/%d" % (epoch + 1, max_epochs), end=" ")
            print("took %f seconds." % (time.time() - start_time))
            print("    Training loss....: %f" % (train_err / train_batches))
            print("    Validation loss..: %f" % (val_err / val_batches))

        #for a, b, c in zip(ex_in, ex_ta, ex_pr):
            #print(" -- Input.....: %s" % a)
            #print("  - Target....: %s" % b)
            #print("  - Prediction: %s" % c)

        super(MLPNetwork, self).train(*args, **kwargs)

    def interactive_predict(self, glove):

        print("Compiling prediction function...")
        prediction = lasagne.layers.get_output(self.network)
        pred_fn = theano.function([self.input_var],
                                  [prediction],
                                  allow_input_downcast=True)

        print("Interactive shell. Type 'exit' to close.")

        while True:

            sentence = input(">> ").lower().strip()

            if sentence == 'exit':
                print("Bye!")
                break

            i = glove.get_sentence_matrix(sentence, max_words=self.max_words_per_sentence)
            i = np.array([i])

            o = pred_fn(i)
            o = o[0][0]
            print(o)

            o = self._get_answer_by_one_hot_vector(o, limit_to=self.output_categories_no)
            print(o)
