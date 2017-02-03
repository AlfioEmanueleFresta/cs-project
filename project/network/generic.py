import lasagne
import theano.tensor as T
import theano
import numpy as np
import time

from project.data import MemoryDataset
from project.helpers import one_hot_decode, CachedTupleGeneratorbatchIterator
from project.laplotter import LossAccPlotter
from project.vectorization.embedding import WordEmbedding


class GenericNetwork:

    def defaults(self):
        return {'verbose': True,
                'include_mask': False,
                'show_plot': True,

                # Training options
                'train_batch_size': 1,
                'train_data_test': True,
                'train_data_shuffle': True,
                'train_data_percentage': 0.7,
                'train_max_epochs': 1000,
                'train_max_epochs_without_improvement': 14,

                # Stop training if the accuracy diverges of more than X% after the first N epochs
                #'train_max_accuracy_diversion': 5,
                #'train_max_accuracy_diversion_min_epochs': 200,

                }

    def __init__(self, input_features_no, output_categories_no, **kwargs):
        kwargs.update({'input_features_no': input_features_no,
                       'output_categories_no': output_categories_no})

        defaults = self.defaults()
        defaults.update(kwargs)

        self.data = {'params': defaults}
        self.training_data = None
        self.training_function = None
        self.prediction_function = None
        self.validation_function = None
        self.glove = None
        self.best_parameters = None
        self.network = None

    def __getattr__(self, item):
        if item in self.data['params']:
            # print("REQ %s=%s" % (item, self.data['params'][item]))
            return self.data['params'][item]
        raise AttributeError

    def build_network(self):
        raise NotImplementedError

    def compile_functions(self):

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
        self.validation_function = val_fn
        self.training_function = train_fn

    def train(self,
              *args, **kwargs
              ):

        train_fn = self.training_function
        val_fn = self.validation_function
        pred_fn = self.prediction_function

        questions_and_answers = self.training_data.get_prepared_data(train_data_percentage=self.train_data_percentage,
                                                                     train_data_shuffle=self.train_data_shuffle,
                                                                     max_sentence_size=self.max_words_per_sentence)

        self.verbose and print("Generating training, validation and testing dataset...")

        train, val, test = questions_and_answers
        train, val, test = CachedTupleGeneratorbatchIterator(train, batch_size=self.train_batch_size), \
                           CachedTupleGeneratorbatchIterator(val, batch_size=self.train_batch_size), \
                           CachedTupleGeneratorbatchIterator(test, batch_size=self.train_batch_size)

        self.verbose and print("Starting training. You can use CTRL-C "
                               "to stop the training process.")

        self.verbose and print("Epoch      Time        Tr. loss   Val. loss  Val. acc.   B  Best acc. ")
        self.verbose and print("---------  ----------  ---------  ---------  ----------  -  ----------")

        best_loss, best_acc, best_epoch = None, 0, None

        try:

            if self.show_plot:
                plotter = LossAccPlotter("Training loss and training accuracy",
                                         show_plot_window=True,
                                         show_averages=False)

            # TODO max_epochs should be used as upper bound -- intelligent early termination.
            for epoch in range(self.train_max_epochs):

                train.reset()
                val.reset()
                test.reset()

                train_err, train_acc, train_batches = 0, 0, 0
                start_time = time.time()

                for batch in train:
                    inputs, mask, targets = batch
                    err,  acc = train_fn(inputs, mask, targets)
                    train_err += err
                    train_acc += acc
                    train_batches += 1

                val_err, val_batches, val_acc = 0, 0, 0
                for batch in val:
                    inputs, mask, targets = batch
                    pred, err, acc = val_fn(inputs, mask, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                train_loss = train_err / train_batches
                train_acc = train_acc / train_batches * 100
                val_loss = val_err / val_batches
                val_acc = val_acc / val_batches * 100

                is_best_loss = False

                if best_loss is None or val_loss < best_loss:
                    is_best_loss = True
                    best_loss = val_loss
                    best_acc = val_acc
                    best_epoch = epoch
                    self.mark_best()
                    distance_to_last_best = 0

                else:
                    distance_to_last_best += 1

                self.verbose and print("%4d/%4d  %9.6fs  %9.6f  "
                                       "%9.6f  %9.5f%%  %s  %9.5f%%" %
                                       (epoch + 1, self.train_max_epochs,
                                        time.time() - start_time,
                                        train_loss, val_loss, val_acc,
                                        "*" if is_best_loss else " ", best_acc))

                # Plot!
                if self.verbose and self.show_plot:
                    plotter.add_values(epoch + 1,
                                       loss_train=train_loss,
                                       acc_train=train_acc,
                                       loss_val=val_loss,
                                       acc_val=val_acc,
                                       redraw=not epoch % 5)

                if distance_to_last_best > self.train_max_epochs_without_improvement:
                    self.verbose and print("Early interruption. Reached %d epochs without "
                                           "any improvement." % distance_to_last_best)
                    raise KeyboardInterrupt

                #if epoch > self.train_max_accuracy_diversion_min_epochs and\
                #    abs(val_acc - train_acc) > self.train_max_accuracy_diversion:
                #    self.verbose and print("Early interruption. Diversion higher than %d percent "
                #                           "after initial %d epochs." % (
                #        self.train_max_accuracy_diversion, self.train_max_accuracy_diversion_min_epochs
                #    ))
                #    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("Training interrupted at epoch %d." % epoch)
            print("Best result (epoch=%d, loss=%9.6f, accuracy"
                  "=%9.5f%%)" % (best_epoch, best_loss, best_acc))

        self.epochs = epoch
        self._get_best()  # Get the best validation score.
        self._test(test)

    def _test(self, test_data):

        if not self.train_data_test:
            self.verbose and print("Skipping testing (train_data_set=False).")
            return

        self.verbose and print("Testing network...", end=" ", flush=True)

        test_data.reset()

        try:
            val_fn = self.validation_function
            test_err, test_acc, test_batches = 0, 0, 0
            for batch in test_data:
                inputs, mask, targets = batch
                _, err, acc = val_fn(inputs, mask, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            self.verbose and print("DONE", flush=True)
            self.test_loss = test_err / test_batches
            self.test_acc = test_acc / test_batches * 100
            self.verbose and print("Test results (loss=%9.6f, accuracy=%9.5f%%)" % (self.test_loss, self.test_acc))

        except KeyboardInterrupt:
            print("SKIPPED")

    def predict(self, input_data):
        raise NotImplementedError

    def save(self, model_filename, json_filename=None):
        if not json_filename:
            json_filename = "%s.json" % model_filename
        self.verbose and print("Saving model to file (%s)..." % model_filename, end=" ", flush=True)
        parameters = lasagne.layers.get_all_param_values(self.network)
        np.savez(model_filename, *parameters)
        self.verbose and print("OK")

    def load_training_data(self, training_data):
        if not self.glove:
            raise ValueError("Need to load a Glove object before loading the training data.")
        self.training_data = training_data
        if len(self.training_data.answers) > self.output_categories_no:
            raise ValueError("You are trying to load %d answers, more than "
                             "'output_categories_no' which is %d." % (len(self.training_data.answers),
                                                                      self.output_categories_no))

    def load_glove(self, glove):
        if not isinstance(glove, WordEmbedding):
            raise ValueError("The glove needs to be a 'Glove' object.")
        self.glove = glove

    def mark_best(self):
        self.best_parameters = lasagne.layers.get_all_param_values(self.network)

    def _get_best(self):
        assert self.best_parameters is not None
        lasagne.layers.set_all_param_values(self.network, self.best_parameters)

    def load(self, model_filename, json_filename=None):
        if not json_filename:
            json_filename = "%s.json" % model_filename
        self.verbose and print("Loading model from file (%s)..." % model_filename, end=" ", flush=True)
        with np.load(model_filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)
        #with open(json_filename, 'rt') as f:
        #    self.data = json.loads(f.read())
        self.verbose and print("OK")

    def interactive_predict(self):
        import readline

        if not self.prediction_function:
            raise ValueError("Can't predict without a 'prediction_function'.")

        pred_fn = self.prediction_function

        self.verbose and print("Interactive shell ready. Type 'exit' to close.")

        while True:

            sentence = input(">> ").strip()

            if sentence.startswith('exit'):
                print("Bye!")
                break

            dataset = "Q: %s\nA: NA\n\n" % sentence
            dataset = MemoryDataset(word_embedding=self.glove,
                                    augmenter=self.training_data.augmenter,
                                    filename=dataset)

            i = dataset.get_prepared_data(train_data_percentage=1,
                                          train_data_shuffle=False,
                                          max_sentence_size=self.max_words_per_sentence,
                                          verbose=self.verbose)
            i = i[0]  # Training split only.
            i = list(i)[0]  # Only first item.

            ts = time.time()

            o = pred_fn([i[0]], [i[1]])  # Question, mask.

            o = o[0][0]
            self.verbose and print(o)
            te = time.time()
            t = te - ts

            o = self._get_answer_by_one_hot_vector(o, limit_to=len(self.training_data.answers))
            print(o[0])  # TODO choose answer randomly

            self.verbose and print("predicted in %.4f seconds" % t)

    def _get_answer_by_index(self, i):
        try:
            return self.training_data.answers[i]
        except IndexError:
            return 'N/A i=%d, n=%d' % (i, len(self.training_data.answers))

    def _get_answer_by_one_hot_vector(self, vector, limit_to=None):
        if limit_to:
            vector = vector[:limit_to]
        return self._get_answer_by_index(one_hot_decode(vector))
