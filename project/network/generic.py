import lasagne
import theano.tensor as T
import theano
import numpy as np
import time
import json
import readline  # Looks unused, but it is not!

from project.data import TrainingData
from project.helpers import one_hot_decode, one_hot_encode
from project.laplotter import LossAccPlotter
from project.vectorization.embedding import WordEmbedding


class GenericNetwork:

    def defaults(self):
        return {'verbose': True,
                'include_mask': False,

                # Training options
                'train_batch_size': 1,
                'train_data_test': True,
                'train_data_shuffle': True,
                'train_data_percentage': 0.7,
                }

    def __init__(self, input_features_no, output_categories_no, **kwargs):
        kwargs.update({'input_features_no': input_features_no,
                       'output_categories_no': output_categories_no})

        defaults = self.defaults()
        defaults.update(kwargs)

        self.data = {'params': defaults}
        self.training_data = None
        self.prediction_function = None
        self.validation_function = None
        self.glove = None
        self.best_parameters = None

    def __getattr__(self, item):
        if item in self.data['params']:
            # print("REQ %s=%s" % (item, self.data['params'][item]))
            return self.data['params'][item]
        raise AttributeError

    def build_network(self):
        raise NotImplementedError

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
        self.validation_function = val_fn

        questions_and_answers = self.get_prepared_training_data()
        self.verbose and print("Starting training. You can use CTRL-C "
                               "to stop the training process.")

        train, val, test = self._get_split_data(questions_and_answers)

        print("Epoch      Time        Tr. loss   Val. loss  Val. acc.   B  Best acc. ")
        print("---------  ----------  ---------  ---------  ----------  -  ----------")

        best_loss, best_acc, best_epoch = None, 0, None

        try:

            plotter = LossAccPlotter("Training loss and training accuracy",
                                     show_plot_window=True,
                                     show_averages=False)

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
                    best_epoch = epoch
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

        except KeyboardInterrupt:
            print("Training interrupted at epoch %d." % epoch)
            print("Best result (epoch=%d, loss=%9.6f, accuracy"
                  "=%9.5f%%)" % (best_epoch, best_loss, best_acc))

        self._test(test)

    def _test(self, test_data):

        if not self.train_data_test:
            print("Skipping testing (train_data_set=False).")
            return

        print("Testing network...", end=" ", flush=True)

        try:
            val_fn = self.validation_function
            test_err, test_acc, test_batches = 0, 0, 0
            for batch in self._iterate_minibatches(test_data):
                inputs, mask, targets = batch
                _, err, acc = val_fn(inputs, mask, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("DONE", flush=True)
            test_loss = test_err / test_batches
            test_acc = test_acc / test_batches * 100
            print("Test results (loss=%9.6f, accuracy=%9.5f%%)" % (test_loss, test_acc))

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
        with open(json_filename, 'wt') as f:
            f.write(json.dumps(self.data))
        self.verbose and print("OK")

    def load_training_data(self, training_data):
        if not self.glove:
            raise ValueError("Need to load a Glove object before loading the training data.")
        self.training_data = training_data
        if len(self.training_data.answers) > self.output_categories_no:
            raise ValueError("You are trying to load %d answers, more than "
                             "'output_categories_no' which is %d." % (len(self.training_data.answers),
                                                                      self.output_categories_no))

    def get_prepared_training_data(self):
        # Note: This is not a generator on purpose (as you want to pre-process!)
        questions, masks, answers = [], [], []
        self.verbose and print("Preparing training data...", end=" ", flush=True)

        for question, answer_id in self.training_data.questions:
            questions.append(self._questions_filter(question))
            masks.append(self._questions_mask_filter(question))
            answers.append(self._answers_filter(answer_id))

        questions, masks, answers = np.array(questions), np.array(masks), np.array(answers)
        self.verbose and print("OK")

        assert questions.shape[0] == masks.shape[0]
        assert questions.shape[0] == answers.shape[0]
        return questions, masks, answers

    def _questions_filter(self, sentence, **kwargs):
        return self.glove.get_sentence_matrix(sentence, max_words=self.max_words_per_sentence, **kwargs)

    def _questions_mask_filter(self, sentence):
        return self._get_mask(self._questions_filter(sentence))

    def _answers_filter(self, answer_id):
        return one_hot_encode(n=self.output_categories_no, i=answer_id,
                              positive=1.0, negative=0.0)  # Force float

    def _iterate_minibatches(self, questions_and_answers,
                             include_mask=False):
        questions, masks, answers = questions_and_answers
        while True:
            assert questions.shape[0] == answers.shape[0]

            this_batch_size = min(self.train_batch_size, questions.shape[0])

            if not this_batch_size:
                break

            these_questions, these_masks, these_answers = questions[:this_batch_size], masks[:this_batch_size], answers[:this_batch_size]
            questions, masks, answers = questions[this_batch_size:], masks[this_batch_size:], answers[this_batch_size:]

            assert these_questions.shape[0] == these_answers.shape[0]
            assert these_questions.shape[0] == these_masks.shape[0]
            assert these_questions.shape[0] <= self.train_batch_size

            if self.include_mask:
                yield these_questions, these_masks, these_answers

            else:
                yield these_questions, these_answers

    def _get_mask(self, vector):
        assert vector.shape == (self.max_words_per_sentence, self.glove.vector_length)
        mask = np.zeros((self.max_words_per_sentence,))
        i = 0
        for element in vector:
            mask[i] = 1
            if np.array_equal(element, self.glove.vectors[self.glove.SYM_END]):
                break
            i += 1
        return mask

    def _get_split_data(self, tuples):

        questions, masks, answers = tuples

        if self.train_data_percentage <= 0 or self.train_data_percentage > 1:
            raise ValueError("train_percentage need to be between 0 and 1.")

        if self.train_data_shuffle:
            indices = np.array(range(questions.shape[0]))
            np.random.shuffle(indices)
            questions, masks, answers = questions[indices], masks[indices], answers[indices]

        all_questions_no = questions.shape[0]
        training_no = int(all_questions_no * self.train_data_percentage)
        training = questions[0:training_no], masks[0:training_no], answers[0:training_no]

        divisor = 2 if self.train_data_test else 1
        validation_no = int((all_questions_no - training_no) / divisor)

        validation = questions[training_no:training_no + validation_no], \
                     masks[training_no:training_no + validation_no], \
                     answers[training_no:training_no + validation_no]

        testing = questions[training_no + validation_no:], \
                  masks[training_no + validation_no:], \
                  answers[training_no + validation_no:]

        return training, validation, testing

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
        with open(json_filename, 'rt') as f:
            self.data = json.loads(f.read())
        self.verbose and print("OK")

    def interactive_predict(self):
        import readline

        self._get_best()  # Get the best validation score.

        if not self.prediction_function:
            raise ValueError("Can't predict without a 'prediction_function'.")

        pred_fn = self.prediction_function

        print("Interactive shell. Type 'exit' to close.")

        while True:

            sentence = input(">> ").strip()

            if sentence.startswith('exit'):
                print("Bye!")
                break

            i = self._questions_filter(sentence, show_workings=True)
            m = self._questions_mask_filter(sentence)

            i = np.array([i])
            m = np.array([m])

            ts = time.time()

            if self.include_mask:
                o = pred_fn(i, m)

            else:
                o = pred_fn(i)

            o = o[0][0]
            print(o)
            te = time.time()
            t = te - ts

            o = self._get_answer_by_one_hot_vector(o, limit_to=len(self.training_data.answers))
            print(o)

            print("predicted in %.4f seconds" % t)

    def _get_answer_by_index(self, i):
        try:
            return self.training_data.answers[i]
        except IndexError:
            return 'N/A i=%d, n=%d' % (i, len(self.training_data.answers))

    def _get_answer_by_one_hot_vector(self, vector, limit_to=None):
        if limit_to:
            vector = vector[:limit_to]
        return self._get_answer_by_index(one_hot_decode(vector))
