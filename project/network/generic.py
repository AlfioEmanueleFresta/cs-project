import lasagne
import theano.tensor as T
import theano
import numpy as np
import os
import time
import json

from project.data import TrainingData
from project.helpers import one_hot_decode, one_hot_encode
from project.vectorization.glove import Glove


class GenericNetwork:

    def defaults(self):
        return {'verbose': True,

                # Training options
                'train_batch_size': 1,
                'train_data_test': False,
                'train_data_shuffle': True,
                'train_data_percentage': 0.6,
                }

    def __init__(self, input_features_no, output_categories_no, **kwargs):
        kwargs.update({'input_features_no': input_features_no,
                       'output_categories_no': output_categories_no})

        defaults = self.defaults()
        defaults.update(kwargs)

        self.data = {'params': defaults}
        self.training_data = None
        self.glove = None

    def __getattr__(self, item):
        if item in self.data['params']:
            # print("REQ %s=%s" % (item, self.data['params'][item]))
            return self.data['params'][item]
        raise AttributeError

    def build_network(self):
        raise NotImplementedError

    def train(self, output_filename):
        raise NotImplementedError

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

    def load_training_data(self, filename):
        if not self.glove:
            raise ValueError("Need to load a Glove object before loading the training data.")
        self.training_data = TrainingData(filename)
        if len(self.training_data.answers) > self.output_categories_no:
            raise ValueError("You are trying to load %d answers, more than "
                             "'output_categories_no' which is %d." % (len(self.training_data.answers),
                                                                      self.output_categories_no))

    def get_prepared_training_data(self):
        # Note: This is not a generator on purpose (as you want to pre-process!)
        questions, answers = [], []
        self.verbose and print("Preparing training data...", end=" ", flush=True)
        for question, answer_id in self.training_data.questions:
            questions.append(self._questions_filter(question))
            answers.append(self._answers_filter(answer_id))
        questions, answers = np.array(questions), np.array(answers)
        self.verbose and print("OK")
        assert questions.shape[0] == answers.shape[0]
        return questions, answers

    def _questions_filter(self, sentence):
        return self.glove.get_sentence_matrix(sentence, max_words=self.max_words_per_sentence,
                                              show_workings=True)

    def _answers_filter(self, answer_id):
        return one_hot_encode(n=self.output_categories_no, i=answer_id,
                              positive=1.0, negative=0.0)  # Force float

    def _iterate_minibatches(self, questions_and_answers):
        questions, answers = questions_and_answers
        while True:
            assert questions.shape[0] == answers.shape[0]

            this_batch_size = min(self.train_batch_size, questions.shape[0])

            if not this_batch_size:
                break

            these_questions, these_answers = questions[:this_batch_size], answers[:this_batch_size]
            questions, answers = questions[this_batch_size:], answers[this_batch_size:]

            assert these_questions.shape[0] == these_answers.shape[0]
            assert these_questions.shape[0] <= self.train_batch_size

            yield these_questions, these_answers

    def _get_split_data(self, tuples):

        questions, answers = tuples

        if self.train_data_percentage <= 0 or self.train_data_percentage > 1:
            raise ValueError("train_percentage need to be between 0 and 1.")

        if self.train_data_shuffle:
            indices = np.array(range(questions.shape[0]))
            np.random.shuffle(indices)
            questions, answers = questions[indices], answers[indices]

        all_questions_no = questions.shape[0]
        training_no = int(all_questions_no * self.train_data_percentage)
        training = questions[0:training_no], answers[0:training_no]

        divisor = 2 if self.train_data_test else 1
        validation_no = int((all_questions_no - training_no) / divisor)

        validation = questions[training_no:training_no + validation_no], answers[training_no:training_no + validation_no]
        testing = questions[training_no + validation_no:], answers[training_no + validation_no:]

        return training, validation, testing

    def load_glove(self, glove):
        if not isinstance(glove, Glove):
            raise ValueError("The glove needs to be a 'Glove' object.")
        self.glove = glove

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

    def interactive_predict(self, glove):
        print("Compiling prediction function...")
        prediction = lasagne.layers.get_output(self.network)
        pred_fn = theano.function([self.input_var],
                                  [prediction],
                                  allow_input_downcast=True)

        print("Interactive shell. Type 'exit' to close.")

        while True:

            sentence = input(">> ").strip()

            if sentence.startswith('exit'):
                print("Bye!")
                break

            i = glove.get_sentence_matrix(sentence, max_words=self.max_words_per_sentence,
                                          show_workings=True)
            i = np.array([i])

            o = pred_fn(i)
            o = o[0][0]
            print(o)

            o = self._get_answer_by_one_hot_vector(o, limit_to=len(self.training_data.answers))
            print(o)

    def _get_answer_by_index(self, i):
        try:
            return self.training_data.answers[i]
        except IndexError:
            return 'N/A i=%d, n=%d' % (i, len(self.training_data.answers))

    def _get_answer_by_one_hot_vector(self, vector, limit_to=None):
        if limit_to:
            vector = vector[:limit_to]
        return self._get_answer_by_index(one_hot_decode(vector))
