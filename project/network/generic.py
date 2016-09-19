import lasagne
import theano.tensor as T
import theano
import numpy as np
import os
import time
import json


class GenericNetwork:

    VERBOSE = True

    def __init__(self,
                 verbose=VERBOSE,
                 *args,
                 **kwargs):

        self.verbose = verbose

        if not hasattr(self, 'network'):
            raise Exception("The __init__ method needs to define a network.")

        if not hasattr(self, 'data'):
            self.data = {}

    def train(self, output_filename):
        self.save(output_filename)

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

    def load_or_train(self, filename, **kwargs):
        if os.path.isfile(filename):
            self.load(filename)
        else:
            self.train(output_filename=filename, **kwargs)

