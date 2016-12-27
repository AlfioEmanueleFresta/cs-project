import lasagne
import theano.tensor as T
from .lstm import LSTMNetwork


class LSTMExperimentalNetwork(LSTMNetwork):

    def __init__(self,
                 **kwargs):

        super(LSTMExperimentalNetwork, self).__init__(**kwargs)

        self.input_var = T.tensor3('inputs')
        self.mask_var = T.matrix('masks')
        self.target_var = T.matrix('targets')

    def defaults(self):
        defaults = super(LSTMExperimentalNetwork, self).defaults()
        defaults.update({
            # Composition window sizes (inclusive)
            'composition_window_min': 2,
            'composition_window_max': 4
        })
        return defaults

    def build_network(self):
        # Building the neural network
        #########################################################

        super(LSTMExperimentalNetwork, self).build_network(input_var=input_var, input_shape=input_shape,
                                                           mask_var=mask_var, mask_shape=mask_shape)
