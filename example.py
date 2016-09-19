from project.helpers import get_options_combinations
from project.vectorization.glove import Glove
from project.network.mlp import MLPNetwork

import sys
import datetime


options = {
    'vector_size': [300],
    'train_percentage': [0.8],
    'max_epochs': [100],
    'dense_layers': [2],
    'dense_layers_size': [500],
    'dense_layers_dropout': [0]
}


i = 0
out = sys.stdout

for options in get_options_combinations(options):

    i += 1

    with open('data/out-%d.txt' % i, 'wt') as o:

        sys.stdout = out
        print(options)

        #sys.stdout = o

        print(datetime.datetime.now())
        print(options)

        vector_size = options['vector_size']

        g = Glove('data/glove.6B.%dd.txt.gz' % vector_size, verbose=True)
        vector_size = g.vector_length

        n = MLPNetwork(word_vector_size=vector_size,
                       max_words_per_sentence=15,
                       dense_layers=options['dense_layers'],
                       dense_layers_size=options['dense_layers_size'],
                       dense_layers_dropout=options['dense_layers_dropout'],
                       output_categories_no=6)

        n.train(training_data_filename='data/training.txt',
                train_percentage=options['train_percentage'],
                glove=g,
                output_filename='test.npz',
                max_epochs=options['max_epochs'])

        n.interactive_predict(glove=g)

