from project.helpers import get_options_combinations
from project.vectorization.glove import Glove
from project.network.mlp import MLPNetwork
import datetime


options = {
    'vector_size': [50],
    'train_percentage': [0.7],
    'max_epochs': [500],
    'dense_layers': [2],
    'dense_layers_size': [500],
    'dense_layers_dropout': [0.0]
}

for options in get_options_combinations(options):

    print(datetime.datetime.now())
    print(options)

    vector_size = options['vector_size']

    g = Glove('data/glove.6B.%dd.txt.gz' % vector_size, verbose=True)
    vector_size = g.vector_length

    n = MLPNetwork(input_features_no=vector_size,
                   output_categories_no=9,
                   max_words_per_sentence=15,
                   dense_layers=options['dense_layers'],
                   dense_layers_size=options['dense_layers_size'],
                   dense_layers_dropout=options['dense_layers_dropout'],
                   train_percentage=options['train_percentage'],
                   train_max_epochs=options['max_epochs'])
    n.build_network()
    n.load_glove(g)
    n.load_training_data('data/training.txt')
    n.train()
    # n.save("x.npz")
    n.interactive_predict(glove=g)



