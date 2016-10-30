from project.helpers import get_options_combinations
from project.network.lstm import LSTMNetwork
from project.vectorization.glove import Glove
from project.network.mlp import MLPNetwork
import datetime


options = {
    'glove_use_cache': [True],
    'vector_size': [100],        # 50 for loading speed, 300 for accuracy.
    'network_class': [LSTMNetwork]
}

for options in get_options_combinations(options):

    print(datetime.datetime.now())
    print(options)

    vector_size = options['vector_size']

    g = Glove('data/glove.6B.%dd.txt' % vector_size,
              verbose=True, use_cache=options['glove_use_cache'])
    vector_size = g.vector_length

    network_class = options['network_class']
    n = network_class(input_features_no=vector_size,
                      output_categories_no=50,
                      max_words_per_sentence=100,
                      train_batch_size=50)
    n.build_network()
    n.load_glove(g)
    n.load_training_data('data/trec.txt')
    try:
        n.train()
    except KeyboardInterrupt:
        print("Training interrupted. Continuing.")
    # n.save("x.npz")
    n.interactive_predict(glove=g)



