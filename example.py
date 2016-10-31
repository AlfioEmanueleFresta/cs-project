from project.data import TrainingData
from project.helpers import get_options_combinations
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding
from project.network.mlp import MLPNetwork
import datetime


options = {
    'glove_use_cache': [True],
    'vector_size': [50],        # 50 for loading speed, 300 for accuracy.
    'network_class': [LSTMNetwork]
}

for options in get_options_combinations(options):

    print(datetime.datetime.now())
    print(options)

    vector_size = options['vector_size']

    g = WordEmbedding('data/glove.6B.%dd.txt' % vector_size,
                      verbose=True, use_cache=True)

    t = TrainingData('data/trec.txt.gz')

    network_class = options['network_class']
    n = network_class(input_features_no=g.vector_length,
                      output_categories_no=len(t.answers),
                      max_words_per_sentence=100,
                      train_batch_size=50)
    n.build_network()
    n.load_glove(g)
    n.load_training_data(t)
    n.train()
    # n.save("x.npz")
    n.interactive_predict(glove=g)
