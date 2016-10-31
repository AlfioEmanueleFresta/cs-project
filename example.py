from project.data import TrainingData
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding
from project.network.mlp import MLPNetwork


g = WordEmbedding('data/glove.6B.50d.txt',
                  verbose=True, use_cache=True)

t = TrainingData('data/trec.txt.gz')

network_class = LSTMNetwork
n = network_class(input_features_no=g.vector_length,
                  output_categories_no=len(t.answers),
                  max_words_per_sentence=100,
                  train_batch_size=50)

n.build_network()

n.load_glove(g)
n.load_training_data(t)

n.train()

# n.save("x.npz")
n.interactive_predict()
