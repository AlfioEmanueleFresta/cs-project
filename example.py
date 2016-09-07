from project.vectorization.glove import Glove
from project.network.lstm import LSTMNetwork


vector_size = 100

g = Glove('data/glove.6B.%dd.txt.gz' % vector_size, verbose=True)
n = LSTMNetwork(word_vector_size=vector_size)

