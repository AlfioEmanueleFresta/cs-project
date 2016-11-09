import argparse

from project.data import TrainingData
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding


parser = argparse.ArgumentParser(description='.')
parser.add_argument('--no-train', dest='train', action='store_false', help='do not train the model.')
parser.add_argument('--no-display', dest='display', action='store_false', help='do not show a plot on screen.')
parser.add_argument('--transient', dest='save', action='store_false', help='do not persist the trained model to disk.')
parser.add_argument('-v', dest='verbose', action='store_true', help='print debug information.')
args = parser.parse_args()


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=args.verbose, use_cache=True)

t = TrainingData('data/prepared/trec.txt.gz')

network_class = LSTMNetwork
n = network_class(input_features_no=g.vector_length,
                  output_categories_no=len(t.answers),
                  max_words_per_sentence=100,
                  train_batch_size=50,
                  verbose=args.verbose,
                  show_plot=args.display)

n.build_network()
n.compile_functions()

n.load_glove(g)
n.load_training_data(t)

model_filename = "data/model.cache.npz"

if args.train:
    n.train()

    if args.save:
        n.save(model_filename)

else:
    n.load(model_filename)

n.interactive_predict()
