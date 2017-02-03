import argparse

from project.augmentation import FakeAugmenter, CombinerAugmenter
from project.data import Dataset, FileDataset
from project.expansion import WangExpander
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding


parser = argparse.ArgumentParser(description='.')
parser.add_argument('--no-train', dest='train', action='store_false', help='do not train the model.')
parser.add_argument('--resume', dest='resume', action='store_true', help='load and resume the training.')
parser.add_argument('--no-display', dest='display', action='store_false', help='do not show a plot on screen.')
parser.add_argument('--transient', dest='save', action='store_false', help='do not persist the trained model to disk.')
parser.add_argument('-v', dest='verbose', action='store_true', help='print debug information.')
args = parser.parse_args()


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=args.verbose, use_cache=True,
                  compute_clusters=False)

#t = Dataset('data/prepared/trec.txt.gz')
t = FileDataset(word_embedding=g,
                filename='data/prepared/tagmynews.txt.gz',
                augmenter=CombinerAugmenter(max_window_size=3,
                                            min_vector_distance=1.45,
                                            glove=g),)

network_class = LSTMNetwork
n = network_class(input_features_no=g.vector_length,
                  output_categories_no=len(t.answers),
                  max_words_per_sentence=325,
                  train_batch_size=1000,
                  verbose=args.verbose,
                  show_plot=args.display,
                  train_max_epochs=5000)

n.build_network()
n.compile_functions()

n.load_glove(g)
n.load_training_data(t)

model_filename = "data/model.cache.npz"

if (not args.train) or args.resume:
    n.load(model_filename)

if args.train:
    n.train()

    if args.save:
        n.save(model_filename)


n.interactive_predict()



## What do we want to do
# [ ] Load the dataset with N samples
# [X] Shuffle the dataset, to randomise the location of the sentences
# [X] Split the dataset into training, validation and test sets
# [X] Once the dataset has been split into groups, extend each sentence of the groups,
#       de facto triplicating, or more, the size of each of the groups
# - See if the improvement has any correlation to the original size of the dataset



## To DO
# At the moment the expansion works by generating for each sentence N new sentences. Instead, it should generate
# a single additional sentence. I.e.
# f(A B C D, N=2) => (A B C D, A B AB C BC ABC).

# Additionally, need to investigate the effect of using the expanded sentence in validation and testing groups.
# - The expanded sentence should *NOT* be used in the latter two as this may skew the validation.

# During prediction, should it utilise only the original or only the expanded sentence?

# Also -- adapt method of distance calculation to discard stupid tuples.

# Further work new idea. The expansion should do something more intelligent, such as exploit
#  the syntax tree of the input sentence to pair words in a more meaningful manner.
