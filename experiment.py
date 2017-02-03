from project.augmentation import FakeAugmenter, CombinerAugmenter
from project.data import FileDataset
from project.helpers import get_options_combinations, log_scale
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding
from datetime import datetime

import numpy as np
import csv


results_file = 'results.csv'

reduction_granularity = 20
repeat_times = 5



datasets = {
    # Name: (Filename, Max Words per sentence)
    # "TREC": ("data/prepared/trec.txt.gz", 200),
    # "Tag My News (Titles and Subtitles)": ("data/prepared/tagmynews.txt.gz", 325),
    "Tag My News (Titles Only)": ("data/prepared/tagmynews-titles-only.txt.gz", 325),
}


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=True, use_cache=True,
                  compute_clusters=False)

options = {
    "dataset": list(datasets.keys()),
    "augmenter": [FakeAugmenter(),
                  CombinerAugmenter(max_window_size=2, glove=g),
                  CombinerAugmenter(max_window_size=3, glove=g)],
    "reduction": log_scale(steps=reduction_granularity),
    "time": list(range(repeat_times)),
}

combinations = get_options_combinations(options)

header = ["Start time", "End time", "Dataset", "Augmentation technique", "Dataset Reduction (r)",
          "Experiment #", "Training Epochs", "Test loss", "Test accuracy"]

with open(results_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

for i, options in enumerate(combinations):

    start_time = datetime.now()
    print(start_time, "Combination %d of %d" % (i + 1, len(combinations)))
    print(options)

    t = FileDataset(word_embedding=g,
                    filename=datasets[options['dataset']][0],
                    augmenter=options['augmenter'],
                    reduce=options['reduction'])

    network_class = LSTMNetwork
    n = network_class(input_features_no=g.vector_length, output_categories_no=len(t.answers),
                      max_words_per_sentence=datasets[options['dataset']][1], train_batch_size=1000,
                      verbose=True, show_plot=False, train_max_epochs=1000)

    n.build_network()
    n.compile_functions()

    n.load_glove(g)
    n.load_training_data(t)

    n.train()

    epochs = n.epochs
    test_loss = n.test_loss
    test_acc = n.test_acc

    end_time = datetime.now()

    line = [start_time, end_time,
            options['dataset'],
            str(options['augmenter']),
            options['reduction'],
            options['time'],
            epochs,
            test_loss,
            test_acc]

    with open(results_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(line)

    print("\n")
