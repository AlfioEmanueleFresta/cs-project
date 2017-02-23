import argparse
import json
from collections import OrderedDict

from project.augmentation import FakeAugmenter, CombinerAugmenter
from project.data import FileDataset
from project.helpers import get_options_combinations, log_scale, get_index_and_increment
from project.network.lstm import LSTMNetwork
from project.vectorization.embedding import WordEmbedding
from datetime import datetime

import csv
import os
import random

base_path = './'

results_file = base_path + 'results.csv'
worker_index_file = base_path + 'workers.index'

reduction_granularity = 30      # How many values between 0 and 1 for `r`
repeat_times = 30               # How many times will each experiment be repeated

parser = argparse.ArgumentParser(description='Run the experiment as a single-task or multi-task script.')
parser.add_argument('--workers', dest='workers', type=int, default=1, help='The overall number of workers.')
parser.add_argument('--worker-id', dest='worker_id', type=int, default=1, help='The ID of this worker.')
args = parser.parse_args()

workers = args.workers
worker_id = args.worker_id

assert workers >= 1             # There needs to be at least one worker
assert worker_id > 0            # The ID must be greater than 1
assert worker_id <= workers     # The ID must be lower or equal to the number of workers

one_worker_per_time = workers != 1      # Should I use one worker per each time?


datasets = {
    # Name: (Filename, Max Words per sentence)
    #"TREC": ("data/prepared/trec.txt.gz", 200),
    #"Tag My News (Titles and Subtitles)": ("data/prepared/tagmynews.txt.gz", 325),
    "Tag My News (Titles Only)": ("data/prepared/tagmynews-titles-only.txt.gz", 325),
}


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=True, use_cache=True,
                  compute_clusters=False)

options = OrderedDict({})
options.update({"dataset": list(datasets.keys())})
options.update({"augmenter": [FakeAugmenter(),
                              CombinerAugmenter(max_window_size=2, glove=g),
                              CombinerAugmenter(max_window_size=3, glove=g)],})
options.update({"reduction": log_scale(steps=reduction_granularity),})
options.update({"time": list(range(repeat_times)),})

combinations = get_options_combinations(options)
print("Generated a list of %d combinations for the experiment." % len(combinations))

if one_worker_per_time:
    print("Multi-worker experiment. Using a section of the list.")
    this_index = worker_id - 1  # Zero-indexed index
    print("worker_id=%d" % worker_id)
    list_no = int(len(combinations) / workers)
    list_start = int(this_index * list_no)

else:
    print("Single-worker experiment. Using entire list of combinations.")
    list_start = 0
    list_no = len(combinations)

print("Slicing the combinations. start_i=%d, no_elements=%d." % (list_start, list_no))

combinations = combinations[list_start:(list_start + list_no)]

header = ["Worker ID",
          "Start time", "End time", "Dataset", "Augmentation technique", "Dataset Reduction (r)",
          "Experiment #", "Training Epochs", "Test loss", "Test accuracy"]

if not os.path.exists(results_file):
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

    line = [worker_id,
            start_time, end_time,
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

    print("OUTPUT: %s" % json.dumps(line))
    print("\n")
