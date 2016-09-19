import gzip
from annoy import AnnoyIndex
import os
import numpy as np

from ..helpers import one_hot_encode, one_hot_decode


class Glove:

    DEFAULT_TREES_NO = 50
    DEFAULT_SEPARATOR = ' '

    SYM_END = 'SYMEND'
    SYM_EMPTY = 'SYMEMPTY'

    SYM_CHARS = ['.', ',', '!', '?']

    ALL_SYM = [SYM_END, SYM_EMPTY] + SYM_CHARS

    def __init__(self, filename, trees_no=DEFAULT_TREES_NO,
                 separator=DEFAULT_SEPARATOR, verbose=False,
                 use_cache=True):
        """
        Instantiate and build a new search tree from a Glove vector
        generated vector file.

        :param filename: The name of the file. May end in .gz if compressed with gzip.
        :param trees_no: The number of search trees to use for the index.
        :param separator: The separator used in the input file. Defaults to the space character.
        :param verbose: Whether to print out some information re. the progress or not. Defaults to False.
        :param use_cache: Whether to look for a .cache file, or create one. Speeds up index loading (no need
                          to rebuild the search trees every time).
        """

        compressed = '.gz' in filename
        opener = gzip.open if compressed else open

        if verbose:
            print("Loading Glove vector %s (compressed=%d)..." % (filename, compressed))

        with opener(filename, 'rt') as input_file:

            first_row = input_file.readline().split(separator)
            input_file.seek(0)  # Don't skip the first row later
            row_length = len(first_row)
            vector_length = row_length - 1

            vector_length += len(self.ALL_SYM)
            self.vector_length = vector_length
            print("Allocating vector of length %d" % self.vector_length)

            self.index = AnnoyIndex(f=self.vector_length)
            self.words = []
            self.vectors = {}
            words_index = 0

            cache_filename = "%s.cache" % filename
            cache_available = use_cache and os.path.isfile(cache_filename)

            all_lines = list(input_file.readlines())

            for row in all_lines:

                row = row.split(separator)
                assert len(row) == row_length  # Check rows consistency.

                word = self.prepare_word(row[0], building=True)
                vector = [float(x) for x in row[1:]]

                # Add one-hot for symbols at the end.
                vector += [0.0 for _ in range(len(self.ALL_SYM))]

                self.words.append(word)
                self.vectors[word] = vector

                if not cache_available:
                    self.index.add_item(words_index, vector)

                words_index += 1

            for symbol in self.ALL_SYM:

                vector = [0.0 for _ in range(row_length - 1)]
                vector += [1.0 if symbol == k else 0.0 for k in self.ALL_SYM]

                self.words.append(symbol)
                self.vectors[symbol] = vector

                if not cache_available:
                    self.index.add_item(words_index, vector)

                words_index += 1

            if cache_available:
                verbose and print("Loading search trees from cache file (%s)..." % cache_filename)
                self.index.load(cache_filename)

            if not cache_available:
                verbose and print("Building %d search trees..." % trees_no)
                self.index.build(n_trees=trees_no)

                if use_cache:
                    verbose and print("Saving search trees to file (%s)..." % cache_filename)
                    self.index.save(cache_filename)

            verbose and print("Loaded %d words with vector length=%d" % (words_index, vector_length))

    def prepare_word(self, word, building=False):
        o = word
        word = word.strip().lower()
        return word

    def get_word_vector(self, word):
        """
        Gets the vector for a given word, if existent.
        :param word: The word.
        :return:
        """
        word = self.prepare_word(word, building=False)
        if word in self.vectors:
            return self.vectors[word]
        return None

    def get_sentence_matrix(self, sentence, max_words):
        # TODO replace zeros with 'stop' vector
        # TODO support end of sentence

        matrix = np.zeros((max_words, self.vector_length))

        # Make all supported symbols their own word.
        for sym in self.SYM_CHARS:
            sentence.replace(sym, " %s" % sym)

        words = sentence.split(' ')
        word_i = 0

        for word in words[:min(len(words), max_words)]:
            vector = self.get_word_vector(word)
            if vector:
                matrix[word_i] = np.array(vector)
            word_i += 1
        return matrix

    def matrix_to_words(self, matrix, try_to_clean=True):
        to_remove = self.get_closest_word(np.zeros(self.vector_length)) if try_to_clean else '**'
        for vector in matrix:
            word = self.get_closest_word(vector)
            if word != to_remove:
                yield word

    def get_words_vectors(self, words):
        for word in words:
            vector = self.get_word_vector(word)
            if vector:
                yield vector

    def get_closest_word(self, vector, **kwargs):
        closest_words = self.get_closest_words(vector, n=1, **kwargs)
        return closest_words[0] if closest_words else None

    def get_closest_words(self, vector, n, include_distances=False):
        """
        Return an ordered list of the n-closest words to a given vector.

        If ``include_distances`` is set to True, return a list of tuples (word, distance).

        :param vector: The vector.
        :param n: The number of closest words to search for.
        :param include_distances: Whether to include the calculated distance or not.
        :return: A list of words or tuples (word, distance).
        """

        closest = self.index.get_nns_by_vector(vector, n=n,
                                               include_distances=include_distances)

        if include_distances:
            return [(self.words[word_id], distance) for word_id, distance in zip(closest[0], closest[1])]

        return [self.words[word_id] for word_id in closest]
