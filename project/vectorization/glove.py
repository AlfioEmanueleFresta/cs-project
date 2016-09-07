import gzip
from annoy import AnnoyIndex
import os


class Glove:

    DEFAULT_TREES_NO = 50
    DEFAULT_SEPARATOR = ' '

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

            self.index = AnnoyIndex(f=vector_length)
            self.words = []
            self.vectors = {}
            words_index = 0

            cache_filename = "%s.cache" % filename
            cache_available = use_cache and os.path.isfile(cache_filename)

            for row in input_file.readlines():

                row = row.split(separator)
                assert len(row) == row_length  # Check rows consistency.

                word = self.prepare_word(row[0], building=True)
                vector = [float(x) for x in row[1:]]

                self.words.append(word)
                self.vectors[word] = vector

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
        return word.strip().lower()

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
