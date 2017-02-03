import numpy as np
import scipy.spatial as spatial


class Augmenter:
    """
    Represents an abstract augmentation method.
    """

    def next(self, sentence):
        """
        A method which takes a single list of words and returns an augmented sentence.
        The augmented sentence is a list of tuples. Each tuple is a combination of words.

        :param words: The original vector.
        :return:
        """
        raise NotImplementedError


class FakeAugmenter(Augmenter):
    """
    Do not augment the input data.
    """

    def next(self, words):
        return [(word,) for word in words]

    def __str__(self):
        return "No Augmentation"


class CombinerAugmenter(Augmenter):

    def __init__(self, max_window_size=3, min_vector_distance=0.0,
                 glove=None):
        self.min_vector_distance = min_vector_distance
        self.max_window_size = max_window_size
        self.glove = glove

    def next(self, sentence):

        augmented_sentence = []
        history = []
        for word in sentence:
            history.append(word)
            for window_size in range(0, min(len(history), self.max_window_size)):
                t = (history[-(window_size + 1)], word)

                # If this is the original word, or if it is over the minimum distance
                if window_size == 0 or self._over_minimum_distance(word, tuple=t):
                    augmented_sentence.append(t)

        return augmented_sentence

    def _over_minimum_distance(self, word, tuple):
        if self.min_vector_distance == 0:
            return True
        word_vector = self.glove.get_word_vector(word=word)
        if not word_vector:
            return True
        tuple_vectors = list(self.glove.get_words_vectors(words=tuple))
        assert len(tuple_vectors) > 0  # Otherwise all words in this tuple are probably not in the WE
        merged_tuple = np.sum(tuple_vectors, axis=0)

        # NWE pass
        #closest_words = self.glove.get_closest_words(merged_tuple, n=5)
        #merged_tuple = self.glove.get_closest_word(merged_tuple)
        #merged_tuple = self.glove.get_word_vector(merged_tuple)

        distance = spatial.distance.euclidean(merged_tuple, word_vector)
        # print(distance >= self.min_vector_distance, "distance", word, tuple, len(tuple_vectors), distance)
        return distance >= self.min_vector_distance

    def __str__(self):
        return "Augmentation with max. window size %d" % self.max_window_size
