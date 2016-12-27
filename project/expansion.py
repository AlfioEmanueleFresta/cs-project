import numpy as np


class Expander:

    def next(self, item):
        """
        A method which takes a single line and returns 0-N expanded lines.
        If the original line is to be kept, it needs to be returned as well.
        :param item: The original line.
        :return:
        """
        raise NotImplementedError


class FakeExpander:
    """
    This is a fake expander -- which does not expand the input provided.
    """

    def next(self, item):
        return item


class WangExpander:
    """
    An expander based on the work by Wang P. in "Semantic expansion using word
    embedding clustering and convolutional neural network for improving short text
    classification".
    """

    def __init__(self, n, distance):
        """
        Initialise the expander.
        :param n: The maximum number of consecutive words to try and match.
        :param distance: The minimum distance to keep the expansion.
        """
        self.n = n
        self.distance = distance
        self.history = []

    def next(self, item):
        """
        Takes a vector representation of a word (in any word embedding) as input,
        and returns all combinations with the last N words, if applicable.

        :param item: A word represented as a vector.
        :return: A list of vectors.
        """

        # Update the history so it contains only the last N items
        self.history.append(item)
        self.history = self.history[-self.n:]

        o = self._get_combinations()
        o = self._combine_combinations(o)
        o = self._filter_combinations(item, o)
        o = np.array(list(o), dtype=np.float32)

        return o

    def _get_combinations(self):
        for i in range(min(self.n, len(self.history))):
            yield self.history[-(i + 1):]

    def _combine_combinations(self, combinations):
        for combination in combinations:
            yield np.sum(combination, axis=0)

    def _filter_combinations(self, item, combinations):
        for combination in combinations:
            distance = np.linalg.norm(combination - item)
            if np.array_equal(combination, item) or distance >= self.distance:
                yield combination
