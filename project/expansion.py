import numpy as np


class Expander:
    """
    Represents an abstract expansion method.
    """

    def next(self, item):
        """
        A method which takes a single vector and returns 0-N vectors,
        in the form of a numpy array of shape (NxS), where S is the shape of the original vector.
        If the original vector is to be kept, it needs to be returned as well.
        :param item: The original vector.
        :return:
        """
        raise NotImplementedError

    def multi(self, items):
        output = []
        for item in items:
            output.append(self.next(item))
        return np.concatenate(output, axis=0)


class FakeExpander(Expander):
    """
    This is a fake expander -- which does not expand the input provided,
    it simply returns what it is fed.
    """

    def next(self, item):
        return np.array([item])


class WangExpander(Expander):
    """
    An expander based on the work by Wang P. in "Semantic expansion using word
    embedding clustering and convolutional neural network for improving short text
    classification".

    It returns all vectors fed as input, plus all combinations of the last 2..N words,
    where N is specified at initialisation.

    All vector combinations are merged into individual vectors using component-wise addition.

    Finally, merged vectors combinations that are closer than `distance` from the original vector
    are ignored and not returned.
    """

    def __init__(self, n, distance, word_embedding):
        """
        Initialise the expander.
        :param n: The maximum number of consecutive words to try and match.
        :param distance: The minimum distance to keep the expansion vector.
        :param word_embedding: The word embedding to use.
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
        o = self._merge_combinations(o)
        o = self._filter_combinations(item, o)
        o = np.array(list(o), dtype=np.float32)

        return o

    def _get_combinations(self):
        for i in range(min(self.n, len(self.history))):
            yield self.history[-(i + 1):]

    def _merge_combinations(self, combinations):
        for combination in combinations:
            yield np.sum(combination, axis=0)

    def _filter_combinations(self, item, combinations):

        ## TODO
        # This is not what is detailed by Wang P and needs changing.
        # For each combination which is NOT the item, he searches the closest cluster centroid.
        ys, ns = 0, 0
        for combination in combinations:
            distance = np.linalg.norm(combination - item)
            if np.array_equal(combination, item) or distance >= self.distance:
                ys += 1
                yield combination
            else:
                ns += 1
        print(ys, ns, ys + ns)
