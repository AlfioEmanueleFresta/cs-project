import numpy as np


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


class CombinerAugmenter(Augmenter):

    def __init__(self, max_window_size=3):
        self.max_window_size = max_window_size

    def next(self, sentence):

        augmented_sentences = []

        for window_size in range(self.max_window_size):
            augmented_sentence = []
            history = []
            for word in sentence:
                history.append(word)
                t = tuple(history[-(window_size + 1):])
                augmented_sentence.append(t)
            augmented_sentences.append(augmented_sentence)

        return augmented_sentences
