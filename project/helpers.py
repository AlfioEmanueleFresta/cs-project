import numpy as np
import itertools as it
import math

from theano import function, config, shared, sandbox
import theano.tensor as T
import theano
import warnings


def log_scale(steps):
    """
    Returns a log scale between 0 and 1 in a given number of steps.
    :param steps: The number of desired values between 0 and 1 (0 excluded).
    :return: A list of real numbers.
    """
    assert steps > 0
    f = lambda x: 1 - math.log10(x)
    values = reversed(list(np.linspace(1, 10, steps + 1))[0:-1])
    return [f(x) for x in values]


def get_options_combinations(options):
    the_product = it.product(*options.values())
    return list([{key: value for (key, value) in zip(options, values)} for values in the_product])


def one_hot(n=100, i=0, positive=1, negative=0):
    if i >= n:
        raise ValueError("Can't one-hot encode index '%d' in vector of size %d." % (i, n))
    return [positive if k == i else negative for k in range(n)]


def one_hot_encode(*args, **kwargs):
    return one_hot(*args, **kwargs)


def one_hot_decode(vector):
    try:
        return np.argmax(vector)
    except AttributeError:
        return vector.index(max(vector))


def get_index_and_increment(file_name):
    try:
        with open(file_name, 'rt') as f:
            current = int(f.readline())
    except FileNotFoundError:
        current = -1
    current += 1
    with open(file_name, 'wt') as f:
        f.write("%d" % (current,))
    return current


def kmeans(X, clusters_no, epochs=500, learning_rate=0.01,
           batch_size=100, verbose=False):
    rng = np.random
    W = rng.randn(clusters_no, X.shape[1])
    X2 = (X**2).sum(1)[:, None]
    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            D = -2 * np.dot(W, X[i:i + batch_size,:].T) + (W**2).sum(1)[:, None] + X2[i:i + batch_size].T
            S = (D == D.min(0)[None, :]).astype("float").T
            W += learning_rate * (np.dot(S.T, X[i:i + batch_size,:]) - S.sum(0)[:, None] * W)
        if verbose:
            print(" k-means, epoch=%d/%d, cost=%f" % (epoch, epochs, D.min(0).sum()))
    return W


def shuffle(l):
    """
    Return a shuffled copy of a Python list.
    :param l: A list to shuffle.
    :return: A copy of the list in random order.
    """
    length = len(l)
    indices = np.random.permutation(length)
    output = []
    for i in indices:
        output.append(l[i])
    return output


def split(l, a_percentage):
    """
    Split a list into two lists A, B of given ratio.
    :param l: The list to split.
    :param a_percentage: The percentage of items to go in A.
    :return: A tuple (A, B).
    """
    assert a_percentage >= 0
    assert a_percentage <= 1
    all_no = len(l)
    a_no = int(all_no * a_percentage)
    return l[0:a_no], l[a_no:]


class ResettableIterator:
    """
    An iterator which can be resetted.
    """

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class CachedBatchIterator(ResettableIterator):
    """
    Create an iterator over some data which returns it in batches of
     a given fixed size. The data is generated and buffered at initialisation,
     to a
    """

    def __init__(self, data, batch_size):
        assert batch_size > 0
        self.i = 0
        self.batch_size = batch_size
        self.data = np.array(list(data))

    def reset(self):
        self.i = 0

    def __next__(self):
        if self.i >= self.data.shape[0] / self.batch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min(start + self.batch_size, self.data.shape[0])

        batch = self.data[start:end]
        self.i += 1

        return batch


class CachedTupleBatchIterator(ResettableIterator):
    """
    >>> from project.helpers import CachedTupleBatchIterator
    >>> a = range(30, 40)
    >>> b = range(60, 70)
    >>> c = range(10, 20)
    >>> t = (a, b, c)
    >>> t
    (range(30, 40), range(60, 70), range(10, 20))
    >>> i = CachedTupleBatchIterator(t, 5)
    >>> i.__next__()
    (array([30, 31, 32, 33, 34]),
     array([60, 61, 62, 63, 64]),
     array([10, 11, 12, 13, 14]))
    """
    def __init__(self, data, batch_size):
        assert batch_size > 0
        self.i = 0
        self.batch_size = batch_size
        self.data = tuple(np.array(list(x)) for x in data)

    def reset(self):
        self.i = 0

    def __next__(self):
        if self.i >= self.data[0].shape[0] / self.batch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min(start + self.batch_size, self.data[0].shape[0])

        batch = tuple(t[start:end] for t in self.data)
        self.i += 1

        return batch


class CachedTupleGeneratorbatchIterator(ResettableIterator):
    """
    >>> from project.helpers import CachedTupleGeneratorbatchIterator
    >>> def generator():
    >>>   for i in range(10):
    >>>     yield i, i, i
    >>>
    >>> i = CachedTupleGeneratorbatchIterator(generator, 5)
    >>> i.__next__()
    (1, 1, 1)
    """
    def __init__(self, generator, batch_size):
        assert batch_size > 0
        self.i = 0
        self.batch_size = batch_size
        self.data = tuple(np.array(x) for x in zip(*generator))


    def reset(self):
        self.i = 0

    def __next__(self):
        if self.i >= self.data[0].shape[0] / self.batch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min(start + self.batch_size, self.data[0].shape[0])

        batch = tuple(t[start:end] for t in self.data)
        self.i += 1

        return batch

