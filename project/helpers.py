import numpy as np
import itertools as it

from theano import function, config, shared, sandbox
import theano.tensor as T
import theano
import warnings


def load_questions_and_answers(filename):

    qa = []

    QUESTION_DELIMITER = 'Q: '
    ANSWER_DELIMITER = 'A: '

    with open(filename, 'rt') as f:

        questions, answers = [], []

        line_no = 0
        for line in f.readlines():

            line = line.rstrip('\n').strip()
            line_no += 1

            if line.startswith(QUESTION_DELIMITER):

                question = line[len(QUESTION_DELIMITER)-1:].strip()
                questions.append(question)

            elif line.startswith(ANSWER_DELIMITER):

                answer = line[len(ANSWER_DELIMITER)-1:].strip()
                answers.append(answer)

            elif not line:

                if (questions and not answers) or (not questions and answers):
                    raise ValueError("Group terminating at line %d has questions but no answers, or viceversa." % line_no)

                if questions and answers:
                    qa.append((questions, answers))

                questions, answers = [], []

            else:
                raise ValueError("Error in line %d: '%s'." % (line_no, line))

    return qa


def get_all_questions_and_answers(qas):
    for questions, answers in qas:
        for q in questions:
            for a in answers:
                yield q, a


def get_all_questions(qas):
    for questions, _ in qas:
        for q in questions:
            yield q


def get_all_answers(qas):
    for _, answers in qas:
        for a in answers:
            yield a


def unique(l):
    l = list(l)
    l = sorted(l)
    r = []
    i = None
    for k in l:
        if i != k:
            i = k
            r.append(k)
    return r


def get_options_combinations(options):
    return [{key: value for (key, value) in zip(options, values)} for values in it.product(*options.values())]


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


def klp_kmeans(data, cluster_num, alpha, epochs=-1, batch=1, verbose=False, use_gpu=False):
    '''
        Theano based implementation, likely to use GPU as well with required Theano
        configurations. Refer to http://deeplearning.net/software/theano/tutorial/using_gpu.html
        for GPU settings
        Inputs:
            data - [instances x variables] matrix of the data.
            cluster_num - number of requisite clusters
            alpha - learning rate
            epoch - how many epoch you want to go on clustering. If not given, it is set with
                Kohonen's suggestion 500 * #instances
            batch - batch size. Larger batch size is better for Theano and GPU utilization
            verbose - True if you want to verbose the algorithm's iterations
        Output:
            W - final cluster centroids
    '''
    if use_gpu:
        config.floatX = 'float32'  # Theano needs this type of data for GPU use

    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.filterwarnings("ignore")

    rng = np.random
    # From Kohonen's paper
    if epochs == -1:
        print(data.shape[0])
        epochs = 500 * data.shape[0]

    if use_gpu == False:
        # Symmbol variables
        X = T.dmatrix('X')
        WIN = T.dmatrix('WIN')

        # Init weights random
        W = theano.shared(rng.randn(cluster_num, data.shape[1]), name="W")
    else:
        # for GPU use
        X = T.matrix('X')
        WIN = T.matrix('WIN')
        W = theano.shared(rng.randn(cluster_num, data.shape[1]).astype(theano.config.floatX), name="W")

    W_old = W.get_value()

    # Find winner unit
    bmu = ((W ** 2).sum(axis=1, keepdims=True) + (X ** 2).sum(axis=1, keepdims=True).T - 2 * T.dot(W, X.T)).argmin(
        axis=0)
    dist = T.dot(WIN.T, X) - WIN.sum(0)[:, None] * W
    err = abs(dist).sum() / X.shape[0]

    update = function([X, WIN], outputs=err, updates=[(W, W + alpha * dist)], allow_input_downcast=True)
    find_bmu = function([X], bmu, allow_input_downcast=True)

    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            update.maker.fgraph.toposort()]):
        print('Used the cpu')

    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              update.maker.fgraph.toposort()]):
        print('Used the gpu')
    else:
        print('ERROR, not able to tell if theano used the cpu or the gpu')
        print(update.maker.fgraph.toposort())

    # Update
    for epoch in range(epochs):
        C = 0
        for i in range(0, data.shape[0], batch):
            batch_data = data[i:i + batch, :]
            D = find_bmu(batch_data)
            # for GPU use
            if use_gpu:
                S = np.zeros([batch, cluster_num], config.floatX)
            else:
                S = np.zeros([batch_data.shape[0], cluster_num])
            S[:, D] = 1
            cost = update(batch_data, S)

        if verbose:
            print("Avg. centroid distance -- ", cost.sum(), "\t EPOCH : ", epoch)

    return W.get_value()