import numpy as np
import itertools as it


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
