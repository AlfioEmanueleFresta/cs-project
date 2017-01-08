from ordered_set import OrderedSet
import gzip

import numpy as np

from project.augmentation import FakeAugmenter
from project.helpers import shuffle, split


class Dataset:
    """
    Represent an data set.
    """

    def __init__(self, word_embedding, filename, augmenter=None):
        """
        Load a data set from a file.

        A file is a number of blocks separated by white lines.
        Each block will contain one or more questions, followed by one or more possible answers.
        The file can contain comments that will be ignored during parsing.

        Question lines start with   'Q: '
        Answer lines start with     'A: '
        Comment lines start with    '# '

        New lines must be in the UNIX format -- ie. "\n" only.

        If the name of the file to read ends in .gz, it will be decompressed using gzip.
        Otherwise it will be read as a UTF-8 encoded text file.

        :param filename: A text filename.
        """
        self.filename = filename
        self.word_embedding = word_embedding
        self.answers = OrderedSet([])
        self.questions = OrderedSet([])
        self.augmenter = augmenter or FakeAugmenter()
        self._load()

    def _add_answer(self, answer_list):
        answer_list = tuple(answer_list)
        return self.answers.add(answer_list)

    def _add_question(self, question, answer_id):
        return self.questions.add((question, answer_id))

    def _add_questions(self, questions, answer_id):
        return [self._add_question(question, answer_id) for question in questions]

    def get_answer(self, answer_id):
        """
        Get the answer set for a given answer ID.

        :param answer_id: A numeric ID.
        :return: A tuple containing alternative answers for the question.
        """
        return self.answers[answer_id]

    def _load(self):
        question_delimiter = 'Q: '
        answer_delimiter = 'A: '
        comment_delimiter = '#'

        open_function = gzip.open if '.gz' in self.filename else open
        with open_function(self.filename, 'rt', encoding='utf-8') as f:
            questions, answers = [], []

            line_no = 0
            for line in f.readlines():
                line = line.rstrip('\n').strip()
                line_no += 1

                if line.startswith(question_delimiter):
                    question = line[len(question_delimiter) - 1:].strip()
                    questions.append(question)

                elif line.startswith(answer_delimiter):
                    answer = line[len(answer_delimiter) - 1:].strip()
                    answers.append(answer)

                elif line.startswith(comment_delimiter):
                    continue

                elif not line:
                    if (questions and not answers) or (not questions and answers):
                        raise ValueError("Group terminating at line %d has questions "
                                         "but no answers, or vice versa." % line_no)

                    if questions and answers:
                        answer_id = self._add_answer(answers)
                        self._add_questions(questions, answer_id)

                    questions, answers = [], []

                else:
                    raise ValueError("Error in line %d: '%s'." % (line_no, line))

        if questions and answers:
            answer_id = self._add_answer(answers)
            self._add_questions(questions, answer_id)

    def __iter__(self):
        for question, answer_id in self.questions:
            yield question, answer_id

    def __repr__(self):
        return "<TrainingData (%s): %d questions with %d answers.>" % (self.filename,
                                                                       len(self.questions), len(self.answers))

    def _process_question(self, question):
        sentence = self.word_embedding.split_sentence_into_words(question)
        return sentence

    def _get_prepared_questions(self):
        for question, answer_id in self.questions:
            yield self._process_question(question), answer_id

    #def _augment_group_with_alternative_questions(self, group):
    #    for question, answer_id in group:
    #        print(">", question, answer_id)
    #        for alternative_question in self.augmenter.next(question):
    #            print(" ", alternative_question, answer_id)
    #            yield alternative_question, answer_id

    def _augment_questions_in_group(self, group):
        for question, answer_id in group:
            print(":", question, answer_id)
            alternative_question = self.augmenter.next(question)
            print(" ", alternative_question, answer_id)
            yield alternative_question, answer_id

    def _replace_words_with_vectors(self, group):
        for question, answer_id in group:
            question = [tuple(self.word_embedding.get_word_vector(word) for word in words_tuple
                              if self.word_embedding.get_word_vector(word) is not None,)
                        for words_tuple in question]
            question = [t for t in question if t]  # Remove (,) tuples
            yield question, answer_id

    def _sum_vector_groups(self, group):
        for question, answer_id in group:
            question = [np.sum(list(vectors_tuple), axis=0)
                        if len(vectors_tuple) > 0 else vectors_tuple[0]
                        for vectors_tuple in question]
            yield question, answer_id

    def _make_sentence_matrices(self, group, max_sentence_size):
        for question, answer_id in group:
            question_matrix = np.tile(self.word_embedding.empty_vector(), (max_sentence_size or len(question), 1))
            assert len(question) <= question_matrix.shape[0]  # otherwise, max_sentence_size is too little.
            for i, vector in enumerate(question):
                question_matrix[i] = vector
            yield question_matrix, answer_id

    def get_prepared_data(self,
                          train_data_percentage,
                          max_sentence_size=200,
                          do_shuffle=True):
        """
        Shuffle and split data into training, validation and test sets.
        :return:
        """

        questions = self._get_prepared_questions()
        questions = list(questions)

        if do_shuffle:
            questions = shuffle(questions)

        training, validation_and_test = split(questions, train_data_percentage)
        validation, test = split(validation_and_test, .5)

        training, validation, test = self._augment_questions_in_group(training), \
                                     self._augment_questions_in_group(validation), \
                                     self._augment_questions_in_group(test)

        training, validation, test = self._replace_words_with_vectors(training), \
                                     self._replace_words_with_vectors(validation), \
                                     self._replace_words_with_vectors(test)

        training, validation, test = self._sum_vector_groups(training), \
                                     self._sum_vector_groups(validation), \
                                     self._sum_vector_groups(test)

        training, validation, test = self._make_sentence_matrices(training, max_sentence_size=max_sentence_size), \
                                     self._make_sentence_matrices(validation, max_sentence_size=max_sentence_size), \
                                     self._make_sentence_matrices(test, max_sentence_size=max_sentence_size)

        return training, validation, test

