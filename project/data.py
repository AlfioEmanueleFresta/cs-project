from ordered_set import OrderedSet
import gzip

import numpy as np

from project.augmentation import FakeAugmenter
from project.helpers import shuffle, split, one_hot_encode


class Dataset:
    """
    Represent an data set.
    """

    def __init__(self, word_embedding, filename, augmenter=None, reduce=1.0):
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
        assert 1 >= reduce > 0
        if reduce < 1:
            self._reduce(ratio=reduce)

    def _reduce(self, ratio):
        total = len(self.questions)
        to_remove = int(total * (1 - ratio))
        self.questions = shuffle(self.questions)
        self.questions = self.questions[to_remove:]

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
        raise NotImplementedError

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
            self.verbose and print("> ORIG", question, answer_id)
            alternative_question = self.augmenter.next(question)
            self.verbose and print("  EXPA", alternative_question, answer_id)
            yield alternative_question, answer_id

    def _replace_words_with_vectors(self, group):
        for question, answer_id in group:
            self.verbose and print("  WOEM", end=" ")
            # TODO nicer!
            question = [tuple(self.word_embedding.get_word_vector(word) for word in words_tuple
                              if self.word_embedding.get_word_vector(word, verbose=self.verbose) is not None,)
                        for words_tuple in question]
            question = [t for t in question if t]  # Remove (,) tuples
            question.append((self.word_embedding.get_word_vector(self.word_embedding.SYM_END),))
            yield question, answer_id

    def _sum_vector_groups(self, group):
        for question, answer_id in group:
            question = [np.sum(list(vectors_tuple), axis=0)
                        if len(vectors_tuple) > 1 else vectors_tuple[0]
                        for vectors_tuple in question]
            yield question, answer_id

    def _one_hot_answers(self, group, output_categories_no):
        for question, answer_id in group:
            assert answer_id < output_categories_no  # output_categories_no too small
            answer = one_hot_encode(i=answer_id, n=output_categories_no,
                                    positive=1.0, negative=0.0)
            answer = np.array(answer)
            yield question, answer

    def _make_sentence_matrices(self, group, max_sentence_size):
        for question, answer_id in group:
            question_matrix = np.tile(self.word_embedding.empty_vector(), (max_sentence_size or len(question), 1))
            assert len(question) <= question_matrix.shape[0]  # otherwise, max_sentence_size is too little.
            for i, vector in enumerate(question):
                question_matrix[i] = vector
            self.verbose and print("  MATR", question_matrix.shape)
            yield question_matrix, answer_id

    def _get_mask_for_question_and_answer(self, question, answer):
        mask = np.zeros((question.shape[0],))
        i = 0
        for element in question:
            mask[i] = 1
            if np.array_equal(element, self.word_embedding.vectors[self.word_embedding.SYM_END]):
                break
            i += 1
        return mask

    def _insert_mask_in_tuple(self, group):
        for question, answer in group:
            new_tuple = question, self._get_mask_for_question_and_answer(question, answer), answer
            yield new_tuple

    def get_prepared_data(self,
                          train_data_percentage,
                          output_categories_no=None,
                          max_sentence_size=200,
                          train_data_shuffle=True,
                          verbose=False):
        """
        Shuffle and split data into training, validation and test sets.
        :return:
        """

        questions = self._get_prepared_questions()
        questions = list(questions)

        output_categories_no = output_categories_no or len(self.answers)
        if output_categories_no < len(self.answers):
            raise ValueError("`output_categories_no` chosen is too small for the dataset provided.")

        if train_data_shuffle:
            questions = shuffle(questions)

        training, validation_and_test = split(questions, train_data_percentage)
        validation, test = split(validation_and_test, .5)

        print(len(training), len(validation), len(test))

        self.verbose = verbose

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

        training, validation, test = self._one_hot_answers(training, output_categories_no), \
                                     self._one_hot_answers(validation, output_categories_no), \
                                     self._one_hot_answers(test, output_categories_no)

        training, validation, test = self._insert_mask_in_tuple(training), \
                                     self._insert_mask_in_tuple(validation), \
                                     self._insert_mask_in_tuple(test)

        return training, validation, test


class FileDataset(Dataset):

    def _load(self):

        open_function = gzip.open if '.gz' in self.filename else open
        with open_function(self.filename, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

        self._load_from_lines(lines)


    def _load_from_lines(self, lines):

        question_delimiter = 'Q: '
        answer_delimiter = 'A: '
        comment_delimiter = '#'

        questions, answers = [], []
        line_no = 0
        for line in lines:
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


class MemoryDataset(FileDataset):

    def _load(self):
        lines = self.filename.split("\n")
        self._load_from_lines(lines)