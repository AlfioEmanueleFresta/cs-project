from ordered_set import OrderedSet
import gzip


class Dataset:
    """
    Represent an data set.
    """

    def __init__(self, filename):
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
        self.answers = OrderedSet([])
        self.questions = OrderedSet([])
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
