import tempfile
from unittest import TestCase

from project.data import Dataset


class TestTrainingData(TestCase):

    def test__load(self):

        temp_filename = tempfile.mkstemp()[1]

        example_input = [
            "",
            "# Ignore me",
            "Q: Question-A1",
            "Q: Question-A2",
            "A: Answer-A1",
            "# This is a comment",
            "A: Answer-A2",
            "A: Answer-A3",
            "",
            "",
            "# Comment, once again",
            "",
            "Q: Question-B",
            "# This is a comment",
            "A: Answer-B1",
            "A: Answer-B2",
            "",
        ]
        example_input = "\n".join(example_input)

        with open(temp_filename, "wt") as f:
            f.write(example_input)

        expected = [
            ("Question-A1", ("Answer-A1", "Answer-A2", "Answer-A3")),
            ("Question-A2", ("Answer-A1", "Answer-A2", "Answer-A3")),
            ("Question-B", ("Answer-B1", "Answer-B2")),
        ]

        d = Dataset(temp_filename)
        for (obtained_question, obtained_answer_id), (expected_question, expected_answers) in zip(d, expected):
            self.assertEqual(obtained_question, expected_question)
            self.assertEqual(d.get_answer(obtained_answer_id), expected_answers)

