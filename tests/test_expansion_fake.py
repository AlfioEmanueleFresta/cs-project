from unittest import TestCase
from project.expansion import FakeExpander

import numpy as np


class TestFakeExpander(TestCase):

    def test_next(self):

        alpha = np.array([1, 2, 3])

        expander = FakeExpander()

        output = expander.next(alpha)
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(np.array_equal(output, np.array([alpha])))


