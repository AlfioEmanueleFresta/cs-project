from unittest import TestCase

from project.expansion import WangExpander

import numpy as np


class TestWangExpander(TestCase):

    def test_next_no_minimum_distance(self):

        # WangExpander w/ n=3
        e = WangExpander(3, 0)

        alpha = np.array([1, 2, 3])
        bravo = np.array([-1, -2, -3])
        charlie = np.array([0, 1, 4])
        delta = np.array([0, 1, -1])

        alpha_bravo = alpha + bravo
        bravo_charlie = bravo + charlie
        charlie_delta = charlie + delta
        alpha_bravo_charlie = alpha + bravo + charlie
        bravo_charlie_delta = charlie + bravo + delta

        # (1) alpha   -> alpha
        o1 = e.next(alpha)
        self.assertTrue(np.array_equal(o1, np.array([alpha])))

        # (2) bravo   -> bravo, alpha + bravo
        o2 = e.next(bravo)
        self.assertTrue(np.array_equal(o2, np.array([bravo, alpha_bravo])))

        # (3) charlie -> charlie, bravo + charlie, alpha + bravo + charlie
        o3 = e.next(charlie)
        self.assertTrue(np.array_equal(o3, np.array([charlie, bravo_charlie, alpha_bravo_charlie])))

        # (4) delta   -> delta, delta + charlie, delta + charlie + bravo
        o4 = e.next(delta)
        self.assertTrue(np.array_equal(o4, np.array([delta, charlie_delta, bravo_charlie_delta])))

    def test__get_combinations(self):
        e = WangExpander(3, 0)
        e.history = ["a", "b", "c"]
        self.assertEqual(list(e._get_combinations()),
                         [["c"], ["b", "c"], ["a", "b", "c"]])

    def test__combine_combinations(self):
        e = WangExpander(3, 0)
        combinations = [[10], [5, 10], [2, 5, 10]]
        combined = e._combine_combinations(combinations)
        combined = list(combined)
        self.assertEqual(combined, [10, 15, 17])

    def test__filter_combinations(self):
        center = np.array([0, 0])

        far1 = np.array([2, 2])
        far2 = np.array([-2, -2])
        borderline = np.array([1, 0])
        close1 = np.array([.99, 0])
        close2 = np.array([0, .5])

        e = WangExpander(4, 1)
        combinations = [far1, center, close1, far2, close2, borderline]
        filtered = e._filter_combinations(center, combinations)
        filtered = list(filtered)

        print(filtered)
        self.assertTrue(np.array_equal(np.array(filtered),
                                       np.array([far1, center, far2, borderline])))

