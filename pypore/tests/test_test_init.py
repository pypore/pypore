"""
Test the functions in :py:mod:`pypore.tests.__init__`.
"""
import unittest

from pypore.tests import pct_diff


class TestPctDiff(unittest.TestCase):
    """
    Tests that the percent difference function works correctly.
    """

    def test_pct_diff(self):
        self.assertEqual(pct_diff(-100.33, 44.2), 144.055)

    def test_integers(self):
        """
        Tests that integers are treated as floats, no integer division.
        """
        self.assertEqual(pct_diff(3, 5), 200. / 3.)
