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
        self.assertAlmostEqual(pct_diff(-100.33, 44.2), 144.054619755)

    def test_integers(self):
        """
        Tests that integers are treated as floats, no integer division.
        """
        self.assertAlmostEqual(pct_diff(3, 5), 200. / 3.)

    def test_zero(self):
        """
        Tests that function handles zeros correctly.
        """
        self.assertEqual(pct_diff(0., 0.), 0.)

        self.assertAlmostEqual(pct_diff(1., 0.), 100.)
        self.assertAlmostEqual(pct_diff(0., 1.), 100.)
