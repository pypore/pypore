import unittest

import numpy as np

from pypore.core import Segment
from pypore.tests.segment_tests import *


class TestSegment(unittest.TestCase, SegmentTests):
    """
    Tests for Segment
    """

    SEGMENT_CLASS = Segment

    def setUp(self):
        self.default_test_data = []

        for data in [np.random.random(100), np.zeros(500), [1, 2, 3, 4, 5, 6]]:
            self.default_test_data.append(
                SegmentTestData(data, np.max(data), np.mean(data), np.min(data), np.shape(data), np.size(data),
                                np.std(data), 1.e6))

    def test_slicing_numpy_array(self):
        """
        Tests that slicing a Segment holding numpy array data works well.
        """
        array = np.random.random(100)

        s = Segment(array)

        # use np.array(s[slice]) so np.testing comparisons work.

        # Regular slices
        np.testing.assert_array_equal(array, np.array(s))
        np.testing.assert_array_equal(array[:], np.array(s[:]))
        np.testing.assert_array_equal(array[:19], np.array(s[:19]))
        np.testing.assert_array_equal(array[55:], np.array(s[55:]))
        np.testing.assert_array_equal(array[1:5], np.array(s[1:5]))
        np.testing.assert_array_equal(array[5:100], np.array(s[5:100]))

        # steps
        np.testing.assert_array_equal(array[::3], np.array(s[::3]))
        np.testing.assert_array_equal(array[::-3], np.array(s[::-3]))

        # Negative indices
        np.testing.assert_array_equal(array[-10:], np.array(s[-10:]))
        np.testing.assert_array_equal(array[:-10], np.array(s[:-10]))
        np.testing.assert_array_equal(array[-95:-5], np.array(s[-95:-5]))

        # Slicing slices
        np.testing.assert_array_equal(array[:][:], np.array(s[:][:]))
        np.testing.assert_array_equal(array[:19][:19], np.array(s[:19][:19]))
        np.testing.assert_array_equal(array[1:15][:10:2], np.array(s[1:15][:10:2]))
        np.testing.assert_array_equal(array[-1:][:], np.array(s[-1:][:]))
        np.testing.assert_array_equal(array[:][:][:], np.array(s[:][:][:]))

    def test_list(self):
        """
        Tests that we can pass in a list to Segment.
        """
        l = [1, 2, 3, 4, 5, 6]

        s = Segment(l)

        self.assertEqual(len(l), len(s))
        self.assertEqual(len(l), s.size)

        # TODO add tests for slicing list, mean, max, etc.

    def test_slice_attributes(self):
        """
        Tests that a sliced Segment has the correct attributes/method returns, like max, min etc.
        """
        array = np.random.random(100)

        sample_rate = 1.e6
        s = Segment(array, sample_rate)

        array_slices = [array[:], array[:50], array[:75][20:]]
        s_slices = [s[:], s[:50], s[:75][20:]]

        for i, array_slice in enumerate(array_slices):
            s_slice = s_slices[i]

            self.assertEqual(array_slice.max(), s_slice.max())
            self.assertEqual(array_slice.mean(), s_slice.mean())
            self.assertEqual(array_slice.min(), s_slice.min())

            self.assertEqual(sample_rate, s_slice.sample_rate)
            self.assertEqual(array_slice.size, s_slice.size)
            self.assertEqual(len(array_slice), len(s_slice))

    def test_sample_rate(self):
        """
        Tests that the sample rate is a named argument. Tests that sample_rate can be None.
        """
        # Test that sample_rate is initialized to zero.
        array = np.random.random(100)

        s = Segment(array)

        self.assertEqual(s.sample_rate, 0.0, "Segment without a sample rate should have the sample rate set to zero.")

        # Test setting the sample_rate.
        sample_rate = 1.2343e9
        s2 = Segment(array, sample_rate=sample_rate)

        self.assertEqual(sample_rate, s2.sample_rate, "Segment's sample_rate incorrect. Should be {0}. Was {"
                                                      "1}".format(sample_rate, s2.sample_rate))
