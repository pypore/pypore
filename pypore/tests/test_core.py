import unittest

import numpy as np

from pypore.core import Segment


class TestSegment(unittest.TestCase):
    """
    Tests for Segment
    """

    def test_slice_type(self):
        """
        Tests that slicing a Segment returns a Segment and getting a single index from a Segment is not a Segment.
        """
        array = np.random.random(10)

        s = Segment(array)
        self.assertTrue(isinstance(s, Segment))

        s2 = s[1:5]
        self.assertTrue(isinstance(s2, Segment), "Slice of Segment object did not return a Segment object.")

        s3 = s[1]
        self.assertFalse(isinstance(s3, Segment), "Single index of Segment should not be a Segment object.")

    def test_slicing_numpy_array(self):
        """
        Tests that slicing a Segment holding numpy array data works well.
        """
        array = np.random.random(100)

        s = Segment(array)

        # use np.array(s[slice]) so np.testing comparisons work.

        # Regular slices
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

    def test_mean(self):
        """
        Tests that the mean method works, that before it's called the _mean field is None, and after it's called the
        _mean field is the mean.
        """
        array = np.random.random(100)

        s = Segment(array)

        self.assertTrue(s._mean is None, "The _mean field of Segment should be None before the user requests the mean.")

        mean_should_be = array.mean()

        mean_was = s.mean()

        self.assertEqual(mean_should_be, mean_was, "Mean of Segment incorrect. Should be {0}. Was {1}.".format(
            mean_should_be, mean_was))

        # Make sure s._mean has been set.
        self.assertEqual(s._mean, mean_should_be, "Segment._mean should be set after the user calls .mean().")

        mean_was = s.mean()

        # Check the mean again, just to be safe
        self.assertEqual(mean_should_be, mean_was, "Mean of Segment incorrect on second try. Should be {0}. Was {"
                                                   "1}.".format(mean_should_be, mean_was))

    def test_min(self):
        """
        Tests that the min method works, that before it's called the _min field is None, and after it's called the
        _min field is the min of the segment.
        """
        array = np.random.random(100)

        s = Segment(array)

        self.assertTrue(s._min is None, "The _min field of Segment should be None before the user requests the min.")

        min_should_be = array.min()

        min_was = s.min()

        self.assertEqual(min_should_be, min_was, "Min of Segment incorrect. Should be {0}. Was {1}.".format(
            min_should_be, min_was))

        # Make sure s._min has been set.
        self.assertEqual(s._min, min_should_be, "Segment._min should be set after the user calls .min().")

        min_was = s.min()

        # Check the min again, just to be safe
        self.assertEqual(min_should_be, min_was, "Min of Segment incorrect on second try. Should be {0}. Was {"
                                                 "1}.".format(min_should_be, min_was))

    def test_max(self):
        """
        Tests that the max method works, that before it's called the _max field is None, and after it's called the
        _max field is the max of the segment.
        """
        array = np.random.random(100)

        s = Segment(array)

        self.assertTrue(s._max is None, "The _max field of Segment should be None before the user requests the max.")

        max_should_be = array.max()

        max_was = s.max()

        self.assertEqual(max_should_be, max_was, "Max of Segment incorrect. Should be {0}. Was {1}.".format(
            max_should_be, max_was))

        # Make sure s._max has been set.
        self.assertEqual(s._max, max_should_be, "Segment._max should be set after the user calls .max().")

        max_was = s.max()

        # Check the max again, just to be safe
        self.assertEqual(max_should_be, max_was, "Max of Segment incorrect on second try. Should be {0}. Was {"
                                                 "1}.".format(max_should_be, max_was))

    def test_size(self):
        """
        Test that size returns the correct size of the Segment.
        """
        array = np.random.random(100)

        s = Segment(array)

        self.assertEqual(array.size, s.size, "Segment size wrong. Should be {0}. Was {1}.".format(array.size, s.size))

    def test_len(self):
        """
        Tests that we can use len(segment) the same as segment.size.
        """
        array = np.random.random(100)

        s = Segment(array)

        self.assertEqual(len(array), len(s), "Segment length wrong. Should be {0}. Was {1}.".format(len(array),
                                                                                                    len(s)))
        self.assertEqual(len(s), s.size, "Segment's length and size differ. Length {0}. Size {1}.".format(len(s),
                                                                                                          s.size))

    def test_sample_rate(self):
        """
        Tests that the sample rate is set as the second argument. Tests that sample_rate can be None.
        """
        # Test that sample_rate is initialized to zero.
        array = np.random.random(100)

        s = Segment(array)

        self.assertEqual(s.sample_rate, 0.0, "Segment without a sample rate should have the sample rate set to zero.")

        # Test setting the sample_rate.
        sample_rate = 1.e6
        s2 = Segment(array, sample_rate)

        self.assertEqual(sample_rate, s2.sample_rate, "Segment's sample_rate incorrect. Should be {0}. Was {"
                                                      "1}".format(sample_rate, s2.sample_rate))
