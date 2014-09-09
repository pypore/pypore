import numpy as np

from pypore.tests import pct_diff


class SegmentTestData(object):
    """
    Data object holding sample data and corresponding attributes so SegmentTests can use it.
    """

    def __init__(self, data, maximum, mean, minimum, shape, size, std, sample_rate=None):
        """
        :param data: Your test data for your Segment subclass. For example, a numpy array.
        :param maximum: The maximum value in your test data.
        :param mean: The mean of your test data.
        :param minimum: The minimum value in your test data.
        :param shape: The shape of your test data.
        :param size: The number of elements in your test data.
        :param std: The standard deviation of your test data.
        :param sample_rate: (Optional) The sampling rate of your test data. Default is 0.0 Hz.
        """
        self.data = data
        self.max = maximum
        self.mean = mean
        self.min = minimum
        self.shape = shape
        self.size = size
        self.std = std

        if sample_rate is None:
            sample_rate = 0.0
        self.sample_rate = sample_rate


class SegmentTests(object):
    """
    General tests that every Segment object and subclass should inherit.

    If you are writing tests for your new type of Segment, inherit SegmentTests as follows:

    >>> from pypore.tests.segment_tests import *
    >>> import unittest
    >>> import numpy as np
    >>> from my_new_segment_module import MyNewSegment # Your Segment subclass
    >>> arr1 = np.random.random(100)
    >>> test_data = SegmentTestData(data=arr1, maximum=np.max(arr1), mean = np.mean(arr1), minimum=np.min(arr1),
    >>>   shape=arr1.shape, size=arr1.size, std=np.std(arr1), sample_rate=1.e6)
    >>> class TestMyNewSegment(unittest.TestCase, SegmentTests):
    >>>   SEGMENT_CLASS = MyNewSegment # you must set this to your Segment subclass
    >>>   default_test_data = [test_data] # set your simple test data as a list.

    Then, all of the unit tests in SegmentTests will also be run in your TestMyNewSegment class. Then, you can add
    more specific test cases to TestMyNewSegment without rewriting all of the generic Segment test cases.

    Examples of subclassing :py:class:`pypore.tests.segment_tests.SegmentTests` can be found in
    :py:class:`pypore.tests.test_core.TestSegment`.
    """

    # The Segment subclass that should be tested by SegmentTests. Subclasses of SegmentTests must set this field.
    SEGMENT_CLASS = None

    # A list of SegmentTestData objects that SegmentTests can use to test your Segment subclass. Subclasses must set
    # this.
    default_test_data = None

    def test_segment_class_is_not_none(self):
        """
        Test that the test subclass has set SEGMENT_CLASS.
        """
        self.assertFalse(self.SEGMENT_CLASS is None,
                         "Subclass {0} of SegmentTests must set self.SEGMENT_CLASS to the Segment subclass they are "
                         "trying to test.".format(
                             self.__class__.__name__))

    def test_default_test_data_is_set(self):
        """
        Test that the test subclass has filled self.default_test_data with a list of test data.
        """
        self.assertFalse(self.default_test_data is None,
                         "Subclass {0} of SegmentTests must set self.default_test_data to a list of simplified test "
                         "data that SegmentTests can use.".format(self.__class__.__name__))
        self.assertTrue(len(self.default_test_data) > 0,
                        "Subclass {0} of SegmentTests must set self.default_test_data to a list of simplified test "
                        "data that SegmentTests can use. This list must contain at least one element of data, "
                        "for example a list of one numpy array.".format(
                            self.__class__.__name__))

    def test_slice_type(self):
        """
        Tests that slicing a Segment returns a Segment and getting a single index from a Segment is not a Segment.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertTrue(isinstance(s, self.SEGMENT_CLASS))

            s2 = s[1:5]
            self.assertTrue(isinstance(s2, self.SEGMENT_CLASS),
                            "Slice of {0} object did not return a {0} object.".format(
                                self.SEGMENT_CLASS.__name__))

            s3 = s[1]
            self.assertFalse(isinstance(s3, self.SEGMENT_CLASS),
                             "Single index of {0} should not be a {0} object.".format(self.SEGMENT_CLASS.__name__))

    def test_slicing(self):
        """
        Tests that slicing works as expected.
        """

        for test_data in self.default_test_data:
            segment = self.SEGMENT_CLASS(test_data.data)

            # get all the data first, to compare the slicing.
            # the idea is that if test_get_all_data works, this should work.
            data = segment[:]

            # check starting points
            np.testing.assert_array_equal(data[0:], segment[0:])
            np.testing.assert_array_equal(data[3:], segment[3:])
            np.testing.assert_array_equal(data[5:], segment[5:])

            # check stopping points
            np.testing.assert_array_equal(data[:0], segment[:0])
            np.testing.assert_array_equal(data[:3], segment[:3])
            np.testing.assert_array_equal(data[:5], segment[:5])

            # check steps
            np.testing.assert_array_equal(data[::2], segment[::2])
            np.testing.assert_array_equal(data[::3], segment[::3])
            np.testing.assert_array_equal(data[::5], segment[::5])

            # check indices
            self.assertEqual(data[0], segment[0])
            self.assertEqual(data[1], segment[1])
            self.assertEqual(data[-1], segment[-1])
            self.assertEqual(data[5], segment[5])

            # check negative step sizes
            np.testing.assert_array_equal(data[::-1], segment[::-1])
            np.testing.assert_array_equal(data[::-2], segment[::-2])
            np.testing.assert_array_equal(data[::-3], segment[::-3])
            np.testing.assert_array_equal(data[::-5], segment[::-5])

            # check the whole thing
            np.testing.assert_array_equal(data[0:8:2], segment[0:8:2])
            np.testing.assert_array_equal(data[1:9:3], segment[1:9:3])
            np.testing.assert_array_equal(data[-8:-1:-1], segment[-8:-1:-1])
            np.testing.assert_array_equal(data[9:1:-1], segment[9:1:-1])
            np.testing.assert_array_equal(data[9:1:-4], segment[9:1:-4])

            try:
                segment.close()
            except(AttributeError):
                pass

    def test_slice_attributes(self):
        """
        Tests that a sliced Segment has the correct attributes/method returns, like max, min etc.
        """
        for test_data in self.default_test_data:
            if isinstance(test_data.data, str):
                # If test_data.data is a string, it is likely a filename, so we will not
                # set the sample rate.
                s = self.SEGMENT_CLASS(test_data.data)
            else:
                s = self.SEGMENT_CLASS(test_data.data, test_data.sample_rate)

            s_slices = [s[:], s[:10], s[:][:10], s[-50:]]

            for i, s_slice in enumerate(s_slices):
                arr = np.array(s_slice)

                self.assertAlmostEqual(arr.max(), s_slice.max(),
                                       msg="Max failed for class {0}. Should be {1}. Was {2}.".format(
                                           self.SEGMENT_CLASS.__name__, arr.max(), s_slice.max()))
                self.assertAlmostEqual(arr.mean(), s_slice.mean(),
                                       msg="Mean failed for class {0}. Should be {1}. Was {2}.".format(
                                           self.SEGMENT_CLASS.__name__, arr.mean(), s_slice.mean()))
                self.assertAlmostEqual(arr.min(), s_slice.min(),
                                       msg="Min failed for class {0}. Should be {1}. Was {2}.".format(
                                           self.SEGMENT_CLASS.__name__, arr.min(), s_slice.min()))

                self.assertAlmostEqual(test_data.sample_rate, s_slice.sample_rate,
                                       msg="Sample rate failed for class {0}. Should be {1}. Was {2}.".format(
                                           self.SEGMENT_CLASS.__name__, test_data.sample_rate, s_slice.sample_rate))
                self.assertEqual(arr.size, s_slice.size,
                                 msg="Size failed for class {0}. Should be {1}. Was {2}.".format(
                                     self.SEGMENT_CLASS.__name__, arr.size, s_slice.size))
                self.assertEqual(len(arr), len(s_slice),
                                 msg="Len failed for class {0}. Should be {1}. Was {2}.".format(
                                     self.SEGMENT_CLASS.__name__, len(arr), len(s_slice)))
                self.assertEqual(arr.shape, s_slice.shape,
                                 msg="Shape failed for class {0}. Should be {1}. Was {2}.".format(
                                     self.SEGMENT_CLASS.__name__, arr.shape, s_slice.shape))

    def test_convert_numpy_array(self):
        """
        Tests that we can convert Segment to a numpy array.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            arr = np.array(s)

            self.assertEqual(s.size, arr.size)
            self.assertEqual(arr.size, test_data.size)

            self.assertEqual(s.shape, arr.shape)
            self.assertEqual(arr.shape, test_data.shape)

            np.testing.assert_array_equal(arr, s)

    def test_iterable(self):
        """
        Tests that the object is iterable.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            # create acutal data array
            data_should_be = np.array(s)
            # create empty array
            data = np.empty(s.size)

            count = 0
            i = 0
            for i, point in enumerate(s):
                data[i] = point
                count += 1
            np.testing.assert_array_equal(data, data_should_be,
                                          "Iterated arrays did not match for class {0}. Should be {1}. Was {2}.".format(
                                              self.SEGMENT_CLASS.__name__, data_should_be, data))

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, test_data.size,
                             "enumerate(segment) did not loop through all elements. It looped through {0}/{1} "
                             "elements.".format(count, test_data.size))
            self.assertEqual(i, test_data.size - 1)

            # Make sure we can loop through a second time with the same results.

            del data
            data = np.empty(s.size)

            count = 0
            i = 0
            for i, point in enumerate(s):
                data[i] = point
                count += 1
            np.testing.assert_array_equal(data, data_should_be,
                                          "On the second pass, iterated arrays did not match for class {0}. Should be {"
                                          "1}. Was {2}.".format(self.SEGMENT_CLASS.__name__, data_should_be, data))

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, test_data.size,
                             "enumerate(segment) did not loop through all elements on the second try. It looped "
                             "through {0}/{1} elements.".format(count, test_data.size))
            self.assertEqual(i, test_data.size - 1)

            # Try iterating a slice of data
            s2 = s[:100]

            data_should_be = np.array(s2)
            del data
            data = np.empty(s2.size)

            print("class: {0}".format(self.SEGMENT_CLASS.__name__))

            count = 0
            i = 0
            for i, point in enumerate(s2):
                data[i] = point
                count += 1
            np.testing.assert_array_equal(data, data_should_be,
                                          "For a sliced Segment, iterated arrays did not match for class {0}. Should "
                                          "be {1}. Was {2}.".format(self.SEGMENT_CLASS.__name__, data_should_be, data))

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, data_should_be.size,
                             "enumerate(segment) did not loop through all elements for a sliced Segment. It looped "
                             "through {0}/{1} elements.".format(count, data_should_be.size))
            self.assertEqual(i, data_should_be.size - 1)

            # Test iterator with negative slice
            s3 = s[-100:]

            data_should_be = np.array(s3)
            del data
            data = np.empty(s3.size)

            print("class: {0}".format(self.SEGMENT_CLASS.__name__))

            count = 0
            i = 0
            for i, point in enumerate(s3):
                data[i] = point
                count += 1
            np.testing.assert_array_equal(data, data_should_be,
                                          "For a negatively sliced Segment, iterated arrays did not match for class {"
                                          "0}. Should "
                                          "be {1}. Was {2}.".format(self.SEGMENT_CLASS.__name__, data_should_be, data))

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, data_should_be.size,
                             "enumerate(segment) did not loop through all elements for a negatively sliced Segment. It "
                             "looped "
                             "through {0}/{1} elements.".format(count, data_should_be.size))
            self.assertEqual(i, data_should_be.size - 1)


    def test_len(self):
        """
        Tests that we can use len(segment) the same as segment.size.
        """
        for data in self.default_test_data:
            s = self.SEGMENT_CLASS(data.data)

            arr = np.array(s)

            self.assertEqual(len(s), len(arr),
                             "Segment's length and size differ for Segment class {0}. Length {1}. Size {2}.".format(
                                 self.SEGMENT_CLASS.__name__, len(s), data.size))

            # Test that the len of a slice is correct
            slice_length = 50
            if len(s) < slice_length:
                len_should_be = len(s)
            else:
                len_should_be = slice_length
            s2 = s[:slice_length]

            self.assertEqual(len(s2), len_should_be,
                             "Segment's length and size differ for Segment class {0}. Length {1}. Size {2}.".format(
                                 self.SEGMENT_CLASS.__name__, len(s2), len_should_be))

    def test_std(self):
        """
        Tests that the standard deviation method works, that before it's called the _std field is none,
        and after it's called the _std field is the standard deviation.
        """

        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertTrue(s._std is None,
                            "The _std field of Segment should be None before the user requests the the standard "
                            "deviation.")

            std_should_be = test_data.std

            std_was = s.std()

            pct = pct_diff(std_should_be, std_was)

            self.assertTrue(pct < 0.1,
                            "Standard deviation of Segment was incorrect. Should be {0}. Was {1}.".format(
                                std_should_be,
                                std_was))

            pct = pct_diff(std_should_be, s._std)

            self.assertTrue(pct < 0.1, "Segment._std should be set after the user calls .std().")

            std_was = s.std()

            pct = pct_diff(std_should_be, std_was)

            # Check the std again, just to be safe.
            self.assertTrue(pct < 0.1,
                            "Standard deviation of Segment was incorrect on the second call. Should be {0}. Was {"
                            "1}.".format(std_should_be, std_was))

    def test_mean(self):
        """
        Tests that the mean method works, that before it's called the _mean field is None, and after it's called the
        _mean field is the mean.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertTrue(s._mean is None,
                            "The _mean field of Segment should be None before the user requests the mean.")

            mean_should_be = test_data.mean

            mean_was = s.mean()

            pct = pct_diff(mean_should_be, mean_was)

            self.assertTrue(pct < 0.1, "Mean of Segment incorrect. Should be {0}. Was {1}.".format(
                mean_should_be, mean_was))

            # Make sure s._mean has been set.
            pct = pct_diff(mean_should_be, s._mean)
            self.assertTrue(pct < 0.1, "Segment._mean should be set after the user calls .mean().")

            mean_was = s.mean()

            pct = pct_diff(mean_should_be, mean_was)

            # Check the mean again, just to be safe
            self.assertTrue(pct < 0.1,
                            "Mean of Segment incorrect on second try. Should be {0}. Was {1}.".format(mean_should_be,
                                                                                                      mean_was))

    def test_min(self):
        """
        Tests that the min method works, that before it's called the _min field is None, and after it's called the
        _min field is the min of the segment.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertTrue(s._min is None,
                            "The _min field of Segment should be None before the user requests the min.")

            min_should_be = test_data.min

            min_was = s.min()

            pct = pct_diff(min_should_be, min_was)

            self.assertTrue(pct < 0.1, "Min of Segment incorrect. Should be {0}. Was {1}.".format(
                min_should_be, min_was))

            pct = pct_diff(min_should_be, s._min)
            # Make sure s._min has been set.
            self.assertTrue(pct < 0.1, "Segment._min should be set after the user calls .min().")

            min_was = s.min()
            pct = pct_diff(min_should_be, min_was)

            # Check the min again, just to be safe
            self.assertTrue(pct < 0.1, "Min of Segment incorrect on second try. Should be {0}. Was {"
                                       "1}.".format(min_should_be, min_was))

    def test_max(self):
        """
        Tests that the max method works, that before it's called the _max field is None, and after it's called the
        _max field is the max of the segment.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertTrue(s._max is None,
                            "The _max field of Segment should be None before the user requests the max.")

            max_should_be = test_data.max

            max_was = s.max()

            pct = pct_diff(max_should_be, max_was)

            self.assertTrue(pct < 0.1, "Max of Segment incorrect. Should be {0}. Was {1}.".format(
                max_should_be, max_was))

            # Make sure s._max has been set.
            pct = pct_diff(max_should_be, s._max)
            self.assertTrue(pct < 0.1, "Segment._max should be set after the user calls .max().")

            max_was = s.max()

            pct = pct_diff(max_should_be, max_was)
            # Check the max again, just to be safe
            self.assertTrue(pct < 0.1, "Max of Segment incorrect on second try. Should be {0}. Was {"
                                       "1}.".format(max_should_be, max_was))

    def test_size(self):
        """
        Test that size returns the correct size of the Segment.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            self.assertEqual(test_data.size, s.size,
                             "Segment size wrong. Should be {0}. Was {1}.".format(test_data.size, s.size))

    def test_shape(self):
        """
        Test that Segment returns the correct shape.
        """
        for test_data in self.default_test_data:
            s = self.SEGMENT_CLASS(test_data.data)

            s_shape = s.shape

            self.assertEqual(test_data.shape, s_shape,
                             "Segment shape incorrect. Should be {0}. Was {1}.".format(test_data.shape,
                                                                                       s_shape))

