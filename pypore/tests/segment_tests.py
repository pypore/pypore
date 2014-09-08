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

            data = s[:]

            count = 0
            i = 0
            for i, point in enumerate(s):
                self.assertEqual(data[i], point)
                count += 1

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, test_data.size,
                             "enumerate(segment) did not loop through all elements. It looped through {0}/{1} "
                             "elements.".format(count, test_data.size))
            self.assertEqual(i, test_data.size - 1)

            # Make sure we can loop through a second time with the same results.

            count = 0
            for i, point in enumerate(s):
                self.assertEqual(data[i], point)
                count += 1

            # Make sure we looped through all of the correct i's
            self.assertEqual(count, test_data.size,
                             "enumerate(segment) did not loop through all elements On the second try. It looped "
                             "through {0}/{1} elements.".format(count, test_data.size))
            self.assertEqual(i, test_data.size - 1)

    def test_len(self):
        """
        Tests that we can use len(segment) the same as segment.size.
        """
        for data in self.default_test_data:
            s = self.SEGMENT_CLASS(data.data)

            self.assertEqual(len(s), data.size,
                             "Segment's length and size differ. Length {0}. Size {1}.".format(len(s), data.size))

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

