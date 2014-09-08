import numpy as np

from pypore.tests.segment_tests import SegmentTests


class ReaderTests(SegmentTests):
    """
    Other test classes for readers should inherit this class in addition to unittest.TestCase.

    See :py:class:`pypore.tests.segment_tests.SegmentTests` for further details.
    """

    def test_non_existing_file_raises(self):
        self.assertRaises(IOError, self.SEGMENT_CLASS, 'this_file_does_not_exist.nope')

    def test_sample_rate(self):
        """
        Tests that the sample rate is initialized properly.
        :return:
        """
        for test_data in self.default_test_data:
            reader = self.SEGMENT_CLASS(test_data.data)

            sample_rate = reader.sample_rate
            sample_rate_diff = np.abs(sample_rate - test_data.sample_rate) / test_data.sample_rate
            reader.close()

            self.assertTrue(sample_rate_diff <= 0.05,
                            "Sample rate read wrong from {0}. Should be {1}, got {2}.".format(test_data.data,
                                                                                              test_data.sample_rate,
                                                                                              sample_rate))

    def test_slicing(self):
        """
        Tests that slicing works as expected.
        """

        for test_data in self.default_test_data:
            reader = self.SEGMENT_CLASS(test_data.data)

            # get all the data first, to compare the slicing.
            # the idea is that if test_get_all_data works, this should work.
            data = reader[:]

            # check starting points
            np.testing.assert_array_equal(data[0:], reader[0:])
            np.testing.assert_array_equal(data[3:], reader[3:])
            np.testing.assert_array_equal(data[5:], reader[5:])

            # check stopping points
            np.testing.assert_array_equal(data[:0], reader[:0])
            np.testing.assert_array_equal(data[:3], reader[:3])
            np.testing.assert_array_equal(data[:5], reader[:5])

            # check steps
            np.testing.assert_array_equal(data[::2], reader[::2])
            np.testing.assert_array_equal(data[::3], reader[::3])
            np.testing.assert_array_equal(data[::5], reader[::5])

            # check indices
            self.assertEqual(data[0], reader[0])
            self.assertEqual(data[1], reader[1])
            self.assertEqual(data[-1], reader[-1])
            self.assertEqual(data[5], reader[5])

            # check negative step sizes
            np.testing.assert_array_equal(data[::-1], reader[::-1])
            np.testing.assert_array_equal(data[::-2], reader[::-2])
            np.testing.assert_array_equal(data[::-3], reader[::-3])
            np.testing.assert_array_equal(data[::-5], reader[::-5])

            # check the whole thing
            np.testing.assert_array_equal(data[0:8:2], reader[0:8:2])
            np.testing.assert_array_equal(data[1:9:3], reader[1:9:3])
            np.testing.assert_array_equal(data[-8:-1:-1], reader[-8:-1:-1])
            np.testing.assert_array_equal(data[9:1:-1], reader[9:1:-1])
            np.testing.assert_array_equal(data[9:1:-4], reader[9:1:-4])

            reader.close()

    def test_close(self):
        """
        Tests that subclasses of AbstractReader have implemented close.
        """
        reader = self.SEGMENT_CLASS(self.default_test_data[0].data)
        reader.close()