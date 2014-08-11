import unittest

from pypore.util import *


class _ItemRet(object):
    def __getitem__(self, item):
        return item


class TestUtil(unittest.TestCase):
    def test_process_range(self):
        length = 100

        # Normal
        self.assertEqual(process_range(None, None, None, length), (0, length, 1))
        self.assertEqual(process_range(0, None, None, length), (0, length, 1))
        self.assertEqual(process_range(1, None, None, length), (1, length, 1))
        self.assertEqual(process_range(None, None, 2, length), (0, length, 2))
        self.assertEqual(process_range(None, None, -1, length), (0, length, -1))

        # out of bounds
        self.assertEqual(process_range(None, 1000000, None, length), (0, length, 1))

        # Negatives
        self.assertEqual(process_range(-1, None, None, length), (length - 1, length, 1))
        self.assertEqual(process_range(-10, None, None, length), (length - 10, length, 1))
        self.assertEqual(process_range(None, -1, None, length), (0, length - 1, 1))

        # Negatives out of bounds
        self.assertEqual(process_range(-5000, None, None, length), (0, length, 1))
        self.assertEqual(process_range(None, -10000, None, length), (0, 0, 1))

        # Errors
        self.assertRaises(ValueError, process_range, None, None, 0, length)

    def test_is_index(self):
        """
        Tests that the is_index method only accepts index like values.
        """

        self.assertTrue(is_index(1))
        self.assertTrue(is_index(int(3923492)))
        self.assertTrue(is_index(long(34843983214892)))
        self.assertTrue(is_index(np.array([1])[0]))

        self.assertFalse(is_index("h"))
        self.assertFalse(is_index('h'))
        self.assertFalse(is_index(5.0))
        self.assertFalse(is_index(float(5)))
        self.assertFalse(is_index(0.5))
        self.assertFalse(is_index(self))
        self.assertFalse(is_index(open))

    def test_interpret_indexing_normal_usage(self):
        item_ret = _ItemRet()

        # # Test normal usage

        args = []  # will contain arguments to interpret_indexing
        expected_results = []  # will contain expected results from interpret_indexing

        # Test selecting single item from single row
        args.append((item_ret[0], (1000,)))
        expected_results.append((
            np.array([0], dtype=np.integer), np.array([1], dtype=np.integer),
            np.array([1], dtype=np.integer),
            (1,)))

        # Test selecting single item from single row
        args.append((item_ret[:], (1000,)))
        expected_results.append((
            np.array([0], dtype=np.integer), np.array([1000], dtype=np.integer),
            np.array([1], dtype=np.integer),
            (1000,)))

        # Test selecting single row
        args.append((item_ret[0], (2, 1000)))
        expected_results.append((
            np.array([0, 0], dtype=np.integer), np.array([1, 1000], dtype=np.integer),
            np.array([1, 1], dtype=np.integer),
            (1, 1000)))

        # Test selecting single row
        args.append((item_ret[1], (2, 1000)))
        expected_results.append((
            np.array([1, 0], dtype=np.integer), np.array([2, 1000], dtype=np.integer),
            np.array([1, 1], dtype=np.integer),
            (1, 1000)))

        # Test selecting single row with negative index
        args.append((item_ret[-1], (2, 1000)))
        expected_results.append((
            np.array([1, 0], dtype=np.integer), np.array([2, 1000], dtype=np.integer),
            np.array([1, 1], dtype=np.integer),
            (1, 1000)))

        # Test selecting single row with negative index
        args.append((item_ret[-5], (10, 1000)))
        expected_results.append((
            np.array([5, 0], dtype=np.integer), np.array([6, 1000], dtype=np.integer),
            np.array([1, 1], dtype=np.integer),
            (1, 1000)))

        # Test selecting single number
        args.append((item_ret[1, 1], (2, 1000)))
        expected_results.append((
            np.array([1, 1], dtype=np.integer), np.array([2, 2], dtype=np.integer), np.array([1, 1], dtype=np.integer),
            (1, 1)))

        args.append((item_ret[:, :], (2, 1000)))
        expected_results.append((
            np.array([0, 0], dtype=np.integer), np.array([2, 1000], dtype=np.integer),
            np.array([1, 1], dtype=np.integer),
            (2, 1000)))

        args.append((item_ret[1:5, 1], (2, 1000)))
        expected_results.append((
            np.array([1, 1], dtype=np.integer), np.array([2, 2], dtype=np.integer), np.array([1, 1], dtype=np.integer),
            (1, 1)))

        args.append((item_ret[1, 0:10], (2, 1000)))
        expected_results.append((
            np.array([1, 0], dtype=np.integer), np.array([2, 10], dtype=np.integer), np.array([1, 1], dtype=np.integer),
            (1, 10)))

        args.append((item_ret[1, 1], (2, 1000)))
        expected_results.append((
            np.array([1, 1], dtype=np.integer), np.array([2, 2], dtype=np.integer), np.array([1, 1], dtype=np.integer),
            (1, 1)))

        # Test each set of arguments against the expected result
        for i, arg in enumerate(args):
            result = interpret_indexing(*arg)
            expected_result = expected_results[i]

            # make sure the starts, stops, steps arrays are equal
            for j in xrange(3):
                np.testing.assert_array_equal(result[j], expected_result[j], "Arrays at index {0} not equal.".format(j))

            # make sure the final shape is correct
            self.assertEqual(result[3], expected_result[3])

    def test_interpret_indexing_errors(self):
        item_ret = _ItemRet()

        # # Test errors
        # Test index errors
        self.assertRaises(IndexError, interpret_indexing, item_ret[100], (1, 3))
        self.assertRaises(IndexError, interpret_indexing, item_ret[1, 100], (100, 3))
        self.assertRaises(IndexError, interpret_indexing, item_ret[0, 0, 500], (1, 3, 1))

        # Test invalid slices
        self.assertRaises(IndexError, interpret_indexing, item_ret[1, 100], (100,))
        self.assertRaises(IndexError, interpret_indexing, item_ret[0, 0, 500], (1, 3))

        # Test for non indexes
        self.assertRaises(TypeError, interpret_indexing, item_ret['a'], (100,))


