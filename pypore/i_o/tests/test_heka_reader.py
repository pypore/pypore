import unittest

from pypore.tests.segment_tests import SegmentTestData
from pypore.i_o.heka_reader import HekaReader
from pypore.i_o.tests.reader_tests import ReaderTests
import pypore.sampledata.testing_files as tf


class TestHekaReader(unittest.TestCase, ReaderTests):
    SEGMENT_CLASS = HekaReader
    default_test_data = [
        SegmentTestData(tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd'), maximum=2.2500000000000003e-11,
                        mean=5.3176916666664804e-12, minimum=-1.5937500000000003e-11, shape=(75000,), size=75000,
                        std=2.7618361051293422e-12, sample_rate=50000.)]

    def test_chunk_size(self):
        """
        Tests that we cannot change the chunk size
        """
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        reader = self.SEGMENT_CLASS(filename)

        self.assertEqual(reader.chunk_size, reader._chunk_size)

        def set_chunk(chunk):
            reader.chunk_size = chunk

        self.assertRaises(AttributeError, set_chunk, 100)

    def test_heka_format_error_raises_binary_file(self):
        """
        Tests that trying to open a completely binary file that doesn't fit the Heka specs raises an IOError.
        """
        # Test with a complete binary file that has no text header.
        filename = tf.get_abs_path('chimera_small.log')
        self.assertRaises(IOError, self.SEGMENT_CLASS, filename)

    def test_heka_format_error_raises_text_file(self):
        """
        Test that opening a text with invalid header text fails.
        """
        filename = tf.get_abs_path('two_line_text.txt')
        self.assertRaises(IOError, self.SEGMENT_CLASS, filename)

    def test_incomplete_heka_file_raises(self):
        """
        Tests that opening a Heka file with incomplete blocks raises an IOError.
        """
        filename = tf.get_abs_path('heka_incomplete.hkd')

        self.assertRaises(IOError, self.SEGMENT_CLASS, filename)
