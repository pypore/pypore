import unittest

from pypore.i_o.heka_reader import HekaReader
from pypore.i_o.tests.reader_tests import ReaderTests
import pypore.sampledata.testing_files as tf


class TestHekaReader(unittest.TestCase, ReaderTests):
    reader_class = HekaReader
    default_test_data_files = [tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')]

    def help_scaling(self):
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        mean = 5.32e-12
        std_dev = 2.76e-12
        return [filename], [mean], [std_dev]

    def help_slicing(self):
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        return [filename]

    def help_sample_rate(self):
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        sample_rate = 50000.0
        return [filename], [sample_rate]

    def help_shape(self):
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        shape = (75000,)
        return [filename], [shape]

    def test_chunk_size(self):
        """
        Tests that we cannot change the chunk size
        :return:
        """
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        reader = self.reader_class(filename)

        self.assertEqual(reader.chunk_size, reader._chunk_size)

        def set_chunk(chunk):
            reader.chunk_size = chunk

        self.assertRaises(AttributeError, set_chunk, 100)
