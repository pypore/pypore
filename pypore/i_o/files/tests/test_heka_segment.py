import unittest

from pypore.tests.segment_tests import SegmentTestData
from pypore.i_o.files.heka_segment import HekaSegment
from pypore.i_o.files.tests.file_segment_tests import FileSegmentTests
import pypore.sampledata.testing_files as tf


class TestHekaSegment(unittest.TestCase, FileSegmentTests):
    SEGMENT_CLASS = HekaSegment
    default_test_data = [SegmentTestData(tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd'), 2.2500000000000003e-11,
                                         5.3176916666664804e-12, -1.5937500000000003e-11, (75000,), 75000,
                                         2.7618361051293422e-12, 50000.)]

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

    def test_get_total_dimension_len(self):
        filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
        reader = self.SEGMENT_CLASS(filename)

        self.assertEqual(75000, reader._get_total_dimension_len())

        # TODO add test with multichannel data

