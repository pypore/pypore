"""
"""
import unittest

from pypore.tests.segment_tests import SegmentTestData
from pypore.i_o.files.chimera_segment import ChimeraSegment
from pypore.i_o.files.tests.file_segment_tests import FileSegmentTests
import pypore.sampledata.testing_files as tf


class TestChimeraSegment(unittest.TestCase, FileSegmentTests):
    SEGMENT_CLASS = ChimeraSegment

    default_test_data = [
        SegmentTestData(tf.get_abs_path('chimera_small.log'), 1.8157181e-08, 1.7765718e-08, 1.7157802e-08, (10,), 10,
                        3.5689698e-10, 4166666.6666666665),
        SegmentTestData(tf.get_abs_path('spheres_20140114_154938_beginning.log'), 1.1914651e-08, 7.5760553e-09,
                        2.7252511e-09, (102400,), 102400, 1.1544473e-09, 6250000.0)]

    def test_constructor_no_mat_spec(self):
        """
        Tests that an IOError is raised when no .mat spec file is next to the .log file.
        """
        test_no_mat_chimera_files = tf.get_abs_path('chimera_empty.log')
        for filename in test_no_mat_chimera_files:
            self.assertRaises(IOError, ChimeraSegment, filename)