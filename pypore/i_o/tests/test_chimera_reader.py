"""
"""
import unittest

from pypore.i_o.chimera_reader import ChimeraReader
from pypore.i_o.tests.reader_tests import ReaderTests
import pypore.sampledata.testing_files as tf


class TestChimeraReader(unittest.TestCase, ReaderTests):
    reader_class = ChimeraReader

    default_test_data_files = [tf.get_abs_path('chimera_small.log')]

    def test_constructor_no_mat_spec(self):
        """
        Tests that an IOError is raised when no .mat spec file is next to the .log file.
        """
        test_no_mat_chimera_files = tf.get_abs_path('chimera_empty.log')
        for filename in test_no_mat_chimera_files:
            self.assertRaises(IOError, ChimeraReader, filename)

    def test_get_all_data(self):
        # Make sure path to chimera file is correct.
        filename = tf.get_abs_path('chimera_small.log')
        chimera_reader = ChimeraReader(filename)

        data = chimera_reader[:]
        self._test_small_chimera_file_help(data)
        chimera_reader.close()

    def _test_small_chimera_file_help(self, data):
        self.assertEqual(data.size, 10, 'Wrong data size returned.')
        self.assertAlmostEqual(data[0], 17.45518, 3)
        self.assertAlmostEqual(data[9], 18.0743, 3)

    def help_sample_rate(self):
        filename = tf.get_abs_path('spheres_20140114_154938_beginning.log')
        sample_rate_should_be = 6.25e6
        return [filename], [sample_rate_should_be]

    def help_shape(self):
        filename = tf.get_abs_path('spheres_20140114_154938_beginning.log')
        shape_should_be = (10,)
        return [filename], [shape_should_be]

    def help_scaling(self):
        filename = tf.get_abs_path('spheres_20140114_154938_beginning.log')
        mean_should_be = 7.57604  # Value gotten from original MATLAB script
        std_should_be = 1.15445  # Value gotten from original MATLAB script
        return [filename], [mean_should_be], [std_should_be]

    def help_slicing(self):
        filename = tf.get_abs_path('spheres_20140114_154938_beginning.log')
        filename2 = tf.get_abs_path('chimera_1event.log')
        filename3 = tf.get_abs_path('chimera_1event_2levels.log')
        return [filename, filename2, filename3]
