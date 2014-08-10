import numpy as np


class ReaderTests(object):
    """
    Other test classes for readers should inherit this class in addition to unittest.TestCase.

    Subclasses **must** use multi-inheritance and also inherit from :py:class:`unittest.TestCase`.

    The subclass **must** have a field 'reader_class' that is a subclass of
    :py:class:`AbstractReader <pypore.i_o.abstract_reader.AbstractReader>`. For example::

        self.reader_class = ChimeraReader

    The subclasses also **must** override the following methods:

    #. :py:func:`help_scaling`
    #. :py:func:`help_scaling_decimated`

    """
    reader_class = None
    default_test_data_files = None

    def test_non_existing_file_raises(self):
        self.assertRaises(IOError, self.reader_class, 'this_file_does_not_exist.nope')

    def test_sample_rate(self):
        """
        Tests that the sample rate is initialized properly.
        :return:
        """
        file_names, sample_rates_should_be = self.help_sample_rate()

        for i, filename in enumerate(file_names):
            sample_rate_should_be = sample_rates_should_be[i]

            reader = self.reader_class(filename)

            sample_rate = reader.sample_rate
            reader.close()

            self.assertEqual(sample_rate, sample_rate_should_be, "Sample rate read wrong from {0}. Should be {1}, got {2}.".format(filename, sample_rate_should_be, sample_rate))

    def help_sample_rate(self):
        """
        Subclasses need to override this and return the following in order.
        :return: The following, in order:

        #. list of file names of the test files
        #. list of corresponding sample rates

        """
        raise NotImplementedError

    def test_shape(self):
        """
        Tests that the shape field is initialized properly.
        """
        file_names, shapes_should_be = self.help_shape()

        for i, filename in enumerate(file_names):
            shape_should_be = shapes_should_be[i]

            reader = self.reader_class

            shape = reader.shape
            reader.close()



    def test_scaling(self):
        """
        Tests that the data is scaled correctly, from a known test file.
        """
        file_names, means_should_be, std_devs_should_be = self.help_scaling()

        for i, filename in enumerate(file_names):
            mean_should_be = means_should_be[i]
            std_dev_should_be = std_devs_should_be[i]

            reader = self.reader_class(filename)

            data = reader[0][:]
            reader.close()

            mean = np.mean(data)

            mean_diff = abs((mean - mean_should_be) / mean_should_be)
            self.assertLessEqual(mean_diff, 0.05,
                                 "Data mean in '{0}' scaled wrong. "
                                 "Should be {1}, got {2}.".format(filename, mean_should_be, mean))

            std_dev = np.std(data)

            std_diff = abs((std_dev - std_dev_should_be) / std_dev_should_be)
            self.assertLessEqual(std_diff, 0.05,
                                 "Baseline in '{0}' scaled wrong. "
                                 "Should be {1}, got {2}.".format(filename, std_dev_should_be, std_dev))

    def help_scaling(self):
        """
        Subclasses should override this method and return the following, in the following order.
        :returns: The following, in order:

        #. list of file names of the test files.
        #. list of known means of the data in the test files to check against the reader_class.
        #. list of known standard deviations of the data to check against the reader_class.

        """
        raise NotImplementedError('Inheritors should override this method.')
