import os


class AbstractReader(object):
    """
    This is an abstract class showing the methods that subclasses must override.

    """
    # fields common to every Reader
    datafile = None

    sample_rate = None
    shape = None

    filename = None
    directory = None

    # extra fields specific to readers should be accessible from
    metadata = None

    # _chunk_size can be used by subclasses when lazy loading data
    # default is ~100kB of 64 bit floating points
    _chunk_size = 12500

    @property
    def chunk_size(self):
        return self._chunk_size

    def __init__(self, filename):
        """
        Opens a data file, reads relevant parameters, and returns then open file and parameters.

        :param StringType filename: Filename to open and read parameters.

        If there was an error opening the files, params will have 'error' key with string description.
        """
        self.filename = os.path.abspath(filename)
        self.simplename = os.path.basename(self.filename)
        self.directory = os.path.dirname(self.filename)
        self.metadata = {}

        self._prepare_file(filename)

    def _prepare_file(self, filename):
        """_prepare_file(filename)

        Subclasses must override this method.

        This method is called from the super class's __init__ method.

        Subclasses should open a data file as

        >>> self.datafile = open(filename, 'r or rb or whatever')

        Additionally, subclasses should initialize the following data:

        self.sample_rate
        self.shape

        :param StringType filename: Filename to open and read parameters.

        """
        raise NotImplementedError

    def close(self):
        """close()

        Closes the file and the reader.
        """
        self.datafile.close()
