"""
Core data types.
"""

import numpy as np


class Segment(object):
    """
    Segment is a segment of current data.

    Slicing a Segment returns a Segment object.

    >>> from pypore.core import Segment
    >>> import numpy as np
    >>> s = Segment(np.random.random(100))
    >>> s_first_fifty = Segment[:50]
    >>> isinstance(s_first_fifty, Segment)
    True

    Attributes:

        * sample_rate - The sampling rate of the segment.
        * shape - The shape of the data.
        * size - Number of data points in the Segment.

    Methods:

        * max - Returns the maximum value of the Segment.
        * mean - Returns the mean of the Segment.
        * min - Returns the minimum value in the Segment.
        * std - Returns the standard deviation of the Segment.

    """

    # Cached values
    _max = None
    _mean = None
    _min = None
    _std = None

    def __array__(self):
        """
        Returns this object as an array.

        This method allows np.array(segment) to work.
        :return: Returns this object as a numpy array.
        """
        return np.array(self._data)

    def __init__(self, data, sample_rate=0.0):
        """
        Initialize a new Segment object with the passed in data.
        :param data: Data defining the segment. Can pass in any data that is slice-able, like a numpy array, list, etc.
        :param sample_rate: Sampling rate of the data, in Hz. Default is 0.0 Hz.
        """
        self._data = data
        self.sample_rate = sample_rate

    def __getitem__(self, item):
        if isinstance(item, int):
            # Return single number.
            return self._data[item]
        else:
            # Return the slice as another Segment object.
            return Segment(self._data[item], self.sample_rate)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return iter(self._data)

    def max(self):
        """
        Returns the max of the Segment's data.
        :return: The maximum value of the Segment's data.
        """
        # Cache the max
        if self._max is None:
            self._max = np.max(self._data)
        return self._max

    def mean(self):
        """
        Returns the mean of the data.
        :return: The mean of the data.
        """
        # Cache the mean
        if self._mean is None:
            self._mean = np.mean(self._data)
        return self._mean

    def min(self):
        """
        Returns the min of the data.
        :return: The minimum value of the data.
        """
        # Cache the min
        if self._min is None:
            self._min = np.min(self._data)
        return self._min

    def std(self):
        """
        :return: The standard deviation of the data.
        """
        # Cache the std deviation.
        if self._std is None:
            self._std = np.std(self._data)
        return self._std

    @property
    def shape(self):
        """
        Returns the shape of the Segment's data.
        :return: The shape of the Segment's data.
        """
        try:
            # First try to get the shape of the data as if _data is a numpy array.
            shape = self._data.shape
            return shape
        except AttributeError:
            # If not, just assume it's 1 dimensional.
            # TODO fix getting dimension if we are passed a list, or a different data type. Might involve converting
            # any data to a numpy array.
            return self.size,

    @property
    def size(self):
        """
        :return: The number of data points in the Segment.
        """
        try:
            size = self._data.size
            return size
        except AttributeError:
            return len(self._data)
