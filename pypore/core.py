"""
Core data types.
"""

import numpy as np


class Segment():
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

        * size - Number of datapoints in the Segment.
        * sample_rate - The sampling rate of the segment.

    Methods:

        * max - Returns the maximum value of the Segment.
        * mean - Returns the mean of the Segment.
        * min - Returns the minimum value in the Segment.
    """

    # Iterator index
    _index = 0

    # Cached values
    _max = None
    _mean = None
    _min = None

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
        :return:
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
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self.size:
            raise StopIteration
        else:
            self._index += 1
            return self[self._index - 1]

    def next(self):
        return self.__next__()

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

    @property
    def shape(self):
        try:
            shape = self._data.shape
            return shape
        except AttributeError:
            return self.size,

    @property
    def size(self):
        try:
            size = self._data.size
            return size
        except AttributeError:
            return len(self._data)
