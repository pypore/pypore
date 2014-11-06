"""
Core pypore types.
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

    # Cached properties
    _ndim = None
    _shape = None
    _size = None

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

            # reduce the sample rate if the slice has steps
            sample_rate = self.sample_rate
            if isinstance(item, slice) and item.step is not None and item.step > 1:
                sample_rate /= item.step
            return Segment(self._data[item], sample_rate)

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
    def ndim(self):
        """
        Returns the number of dimensions in the data set.
        """
        if self._ndim is None:
            try:
                self._ndim = self._data.ndim
            except AttributeError:
                self._ndim = np.ndim(self._data)
        return self._ndim

    @property
    def shape(self):
        """
        Returns the shape of the Segment's data.
        :return: The shape of the Segment's data.
        """
        if self._shape is None:
            try:
                # First try to get the shape of the data as if _data is a numpy array.
                self._shape = self._data.shape
            except AttributeError:
                self._shape = np.shape(self._data)
        return self._shape

    @property
    def size(self):
        """
        :return: The number of data points in the Segment.
        """
        if self._size is None:
            try:
                self._size = self._data.size
            except AttributeError:
                self._size = np.size(self._data)
        return self._size


class Extractor(object):
    """
    Base Extractor object defining the methods and attributes of Extractors.

    An Extractor searches through a :py:class:`pypore.core.Segment` or list of :py:class:`pypore.core.Segment`s and
    extracts a list of :py:class:`pypore.core.Segment`s based on search parameters.

    """

    def search(self, segment):
        """
        A extractor is responsible for taking in one or many :py:class:`pypore.core.Segment`s and returning a list of
        :py:class:`pypore.core.Segment`s.
        :param segment: One or many :py:class:`pypore.core.Segment`s. If a list of :py:class:`pypore.core.Segment`s
        is passed in, the :py:class:`pypore.core.Segment`s are assumed to be contiguous.
        :return: A list of :py:class:`pypore.core.Segment`s found by the extractor.
        """
        raise NotImplementedError("Extractor is an abstract base class. Extend it and override the search method.")
