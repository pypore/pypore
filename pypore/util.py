import numpy as np

import sys

ALLOWED_INDEX_TYPES = [int]
if sys.version_info < (3, 0):
    ALLOWED_INDEX_TYPES.append(long)

try:
    xrange
except NameError:
    xrange = range


def process_range(start, stop, step, length):
    """
    Returns the start, stop, and step of a slice. Ensures that start, stop, and step
    are given a value, which can be based on the length of the dimension of interest.

    :param start: The start of a range.
    :param stop: The stopping point of the range.
    :param step:
    :param length: The max length of the range.
    :return: Tuple of the following:

        #. start - the starting index of the range
        #. stop - the ending index of the range [start, stop)
        #. step - the step size of the range

    """
    if step is None:
        step = 1
    elif step == 0:
        raise ValueError("Step cannot be 0.")

    if start is None:
        start = 0
    elif start < 0:
        start += length
    if start < 0:
        start = 0
    if start > length:
        start = length

    if stop is None or stop > length:
        stop = length
    if stop < 0:
        stop += length
    if stop < 0:
        stop = 0

    return start, stop, step


def is_index(index):
    """Checks if an object can work as an index or not."""

    if type(index) in ALLOWED_INDEX_TYPES:
        return True
    elif isinstance(index, np.integer):
        return True

    return False


def interpret_indexing(keys, obj_shape):
    """
    Interprets slice information sent to __getitem__.

    :param keys: Slice info sent as parameter to __getitem__.
    :param obj_shape: The shape of the object you are trying to slice.
    :return: Returns a tuple of the following things.

        #. starts - numpy array of starting indices for each slice dimension.
        #. stops - numpy array of stopping indices for each slice dimension.
        #. steps - numpy array of step size for each slice dimension.
        #. shape - Final shape that slice will assume.
    """
    max_keys = len(obj_shape)
    shape = (max_keys,)
    starts = np.empty(shape=shape, dtype=np.integer)
    stops = np.empty(shape=shape, dtype=np.integer)
    steps = np.empty(shape=shape, dtype=np.integer)
    # make the keys a tuple if not already
    if not isinstance(keys, tuple):
        keys = (keys,)
    nkeys = len(keys)
    # check if we were asked for too many dimensions
    if nkeys > max_keys:
        raise IndexError("Too many indices for shape {0}.".format(obj_shape))
    dim = 0
    for key in keys:
        # Check if it's an index
        if is_index(key):
            if abs(key) >= obj_shape[dim]:
                raise IndexError("Index out of range.")
            if key < 0:
                key += obj_shape[dim]
            start, stop, step = process_range(key, key + 1, 1, obj_shape[dim])

        elif isinstance(key, slice):
            start, stop, step = process_range(key.start, key.stop, key.step, obj_shape[dim])
        else:
            raise TypeError("Non-valid index or slice {0}".format(key))
        starts[dim] = start
        stops[dim] = stop
        steps[dim] = step
        dim += 1

    # finish the extra dimensions
    if dim < max_keys:
        for j in range(dim, max_keys):
            starts[j] = 0
            stops[j] = obj_shape[j]
            steps[j] = 1

    # compute the new shape for the slice
    shape = []
    for j in range(max_keys):
        # new_dim = abs(stops[j] - starts[j])/steps[j]
        new_dim = len(xrange(starts[j], stops[j], steps[j]))
        shape.append(new_dim)

    return starts, stops, steps, tuple(shape)
