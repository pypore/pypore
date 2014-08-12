import os

import numpy as np

from pypore.i_o.abstract_reader import AbstractReader

# Data types list, in order specified by the HEKA file header v2.0.
# Using big-endian.
# Code 0=uint8,1=uint16,2=uint32,3=int8,4=int16,5=int32,
# 6=single,7=double,8=string64,9=string512
from pypore.util import interpret_indexing

HEKA_ENCODINGS = [np.dtype('>u1'), np.dtype('>u2'), np.dtype('>u4'),
                  np.dtype('>i1'), np.dtype('>i2'), np.dtype('>i4'),
                  np.dtype('>f4'), np.dtype('>f8'), np.dtype('>S64'),
                  np.dtype('>S512'), np.dtype('<u2')]

HEKA_DATATYPE = dt = np.dtype('>i2')  # int16

# Stupid python 3, dropping xrange....
try:
    xrange
except NameError:
    xrange = range


def _get_param_list_byte_length(param_list):
    """
    Returns the length in bytes of the sum of all the parameters in the list.
    Here, list[i][0] = param, list[i][1] = np.dtype
    """
    size = 0
    for i in param_list:
        size = size + i[1].itemsize
    return size


class HekaReader(AbstractReader):
    def __getitem__(self, item):
        # first we have to interpret the selection
        starts, stops, steps, shape = interpret_indexing(item, self.shape)

        # get the data for the selection
        return self.get_data_from_selection(starts, stops, steps, shape)

    def get_data_from_selection(self, starts, stops, steps, shape):
        """
        Returns the requested data.
        :param starts:
        :param stops:
        :param steps:
        :param shape:
        :return:
        """
        # find the block number that contains the start of the selection
        start_block_number = starts[0] // self._chunk_size

        # skip to that block, from the start of the binary data
        self.datafile.seek(self.per_file_header_length + start_block_number * self.total_bytes_per_block)

        # how far into the block is the first data point
        remainder = starts[0] % self._chunk_size

        n_points = len(xrange(starts[0], stops[0], steps[0]))

        # if we are dealing with a single integer, just return it
        if n_points == 1:
            chunk = self._read_heka_next_block()[0]
            values = chunk[remainder]
        else:
            step_size = steps[0]
            count = 0
            # only read channel 1 for now
            # TODO fix for multichannel
            values = np.empty(shape=(n_points,))
            while count < n_points:
                # only read channel 1 for now
                # TODO fix for multichannel
                chunk = self._read_heka_next_block()[0]

                chunk = chunk[remainder::step_size]
                chunk_size = chunk.size
                if (n_points - count) < chunk_size:
                    chunk = chunk[:n_points - count]
                    chunk_size = chunk.size
                values[count:count + chunk_size] = chunk[:]
                remainder = (remainder + step_size - self._chunk_size) % step_size
                count += chunk_size

        return values

    def _prepare_file(self, filename):
        """
        Implementation of :py:func:`prepare_data_file` for Heka ".hkd" files.
        """
        self.datafile = open(filename, 'rb')

        # Check that the first line is as expected
        line = self.datafile.readline()
        if not 'Nanopore Experiment Data File V2.0' in line:
            self.datafile.close()
            raise IOError('Heka data file format not recognized.')

        # Just skip over the file header text, should be always the same.
        while True:
            line = self.datafile.readline()
            if 'End of file format' in line:
                break

        # So now datafile should be at the binary data.

        # # Read binary header parameter lists
        self.per_file_param_list = self._read_heka_header_param_list(np.dtype('>S64'))
        self.per_block_param_list = self._read_heka_header_param_list(np.dtype('>S64'))
        self.per_channel_param_list = self._read_heka_header_param_list(np.dtype('>S64'))
        self.channel_list = self._read_heka_header_param_list(np.dtype('>S512'))

        # # Read per_file parameters
        self.per_file_params = self._read_heka_header_params(self.per_file_param_list)

        # # Calculate sizes of blocks, channels, etc
        self.per_file_header_length = self.datafile.tell()

        # Calculate the block lengths
        self.per_channel_per_block_length = _get_param_list_byte_length(self.per_channel_param_list)
        self.per_block_length = _get_param_list_byte_length(self.per_block_param_list)

        self.channel_list_number = len(self.channel_list)

        self.header_bytes_per_block = self.per_channel_per_block_length * self.channel_list_number
        self.data_bytes_per_block = self.per_file_params['Points per block'] * 2 * self.channel_list_number
        self.total_bytes_per_block = self.header_bytes_per_block + self.data_bytes_per_block + self.per_block_length

        # Calculate number of points per channel
        self.file_size = os.path.getsize(filename)
        self.num_blocks_in_file = int((self.file_size - self.per_file_header_length) / self.total_bytes_per_block)
        remainder = (self.file_size - self.per_file_header_length) % self.total_bytes_per_block
        if not remainder == 0:
            self.datafile.close()
            raise IOError('Heka file ends with incomplete block')
        self._chunk_size = self.per_file_params['Points per block']
        self.points_per_channel_total = self._chunk_size * self.num_blocks_in_file

        # TODO change for multichannel
        self.shape = (self.points_per_channel_total,)

        self.sample_rate = 1.0 / self.per_file_params['Sampling interval']

    def _read_heka_next_block(self):
        """
        Reads the next block of heka data.
        Returns a dictionary with 'data', 'per_block_params', and 'per_channel_params'.
        """
        # Read block header
        per_block_params = self._read_heka_header_params(self.per_block_param_list)
        if per_block_params is None:
            return [np.empty(0)]

        # Read per channel header
        per_channel_block_params = []
        for _ in self.channel_list:  # underscore used for discarded parameters
            channel_params = {}
            # i[0] = name, i[1] = datatype
            for i in self.per_channel_param_list:
                channel_params[i[0]] = np.fromfile(self.datafile, i[1], 1)[0]
            per_channel_block_params.append(channel_params)

        # Read data
        data = []
        for i in xrange(0, len(self.channel_list)):
            values = np.fromfile(self.datafile, dt, count=self._chunk_size) * \
                     per_channel_block_params[i][
                         'Scale']
            # get rid of nan's
            # values[np.isnan(values)] = 0
            data.append(values)

        return data

    def _read_heka_header_param_list(self, datatype):
        """
        Reads the binary parameter list of the following format:
            3 null bytes
            1 byte uint8 - how many params following
            params - 1 byte uint8 - code for datatype (eg encoding[code])
                     datatype.intemsize bytes - name the parameter
        Returns a list of parameters, with
            item[0] = name
            item[1] = numpy datatype
        """
        param_list = []
        self.datafile.read(3)  # read null characters?
        dt = np.dtype('>u1')
        num_params = np.fromfile(self.datafile, dt, 1)[0]
        for _ in xrange(0, num_params):
            type_code = np.fromfile(self.datafile, dt, 1)[0]
            name = np.fromfile(self.datafile, datatype, 1)[0].strip()
            param_list.append([name, HEKA_ENCODINGS[type_code]])
        return param_list

    def _read_heka_header_params(self, param_list):
        params = {}
        # pair[0] = name, pair[1] = np.datatype
        for pair in param_list:
            array = np.fromfile(self.datafile, pair[1], 1)
            if array.size > 0:
                params[pair[0]] = array[0]
            else:
                return None
        return params