import os

import scipy.io as sio
import numpy as np

from pypore.i_o.abstract_reader import AbstractReader


# ctypedef np.float_t DTYPE_t
from pypore.util import interpret_indexing

CHIMERA_DATA_TYPE = np.dtype('<u2')

# Stupid python 3, dropping xrange....
try:
    xrange
except NameError:
    xrange = range


class ChimeraReader(AbstractReader):
    """
    Reader class that reads .log files (with corresponding .mat files) produced by the Chimera acquisition software
    at UPenn.
    """
    specs_file = None

    # parameters from the specsfile
    adc_bits = None
    adc_v_ref = None
    current_offset = None
    tia_gain = None
    pre_adc_gain = None
    bit_mask = None

    def __getitem__(self, item):
        """
        Examples
        --------

        ::

            array1 = reader[0]          # simple selection
            array2 = reader[4:1400:3]   # slice selection
            array3 = reader[1, :]     # general slice selection, although chimera is only 1 channel, so this will fail

        :param item:
        :return:
        """
        # first we have to interpret the selection
        starts, stops, steps, shape = interpret_indexing(item, self.shape)

        # get the data for the selection
        return self.get_data_from_selection(starts, stops, steps, shape)

    def get_data_from_selection(self, starts, stops, steps, shape):
        # for chimera, we just need one channel

        # how expensive is using this here?
        my_range = xrange(starts[0], stops[0], steps[0])

        n_points = len(my_range)
        if n_points == 0:
            return np.zeros(0, dtype=CHIMERA_DATA_TYPE)

        # if the step size is < 0, we need to figure out what the first point actually is
        if steps[0] > 0:
            start = starts[0]
            stop = stops[0]
            negative_step = False
        else:
            start = my_range[-1]
            stop = my_range[0]
            negative_step = True
        step_size = abs(steps[0])

        # go to the start of the shape
        self.datafile.seek(start * CHIMERA_DATA_TYPE.itemsize)

        # if we are dealing with a single integer, just return it
        if n_points == 1:
            values = np.fromfile(self.datafile, CHIMERA_DATA_TYPE, n_points)
        elif step_size == 1:
            # if the step size is 1, do normal read
            values = np.fromfile(self.datafile, CHIMERA_DATA_TYPE, n_points)
        else:
            values = np.empty(shape=(n_points,), dtype=CHIMERA_DATA_TYPE)
            # otherwise, read is a little more complicated.

            remainder = 0
            count = 0
            while count < n_points:
                # Compute the chunk size of stuff needed
                chunk = np.fromfile(self.datafile, CHIMERA_DATA_TYPE, self._chunk_size)
                real_chunk_size = chunk.size

                chunk = chunk[remainder::step_size]
                chunk_size = chunk.size
                if (n_points - count) < chunk_size:
                    chunk = chunk[:n_points - count]
                    chunk_size = chunk.size
                values[count:count + chunk_size] = chunk[:]
                remainder = (remainder + step_size - real_chunk_size) % step_size
                count += chunk_size

        # scale the values
        values = self._scale_raw_chimera(values)
        if negative_step:
            values = values[::-1]

        if n_points == 1:
            return values[0]

        return values

    def _scale_raw_chimera(self, values):
        """
        Scales the raw chimera data to correct scaling.

        :param values: numpy array of Chimera values. (raw <u2 datatype)
        :returns: Array of scaled Chimera values (np.float datatype)
        """
        values &= self.bit_mask
        values = values.astype(np.float, copy=False)
        values *= self.scale_multiplication
        values += self.scale_addition

        return values

    def _prepare_file(self, filename):
        """
        Implementation of :py:func:`prepare_data_file` for Chimera ".log" files with the associated ".mat" file.
        """
        # remove 'log' append 'mat'
        specs_filename = filename[:-len('log')] + 'mat'
        # load the matlab file with parameters for the runs
        try:
            self.specs_file = sio.loadmat(specs_filename)
        except IOError:
            raise IOError(
                "Error opening " + filename + ", Chimera .mat specs file of same name must be located in same folder.")

        self.datafile = open(filename, 'rb')

        # Calculate number of points per channel
        file_size = os.path.getsize(filename)
        points_per_channel_total = file_size / CHIMERA_DATA_TYPE.itemsize
        self.shape = (points_per_channel_total,)

        self.adc_bits = self.specs_file['SETUP_ADCBITS'][0][0]
        self.adc_v_ref = self.specs_file['SETUP_ADCVREF'][0][0]
        self.current_offset = self.specs_file['SETUP_pAoffset'][0][0]
        self.tia_gain = self.specs_file['SETUP_TIAgain'][0][0]
        self.pre_adc_gain = self.specs_file['SETUP_preADCgain'][0][0]

        self.bit_mask = (2 ** 16) - 1 - (2 ** (16 - self.adc_bits) - 1)

        self.sample_rate = 1.0 * self.specs_file['SETUP_ADCSAMPLERATE'][0][0]

        # calculate the scaling factor from raw data
        self.scale_multiplication = 1.e9 * (2 * self.adc_v_ref / 2 ** 16) / (self.pre_adc_gain * self.tia_gain)
        self.scale_addition = 1.e9 * (self.current_offset - self.adc_v_ref / (self.pre_adc_gain * self.tia_gain))

