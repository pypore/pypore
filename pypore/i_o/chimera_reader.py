import os

import scipy.io as sio
import numpy as np

from pypore.i_o.abstract_reader import AbstractReader


# ctypedef np.float_t DTYPE_t

CHIMERA_DATA_TYPE = np.dtype('<u2')
# mantissa is 23 bits for np.float32, well above 16 bit from raw
CHIMERA_OUTPUT_DATA_TYPE = np.float32

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
        return self._scale_raw_chimera(np.array(self.memmap[item]))

    def _scale_raw_chimera(self, values):
        """
        Scales the raw chimera data to correct scaling.

        :param values: numpy array of Chimera values. (raw <u2 datatype)
        :returns: Array of scaled Chimera values (np.float datatype)
        """
        values &= self.bit_mask
        values = values.astype(CHIMERA_OUTPUT_DATA_TYPE, copy=False)
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

        self.bit_mask = np.array((2 ** 16) - 1 - (2 ** (16 - self.adc_bits) - 1), dtype=CHIMERA_DATA_TYPE)

        self.sample_rate = 1.0 * self.specs_file['SETUP_ADCSAMPLERATE'][0][0]

        # calculate the scaling factor from raw data
        self.scale_multiplication = np.array(
            (2 * self.adc_v_ref / 2 ** 16) / (self.pre_adc_gain * self.tia_gain), dtype=CHIMERA_OUTPUT_DATA_TYPE)
        self.scale_addition = np.array(self.current_offset - self.adc_v_ref / (self.pre_adc_gain * self.tia_gain),
                                       dtype=CHIMERA_OUTPUT_DATA_TYPE)

        # Use numpy memmap. Note this will fail for files > 4GB on 32 bit systems.
        # If you run into this, a more extreme lazy loading solution will be needed.
        self.memmap = np.memmap(self.datafile, dtype=CHIMERA_DATA_TYPE, mode='r', shape=self.shape)

