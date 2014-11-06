import cProfile
import numpy as np

from pypore.i_o.files.chimera_segment import ChimeraSegment
from pypore.i_o.files.heka_segment import HekaSegment
import pypore.sampledata.testing_files as tf


def _reader_tasks(reader):
    for _ in xrange(1000):
        data = np.array(reader[:])
        data = np.array(reader[::-1])
        data = np.array(reader[100])
        data = np.array(reader[::-2])
        data = np.array(reader[::3])


def profile_chimera():
    filename = tf.get_abs_path('spheres_20140114_154938_beginning.log')
    reader = ChimeraSegment(filename)
    _reader_tasks(reader)
    reader.close()


def profile_heka():
    filename = tf.get_abs_path('heka_1.5s_mean5.32p_std2.76p.hkd')
    reader = HekaSegment(filename)
    _reader_tasks(reader)
    reader.close()

if __name__ == '__main__':
    print("Profiling ChimeraReader")
    cProfile.run('profile_chimera()', sort='cumtime')

    print("Profiling HekaReader")
    cProfile.run('profile_heka()', sort='cumtime')
