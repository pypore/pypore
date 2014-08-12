from pypore.i_o.abstract_reader import AbstractReader

import unittest


class TestAbstractReader(unittest.TestCase):
    def test_not_implemented(self):
        """
        Tests that we cannot instantiate AbstractReader
        :return:
        """
        self.assertRaises(NotImplementedError, AbstractReader, 'hi.txt')
