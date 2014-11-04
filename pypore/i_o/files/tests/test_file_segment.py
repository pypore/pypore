import unittest

from pypore.i_o.files.file_segment import FileSegment


class TestFileSegment(unittest.TestCase):
    def test_not_implemented(self):
        """
        Tests that we cannot instantiate :py:class:`pypore.i_o.files.FileSegment`.
        :return:
        """
        self.assertRaises(NotImplementedError, FileSegment, 'hi.txt')
