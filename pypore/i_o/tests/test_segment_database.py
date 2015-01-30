import unittest
import os

import numpy as np

from pypore.i_o.segment_database import SegmentDatabase
from pypore.sampledata.testing_files import get_abs_path
from pypore.core import Segment, MetaSegment
from pypore.util import slice_combine


class TestSegmentDatabase(unittest.TestCase):
    def test_opening_nonexisting_file(self):
        filename = 'TSD_test_opening_non_existing_file.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        database = SegmentDatabase.open_file(filename, mode='w')

        database.close()

        os.remove(filename)

    def test_opening_existing_file(self):
        filename = get_abs_path('db_1.h5')

        database = SegmentDatabase.open_file(filename, mode='r')

        # TODO add assert for data equality

        database.close()

    def test_existing_db_has_segment_or_meta_segment(self):
        filename = get_abs_path('db_1.h5')

        database = SegmentDatabase.open_file(filename)

        segment = database.segment

        self.assertTrue(isinstance(segment, Segment) or isinstance(segment, MetaSegment))

        # TODO add assert for data equality

        database.close()

    def test_setting_segment(self):
        """
        Tests that we can set the segment.
        """
        filename = 'TSD_test_setting_segment.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename)

        segment = Segment(np.random.random(100))

        db.segment = segment

        self.assertTrue(isinstance(db.segment, Segment))

        np.testing.assert_array_equal(segment, db.segment)

        db.close()

        os.remove(filename)

    def test_setting_segment_read_only_raises(self):
        """
        Tests that setting a segment in a file opened as read only raises an exception.
        """
        filename = 'TSD_test_setting_segment_read_only_raises.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')
        db.close()

        db = SegmentDatabase.open_file(filename, mode='r')

        def t():
            db.segment = Segment(np.random.random(100))

        self.assertRaises(IOError, t)

        os.remove(filename)

    def test_setting_segment_in_sub_db_read_only_raises(self):
        """
        Tests that trying to set a segment in a sub database opened in read-only mode raises an IOError.
        """
        filename = 'TSD_test_setting_segment_in_sub_db_read_only_raises.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')
        db.segment = Segment(np.random.random(100))
        db.append(0, 50)
        db.close()

        db = SegmentDatabase.open_file(filename, mode='r')

        def t(database):
            database[0].segment = Segment(np.random.random(100))

        self.assertEqual(db.size, 1)

        self.assertRaises(IOError, t, db)

        os.remove(filename)

    def test_appending_segment_read_only_raises(self):
        """
        Tests that appending a segment to a read-only database raises an error.
        """
        filename = 'TSD_test_appending_segment_read_only_raises.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')
        db.segment = Segment(np.random.random(100))
        db.close()

        db = SegmentDatabase.open_file(filename, mode='r')

        def t(database):
            database.append(0, 50)

        self.assertRaises(IOError, t, db)

        db.close()

        os.remove(filename)

    def test_appending_segment_in_sub_db_read_only_rasises(self):
        """
        Tests that appending a segment to a sub db opened in read-only mode raises an exception.
        """
        filename = 'TSD_test_appending_segment_in_sub_db_read_only_raises.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')
        db.segment = Segment(np.random.random(100))
        db.append(10, 99)
        db.close()

        db = SegmentDatabase.open_file(filename, mode='r')

        def t(database):
            database.append(0, 50)

        self.assertEqual(db.size, 1)

        self.assertRaises(IOError, t, db[0])

        db.close()

        os.remove(filename)

    def test_setting_segment_in_sub_db(self):
        """
        Tests that we can set the segment of a sub database
        """
        filename = 'TSD_test_setting_segment_in_sub_db.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')

        db.segment = Segment(np.random.random(100))

        db.append(0, 50)

        np.testing.assert_array_equal(db.segment[0:50], db[0].segment,
                                      err_msg="Appended child segment incorrect. Should be {0}, was {1}."
                                      .format(db.segment[0:50], db[0].segment))

        seg2 = SegmentDatabase(np.random.random(55))

        db[0].segment = seg2

        np.testing.assert_array_equal(seg2, db[0].segment,
                                      err_msg="Set child segment incorrect. Should be {0], was {1}.".format(seg2, db[
                                          0].segment))

        # TODO test that metadata attributes of the parent db were updated correctly.

        os.remove(filename)

    def test_append_segment_without_data_with_parent_segment(self):
        """
        Tests that appending a Segment works without passing a data parameter, when there is a parent Segment.
        """
        filename = 'TSD_test_append_segment_without_data_with_parent_segment.test.h5'
        if os.path.exists(filename):
            os.remove(filename)

        database = SegmentDatabase.open_file(filename)

        segment = Segment(np.random.random(100))

        database.segment = segment

        # Make sure there aren't any sub SegmentDatabases
        self.assertEqual(database.size, 0)

        sub_slice = slice(1, 89)

        # Append slice from [1:89]
        database.append(sub_slice.start, sub_slice.stop)

        # Make sure sub_segment was appended correctly
        self.assertEqual(database.size, 1)

        sub_db = database[0]

        self.assertTrue(isinstance(sub_db), SegmentDatabase)
        np.testing.assert_array_equal(sub_db.segment, database.segment[sub_slice])

        self.assertEqual(sub_db.size, 0)

        # TODO test sub attributes

        os.remove(filename)

    def test_db_root_has_no_parent(self):
        """
        The root SegmentDatabase should have no parent.
        """
        filename = "TSD_db_root_has_no_parent.test.h5"
        if os.path.exists(filename):
            os.remove(filename)

        # Test on new db
        db = SegmentDatabase.open_file(filename)

        self.assertTrue(db.parent is None)

        db.append(data=Segment(np.random.random(100)))

        self.assertTrue(db.parent is None)

        db.close()
        os.remove(filename)

    def test_child_db_has_parent(self):
        """
        Tests that a child database has a parent.
        """
        filename = "TSD_test_child_db_has_parent.test.h5"
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename)

        parent_segment = Segment(np.random.random(100))
        db.segment = parent_segment

        db.append(0, 50)

        child_db = db[0]

        parent_db = child_db.parent

        self.assertTrue(parent_db is not None)
        self.assertEqual(parent_db.size, 1)

        np.testing.assert_array_equal(db.segment, parent_db.segment)

        db.close()
        os.remove(filename)

    def test_append_segments_to_sub_db(self):
        """
        Tests that you can append databases to subDBs.
        """
        filename = "TSD_test_append_segments_to_sub_db.test.h5"
        if os.path.exists(filename):
            os.remove(filename)

        db = SegmentDatabase.open_file(filename, mode='w')

        db.segment = Segment(np.random.random(100))

        stop = 95
        slices = []

        depth = 5

        curr_db = db

        # Go 5 levels deeper, total of 6 levels
        for i in range(5):
            slic = slice(0, stop)
            slices.append(slic)

            curr_db.append(slic.start, slic.stop)

            # Move a level down
            curr_db = curr_db[0]
            stop -= 10

        curr_db = db
        # Test each level
        for i in range(depth):
            # Move down a level
            curr_db = curr_db[0]

            # combine the slices at all levels to get a total slice
            slice_combined = slice_combine(len(db.segment), slices[0:i])

            np.testing.assert_array_equal(db.segment[slice_combined], curr_db.segment)

            # TODO test attributes each level

        os.remove(filename)
