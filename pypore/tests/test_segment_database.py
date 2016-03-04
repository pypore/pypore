
import unittest
import os

import numpy as np

from pypore.core import SegmentDatabase
from pypore.sampledata.testing_files import get_abs_path
from pypore.core import Segment, MetaSegment
from pypore.util import slice_combine


class TestSegmentDatabase(unittest.TestCase):

    def test_setting_segment(self):
        """
        Tests that we can set the segment.
        """
        db = SegmentDatabase()

        segment = Segment(np.random.random(100))

        db.segment = segment

        self.assertTrue(isinstance(db.segment, Segment))

        np.testing.assert_array_equal(segment, db.segment)

    def test_setting_segment_in_sub_db(self):
        """
        Tests that we can set the segment of a sub database
        """
        db = SegmentDatabase()

        db.segment = Segment(np.random.random(100))

        db.add_segment(0, 50)

        np.testing.assert_array_equal(db.segment[0:50], db[0].segment,
                                      err_msg="Appended child segment incorrect. Should be {0}, was {1}."
                                      .format(db.segment[0:50], db[0].segment))

        seg2 = Segment(np.random.random(55))

        db[0].segment = seg2

        np.testing.assert_array_equal(seg2, db[0].segment,
                                      err_msg="Set child segment incorrect. Should be {0}, was {1}.".format(seg2, db[0].segment))

        # TODO test that metadata attributes of the parent db were updated correctly.

    def test_append_segment_without_data_with_parent_segment(self):
        """
        Tests that appending a Segment works without passing a data parameter, when there is a parent Segment.
        """
        database = SegmentDatabase()

        segment = Segment(np.random.random(100))

        database.segment = segment

        # Make sure there aren't any sub SegmentDatabases
        self.assertEqual(database.size, 0)

        sub_slice = slice(1, 89)

        # Append slice from [1:89]
        database.add_segment(sub_slice.start, sub_slice.stop)

        # Make sure sub_segment was appended correctly
        self.assertEqual(database.size, 1)

        sub_db = database[0]

        self.assertTrue(isinstance(sub_db, SegmentDatabase))
        np.testing.assert_array_equal(sub_db.segment, database.segment[sub_slice])

        self.assertEqual(sub_db.size, 0)

        # TODO test sub attributes

    def test_db_root_has_no_parent(self):
        """
        The root SegmentDatabase should have no parent.
        """
        # Test on new db
        db = SegmentDatabase()

        self.assertTrue(db.parent is None)

        db.add_segment(segment=Segment(np.random.random(100)))

        self.assertTrue(db.parent is None)

    def test_child_db_has_parent(self):
        """
        Tests that a child database has a parent.
        """
        db = SegmentDatabase()

        parent_segment = Segment(np.random.random(100))
        db.segment = parent_segment

        db.add_segment(0, 50)

        child_db = db[0]

        parent_db = child_db.parent

        self.assertTrue(parent_db is not None)
        self.assertEqual(parent_db.size, 1)

        np.testing.assert_array_equal(db.segment, parent_db.segment)

    def test_append_segments_to_sub_db(self):
        """
        Tests that you can append databases to subDBs.
        """
        db = SegmentDatabase()

        db.segment = Segment(np.random.random(100))

        stop = 95
        slices = []

        curr_db = db

        depth = 5

        # Go 5 levels deeper, total of 6 levels
        for i in range(depth):
            slic = slice(0, stop)
            slices.append(slic)

            curr_db.add_segment(slic.start, slic.stop)

            # Move a level down
            curr_db = curr_db[0]
            stop -= 10

        curr_db = db
        # Test each level
        for i in range(depth):
            # Move down a level
            curr_db = curr_db[0]

            # combine the slices at all levels to get a total slice
            slice_combined = slice_combine(len(db.segment), *slices[0:i+1])

            np.testing.assert_array_equal(db.segment[slice_combined], curr_db.segment)

            # TODO test attributes each level

    def test_append_segment_with_segment_no_slice(self):
        """
        Tests appending Segments with their own data and without specifying the
        location in the parent data.
        """
        db = SegmentDatabase()
        db.segment = Segment(np.random.random(100))

        segments = []

        n = 50

        for i in range(n):
            self.assertEqual(db.size, i)

            segment = Segment(np.random.random(50))
            segments.append(segment)
            db.add_segment(segment=segment)

        self.assertEqual(db.size, n)

        for i in range(n):
            np.testing.assert_array_equal(segments[i], db[i].segment)

        # TODO test metadata
