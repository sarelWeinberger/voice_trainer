"""Regression checks for the missing under-chin region and temporal context."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'root/voice_app'))
from neck_v1_cv import neck_context_box, select_crop_frames, select_frames


class EvidenceRegionsTests(unittest.TestCase):
    def test_crop_includes_entire_submental_and_jaw_regions(self):
        rois = dict(anterior=[270, 300, 340, 430], submental=[250, 230, 360, 300],
                    jaw=[220, 180, 390, 260], left_lateral=[215, 270, 280, 430],
                    right_lateral=[340, 270, 405, 430])
        box = neck_context_box(rois, 640, 480)
        for roi in rois.values():
            self.assertLessEqual(box[0], roi[0]); self.assertLessEqual(box[1], roi[1])
            self.assertGreaterEqual(box[2], roi[2]); self.assertGreaterEqual(box[3], roi[3])
        self.assertLess(box[1], 230)

    def test_crop_is_clipped_and_missing_regions_are_not_invented(self):
        self.assertEqual(neck_context_box({'submental': [0, 0, 100, 100]}, 100, 100), [0, 0, 100, 100])
        self.assertIsNone(neck_context_box({}, 640, 480))
        self.assertIsNone(neck_context_box({'jaw': [20, 20, 40, 40]}, 640, 480))
        self.assertIsNone(neck_context_box({'submental': [0, 0, 0, 0]}, 640, 480))
        self.assertIsNone(neck_context_box({'submental': [0, 0, 0, 0], 'jaw': [20, 20, 40, 40]}, 640, 480))

    def test_held_behavior_has_crops_at_beginning_middle_and_end(self):
        rows = [dict(t=i/5, features={}) for i in range(100)]
        selected = select_frames(rows, 16)
        crops = select_crop_frames(rows, selected)
        self.assertIn(0, crops); self.assertIn(99, crops)
        self.assertTrue(any(40 <= i <= 60 for i in crops))
        self.assertLessEqual(len(crops), 5)
        self.assertEqual(select_crop_frames(rows, []), [])
        self.assertEqual(select_crop_frames(rows, selected, 0), [])

    def test_crop_includes_both_anterior_and_lateral_event_peaks(self):
        rows = [dict(t=i/5, features={}) for i in range(100)]
        rows[8]['features']['anterior_motion'] = 5
        rows[83]['features']['lateral_motion'] = 8
        selected = select_frames(rows, 16)
        crops = select_crop_frames(rows, selected)
        self.assertTrue({0, 8, 83, 99}.issubset(crops))
        self.assertTrue(set(crops).issubset(selected))

    def test_stable_video_unused_slots_do_not_cluster_in_first_second(self):
        rows = [dict(t=i/5, features={}) for i in range(100)]
        selected = select_frames(rows, 16)
        self.assertLessEqual(sum(i < 10 for i in selected), 3)
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected, select_frames(rows, 16))


if __name__ == '__main__': unittest.main()
