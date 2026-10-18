import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_observable_change import compare_observable_windows
from neck_v1_cv import select_frames


def measured(t,jaw=.1):
    return dict(t=t,features={'jaw_opening':jaw,'head_pitch_deg':0},
        visibility={'face':1,'jaw':1},lighting=1,camera_stability=1)


class ObservableChangeTests(unittest.TestCase):
    def test_initial_reference_and_later_delta_without_effort_inference(self):
        rows=[measured(i/5,.1 if i<25 else .3) for i in range(85)]
        change=compare_observable_windows(rows)
        self.assertEqual(change['reference']['end'],5)
        self.assertFalse(change['reference']['assumed_relaxed'])
        self.assertAlmostEqual(change['windows'][0]['measurements']['jaw_opening']['delta_from_reference'],.2)
        self.assertIsNone(change['effort_score'])

    def test_short_recording_retains_reference_and_comparison(self):
        change=compare_observable_windows([measured(i/5) for i in range(25)])
        self.assertAlmostEqual(change['reference']['end'],1.667,places=3)
        self.assertGreater(len(change['windows']),0)
        self.assertEqual(change['windows'][0]['measurements']['jaw_opening']['delta_from_reference'],0)

    def test_missing_reference_and_untracked_features_never_become_zero(self):
        rows=[measured(i/5) for i in range(85)]
        for row in rows[:25]:row['visibility']['face']=0
        change=compare_observable_windows(rows)
        measurement=change['windows'][0]['measurements']['jaw_opening']
        self.assertIsNone(measurement['delta_from_reference'])
        self.assertEqual(measurement['reason'],'reference_not_assessable')
        self.assertIsNone(change['windows'][0]['measurements']['lateral_motion']['value'])

    def test_gaps_and_camera_motion_prevent_comparison(self):
        rows=[measured(i/5) for i in range(85) if i<25 or i%5==0]
        change=compare_observable_windows(rows)
        self.assertIsNone(change['windows'][0]['measurements']['jaw_opening']['value'])
        rows=[measured(i/5) for i in range(85)]
        for row in rows[25:]:row['camera_stability']=.1
        self.assertIsNone(compare_observable_windows(rows)['windows'][0]['measurements']['jaw_opening']['value'])

    def test_temporal_selection_cannot_be_crowded_out_by_early_extrema(self):
        rows=[measured(i/5) for i in range(100)]
        for key,i in zip(('anterior_motion','lateral_motion','head_motion','shoulder_motion'),range(1,5)):
            rows[i]['features'][key]=100
        selected=select_frames(rows,8)
        self.assertEqual(len(selected),8)
        self.assertIn(99,selected)
        self.assertGreaterEqual(sum(i>=40 for i in selected),3)
        self.assertEqual(select_frames(rows,0),[])
        self.assertEqual(select_frames(rows,1),[0])


if __name__=='__main__':unittest.main()
