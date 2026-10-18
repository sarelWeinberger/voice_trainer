import json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from head_motion import parse_trace,analyze_trace,analyze_payload

def trace(hz=4.4,fps=30,camera=False):
 rows=[]
 for t in np.arange(0,6,1/fps):
  v=.003*np.sin(2*np.pi*hz*t)
  rows.append([float(t),0 if camera else float(v),0,float(v) if camera else 0,0,1])
 return rows
class Tests(unittest.TestCase):
 def test_oscillation_frequency_and_camera_control(self):
  result=analyze_trace(trace());self.assertEqual(result['status'],'measured')
  fast=[w for w in result['windows'] if w['end']-w['start']<3]
  self.assertTrue(all(abs(w['dominant_frequency_hz']-4.4)<.6 for w in fast))
  self.assertTrue(all(w['amplitude_percent_image_width']>.1 for w in fast))
  camera=analyze_trace(trace(camera=True))
  self.assertTrue(all(w['amplitude_percent_image_width']==0 for w in camera['windows']))
  self.assertTrue(all(w['dominant_frequency_hz'] is None for w in camera['windows']))
  self.assertIsNone(result['effort_score'])
 def test_normal_slow_movement_is_not_fast_oscillation(self):
  result=analyze_trace(trace(hz=.5))
  short=[w for w in result['windows'] if w['end']-w['start']<3]
  self.assertTrue(all(w['dominant_frequency_hz'] is None for w in short))
  slow=[w for w in result['windows'] if w['end']-w['start']>=6]
  self.assertTrue(slow);self.assertAlmostEqual(slow[0]['dominant_frequency_hz'],.5,places=1)
 def test_sparse_or_missing_tracking_stays_unavailable(self):
  self.assertEqual(analyze_trace(trace(fps=5))['status'],'insufficient_data')
  rows=trace()
  for row in rows:row[5]=0
  result=analyze_trace(rows);self.assertEqual(result['status'],'insufficient_data')
  self.assertTrue(all(w['amplitude_percent_image_width'] is None for w in result['windows']))
 def test_gaps_cannot_be_interpolated_into_a_measurement(self):
  rows=[r for i,r in enumerate(trace()) if i%10<4]
  self.assertEqual(analyze_trace(rows)['status'],'insufficient_data')
 def test_isolated_camera_frame_gaps_use_observed_times(self):
  rows=[r for i,r in enumerate(trace()) if not (21<=i%60<=24)]
  result=analyze_trace(rows)
  self.assertEqual(result['status'],'measured')
  self.assertTrue(all(w['status']=='measured' for w in result['windows']))
  self.assertTrue(all(abs(w['dominant_frequency_hz']-4.4)<.1 for w in result['windows'] if w['end']-w['start']<3))
  self.assertTrue(all(w['method']=='observed_timestamp_periodicity_check' for w in result['windows']))
 def test_validation_rejects_nonfinite_unordered_and_oversize(self):
  def payload(rows):return json.dumps(dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=rows))
  self.assertEqual(len(parse_trace(payload(trace()))),180)
  for rows in [[[0,0,0,0,0,float('nan')]],[[1,0,0,0,0,1],[.5,0,0,0,0,1]],[[26,0,0,0,0,1]],[[0,0,0,0,0,True]],[[0,0,0,0,0,1]]*901]:
   with self.assertRaises(ValueError):parse_trace(payload(rows))
 def test_acquisition_failure_is_preserved_without_untrusted_diagnostic_text(self):
  payload=dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=[],capture_reason='face_reference_unavailable')
  self.assertEqual(analyze_payload(json.dumps(payload))['capture_reason'],'face_reference_unavailable')
  payload['capture_reason']='arbitrary client text'
  self.assertNotIn('capture_reason',analyze_payload(json.dumps(payload)))
 def test_tracking_diagnostics_are_bounded_and_persisted(self):
  payload=dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=trace(),
   tracking_diagnostics=dict(head_missing_samples=40,background_missing_samples=9999,remaining_head_points=2,remaining_background_points=True))
  result=analyze_payload(json.dumps(payload))
  self.assertEqual(result['tracking_diagnostics'],dict(head_missing_samples=40,remaining_head_points=2))
 def test_reinitialized_tracks_are_not_fitted_across_a_position_jump(self):
  rows=[r+[0 if i<45 else 1] for i,r in enumerate(trace())]
  for r in rows[45:]:r[1]+=1
  payload=json.dumps(dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=rows))
  report=analyze_payload(payload)
  self.assertEqual([w['reason'] for w in report['windows'][:2]],['tracking_reinitialized']*2)
  later=[w for w in report['windows'] if w['start']>=2]
  self.assertTrue(all(w['status']=='measured' for w in later))
  self.assertTrue(all(w['amplitude_percent_image_width']<.3 for w in later))
  rows[50][6]=0
  with self.assertRaises(ValueError):parse_trace(json.dumps(dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=rows)))
 def test_random_and_correlated_noise_do_not_receive_a_frequency(self):
  for seed in range(12):
   rng=np.random.default_rng(seed);noise=rng.normal(0,.0004,(180,2))
   for correlated in (False,True):
    values=noise.copy()
    if correlated:
     for i in range(1,len(values)):values[i]=.65*values[i-1]+values[i]
    rows=[[i/30,float(v[0]),float(v[1]),0,0,1] for i,v in enumerate(values)]
    self.assertTrue(all(w['dominant_frequency_hz'] is None for w in analyze_trace(rows)['windows']))
 def test_high_frequency_when_sampling_supports_it(self):
  report=analyze_trace(trace(hz=10,fps=30))
  self.assertTrue(all(abs(w['dominant_frequency_hz']-10)<.1 for w in report['windows'] if w['end']-w['start']<3))
 def test_fast_oscillation_survives_larger_slow_posture_motion(self):
  rows=trace()
  for row in rows:row[1]+=.02*np.sin(2*np.pi*.5*row[0])
  fast=[w for w in analyze_trace(rows)['windows'] if w['end']-w['start']<3]
  self.assertTrue(all(w['periodicity_status']=='supported' for w in fast))
  self.assertTrue(all(abs(w['dominant_frequency_hz']-4.4)<.1 for w in fast))
 def test_background_dominance_is_not_published_as_head_periodicity(self):
  rows=trace()
  for row in rows:row[3]=row[1]*4
  fast=[w for w in analyze_trace(rows)['windows'] if w['end']-w['start']<3]
  self.assertTrue(all(w['periodicity_status']=='camera_confounded' for w in fast))
  self.assertTrue(all(w['dominant_frequency_hz'] is None for w in fast))
 def test_slow_off_grid_motion_is_not_misreported_at_the_fast_band_edge(self):
  for hz in (.3,.6,.8):
   rows=[[float(t),float(.003*np.sin(2*np.pi*hz*t)),0,0,0,1] for t in np.arange(0,12,1/30)]
   windows=analyze_trace(rows)['windows']
   self.assertTrue(all(w['dominant_frequency_hz'] is None for w in windows if w['end']-w['start']<3))
   appropriate=[w for w in windows if w['end']-w['start']>=(10 if hz<.5 else 6) and w['periodicity_status']=='supported']
   self.assertTrue(appropriate)
   self.assertTrue(all(abs(w['dominant_frequency_hz']-hz)<.06 for w in appropriate))
if __name__=='__main__':unittest.main()
