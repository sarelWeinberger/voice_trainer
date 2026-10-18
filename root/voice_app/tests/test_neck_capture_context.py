"""Cross-modal evidence routing and truthful capture completeness."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_capture_context import parse_transport,transport_evidence,movement_evidence
from neck_effort_engine import NeckEffortEngine
from test_neck_effort_engine import FakeProcessor,prediction


def counters(**values):
    data=dict(version='neck-capture-transport-v1',captured_frames=30,acknowledged_frames=25,
        encoding_failed_frames=0,overflow_frames=0,retry_count=1,failed_batches=0,unconfirmed_frames=5)
    data.update(values);return data


def movement():
    return dict(version='head-movement-stability-v3',status='measured',sample_count=180,effective_fps=30,
        windows=[dict(start=start,end=start+2,status='measured',motion_rms_percent_image_width=.2+start*.1,
                      dominant_frequency_hz=None,periodicity_status='not_established',band_hz=[1.5,12],
                      tracking_coverage=1,temporal_coverage=1) for start in (0,1,2,3)])


class CaptureContextTests(unittest.TestCase):
    def test_counter_validation_and_lost_acknowledgement_reconciliation(self):
        parsed=parse_transport(json.dumps(counters()))
        self.assertTrue(transport_evidence(parsed,30)['complete'])
        self.assertEqual(transport_evidence(parsed,25)['missing_frames'],5)
        for value in (dict(counters(),acknowledged_frames=31),dict(counters(),retry_count=True),dict(counters(),patient='secret')):
            with self.assertRaises(ValueError):parse_transport(json.dumps(value))

    def test_same_band_comparison_preserves_irregular_and_missing_measurements(self):
        report=movement();report['windows'].append(dict(start=4,end=6,status='insufficient_data'))
        evidence=movement_evidence(report,0,5)
        comparison=evidence['comparisons'][0]
        self.assertEqual(comparison['measured_windows'],4)
        self.assertEqual(comparison['first']['start'],0)
        self.assertEqual(comparison['largest']['start'],3)
        self.assertIsNone(comparison['largest']['dominant_frequency_hz'])
        self.assertEqual(movement_evidence(report,10,15)['status'],'unavailable')

    def test_native_server_estimator_numbers_reach_context(self):
        from head_motion import analyze_payload
        from test_head_motion import trace
        report=analyze_payload(json.dumps(dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=trace())))
        self.assertEqual(movement_evidence(report,0,6)['status'],'measured')

    def test_movement_reaches_interpreter_before_grading_without_forcing_score(self):
        seen=[]
        def interpreter(context,images):
            seen.append(context)
            result=prediction((0,0,0,0,0))
            for item in result.dimensions.__dict__.values():item.frame_indices=context['selected_indices'][:2]
            return result
        engine=NeckEffortEngine(processor=FakeProcessor(),interpreter=interpreter)
        try:
            engine.start_recording_session('evidence')
            for i in range(30):engine.process_frame(np.ones((100,100,3),np.uint8)*100,i/5)
            result=engine.finalize(movement=movement(),capture_transport=counters())
            self.assertEqual(seen[0]['head_movement']['status'],'measured')
            self.assertTrue(seen[0]['diagnostics']['head_motion_context_supplied'])
            self.assertEqual(result['neckAnalysis']['dimensions']['posture_compensation']['score'],0)
            self.assertEqual(result['neckAnalysis']['movement_stability']['sample_count'],180)
            self.assertTrue(result['neckAnalysis']['diagnostics']['capture_transport']['complete'])
        finally:engine.close()

    def test_missing_images_stay_explicit_and_model_failure_retains_motion(self):
        from neck_v1_openai import NeckProviderError
        engine=NeckEffortEngine(processor=FakeProcessor(),interpreter=lambda *args:(_ for _ in ()).throw(NeckProviderError('failed')))
        try:
            engine.start_recording_session('partial')
            for i in range(25):engine.process_frame(np.ones((100,100,3),np.uint8)*100,i/5)
            result=engine.finalize(movement=movement(),capture_transport=counters())['neckAnalysis']
            self.assertEqual(result['status'],'error')
            self.assertIn('capture_frames_missing',result['quality']['warnings'])
            self.assertEqual(result['diagnostics']['capture_transport']['missing_frames'],5)
            self.assertEqual(result['movement_stability']['status'],'measured')
            self.assertTrue(all(d['score'] is None for d in result['dimensions'].values()))
        finally:engine.close()


if __name__=='__main__':unittest.main()
